// Attention with Grouped-Query Attention (GQA) + optional sliding window.
//
// Generalizes our existing attention.wgsl for the Gemma/Llama/Qwen family.
//
// When num_kv_heads == num_q_heads: standard MHA (what our old attention did).
// When num_kv_heads  < num_q_heads: GQA — multiple Q heads share one K/V head.
// When num_kv_heads == 1:           MQA — all Q heads share a single K/V.
// When window_size  > 0 and < seq:  sliding-window causal mask.
// When window_size == 0 or >= seq:  regular causal mask (attend to all prior).
//
// Layout:
//
//   Q: [seq_len, num_q_heads,  head_dim]   row-major, contiguous
//   K: [seq_len, num_kv_heads, head_dim]
//   V: [seq_len, num_kv_heads, head_dim]
//   output: [seq_len, num_q_heads, head_dim]
//
// Q head h reads from kv head h / group_size where group_size = num_q_heads / num_kv_heads.
// So if num_q_heads=4 and num_kv_heads=1 (MQA), every Q head uses kv_head=0.
// If num_q_heads=8 and num_kv_heads=4 (GQA-2), heads 0,1 use kv=0, heads 2,3 use kv=1, etc.
//
// Three passes (matches attention.wgsl layout):
//
//   Pass 1 (qk_scores): compute Q·K^T / sqrt(head_dim), apply causal + sliding mask.
//   Pass 2 (softmax_rows): row-wise softmax over scores.
//   Pass 3 (attend): weighted sum of V rows by softmax probs.
//
// Each pass uses a bind group with the storage buffers it needs.
//
// NOTE: Gemma 3's pre-attention softcapping (attn_logit_softcapping) is NOT applied
// here — it's None for Gemma 3 1B per config. For Gemma 4 it's 50.0 and needs a
// fused tanh(scores / cap) * cap step before softmax. Add in a follow-up if needed.

struct Params1 {
  seq_len:       u32,
  num_q_heads:   u32,
  num_kv_heads:  u32,
  head_dim:      u32,
  window_size:   u32,  // 0 = no sliding window (pure causal)
  inv_sqrt_scale: f32, // 1/sqrt(query_pre_attn_scalar). Caller pre-computes.
  _pad2:         u32,
  _pad3:         u32,
};

@group(0) @binding(0) var<storage, read>       q1:       array<f32>;
@group(0) @binding(1) var<storage, read>       k1:       array<f32>;
@group(0) @binding(2) var<storage, read_write> scores:   array<f32>;
@group(0) @binding(3) var<uniform>             params1:  Params1;

// Pass 1: compute Q·K^T / sqrt(head_dim) with causal + sliding window mask.
// Output shape: [num_q_heads, seq_len, seq_len].
// Dispatch: (seq_len, seq_len, num_q_heads).
@compute @workgroup_size(8, 8, 1)
fn qk_scores(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let i = gid.x;   // query position
  let j = gid.y;   // key position
  let h = gid.z;   // query head

  let S  = params1.seq_len;
  let HQ = params1.num_q_heads;
  let HK = params1.num_kv_heads;
  let D  = params1.head_dim;
  let W  = params1.window_size;

  if (i >= S || j >= S || h >= HQ) { return; }

  let group_size = HQ / HK;
  let kv_h       = h / group_size;

  // Causal mask: can't attend to future.
  // Sliding window: also can't attend to positions > W tokens back.
  var masked = j > i;
  if (W > 0u) {
    if (i > j && (i - j) > W) { masked = true; }
  }

  let scores_idx = h * S * S + i * S + j;
  if (masked) {
    scores[scores_idx] = -1e4;  // FP16-safe large negative (same as attention.wgsl fix)
    return;
  }

  // Dot product: Q[i, h, :] · K[j, kv_h, :]
  var dot: f32 = 0.0;
  let q_base = i * HQ * D + h    * D;
  let k_base = j * HK * D + kv_h * D;
  for (var d: u32 = 0u; d < D; d = d + 1u) {
    dot = dot + q1[q_base + d] * k1[k_base + d];
  }
  scores[scores_idx] = dot * params1.inv_sqrt_scale;
}

// Pass 2: row-wise softmax. Same shape semantics as attention.wgsl.

struct Params2 {
  seq_len: u32,
  _pad1:   u32,
  _pad2:   u32,
  _pad3:   u32,
};

@group(1) @binding(0) var<storage, read_write> scores2: array<f32>;
@group(1) @binding(1) var<uniform>             params2: Params2;

var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(256)
fn softmax_rows(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  // group_id.x = row index; group_id.y = head index (passed via workgroups dispatch)
  let row = group_id.x;
  let h   = group_id.y;
  let S   = params2.seq_len;
  if (row >= S) { return; }
  let base = h * S * S + row * S;
  let tid = local_id.x;

  // Step 1: find max
  if (tid == 0u) {
    var m: f32 = -1e30;
    for (var c: u32 = 0u; c < S; c = c + 1u) {
      let v = scores2[base + c];
      if (v > m) { m = v; }
    }
    shared_max = m;
  }
  workgroupBarrier();

  // Step 2: exp(x - max), summed
  var col = tid;
  loop {
    if (col >= S) { break; }
    scores2[base + col] = exp(scores2[base + col] - shared_max);
    col = col + 256u;
  }
  workgroupBarrier();

  if (tid == 0u) {
    var s: f32 = 0.0;
    for (var c: u32 = 0u; c < S; c = c + 1u) {
      s = s + scores2[base + c];
    }
    shared_sum = s;
  }
  workgroupBarrier();

  // Step 3: divide by sum
  col = tid;
  let inv_sum = 1.0 / max(shared_sum, 1e-20);
  loop {
    if (col >= S) { break; }
    scores2[base + col] = scores2[base + col] * inv_sum;
    col = col + 256u;
  }
}

// Pass 3: weighted sum of V by softmax probabilities.

struct Params3 {
  seq_len:      u32,
  num_q_heads:  u32,
  num_kv_heads: u32,
  head_dim:     u32,
};

@group(2) @binding(0) var<storage, read>       probs:   array<f32>;
@group(2) @binding(1) var<storage, read>       v3:      array<f32>;
@group(2) @binding(2) var<storage, read_write> output:  array<f32>;
@group(2) @binding(3) var<uniform>             params3: Params3;

@compute @workgroup_size(8, 8, 1)
fn attend(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let i = gid.x;   // query position
  let d = gid.y;   // output dim within head
  let h = gid.z;   // query head

  let S  = params3.seq_len;
  let HQ = params3.num_q_heads;
  let HK = params3.num_kv_heads;
  let D  = params3.head_dim;

  if (i >= S || d >= D || h >= HQ) { return; }

  let group_size = HQ / HK;
  let kv_h = h / group_size;

  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < S; j = j + 1u) {
    let p = probs[h * S * S + i * S + j];
    let v = v3[j * HK * D + kv_h * D + d];
    sum = sum + p * v;
  }

  output[i * HQ * D + h * D + d] = sum;
}
