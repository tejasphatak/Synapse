// Scaled dot-product attention kernel for GPT-2
// Implements: softmax(Q·Kᵀ / √d_k) · V
//
// For GPT-2 small: head_dim = 64, num_heads = 12
// This kernel processes one attention head at a time.

struct AttentionParams {
  seq_len: u32,
  head_dim: u32,
  scale: f32,     // 1.0 / sqrt(head_dim)
  _pad: u32,
}

// ──────────────────────────────────────────────────────────────────
// Pass 1: Compute QKᵀ scores and write to scores buffer
// ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;       // [seq_len, head_dim]
@group(0) @binding(2) var<storage, read> K: array<f32>;       // [seq_len, head_dim]
@group(0) @binding(3) var<storage, read_write> scores: array<f32>; // [seq_len, seq_len]

@compute @workgroup_size(8, 8)
fn compute_scores(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let i = global_id.x; // query position
  let j = global_id.y; // key position

  if (i >= params.seq_len || j >= params.seq_len) {
    return;
  }

  // Causal mask: positions can only attend to earlier positions (and themselves)
  if (j > i) {
    scores[i * params.seq_len + j] = -1e9;
    return;
  }

  // Dot product: Q[i] · K[j]
  var dot: f32 = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
    dot = dot + Q[i * params.head_dim + d] * K[j * params.head_dim + d];
  }

  scores[i * params.seq_len + j] = dot * params.scale;
}

// ──────────────────────────────────────────────────────────────────
// Pass 2: Row-wise softmax over scores
// Each workgroup handles one row (one query position)
// ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> params2: AttentionParams;
@group(0) @binding(1) var<storage, read_write> scores2: array<f32>; // [seq_len, seq_len]

var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(256)
fn softmax_rows(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(num_workgroups) num_groups: vec3<u32>,
) {
  let row = group_id.x;
  if (row >= params2.seq_len) {
    return;
  }

  let tid = local_id.x;
  let base = row * params2.seq_len;

  // Step 1: Find max (for numerical stability)
  var local_max: f32 = -1e30;
  var col = tid;
  while (col < params2.seq_len) {
    local_max = max(local_max, scores2[base + col]);
    col = col + 256u;
  }

  // Simple reduction using workgroup atomics isn't available,
  // so we use the first thread for final reduction after barrier
  // For POC, we compute max in first thread
  workgroupBarrier();

  if (tid == 0u) {
    var row_max: f32 = -1e30;
    for (var c: u32 = 0u; c < params2.seq_len; c = c + 1u) {
      row_max = max(row_max, scores2[base + c]);
    }
    shared_max = row_max;
  }

  workgroupBarrier();

  // Step 2: Compute exp(x - max) and sum
  col = tid;
  while (col < params2.seq_len) {
    scores2[base + col] = exp(scores2[base + col] - shared_max);
    col = col + 256u;
  }

  workgroupBarrier();

  if (tid == 0u) {
    var row_sum: f32 = 0.0;
    for (var c: u32 = 0u; c < params2.seq_len; c = c + 1u) {
      row_sum = row_sum + scores2[base + c];
    }
    shared_sum = row_sum;
  }

  workgroupBarrier();

  // Step 3: Normalize
  col = tid;
  while (col < params2.seq_len) {
    scores2[base + col] = scores2[base + col] / shared_sum;
    col = col + 256u;
  }
}

// ──────────────────────────────────────────────────────────────────
// Pass 3: Weighted sum — output = softmax_scores · V
// ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> params3: AttentionParams;
@group(0) @binding(1) var<storage, read> attn_weights: array<f32>; // [seq_len, seq_len]
@group(0) @binding(2) var<storage, read> V: array<f32>;            // [seq_len, head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [seq_len, head_dim]

@compute @workgroup_size(8, 8)
fn weighted_sum(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let i = global_id.x; // sequence position
  let d = global_id.y; // head dimension

  if (i >= params3.seq_len || d >= params3.head_dim) {
    return;
  }

  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < params3.seq_len; j = j + 1u) {
    sum = sum + attn_weights[i * params3.seq_len + j] * V[j * params3.head_dim + d];
  }

  output[i * params3.head_dim + d] = sum;
}
