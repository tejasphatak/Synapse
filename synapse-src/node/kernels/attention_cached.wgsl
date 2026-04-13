// Cached attention kernel — single query against KV cache
//
// During autoregressive decoding, we only need to compute attention for
// the NEW token (Q is [1, head_dim]) against ALL cached keys/values
// (K_cache is [cache_len, head_dim], V_cache is [cache_len, head_dim]).
//
// This reduces attention from O(seq^2) to O(seq) per step.

struct CachedAttentionParams {
  cache_len: u32,    // number of valid positions in KV cache (including current)
  head_dim: u32,     // dimension per head (64 for GPT-2)
  scale: f32,        // 1.0 / sqrt(head_dim)
  _pad: u32,
}

// ──────────────────────────────────────────────────────────────────
// Pass 1: Compute Q·K_cache^T scores for single query
// Q: [1, head_dim], K_cache: [cache_len, head_dim] → scores: [1, cache_len]
// ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> params: CachedAttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;            // [1, head_dim]
@group(0) @binding(2) var<storage, read> K_cache: array<f32>;      // [cache_len, head_dim]
@group(0) @binding(3) var<storage, read_write> scores: array<f32>; // [cache_len]

@compute @workgroup_size(256)
fn compute_scores_cached(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let j = global_id.x; // key position in cache
  if (j >= params.cache_len) {
    return;
  }

  // No causal mask needed — all cached positions are valid for this query
  // (the cache only contains positions <= current position)

  var dot: f32 = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d = d + 1u) {
    dot = dot + Q[d] * K_cache[j * params.head_dim + d];
  }

  scores[j] = dot * params.scale;
}

// ──────────────────────────────────────────────────────────────────
// Pass 2: Softmax over single row [cache_len]
// Simpler than full attention — only one row to normalize
// ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> params2: CachedAttentionParams;
@group(0) @binding(1) var<storage, read_write> scores2: array<f32>; // [cache_len]

var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(256)
fn softmax_cached(
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let tid = local_id.x;

  // Step 1: Find max (thread 0 does full scan for correctness)
  workgroupBarrier();

  if (tid == 0u) {
    var row_max: f32 = -1e30;
    for (var c: u32 = 0u; c < params2.cache_len; c = c + 1u) {
      row_max = max(row_max, scores2[c]);
    }
    shared_max = row_max;
  }

  workgroupBarrier();

  // Step 2: Compute exp(x - max)
  var col = tid;
  while (col < params2.cache_len) {
    scores2[col] = exp(scores2[col] - shared_max);
    col = col + 256u;
  }

  workgroupBarrier();

  // Step 3: Sum (thread 0)
  if (tid == 0u) {
    var row_sum: f32 = 0.0;
    for (var c: u32 = 0u; c < params2.cache_len; c = c + 1u) {
      row_sum = row_sum + scores2[c];
    }
    shared_sum = row_sum;
  }

  workgroupBarrier();

  // Step 4: Normalize
  col = tid;
  while (col < params2.cache_len) {
    scores2[col] = scores2[col] / shared_sum;
    col = col + 256u;
  }
}

// ──────────────────────────────────────────────────────────────────
// Pass 3: Weighted sum — output = softmax_scores · V_cache
// scores: [cache_len], V_cache: [cache_len, head_dim] → output: [1, head_dim]
// ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> params3: CachedAttentionParams;
@group(0) @binding(1) var<storage, read> attn_weights: array<f32>; // [cache_len]
@group(0) @binding(2) var<storage, read> V_cache: array<f32>;      // [cache_len, head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [1, head_dim]

@compute @workgroup_size(256)
fn weighted_sum_cached(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let d = global_id.x; // head dimension index
  if (d >= params3.head_dim) {
    return;
  }

  var sum: f32 = 0.0;
  for (var j: u32 = 0u; j < params3.cache_len; j = j + 1u) {
    sum = sum + attn_weights[j] * V_cache[j * params3.head_dim + d];
  }

  output[d] = sum;
}
