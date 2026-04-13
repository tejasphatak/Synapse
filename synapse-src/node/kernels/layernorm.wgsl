// Layer normalization kernel for GPT-2
// Implements: y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Uses two-pass algorithm for numerical stability:
//   Pass 1: Compute mean
//   Pass 2: Compute variance using mean, then normalize
//
// Input:  x     [seq_len, hidden_size]
// Params: gamma  [hidden_size], beta [hidden_size]
// Output: out   [seq_len, hidden_size]

struct LayerNormParams {
  seq_len: u32,
  hidden_size: u32,
  eps: f32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> params: LayerNormParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Workgroup shared memory — MUST be at module scope
var<workgroup> partial_sums: array<f32, 256>;
var<workgroup> row_mean: f32;
var<workgroup> row_inv_std: f32;

// Each workgroup processes one row (one token position).
// Workgroup size = 256 threads to handle hidden_size up to 768+.

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = group_id.x;
  if (row >= params.seq_len) {
    return;
  }

  let tid = local_id.x;
  let base = row * params.hidden_size;

  // ── Pass 1: Compute mean ──────────────────────────────────────

  var local_sum: f32 = 0.0;
  var idx = tid;
  while (idx < params.hidden_size) {
    local_sum = local_sum + input[base + idx];
    idx = idx + 256u;
  }

  // Workgroup reduction for sum
  partial_sums[tid] = local_sum;

  workgroupBarrier();

  // Tree reduction
  var stride: u32 = 128u;
  while (stride > 0u) {
    if (tid < stride) {
      partial_sums[tid] = partial_sums[tid] + partial_sums[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid == 0u) {
    row_mean = partial_sums[0] / f32(params.hidden_size);
  }

  workgroupBarrier();

  // ── Pass 2: Compute variance ──────────────────────────────────

  var local_var_sum: f32 = 0.0;
  idx = tid;
  while (idx < params.hidden_size) {
    let diff = input[base + idx] - row_mean;
    local_var_sum = local_var_sum + diff * diff;
    idx = idx + 256u;
  }

  partial_sums[tid] = local_var_sum;

  workgroupBarrier();

  stride = 128u;
  while (stride > 0u) {
    if (tid < stride) {
      partial_sums[tid] = partial_sums[tid] + partial_sums[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid == 0u) {
    let variance = partial_sums[0] / f32(params.hidden_size);
    row_inv_std = 1.0 / sqrt(variance + params.eps);
  }

  workgroupBarrier();

  // ── Normalize + scale + shift ─────────────────────────────────

  idx = tid;
  while (idx < params.hidden_size) {
    let normalized = (input[base + idx] - row_mean) * row_inv_std;
    output[base + idx] = gamma[idx] * normalized + beta[idx];
    idx = idx + 256u;
  }
}
