// RMSNorm — Root Mean Square Layer Normalization.
//
// Used by Gemma 3 / 4, Llama, Qwen, Mistral, and most post-2023 LLMs.
// Simpler than LayerNorm: no mean subtraction, no beta bias.
//
//   y[i] = x[i] / sqrt(mean(x[i]^2) + eps) * gamma[i]
//
// Equivalent numpy:
//   rms = sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
//   y = x / rms * gamma
//
// Optimizations over a naive port:
//
//   1. Two-pass workgroup reduction (like our existing layernorm.wgsl),
//      with the second pass fused into the normalize+scale step so we
//      only read x twice total instead of three times.
//
//   2. FP32 accumulation of the sum-of-squares even when x is bfloat16
//      at rest. Summing 1000+ squared values in FP16 would overflow;
//      FP32 accum is mandatory for numerical parity with Gemma's
//      reference implementation.
//
//   3. workgroup_size(256) — matches our layernorm.wgsl. Each thread
//      handles (hidden_size / 256) elements in stride=256 chunks. At
//      hidden=1152 (Gemma 3 1B) that's ~5 elements per thread.
//
//   4. Shared workgroup memory for the reduction — one atomic-free
//      barrier instead of N round-trips to global memory.
//
// Invariants assumed by caller:
//   - hidden_size is a multiple of workgroup_size (256). Gemma uses
//     1152 (1B) and 1536 (E2B) which are not — so the tail-handling
//     loop below uses a bounds check.
//   - gamma buffer length == hidden_size.
//   - eps is typically 1e-6 for Gemma (smaller than GPT-2's 1e-5).

struct Params {
  seq_len:     u32,
  hidden_size: u32,
  eps:         f32,
  _pad:        f32,  // 16-byte align
};

@group(0) @binding(0) var<storage, read>       input:   array<f32>;
@group(0) @binding(1) var<storage, read>       gamma:   array<f32>;
@group(0) @binding(2) var<storage, read_write> output:  array<f32>;
@group(0) @binding(3) var<uniform>             params:  Params;

var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_rrms: f32;

@compute @workgroup_size(256)
fn rmsnorm(
  @builtin(workgroup_id) group_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = group_id.x;
  if (row >= params.seq_len) { return; }

  let tid = local_id.x;
  let H = params.hidden_size;
  let base = row * H;

  // Pass 1: partial sum-of-squares in FP32 accumulation.
  var local_sq: f32 = 0.0;
  var i = tid;
  loop {
    if (i >= H) { break; }
    let v = input[base + i];
    local_sq = local_sq + v * v;
    i = i + 256u;
  }
  shared_sum[tid] = local_sq;
  workgroupBarrier();

  // Tree reduction across workgroup. Log2(256) = 8 steps.
  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  // Thread 0 finalizes the reciprocal-rms and broadcasts via shared mem.
  if (tid == 0u) {
    let mean_sq = shared_sum[0] / f32(H);
    shared_rrms = 1.0 / sqrt(mean_sq + params.eps);
  }
  workgroupBarrier();

  // Pass 2: normalize + scale by gamma, fused write.
  let rrms = shared_rrms;
  i = tid;
  loop {
    if (i >= H) { break; }
    output[base + i] = input[base + i] * rrms * gamma[i];
    i = i + 256u;
  }
}
