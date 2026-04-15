// Elementwise multiply — Hadamard product.
//
//   c[i] = a[i] * b[i]
//
// Needed for Gemma / Llama / Qwen gated MLP: the `gelu(gate) * up`
// fan-in step. Our existing kernels have residual_add (a+b) but no
// corresponding multiply. Trivial kernel, worth having as a named
// primitive for readability downstream.
//
// Usage:
//   inputs a, b: same shape, any 1D-flattened tensor
//   output c: same shape, can alias with a (in-place: c == a)
//
// Optimizations:
//   - Workgroup size 256 (matches the rest of our kernels).
//   - Vectorized load optional — WGSL auto-coalesces well on modern
//     drivers for array<f32> sequential access; no need to manually
//     pack into vec4 unless we profile a hotspot.

struct Params {
  length: u32,
  _pad1:  u32,
  _pad2:  u32,
  _pad3:  u32,
};

@group(0) @binding(0) var<storage, read>       a:      array<f32>;
@group(0) @binding(1) var<storage, read>       b:      array<f32>;
@group(0) @binding(2) var<storage, read_write> c:      array<f32>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(256)
fn elementwise_mul(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let i = gid.x;
  if (i >= params.length) { return; }
  c[i] = a[i] * b[i];
}
