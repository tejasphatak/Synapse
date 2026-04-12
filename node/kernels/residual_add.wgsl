// Element-wise residual addition: output = a + b
// Used for residual connections in transformer blocks:
//   h = h + Attention(LayerNorm(h))
//   h = h + MLP(LayerNorm(h))

struct Params {
  total_elements: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= params.total_elements) {
    return;
  }

  output[idx] = a[idx] + b[idx];
}
