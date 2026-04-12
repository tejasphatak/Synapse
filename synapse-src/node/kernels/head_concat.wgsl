// Copy a single head's output [seq_len, head_dim] into the concatenated
// output buffer [seq_len, hidden_size] at the correct column offset.

struct Params {
  seq_len: u32,
  head_dim: u32,
  hidden_size: u32,
  head_offset: u32,     // headIdx * headDim
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> head_data: array<f32>;      // [seq_len, head_dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;   // [seq_len, hidden_size]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = params.seq_len * params.head_dim;
  if (idx >= total) {
    return;
  }

  let s = idx / params.head_dim;
  let d = idx % params.head_dim;

  let dst_idx = s * params.hidden_size + params.head_offset + d;
  output[dst_idx] = head_data[idx];
}
