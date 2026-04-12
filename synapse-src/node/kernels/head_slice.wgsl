// Extract a single attention head's Q, K, or V slice from the combined QKV buffer.
// QKV layout: [seq_len, 3 * hidden_size]
// For Q: sectionOffset=0, K: sectionOffset=hidden, V: sectionOffset=2*hidden
// Head h: columns [sectionOffset + h*headDim : sectionOffset + (h+1)*headDim]

struct Params {
  seq_len: u32,
  qkv_cols: u32,        // 3 * hidden_size
  section_offset: u32,   // 0 for Q, hidden for K, 2*hidden for V
  head_offset: u32,      // headIdx * headDim
}

struct Params2 {
  head_dim: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> params2: Params2;
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;  // [seq_len, head_dim]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = params.seq_len * params2.head_dim;
  if (idx >= total) {
    return;
  }

  let s = idx / params2.head_dim;    // sequence position
  let d = idx % params2.head_dim;    // dimension within head

  let src_idx = s * params.qkv_cols + params.section_offset + params.head_offset + d;
  output[idx] = qkv[src_idx];
}
