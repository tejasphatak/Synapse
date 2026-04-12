// Broadcast bias addition: matrix[row, col] += bias[col]
// matrix is [rows, cols], bias is [cols]

struct Params {
  rows: u32,
  cols: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> matrix: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = params.rows * params.cols;
  if (idx >= total) {
    return;
  }

  let col = idx % params.cols;
  matrix[idx] = matrix[idx] + bias[col];
}
