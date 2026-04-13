// Matrix multiplication with transposed B: C = A × Bᵀ
// A is [M, K], B is [N, K] (stored row-major), C is [M, N]
// Used for lm_head where weight is [vocab_size, hidden_size]

struct Dimensions {
  M: u32,
  K: u32,
  N: u32,
  _pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE_SIZE: u32 = 8u;

var<workgroup> tileA: array<array<f32, 8>, 8>;
var<workgroup> tileB: array<array<f32, 8>, 8>;

@compute @workgroup_size(8, 8)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = global_id.x;
  let col = global_id.y;
  let localRow = local_id.x;
  let localCol = local_id.y;

  var sum: f32 = 0.0;

  let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    // Load tile of A [M, K]
    let aCol = t * TILE_SIZE + localCol;
    if (row < dims.M && aCol < dims.K) {
      tileA[localRow][localCol] = A[row * dims.K + aCol];
    } else {
      tileA[localRow][localCol] = 0.0;
    }

    // Load tile of Bᵀ — B is [N, K], so Bᵀ[k][n] = B[n * K + k]
    let bRow = t * TILE_SIZE + localRow;
    if (bRow < dims.K && col < dims.N) {
      tileB[localRow][localCol] = B[col * dims.K + bRow];
    } else {
      tileB[localRow][localCol] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
      sum = sum + tileA[localRow][k] * tileB[k][localCol];
    }

    workgroupBarrier();
  }

  if (row < dims.M && col < dims.N) {
    C[row * dims.N + col] = sum;
  }
}
