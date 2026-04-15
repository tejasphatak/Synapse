// RoPE — Rotary Positional Embedding.
//
// Used by Gemma, Llama, Qwen, Mistral, etc. Rotates pairs of dimensions
// in Q and K by an angle that depends on (position, dim_pair_index).
// Encodes position information multiplicatively instead of adding
// learned positional embeddings.
//
// Math, per head, per position p, per dim pair (2i, 2i+1):
//
//     θ_i(p) = p · base ^ (-2i / head_dim)
//     (x_{2i}, x_{2i+1}) → ( x_{2i} cos θ − x_{2i+1} sin θ,
//                             x_{2i} sin θ + x_{2i+1} cos θ )
//
// Gemma uses base = 10000 for lower layers and a scaled base for the
// sliding-window attention; we read the precomputed cos/sin cache from
// a buffer so layer-specific bases are baked in caller-side.
//
// Layout conventions matching our existing attention kernels:
//
//   input shape:  [seq_len, num_heads, head_dim]  (row-major, contig)
//   cos_cache:    [max_pos, head_dim / 2]          ( precomputed per-position cos values for each dim pair )
//   sin_cache:    [max_pos, head_dim / 2]          ( same for sin )
//
// The cos/sin caches are computed once (CPU or GPU) at model init and
// reused across all forward passes. head_dim/2 entries per position
// because each entry applies to a pair.
//
// Gemma 3's rotation is the ROTATE_HALF variant (HF default):
//   out[:, i]        = x[:, i]        * cos - x[:, i + d/2] * sin
//   out[:, i + d/2]  = x[:, i + d/2]  * cos + x[:, i]       * sin
// for i in [0, d/2). Proven by CPU-vs-HF parity (research/gemma_parity).
//
// Optimizations:
//   - Each thread handles one (head, pair) for one position. Workgroup
//     size 256 processes positions in tiles.
//   - Two global reads (x_even, x_odd), one cos, one sin, two writes.
//     Minimum memory traffic.
//   - cos/sin precomputed → no trig on GPU.
//   - Apply in-place: input buffer == output buffer is supported.

struct Params {
  seq_len:   u32,
  num_heads: u32,
  head_dim:  u32,     // must be even
  start_pos: u32,     // position offset for cached decode
};

@group(0) @binding(0) var<storage, read_write> x:         array<f32>;
@group(0) @binding(1) var<storage, read>       cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read>       sin_cache: array<f32>;
@group(0) @binding(3) var<uniform>             params:    Params;

@compute @workgroup_size(256)
fn rope(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let pair_idx = gid.x;   // 0 .. seq*heads*(head_dim/2)
  let S = params.seq_len;
  let H = params.num_heads;
  let D = params.head_dim;
  let D2 = D / 2u;

  let total = S * H * D2;
  if (pair_idx >= total) { return; }

  // Decode (position, head, pair) from linear index
  let pos_in_seq = pair_idx / (H * D2);
  let rem1       = pair_idx % (H * D2);
  let head       = rem1 / D2;
  let pair       = rem1 % D2;

  let abs_pos = pos_in_seq + params.start_pos;

  // Fetch cos/sin for this (abs_pos, pair)
  let c = cos_cache[abs_pos * D2 + pair];
  let s = sin_cache[abs_pos * D2 + pair];

  // rotate_half: element `pair` pairs with element `pair + D/2`.
  let head_base = pos_in_seq * H * D + head * D;
  let x_lo = x[head_base + pair];
  let x_hi = x[head_base + pair + D2];

  x[head_base + pair]       = x_lo * c - x_hi * s;
  x[head_base + pair + D2]  = x_hi * c + x_lo * s;
}
