// Gemma token-embedding gather + scale.
//
// Gemma 3 embeds tokens and then multiplies by sqrt(hidden_size):
//   out[i, j] = embed_table[token_ids[i], j] * sqrt(hidden_size)
//
// No positional embedding — RoPE handles that later. GPT-2's embed.wgsl
// does token + positional, so we keep this kernel separate for clarity.
//
// Writing the gather on-GPU avoids a CPU readback of the full embedding
// matrix (262k * 1152 * 4B = 1.2GB for Gemma 3 1B), which exceeds the
// WebGPU maxBufferSize limit on most devices.
//
// Dispatch: one thread per output element. workgroup_size(256) matches
// our other kernels.

struct Params {
  seq_len:     u32,
  hidden_size: u32,
  scale:       f32,
  _pad:        u32,
};

@group(0) @binding(0) var<uniform>             params:    Params;
@group(0) @binding(1) var<storage, read>       token_ids: array<u32>;    // [seq_len]
@group(0) @binding(2) var<storage, read>       embed_tbl: array<f32>;    // [vocab, hidden]
@group(0) @binding(3) var<storage, read_write> output:    array<f32>;    // [seq_len, hidden]

@compute @workgroup_size(256)
fn gemma_embed(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  let total = params.seq_len * params.hidden_size;
  if (idx >= total) { return; }

  let H = params.hidden_size;
  let pos = idx / H;
  let dim = idx % H;

  let tok = token_ids[pos];
  let src = tok * H + dim;
  output[idx] = embed_tbl[src] * params.scale;
}
