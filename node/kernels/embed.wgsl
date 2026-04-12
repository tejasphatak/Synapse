// Token + positional embedding lookup for GPT-2
// For each position i in the sequence:
//   output[i] = token_embedding[token_ids[i]] + position_embedding[i]
//
// token_embedding: [vocab_size, hidden_size] = [50257, 768]
// position_embedding: [max_seq_len, hidden_size] = [1024, 768]
// output: [seq_len, hidden_size]

struct EmbedParams {
  seq_len: u32,
  hidden_size: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: EmbedParams;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read> token_emb: array<f32>;   // [vocab_size, hidden_size]
@group(0) @binding(3) var<storage, read> pos_emb: array<f32>;     // [max_seq_len, hidden_size]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [seq_len, hidden_size]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = params.seq_len * params.hidden_size;

  if (idx >= total) {
    return;
  }

  let pos = idx / params.hidden_size;   // sequence position
  let dim = idx % params.hidden_size;   // hidden dimension

  let token_id = token_ids[pos];

  // Lookup token embedding + positional embedding
  let tok_val = token_emb[token_id * params.hidden_size + dim];
  let pos_val = pos_emb[pos * params.hidden_size + dim];

  output[idx] = tok_val + pos_val;
}
