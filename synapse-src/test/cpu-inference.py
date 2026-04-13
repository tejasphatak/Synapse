"""
Minimal CPU-only GPT-2 inference using the Synapse shard files.
No PyTorch model — reads weights directly from shard binaries.
This verifies the weight layout matches what the WebGPU pipeline expects.
"""
import json, struct, math, sys
import numpy as np

SHARDS_DIR = "model/shards"

def load_manifest():
    with open(f"{SHARDS_DIR}/manifest.json") as f:
        return json.load(f)

def load_tensor(manifest, name):
    """Load a tensor from shard files, dequantize to float32."""
    entry = next(t for t in manifest["tensors"] if t["name"] == name)
    fname = f"{SHARDS_DIR}/{entry['file']}"

    with open(fname, "rb") as f:
        f.seek(entry["offset"])
        raw = f.read(entry["size"])

    if entry["dtype"] == "float16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif entry["dtype"] == "float32":
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {entry['dtype']}")

    return arr.reshape(entry["shape"])

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def attention(q, k, v, head_dim):
    """Single-head attention with causal mask."""
    scores = q @ k.T / math.sqrt(head_dim)
    seq_len = scores.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ v

def forward_layer(hidden, layer_idx, manifest):
    """Run one transformer layer."""
    prefix = f"transformer.h.{layer_idx}"
    hidden_size = manifest["hidden_size"]
    num_heads = manifest["num_heads"]
    head_dim = manifest["head_dim"]

    # Pre-attention LayerNorm
    ln1_w = load_tensor(manifest, f"{prefix}.ln_1.weight")
    ln1_b = load_tensor(manifest, f"{prefix}.ln_1.bias")
    ln1_out = layer_norm(hidden, ln1_w, ln1_b)

    # QKV projection
    c_attn_w = load_tensor(manifest, f"{prefix}.attn.c_attn.weight")  # [768, 2304]
    c_attn_b = load_tensor(manifest, f"{prefix}.attn.c_attn.bias")    # [2304]
    qkv = ln1_out @ c_attn_w + c_attn_b  # [seq, 2304]

    q = qkv[:, :hidden_size]
    k = qkv[:, hidden_size:2*hidden_size]
    v = qkv[:, 2*hidden_size:]

    # Multi-head attention
    attn_out = np.zeros_like(q)
    for h in range(num_heads):
        s, e = h*head_dim, (h+1)*head_dim
        attn_out[:, s:e] = attention(q[:, s:e], k[:, s:e], v[:, s:e], head_dim)

    # Output projection
    c_proj_w = load_tensor(manifest, f"{prefix}.attn.c_proj.weight")  # [768, 768]
    c_proj_b = load_tensor(manifest, f"{prefix}.attn.c_proj.bias")
    proj_out = attn_out @ c_proj_w + c_proj_b

    # Residual
    hidden = hidden + proj_out

    # Pre-FFN LayerNorm
    ln2_w = load_tensor(manifest, f"{prefix}.ln_2.weight")
    ln2_b = load_tensor(manifest, f"{prefix}.ln_2.bias")
    ln2_out = layer_norm(hidden, ln2_w, ln2_b)

    # FFN
    fc_w = load_tensor(manifest, f"{prefix}.mlp.c_fc.weight")  # [768, 3072]
    fc_b = load_tensor(manifest, f"{prefix}.mlp.c_fc.bias")
    fc_out = gelu(ln2_out @ fc_w + fc_b)

    proj_w = load_tensor(manifest, f"{prefix}.mlp.c_proj.weight")  # [3072, 768]
    proj_b = load_tensor(manifest, f"{prefix}.mlp.c_proj.bias")
    ffn_out = fc_out @ proj_w + proj_b

    # Residual
    hidden = hidden + ffn_out
    return hidden

def main():
    manifest = load_manifest()
    print(f"Model: {manifest['model']}, layers: {manifest['num_layers']}, hidden: {manifest['hidden_size']}")

    # Token IDs for "Hello Hello Synapse!"
    token_ids = [15496, 18435, 16065, 7512, 0]
    seq_len = len(token_ids)

    # Embedding
    wte = load_tensor(manifest, "transformer.wte.weight")  # [50257, 768]
    wpe = load_tensor(manifest, "transformer.wpe.weight")  # [1024, 768]

    hidden = wte[token_ids] + wpe[:seq_len]
    print(f"After embedding: shape={hidden.shape}, mean={hidden.mean():.4f}, std={hidden.std():.4f}")
    print(f"  first 5 values: {hidden[0,:5]}")

    # Run all 12 layers
    for l in range(manifest["num_layers"]):
        hidden = forward_layer(hidden, l, manifest)
        print(f"After layer {l}: mean={hidden.mean():.4f}, std={hidden.std():.4f}, first5={hidden[0,:5]}")

    # Final LayerNorm
    ln_f_w = load_tensor(manifest, "transformer.ln_f.weight")
    ln_f_b = load_tensor(manifest, "transformer.ln_f.bias")
    hidden = layer_norm(hidden, ln_f_w, ln_f_b)

    # LM head: hidden @ wte.T (weight tying) or lm_head.weight.T
    lm_head = load_tensor(manifest, "lm_head.weight")  # [50257, 768]
    logits = hidden @ lm_head.T  # [seq, 50257]

    # Sample from last position
    last_logits = logits[-1]
    top5_idx = np.argsort(last_logits)[-5:][::-1]
    print(f"\nTop 5 next tokens:")
    for idx in top5_idx:
        print(f"  logit={last_logits[idx]:.2f} token_id={idx}")

    # Temperature sampling
    probs = softmax(last_logits / 0.8)
    sampled = np.random.choice(len(probs), p=probs)
    print(f"\nSampled token: {sampled}")

if __name__ == "__main__":
    main()
