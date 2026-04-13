"""
CPU simulation of the full distributed Synapse pipeline WITH quantization.
Compares output quality: exact float32 vs int8-quantized at shard boundaries.

This proves that int8 quantization between nodes doesn't degrade output quality.
"""
import json, math, sys
import numpy as np

SHARDS_DIR = "model/shards"

def load_manifest():
    with open(f"{SHARDS_DIR}/manifest.json") as f:
        return json.load(f)

def load_tensor(manifest, name):
    entry = next(t for t in manifest["tensors"] if t["name"] == name)
    with open(f"{SHARDS_DIR}/{entry['file']}", "rb") as f:
        f.seek(entry["offset"])
        raw = f.read(entry["size"])
    if entry["dtype"] == "float16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(entry["shape"])
    return np.frombuffer(raw, dtype=np.float32).reshape(entry["shape"])

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
    scores = q @ k.T / math.sqrt(head_dim)
    seq_len = scores.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    weights = softmax(scores + mask, axis=-1)
    return weights @ v

def forward_layer(hidden, layer_idx, manifest):
    prefix = f"transformer.h.{layer_idx}"
    hidden_size = manifest["hidden_size"]
    num_heads = manifest["num_heads"]
    head_dim = manifest["head_dim"]

    ln1_out = layer_norm(hidden,
        load_tensor(manifest, f"{prefix}.ln_1.weight"),
        load_tensor(manifest, f"{prefix}.ln_1.bias"))
    qkv = ln1_out @ load_tensor(manifest, f"{prefix}.attn.c_attn.weight") + \
          load_tensor(manifest, f"{prefix}.attn.c_attn.bias")

    q, k, v = qkv[:,:hidden_size], qkv[:,hidden_size:2*hidden_size], qkv[:,2*hidden_size:]
    attn_out = np.zeros_like(q)
    for h in range(num_heads):
        s, e = h*head_dim, (h+1)*head_dim
        attn_out[:,s:e] = attention(q[:,s:e], k[:,s:e], v[:,s:e], head_dim)

    proj_out = attn_out @ load_tensor(manifest, f"{prefix}.attn.c_proj.weight") + \
               load_tensor(manifest, f"{prefix}.attn.c_proj.bias")
    hidden = hidden + proj_out

    ln2_out = layer_norm(hidden,
        load_tensor(manifest, f"{prefix}.ln_2.weight"),
        load_tensor(manifest, f"{prefix}.ln_2.bias"))
    fc_out = gelu(ln2_out @ load_tensor(manifest, f"{prefix}.mlp.c_fc.weight") + \
                  load_tensor(manifest, f"{prefix}.mlp.c_fc.bias"))
    ffn_out = fc_out @ load_tensor(manifest, f"{prefix}.mlp.c_proj.weight") + \
              load_tensor(manifest, f"{prefix}.mlp.c_proj.bias")
    return hidden + ffn_out

def quantize_int8(tensor):
    """Per-tensor int8 quantization (current Synapse approach)."""
    abs_max = np.abs(tensor).max()
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int8), 0.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, scale

def dequantize_int8(quantized, scale):
    return quantized.astype(np.float32) * scale

def quantize_int8_perchannel(tensor):
    """Per-row (per-token) int8 quantization — better dynamic range."""
    # Each row gets its own scale factor
    abs_max = np.abs(tensor).max(axis=-1, keepdims=True)
    abs_max = np.maximum(abs_max, 1e-8)  # avoid division by zero
    scales = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scales), -127, 127).astype(np.int8)
    return quantized, scales.flatten()

def dequantize_int8_perchannel(quantized, scales):
    return quantized.astype(np.float32) * scales.reshape(-1, 1)

def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_pipeline(token_ids, manifest, quantize_at_layers=None, mode="per_tensor"):
    """Run full GPT-2 forward pass, optionally quantizing at shard boundaries."""
    seq_len = len(token_ids)
    wte = load_tensor(manifest, "transformer.wte.weight")
    wpe = load_tensor(manifest, "transformer.wpe.weight")
    hidden = wte[token_ids] + wpe[:seq_len]

    for l in range(manifest["num_layers"]):
        hidden = forward_layer(hidden, l, manifest)

        if quantize_at_layers and l in quantize_at_layers:
            if mode == "per_channel":
                q, scales = quantize_int8_perchannel(hidden)
                hidden = dequantize_int8_perchannel(q, scales)
            else:
                q, scale = quantize_int8(hidden)
                hidden = dequantize_int8(q, scale)

    ln_f_w = load_tensor(manifest, "transformer.ln_f.weight")
    ln_f_b = load_tensor(manifest, "transformer.ln_f.bias")
    hidden = layer_norm(hidden, ln_f_w, ln_f_b)
    lm_head = load_tensor(manifest, "lm_head.weight")
    logits = hidden @ lm_head.T
    return logits

def main():
    manifest = load_manifest()
    prompts = {
        "hello": [15496],                           # "Hello"
        "the capital": [464, 3139],                  # "The capital"
        "synapse": [15496, 18435, 16065, 7512, 0],  # "Hello Hello Synapse!"
    }

    # Shard boundary at layer 5 (node 0: layers 0-5, node 1: layers 6-11)
    shard_boundary = [5]

    print("=" * 70)
    print("QUANTIZATION QUALITY TEST: float32 vs int8 at shard boundary")
    print("=" * 70)

    for name, tokens in prompts.items():
        exact = run_pipeline(tokens, manifest, quantize_at_layers=None)
        quant = run_pipeline(tokens, manifest, quantize_at_layers=shard_boundary)

        exact_last = exact[-1]
        quant_last = quant[-1]

        cos = cosine_sim(exact_last, quant_last)
        top5_exact = np.argsort(exact_last)[-5:][::-1]
        top5_quant = np.argsort(quant_last)[-5:][::-1]
        top1_match = top5_exact[0] == top5_quant[0]
        top5_overlap = len(set(top5_exact) & set(top5_quant))

        print(f"\nPrompt: '{name}' ({len(tokens)} tokens)")
        print(f"  Logits cosine similarity: {cos:.6f}")
        print(f"  Top-1 match: {top1_match}")
        print(f"  Top-5 overlap: {top5_overlap}/5")
        print(f"  Max logit diff: {np.abs(exact_last - quant_last).max():.4f}")
        print(f"  Exact  top-5: {list(top5_exact)}")
        print(f"  Quant  top-5: {list(top5_quant)}")

    # Per-channel quantization
    print(f"\n{'=' * 70}")
    print("PER-CHANNEL INT8 at shard boundary (layer 5)")
    print("=" * 70)
    for name, tokens in prompts.items():
        exact = run_pipeline(tokens, manifest)
        quant = run_pipeline(tokens, manifest, quantize_at_layers=shard_boundary, mode="per_channel")
        cos = cosine_sim(exact[-1], quant[-1])
        top5_exact = np.argsort(exact[-1])[-5:][::-1]
        top5_quant = np.argsort(quant[-1])[-5:][::-1]
        top1_match = top5_exact[0] == top5_quant[0]
        top5_overlap = len(set(top5_exact) & set(top5_quant))
        max_diff = np.abs(exact[-1] - quant[-1]).max()
        print(f"  '{name}': cosine={cos:.6f}, top-1={top1_match}, top-5={top5_overlap}/5, max_diff={max_diff:.4f}")

    # Multi-hop test: every layer
    print(f"\n{'=' * 70}")
    print("WORST CASE: per-tensor int8 at EVERY layer")
    print("=" * 70)
    all_boundaries = list(range(12))
    for name, tokens in prompts.items():
        exact = run_pipeline(tokens, manifest)
        quant_pt = run_pipeline(tokens, manifest, quantize_at_layers=all_boundaries, mode="per_tensor")
        quant_pc = run_pipeline(tokens, manifest, quantize_at_layers=all_boundaries, mode="per_channel")
        cos_pt = cosine_sim(exact[-1], quant_pt[-1])
        cos_pc = cosine_sim(exact[-1], quant_pc[-1])
        t1_pt = np.argsort(exact[-1])[-1] == np.argsort(quant_pt[-1])[-1]
        t1_pc = np.argsort(exact[-1])[-1] == np.argsort(quant_pc[-1])[-1]
        print(f"  '{name}': per_tensor cos={cos_pt:.6f} top1={t1_pt} | per_channel cos={cos_pc:.6f} top1={t1_pc}")

if __name__ == "__main__":
    main()
