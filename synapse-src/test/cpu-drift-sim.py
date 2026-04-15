"""
Local drift simulator: runs GPT-2 117M inference the same way the distributed
pipeline does, with INT8 quantization injected at each simulated shard boundary.
Compares output vs pure-FP32 baseline. Delta is exactly the wire-quantization
drift our 6-shard pipeline experiences.

Run:
  python3 test/cpu-drift-sim.py [--shards N]

Shard count defaults to 6 (matches current live coord config).
"""
import json, struct, sys, argparse
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
    if entry["dtype"] == "float32":
        return np.frombuffer(raw, dtype=np.float32).reshape(entry["shape"])
    raise ValueError(entry["dtype"])


def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def gelu(x):
    # GPT-2's approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(hidden, W_qkv, b_qkv, W_proj, b_proj, num_heads=12):
    seq, d = hidden.shape
    head_dim = d // num_heads
    qkv = hidden @ W_qkv + b_qkv  # [seq, 3*d]
    q, k, v = qkv[:, :d], qkv[:, d:2*d], qkv[:, 2*d:]
    q = q.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)
    mask = np.triu(np.full((seq, seq), -1e9), k=1)
    scores += mask
    attn = softmax(scores) @ v  # [heads, seq, head_dim]
    out = attn.transpose(1, 0, 2).reshape(seq, d)
    return out @ W_proj + b_proj


def forward_layer(hidden, l, manifest):
    p = f"transformer.h.{l}"
    ln1_w = load_tensor(manifest, f"{p}.ln_1.weight")
    ln1_b = load_tensor(manifest, f"{p}.ln_1.bias")
    attn_w = load_tensor(manifest, f"{p}.attn.c_attn.weight")
    attn_b = load_tensor(manifest, f"{p}.attn.c_attn.bias")
    proj_w = load_tensor(manifest, f"{p}.attn.c_proj.weight")
    proj_b = load_tensor(manifest, f"{p}.attn.c_proj.bias")
    ln2_w = load_tensor(manifest, f"{p}.ln_2.weight")
    ln2_b = load_tensor(manifest, f"{p}.ln_2.bias")
    fc_w = load_tensor(manifest, f"{p}.mlp.c_fc.weight")
    fc_b = load_tensor(manifest, f"{p}.mlp.c_fc.bias")
    mp_w = load_tensor(manifest, f"{p}.mlp.c_proj.weight")
    mp_b = load_tensor(manifest, f"{p}.mlp.c_proj.bias")

    h1 = layer_norm(hidden, ln1_w, ln1_b)
    a = attention(h1, attn_w, attn_b, proj_w, proj_b)
    hidden = hidden + a
    h2 = layer_norm(hidden, ln2_w, ln2_b)
    ff = gelu(h2 @ fc_w + fc_b) @ mp_w + mp_b
    return hidden + ff


def int8_roundtrip(x):
    """Match synapse-src/protocol/quantize.js int8 per-tensor quantization."""
    scale = np.max(np.abs(x)) / 127.0
    if scale == 0:
        return x.copy()
    q = np.round(x / scale).clip(-128, 127).astype(np.int8)
    return q.astype(np.float32) * scale


def run(token_ids, manifest, shard_boundaries=None):
    """shard_boundaries: set of layer indices AFTER which to inject INT8 roundtrip.
    None = pure FP32 baseline."""
    wte = load_tensor(manifest, "transformer.wte.weight")
    wpe = load_tensor(manifest, "transformer.wpe.weight")
    hidden = wte[token_ids] + wpe[:len(token_ids)]
    for l in range(manifest["num_layers"]):
        hidden = forward_layer(hidden, l, manifest)
        if shard_boundaries and l in shard_boundaries:
            hidden = int8_roundtrip(hidden)
    ln_f_w = load_tensor(manifest, "transformer.ln_f.weight")
    ln_f_b = load_tensor(manifest, "transformer.ln_f.bias")
    hidden = layer_norm(hidden, ln_f_w, ln_f_b)
    lm_head = load_tensor(manifest, "lm_head.weight")
    logits = hidden @ lm_head.T
    return hidden, logits


def compare_outputs(label, hidden_a, logits_a, hidden_b, logits_b, top_k=5):
    """Compare two runs' final hidden + top-K logits."""
    last_a, last_b = logits_a[-1], logits_b[-1]
    top_a = np.argsort(last_a)[-top_k:][::-1]
    top_b = np.argsort(last_b)[-top_k:][::-1]
    cos = (hidden_a[-1] @ hidden_b[-1]) / (np.linalg.norm(hidden_a[-1]) * np.linalg.norm(hidden_b[-1]) + 1e-9)
    print(f"\n=== {label} ===")
    print(f"hidden cosine(a, b): {cos:.4f}")
    print(f"hidden max |diff|:   {np.max(np.abs(hidden_a[-1] - hidden_b[-1])):.4f}")
    print(f"top5 baseline:       {list(top_a)}")
    print(f"top5 quantized:      {list(top_b)}")
    print(f"top1 match:          {top_a[0] == top_b[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=6, help="Number of shards (num_shards)")
    args = ap.parse_args()

    manifest = load_manifest()
    num_layers = manifest["num_layers"]
    N = args.shards
    layers_per_shard = num_layers // N
    # Boundaries are AFTER the last layer of each shard except the last shard
    boundaries = set((i + 1) * layers_per_shard - 1 for i in range(N - 1))
    print(f"Model: GPT-2 {num_layers}-layer, simulating {N}-shard pipeline")
    print(f"Shard boundaries (roundtrip after layer): {sorted(boundaries)}")

    token_ids = [464, 6881, 318, 5909, 290, 3716]  # "The universe is vast and complex"
    print(f"Prompt tokens: {token_ids}")

    # Baseline (pure FP32)
    h_base, l_base = run(token_ids, manifest, shard_boundaries=None)
    print(f"\nBaseline FP32 final hidden: mean={h_base[-1].mean():.4f} std={h_base[-1].std():.4f}")

    # Simulated N-shard with INT8 roundtrips
    h_sim, l_sim = run(token_ids, manifest, shard_boundaries=boundaries)
    print(f"Simulated INT8 final hidden: mean={h_sim[-1].mean():.4f} std={h_sim[-1].std():.4f}")

    compare_outputs(f"{N}-shard INT8 vs FP32 baseline", h_base, l_base, h_sim, l_sim)

    # Also test 2-shard and 3-shard for curve
    for n in [2, 3, 4, 6, 8, 12]:
        if n == N:
            continue
        lps = num_layers // n
        bnds = set((i + 1) * lps - 1 for i in range(n - 1))
        h, lg = run(token_ids, manifest, shard_boundaries=bnds)
        cos = (h_base[-1] @ h[-1]) / (np.linalg.norm(h_base[-1]) * np.linalg.norm(h[-1]) + 1e-9)
        top1_base = np.argmax(l_base[-1])
        top1_sim = np.argmax(lg[-1])
        print(f"  {n}-shard INT8 sim: hidden cosine={cos:.4f}  top1 match={top1_base == top1_sim}  (base={top1_base}, sim={top1_sim})")


if __name__ == "__main__":
    main()
