"""
Full distributed pipeline simulation on CPU.
Stacks ALL optimizations: per-channel int8, delta encoding, early exit metrics, head importance.
Simulates 2-shard inference with quantized boundary exactly as phones would run it.
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

def quantize_perchannel(tensor):
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    abs_max = np.abs(tensor).max(axis=-1, keepdims=True)
    abs_max = np.maximum(abs_max, 1e-8)
    scales = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scales), -127, 127).astype(np.int8)
    return quantized, scales

def dequantize_perchannel(quantized, scales):
    return quantized.astype(np.float32) * scales

def cosine_sim(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def forward_layer(hidden, layer_idx, manifest, head_norms_out=None):
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
        head_out = attention(q[:,s:e], k[:,s:e], v[:,s:e], head_dim)
        attn_out[:,s:e] = head_out
        if head_norms_out is not None:
            head_norms_out[h] = np.linalg.norm(head_out)

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

def generate_tokens(prompt_ids, manifest, max_tokens=10):
    """Autoregressive generation simulating the full distributed pipeline."""
    num_layers = manifest["num_layers"]
    num_heads = manifest["num_heads"]
    hidden_size = manifest["hidden_size"]
    shard_boundary = 5  # node 0: layers 0-5, node 1: layers 6-11

    wte = load_tensor(manifest, "transformer.wte.weight")
    wpe = load_tensor(manifest, "transformer.wpe.weight")
    ln_f_w = load_tensor(manifest, "transformer.ln_f.weight")
    ln_f_b = load_tensor(manifest, "transformer.ln_f.bias")
    lm_head = load_tensor(manifest, "lm_head.weight")

    token_ids = list(prompt_ids)
    prev_activation = None  # for delta encoding tracking

    # Metrics
    metrics = {
        "quant_cosines": [],
        "delta_sparsities": [],
        "early_exit_cosines": [],
        "head_importance": np.zeros((num_layers, num_heads)),
        "head_importance_count": 0,
    }

    for gen_step in range(max_tokens):
        seq_len = len(token_ids)
        hidden = wte[token_ids] + wpe[:seq_len]

        prev_layer_hidden = None
        for l in range(num_layers):
            head_norms = np.zeros(num_heads)
            hidden = forward_layer(hidden, l, manifest, head_norms)

            # Track head importance
            metrics["head_importance"][l] += head_norms
            metrics["head_importance_count"] += 1

            # Track early exit convergence (cosine between consecutive layers)
            if prev_layer_hidden is not None:
                # Compare last token position
                cos = cosine_sim(prev_layer_hidden[-1:], hidden[-1:])
                metrics["early_exit_cosines"].append((l, cos))
            prev_layer_hidden = hidden.copy()

            # Simulate shard boundary quantization
            if l == shard_boundary:
                q, scales = quantize_perchannel(hidden)
                reconstructed = dequantize_perchannel(q, scales)
                cos = cosine_sim(hidden, reconstructed)
                metrics["quant_cosines"].append(cos)

                # Delta encoding metric
                if prev_activation is not None and hidden.shape == prev_activation.shape:
                    delta = hidden - prev_activation
                    sparsity = (np.abs(delta) < 0.01).mean()
                    metrics["delta_sparsities"].append(sparsity)

                prev_activation = hidden.copy()
                hidden = reconstructed  # use quantized version (realistic)

        # Final layer norm + lm_head
        hidden = layer_norm(hidden, ln_f_w, ln_f_b)
        logits = hidden @ lm_head.T
        last_logits = logits[-1]

        # Temperature sampling
        probs = softmax(last_logits / 0.8)
        next_token = np.argmax(probs)  # greedy for reproducibility
        token_ids.append(int(next_token))

        if next_token == 50256:  # EOS
            break

    return token_ids, metrics

def main():
    manifest = load_manifest()

    prompts = {
        "Hello": [15496],
        "The capital of France": [464, 3139, 286, 4881],
        "Once upon a": [7454, 2402, 257],
    }

    print("=" * 70)
    print("FULL PIPELINE SIMULATION — all optimizations stacked")
    print("=" * 70)

    for name, ids in prompts.items():
        print(f"\nPrompt: '{name}' ({len(ids)} tokens)")
        tokens, metrics = generate_tokens(ids, manifest, max_tokens=8)

        # Results
        gen_tokens = tokens[len(ids):]
        print(f"  Generated {len(gen_tokens)} tokens: {gen_tokens}")

        # Quantization quality at shard boundary
        if metrics["quant_cosines"]:
            avg_cos = np.mean(metrics["quant_cosines"])
            min_cos = np.min(metrics["quant_cosines"])
            print(f"  Shard boundary quantization: avg={avg_cos:.6f}, min={min_cos:.6f}")

        # Delta encoding sparsity
        if metrics["delta_sparsities"]:
            avg_sp = np.mean(metrics["delta_sparsities"])
            print(f"  Delta sparsity (consecutive steps): {avg_sp*100:.1f}%")

        # Early exit convergence
        if metrics["early_exit_cosines"]:
            by_layer = {}
            for l, cos in metrics["early_exit_cosines"]:
                by_layer.setdefault(l, []).append(cos)
            print(f"  Early exit convergence (avg cosine per layer):")
            for l in sorted(by_layer.keys()):
                avg = np.mean(by_layer[l])
                print(f"    Layer {l:2d}: {avg:.6f}" + (" ← potential early exit" if avg > 0.999 else ""))

        # Head importance
        if metrics["head_importance_count"] > 0:
            imp = metrics["head_importance"] / metrics["head_importance_count"]
            print(f"  Head importance (top 3 most/least important per layer):")
            for l in [0, 5, 11]:  # sample layers
                ranked = np.argsort(imp[l])
                top3 = ranked[-3:][::-1]
                bot3 = ranked[:3]
                print(f"    Layer {l:2d}: top={list(top3)} ({imp[l][top3].mean():.2f}), "
                      f"bottom={list(bot3)} ({imp[l][bot3].mean():.2f}), "
                      f"ratio={imp[l][top3].mean()/max(imp[l][bot3].mean(), 1e-8):.1f}x")

    print(f"\n{'=' * 70}")
    print("DONE")

if __name__ == "__main__":
    main()
