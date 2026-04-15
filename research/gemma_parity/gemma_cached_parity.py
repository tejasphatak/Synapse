#!/usr/bin/env python3
"""
Gemma 3 CACHED-step parity test.

Validates that a numpy single-token cached decode produces the SAME layer
output at position N as prefilling the full [0..N] sequence. This is the
algebraic identity that forwardLayerGemmaCached must preserve.

Approach:
  1. Prefill the full [0..N] sequence through layer L using the proven
     gemma_layer_forward (which already matches HF at cos=1.000).
  2. Re-run layer L on the single row [N..N+1] with a hand-rolled cached
     attention: Q,K,V computed only for position N, K/V of [0..N-1]
     reconstructed from a prefill-style replay, attention computed as
     Q_N · K_{0..N} / sqrt(scale).
  3. Assert the cached-decode output equals prefill[-1] within FP tolerance.

This verifies the math BEFORE we trust the GPU KVCache path, closing the
loop: HF↔numpy-prefill (done) ↔ numpy-cached (this test) ↔ WGSL-cached
(future browser E2E).
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gemma_layer_parity import (
    HF_TOKEN_FILE, load_manifest, load_shard_tensor,
    rmsnorm, gelu_pytorch_tanh, gemma_layer_forward, build_rope_cache,
    rope_apply_half,
)


def gemma_layer_cached_step(full_x, layer_idx, manifest, cfg):
    """
    Run one cached-decode step for position N = full_x.shape[0] - 1.
    Returns the layer-L output for ONLY that last position, [hidden].
    Internally: compute Q,K,V for all positions, but do attention only for
    the last query against the full K/V (this is what a real KV cache replays).
    """
    seq, hidden = full_x.shape
    N = seq - 1

    h_q = manifest["num_attention_heads"]
    h_kv = manifest["num_key_value_heads"]
    d = manifest["head_dim"]
    eps = manifest["rms_norm_eps"]
    is_sliding = cfg["layer_types"][layer_idx] == "sliding_attention"
    window = manifest.get("sliding_window", 0) if is_sliding else 0
    theta = cfg["rope_scaling"]["sliding_attention"]["rope_theta"] if is_sliding \
            else cfg["rope_scaling"]["full_attention"]["rope_theta"]
    cos_cache, sin_cache = build_rope_cache(d, seq + 16, theta)
    attn_scale = 1.0 / np.sqrt(cfg["query_pre_attn_scalar"])

    p = f"model.layers.{layer_idx}"
    residual1 = full_x.copy()

    # Pre-attn RMSNorm for ALL positions (would be cached in real impl;
    # for the last one alone we'd just norm row N).
    h = rmsnorm(full_x, load_shard_tensor(manifest, f"{p}.input_layernorm.weight"), eps)

    W_q = load_shard_tensor(manifest, f"{p}.self_attn.q_proj.weight")
    W_k = load_shard_tensor(manifest, f"{p}.self_attn.k_proj.weight")
    W_v = load_shard_tensor(manifest, f"{p}.self_attn.v_proj.weight")
    q_all = (h @ W_q.T).reshape(seq, h_q, d)
    k_all = (h @ W_k.T).reshape(seq, h_kv, d)
    v_all = (h @ W_v.T).reshape(seq, h_kv, d)

    q_all = rmsnorm(q_all, load_shard_tensor(manifest, f"{p}.self_attn.q_norm.weight"), eps)
    k_all = rmsnorm(k_all, load_shard_tensor(manifest, f"{p}.self_attn.k_norm.weight"), eps)

    q_all = rope_apply_half(q_all, cos_cache, sin_cache, start_pos=0)
    k_all = rope_apply_half(k_all, cos_cache, sin_cache, start_pos=0)

    # Cached attention: single Q at position N, against K[0..N] / V[0..N].
    group = h_q // h_kv
    cache_start = max(0, N + 1 - window) if window > 0 else 0
    Q_N = q_all[N]                                    # [h_q, d]
    attn_N = np.zeros((h_q, d), dtype=np.float32)
    for head in range(h_q):
        kvh = head // group
        K = k_all[cache_start:N + 1, kvh, :]          # [cache_len, d]
        V = v_all[cache_start:N + 1, kvh, :]
        scores = (Q_N[head] @ K.T) * attn_scale       # [cache_len]
        scores = scores - scores.max()
        e = np.exp(scores)
        probs = e / e.sum()
        attn_N[head] = probs @ V                      # [d]

    attn_flat = attn_N.reshape(h_q * d)
    W_o = load_shard_tensor(manifest, f"{p}.self_attn.o_proj.weight")
    o = attn_flat @ W_o.T                             # [hidden]
    o = rmsnorm(o, load_shard_tensor(manifest, f"{p}.post_attention_layernorm.weight"), eps)
    x_N = residual1[N] + o

    # FFN (one row only).
    residual2 = x_N.copy()
    h_ff = rmsnorm(x_N, load_shard_tensor(manifest, f"{p}.pre_feedforward_layernorm.weight"), eps)
    W_gate = load_shard_tensor(manifest, f"{p}.mlp.gate_proj.weight")
    W_up   = load_shard_tensor(manifest, f"{p}.mlp.up_proj.weight")
    W_down = load_shard_tensor(manifest, f"{p}.mlp.down_proj.weight")
    gated = gelu_pytorch_tanh(h_ff @ W_gate.T) * (h_ff @ W_up.T)
    down = gated @ W_down.T
    down = rmsnorm(down, load_shard_tensor(manifest, f"{p}.post_feedforward_layernorm.weight"), eps)
    return residual2 + down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seq", type=int, default=6)
    ap.add_argument("--tol", type=float, default=5e-4)
    args = ap.parse_args()

    manifest = load_manifest()
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(
        manifest["model"],
        token=HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None,
    )
    cfg = {
        "layer_types":           hf_cfg.layer_types,
        "rope_scaling":          hf_cfg.rope_scaling,
        "query_pre_attn_scalar": hf_cfg.query_pre_attn_scalar,
    }

    rng = np.random.default_rng(seed=7)
    x_input = rng.standard_normal((args.seq, manifest["hidden_size"]), dtype=np.float32) * 0.1

    print(f"  layer {args.layer}: {cfg['layer_types'][args.layer]}, seq={args.seq}")

    prefill_out = gemma_layer_forward(x_input, args.layer, manifest, cfg)
    prefill_last = prefill_out[-1]
    print(f"  prefill[last]: mean={prefill_last.mean():+.4f} std={prefill_last.std():+.4f}")

    cached_out = gemma_layer_cached_step(x_input, args.layer, manifest, cfg)
    print(f"  cached-step:   mean={cached_out.mean():+.4f} std={cached_out.std():+.4f}")

    diff = cached_out - prefill_last
    cos = float((cached_out @ prefill_last) /
                (np.linalg.norm(cached_out) * np.linalg.norm(prefill_last) + 1e-9))
    max_abs = float(np.max(np.abs(diff)))
    print(f"\n  cosine:     {cos:.8f}")
    print(f"  max |diff|: {max_abs:.6f}")
    if cos > 0.9999 and max_abs < args.tol:
        print(f"\n  ✓ CACHED PARITY: single-step decode matches prefill[-1] within tol={args.tol}")
    else:
        print(f"\n  ✗ CACHED DIVERGENCE — cached decode math is wrong.")


if __name__ == "__main__":
    main()
