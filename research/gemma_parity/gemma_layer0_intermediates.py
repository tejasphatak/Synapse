#!/usr/bin/env python3
"""Dump numpy intermediates for layer 0 (sliding attention) after each
sub-kernel. Serves as the golden reference for live WGSL bisection."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gemma_layer_parity import (
    HF_TOKEN_FILE, load_manifest, load_shard_tensor,
    rmsnorm, gelu_pytorch_tanh, build_rope_cache, rope_apply_half,
    attention_gqa,
)


def stats(x, name):
    f = x.flatten()
    print(f"  {name:30s} rms={np.sqrt((f**2).mean()):9.4f}  "
          f"min={f.min():10.3f}  max={f.max():10.3f}  n={f.size}")


def main():
    import urllib.request, json
    req = urllib.request.Request(
        "http://34.82.32.123:8080/api/tokenize",
        data=json.dumps({"text": "The universe is"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        input_ids = json.loads(r.read())["tokenIds"]
    print(f"  ids: {input_ids}")

    m = load_manifest()
    from transformers import AutoConfig
    hf = AutoConfig.from_pretrained(m["model"],
        token=HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None)
    layer_types = hf.layer_types
    rope_scaling = hf.rope_scaling
    qpa = hf.query_pre_attn_scalar

    h_q, h_kv = m["num_attention_heads"], m["num_key_value_heads"]
    d, H = m["head_dim"], m["hidden_size"]
    eps = m["rms_norm_eps"]
    is_sliding = layer_types[0] == "sliding_attention"
    window = m.get("sliding_window", 0) if is_sliding else 0
    theta = rope_scaling["sliding_attention"]["rope_theta"] if is_sliding \
            else rope_scaling["full_attention"]["rope_theta"]
    cos, sin = build_rope_cache(d, len(input_ids) + 8, theta)
    attn_scale = 1.0 / np.sqrt(qpa)

    # Embed
    embed = load_shard_tensor(m, "model.embed_tokens.weight")
    x = embed[np.array(input_ids, dtype=np.int64)].astype(np.float32) * np.sqrt(H)
    stats(x, "embed")

    # Layer 0 sub-steps
    p = "model.layers.0"
    residual1 = x.copy()
    h = rmsnorm(x, load_shard_tensor(m, f"{p}.input_layernorm.weight"), eps)
    stats(h, "ln1 (input_layernorm)")

    Wq = load_shard_tensor(m, f"{p}.self_attn.q_proj.weight")
    Wk = load_shard_tensor(m, f"{p}.self_attn.k_proj.weight")
    Wv = load_shard_tensor(m, f"{p}.self_attn.v_proj.weight")
    q = (h @ Wq.T).reshape(len(input_ids), h_q, d)
    k = (h @ Wk.T).reshape(len(input_ids), h_kv, d)
    v = (h @ Wv.T).reshape(len(input_ids), h_kv, d)
    stats(q, "q (post q_proj)")
    stats(k, "k (post k_proj)")
    stats(v, "v (post v_proj)")

    q = rmsnorm(q, load_shard_tensor(m, f"{p}.self_attn.q_norm.weight"), eps)
    k = rmsnorm(k, load_shard_tensor(m, f"{p}.self_attn.k_norm.weight"), eps)
    stats(q, "q (post q_norm)")
    stats(k, "k (post k_norm)")

    q = rope_apply_half(q, cos, sin, 0)
    k = rope_apply_half(k, cos, sin, 0)
    stats(q, "q (post rope)")
    stats(k, "k (post rope)")

    attn = attention_gqa(q, k, v, window_size=window, scale=attn_scale).reshape(len(input_ids), h_q * d)
    stats(attn, "attn_gqa output")

    Wo = load_shard_tensor(m, f"{p}.self_attn.o_proj.weight")
    o = attn @ Wo.T
    stats(o, "o_proj output")

    o_norm = rmsnorm(o, load_shard_tensor(m, f"{p}.post_attention_layernorm.weight"), eps)
    stats(o_norm, "o_proj post_attn_ln")

    after_attn = residual1 + o_norm
    stats(after_attn, "residual1 + oNorm")

    residual2 = after_attn.copy()
    ln2 = rmsnorm(after_attn, load_shard_tensor(m, f"{p}.pre_feedforward_layernorm.weight"), eps)
    stats(ln2, "ln2 (pre_ff_ln)")

    Wg = load_shard_tensor(m, f"{p}.mlp.gate_proj.weight")
    Wu = load_shard_tensor(m, f"{p}.mlp.up_proj.weight")
    Wd = load_shard_tensor(m, f"{p}.mlp.down_proj.weight")
    gate = ln2 @ Wg.T
    up = ln2 @ Wu.T
    stats(gate, "gate (pre-gelu)")
    stats(up, "up")
    g_gelu = gelu_pytorch_tanh(gate)
    stats(g_gelu, "gelu(gate)")
    gated = g_gelu * up
    stats(gated, "gelu(gate) * up")
    down = gated @ Wd.T
    stats(down, "down_proj")
    down_norm = rmsnorm(down, load_shard_tensor(m, f"{p}.post_feedforward_layernorm.weight"), eps)
    stats(down_norm, "down post_ff_ln")
    out = residual2 + down_norm
    stats(out, "layer 0 FINAL")


if __name__ == "__main__":
    main()
