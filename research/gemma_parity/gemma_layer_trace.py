#!/usr/bin/env python3
"""Dump per-layer mean/std/rms on the numpy reference for the same prompt
the live fleet runs, so we can compare against what the WGSL pipeline
reports via sub_kernel_trace / pre_lmhead_hidden events.

Also prints pre-lmhead hidden (after final norm) so we can match
directly against the live `pre_lmhead_hidden` log event."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from gemma_layer_parity import (
    HF_TOKEN_FILE, load_manifest, load_shard_tensor,
    gemma_layer_forward, rmsnorm,
)


def trace(prompt="The universe is"):
    import urllib.request, json
    # Tokenize via coord for exact fidelity
    coord = "http://34.82.32.123:8080"
    req = urllib.request.Request(
        f"{coord}/api/tokenize", data=json.dumps({"text": prompt}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        input_ids = json.loads(r.read())["tokenIds"]
    print(f"  prompt: {prompt!r}  ids: {input_ids}")

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

    hidden = manifest["hidden_size"]
    embed = load_shard_tensor(manifest, "model.embed_tokens.weight")
    x = embed[np.array(input_ids, dtype=np.int64)].astype(np.float32) * np.sqrt(hidden)
    print(f"  embed:  rms={np.sqrt((x**2).mean()):.4f}  min={x.min():.2f} max={x.max():.2f}")

    for l in range(manifest["num_layers"]):
        x = gemma_layer_forward(x, l, manifest, cfg)
        last = x[-1]
        print(f"  layer{l:2d}: rms={np.sqrt((last**2).mean()):8.4f} "
              f"min={last.min():8.2f} max={last.max():8.2f} (type={cfg['layer_types'][l][:8]})")

    # Pre-lmhead: final norm on last-position vector
    norm_g = load_shard_tensor(manifest, "model.norm.weight")
    pre = rmsnorm(x, norm_g, manifest["rms_norm_eps"])[-1]
    print(f"\n  pre_lmhead (after final norm): rms={np.sqrt((pre**2).mean()):.4f} "
          f"min={pre.min():.2f} max={pre.max():.2f}")


if __name__ == "__main__":
    trace()
