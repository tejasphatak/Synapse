#!/usr/bin/env python3
"""
Gemma 3 FULL-forward CPU parity test.

Extends gemma_layer_parity.py: instead of a single layer, run the entire
text transformer (embed → all 26 layers → final norm → lm_head) in numpy
using our shard-format weights, then compare against HF's full forward.

Adds the bits the layer-level test skipped:
  - token-id → embedding lookup
  - embedding scale by sqrt(hidden_size)
  - final model.norm
  - lm_head via tied weights (embed_tokens.weight.T)
  - logits argmax comparison (top-1 parity under greedy decode)

Usage:
  python3 gemma_full_parity.py [--seq 4]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gemma_layer_parity import (
    SHARDS_DIR, HF_TOKEN_FILE,
    load_manifest, load_shard_tensor,
    gemma_layer_forward, rmsnorm,
)


def ours_full_forward(input_ids, manifest, cfg):
    """
    input_ids: [seq] numpy int64
    Returns logits [seq, vocab].
    """
    hidden = manifest["hidden_size"]
    eps = manifest["rms_norm_eps"]
    num_layers = manifest["num_layers"]

    # Embed + scale
    embed = load_shard_tensor(manifest, "model.embed_tokens.weight")  # [vocab, hidden]
    x = embed[input_ids].astype(np.float32) * np.sqrt(hidden)

    # All layers
    for l in range(num_layers):
        x = gemma_layer_forward(x, l, manifest, cfg)

    # Final norm + lm_head (tied)
    final_gamma = load_shard_tensor(manifest, "model.norm.weight")
    x = rmsnorm(x, final_gamma, eps)
    logits = x @ embed.T  # tied: lm_head.weight = embed_tokens.weight
    return logits


def hf_full_forward(input_ids, hf_id="google/gemma-3-1b-it"):
    import torch
    from transformers import AutoModelForCausalLM
    tok = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32, token=tok)
    model.eval()
    ids = torch.from_numpy(input_ids).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=ids)
    return out.logits.squeeze(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4)
    args = ap.parse_args()

    manifest = load_manifest()
    assert manifest["arch"] == "gemma"
    print(f"  model: {manifest['model']}  layers: {manifest['num_layers']}  hidden: {manifest['hidden_size']}")
    print(f"  seq: {args.seq}")

    # Grab per-layer config from HF (layer_types, rope_scaling, query_pre_attn_scalar)
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(
        manifest["model"],
        token=HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None,
    )
    cfg = {
        "layer_types":            hf_cfg.layer_types,
        "rope_scaling":           hf_cfg.rope_scaling,
        "query_pre_attn_scalar":  hf_cfg.query_pre_attn_scalar,
    }

    rng = np.random.default_rng(seed=42)
    input_ids = rng.integers(10, 100, size=args.seq, dtype=np.int64)
    print(f"  input_ids: {input_ids.tolist()}")

    print("\n  running OUR full forward (numpy, mirrors WGSL composition)...")
    import time
    t0 = time.time()
    ours = ours_full_forward(input_ids, manifest, cfg)
    print(f"    ours logits: {ours.shape}  argmax[last]={int(ours[-1].argmax())}  elapsed={time.time()-t0:.1f}s")

    print("\n  running HF full forward...")
    t0 = time.time()
    ref = hf_full_forward(input_ids)
    print(f"    ref  logits: {ref.shape}  argmax[last]={int(ref[-1].argmax())}  elapsed={time.time()-t0:.1f}s")

    print("\n  ─── comparison ───")
    diff = ours - ref
    cos = float((ours.flatten() @ ref.flatten()) /
                (np.linalg.norm(ours) * np.linalg.norm(ref) + 1e-9))
    print(f"    cosine(logits):      {cos:.6f}")
    print(f"    max |diff|:          {float(np.max(np.abs(diff))):.4f}")
    print(f"    rms(diff):           {float(np.sqrt((diff**2).mean())):.5f}")

    ours_top1 = ours.argmax(axis=-1)
    ref_top1  = ref.argmax(axis=-1)
    match = int((ours_top1 == ref_top1).sum())
    print(f"    top-1 agreement:     {match}/{args.seq} positions")
    print(f"    ours_top1: {ours_top1.tolist()}")
    print(f"    ref_top1:  {ref_top1.tolist()}")

    if cos > 0.999 and match == args.seq:
        print("\n  ✓ FULL PARITY: end-to-end kernel composition matches HF.")
    else:
        print("\n  ✗ DIVERGENCE: layer math matched but full-forward does not.")


if __name__ == "__main__":
    main()
