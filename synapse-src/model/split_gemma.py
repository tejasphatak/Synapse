#!/usr/bin/env python3
"""
Gemma 3 / Gemma 4 → Synapse shard converter.

Focused split script for the Gemma family — handles the specifics
(MQA/GQA, RoPE, RMSNorm, sliding-window-per-layer, text-only extraction
from multi-modal checkpoints) in one place rather than shoehorning
them into the GPT-2-centric split.py.

Usage:
  python split_gemma.py --model google/gemma-3-1b-it --num-shards 3
  python split_gemma.py --model google/gemma-3-4b-it --num-shards 6
  python split_gemma.py --model google/gemma-4-E2B-it --num-shards 4

Requires a HuggingFace token in ~/.huggingface_token and the target
model's license accepted on the HF account.

Outputs to synapse-src/model/shards/:
  - manifest.json           — tensor index + Gemma-specific config
  - shared.bin              — embed, final norm, lm_head (if separate)
  - shard_N.bin             — layers assigned to shard N
  - rope_cos.bin            — precomputed cos cache [max_pos, head_dim/2]
  - rope_sin.bin            — precomputed sin cache [max_pos, head_dim/2]
"""

import argparse, json, os, sys, math
from pathlib import Path

try:
    import numpy as np
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download, list_repo_files
except ImportError as e:
    print(f"Missing dep: {e}. Install with:")
    print("  pip install safetensors huggingface_hub numpy")
    sys.exit(1)


SHARDS_DIR = Path(__file__).parent / "shards"
HF_TOKEN_FILE = Path.home() / ".huggingface_token"


def read_hf_token():
    if HF_TOKEN_FILE.exists():
        return HF_TOKEN_FILE.read_text().strip()
    return os.environ.get("HF_TOKEN")


# ─── Weight-name mapping: Gemma HF → Synapse manifest ─────────────
#
# Gemma 3 text model (gemma3_text) naming:
#   model.embed_tokens.weight                              [vocab, hidden]
#   model.layers.{L}.input_layernorm.weight                [hidden]
#   model.layers.{L}.self_attn.q_proj.weight               [hq*head_dim, hidden]
#   model.layers.{L}.self_attn.k_proj.weight               [hkv*head_dim, hidden]
#   model.layers.{L}.self_attn.v_proj.weight               [hkv*head_dim, hidden]
#   model.layers.{L}.self_attn.o_proj.weight               [hidden, hq*head_dim]
#   model.layers.{L}.self_attn.q_norm.weight               [head_dim]   ← Gemma-specific Q/K norm
#   model.layers.{L}.self_attn.k_norm.weight               [head_dim]
#   model.layers.{L}.post_attention_layernorm.weight       [hidden]
#   model.layers.{L}.pre_feedforward_layernorm.weight      [hidden]     ← Gemma 3 has pre + post FFN norms
#   model.layers.{L}.post_feedforward_layernorm.weight     [hidden]
#   model.layers.{L}.mlp.gate_proj.weight                  [ffn, hidden]
#   model.layers.{L}.mlp.up_proj.weight                    [ffn, hidden]
#   model.layers.{L}.mlp.down_proj.weight                  [hidden, ffn]
#   model.norm.weight                                       [hidden]
#
# Gemma uses WEIGHT TYING: lm_head.weight == embed_tokens.weight. No
# separate lm_head tensor in the safetensors. The pipeline reuses embed
# transposed at output.
#
# Gemma 4 multi-modal: text transformer is under "language_model." prefix.
# We strip that so downstream code sees Gemma-3-like naming.


def strip_multimodal_prefix(name):
    """Gemma 4 has 'language_model.model.layers...' — strip the wrapper."""
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def parse_layer_idx(name):
    """Return the layer index if this tensor belongs to a layer, else None."""
    marker = "model.layers."
    idx = name.find(marker)
    if idx < 0:
        return None
    tail = name[idx + len(marker):]
    dot = tail.find(".")
    try:
        return int(tail[:dot])
    except ValueError:
        return None


def is_shared(name):
    """True if tensor belongs to shared weights (embed, norm) not a layer."""
    return parse_layer_idx(name) is None


# ─── RoPE cache ───────────────────────────────────────────────────

def compute_rope_cache(max_pos, head_dim, base=10000.0):
    """
    Return (cos, sin) each shape [max_pos, head_dim / 2], float32.

    HF Gemma 3 uses the "rotate_half" layout (concat halves, not interleaved pairs):
      out[:, :half] = x[:, :half] * cos - x[:, half:] * sin
      out[:, half:] = x[:, half:] * cos + x[:, :half] * sin

    Formula: θ_i(p) = p * base^(-2i / head_dim), i in [0, head_dim/2)
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    half = head_dim // 2
    freqs = base ** (-2.0 * np.arange(half, dtype=np.float64) / head_dim)
    positions = np.arange(max_pos, dtype=np.float64).reshape(-1, 1)
    angles = positions * freqs.reshape(1, -1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


# ─── Download & convert ───────────────────────────────────────────

def download_model(hf_id, token):
    """Pull config + weights from HF to local cache. Returns (config dict, weights file path).

    Uses transformers AutoConfig so fields that the architecture fills in at
    runtime (layer_types, rope_scaling dual bases) are populated, not just
    the raw JSON which omits them.
    """
    from transformers import AutoConfig
    auto = AutoConfig.from_pretrained(hf_id, token=token)
    tc = getattr(auto, "text_config", auto)
    config = tc.to_dict() if hasattr(tc, "to_dict") else dict(tc)

    # Find the actual safetensors file(s)
    files = list_repo_files(hf_id, token=token)
    st_files = [f for f in files if f.endswith(".safetensors")]
    if not st_files:
        raise RuntimeError(f"No safetensors in {hf_id}")
    if len(st_files) == 1:
        path = hf_hub_download(hf_id, st_files[0], token=token)
        return config, [path]
    # Multiple shards — download all, caller concatenates logical tensors
    paths = [hf_hub_download(hf_id, f, token=token) for f in st_files]
    return config, paths


def load_all_tensors(paths):
    """Open all safetensors files and return a combined name → (raw name, file) map.

    Uses framework="pt" because Gemma weights are bfloat16 which numpy
    doesn't recognize natively. PyTorch handles bf16 and we convert on fetch.
    """
    tensor_refs = {}
    for path in paths:
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                clean = strip_multimodal_prefix(name)
                tensor_refs[clean] = (name, path)
    return tensor_refs


def fetch_tensor(tensor_refs, name):
    """Materialise a tensor as float32 numpy array."""
    import torch
    if name not in tensor_refs:
        return None
    raw_name, path = tensor_refs[name]
    with safe_open(path, framework="pt") as f:
        t = f.get_tensor(raw_name)
    # Upcast any dtype (bfloat16, float16, int8) to float32 via torch
    return t.to(torch.float32).cpu().numpy()


# ─── Pick Gemma config fields ─────────────────────────────────────

def extract_gemma_text_config(config):
    """
    Pull the text transformer config out of either a Gemma-3-text-only
    config (flat) or a Gemma-4 multi-modal config (nested under text_config).
    """
    tc = config.get("text_config", config) if isinstance(config, dict) else config
    fields = {
        "vocab_size":                tc.get("vocab_size"),
        "hidden_size":               tc.get("hidden_size"),
        "num_hidden_layers":         tc.get("num_hidden_layers"),
        "num_attention_heads":       tc.get("num_attention_heads"),
        "num_key_value_heads":       tc.get("num_key_value_heads", tc.get("num_attention_heads")),
        "head_dim":                  tc.get("head_dim"),
        "intermediate_size":         tc.get("intermediate_size"),
        "max_position_embeddings":   tc.get("max_position_embeddings", 8192),
        "rope_theta":                tc.get("rope_theta", 10000.0),
        "rope_scaling":              tc.get("rope_scaling") or tc.get("rope_parameters"),
        "sliding_window":            tc.get("sliding_window", None),
        "rms_norm_eps":              tc.get("rms_norm_eps", 1e-6),
        "hidden_activation":         tc.get("hidden_activation", "gelu_pytorch_tanh"),
        "attn_logit_softcapping":    tc.get("attn_logit_softcapping", None),
        "final_logit_softcapping":   tc.get("final_logit_softcapping", None),
        "tie_word_embeddings":       tc.get("tie_word_embeddings", True),
        "query_pre_attn_scalar":     tc.get("query_pre_attn_scalar", tc.get("head_dim")),
        "layer_types":               tc.get("layer_types", None),
        "sliding_window_pattern":    tc.get("sliding_window_pattern", None),
    }
    # Derive head_dim if missing
    if not fields["head_dim"]:
        fields["head_dim"] = fields["hidden_size"] // fields["num_attention_heads"]
    return fields


# ─── Shard writer ─────────────────────────────────────────────────

def write_fp16_tensor(f, array):
    """Write a numpy array as float16 bytes to an open file. Returns byte count."""
    fp16 = array.astype(np.float16)
    raw = fp16.tobytes()
    f.write(raw)
    return len(raw)


def build_shards(tensor_refs, text_cfg, num_shards, dtype="float16"):
    """
    Walk all tensors, write shared.bin + shard_N.bin files, return manifest dict.
    """
    SHARDS_DIR.mkdir(exist_ok=True)
    num_layers = text_cfg["num_hidden_layers"]
    layers_per_shard = num_layers // num_shards
    if num_layers % num_shards != 0:
        print(f"  warning: {num_layers} layers don't divide evenly into {num_shards} shards — last shard takes the remainder")

    # Assign each layer to a shard (contiguous ranges).
    layer_to_shard = {}
    shard_layer_ranges = {}
    for s in range(num_shards):
        start = s * layers_per_shard
        end = (s + 1) * layers_per_shard - 1 if s < num_shards - 1 else num_layers - 1
        shard_layer_ranges[s] = (start, end)
        for L in range(start, end + 1):
            layer_to_shard[L] = s

    manifest = {
        "model": text_cfg.get("model_id", "gemma"),
        "arch":  "gemma",
        "num_layers": num_layers,
        "hidden_size": text_cfg["hidden_size"],
        "num_attention_heads": text_cfg["num_attention_heads"],
        "num_key_value_heads": text_cfg["num_key_value_heads"],
        "head_dim": text_cfg["head_dim"],
        "intermediate_size": text_cfg["intermediate_size"],
        "vocab_size": text_cfg["vocab_size"],
        "max_seq_len": text_cfg["max_position_embeddings"],
        "rope_theta": text_cfg["rope_theta"],
        "rope_scaling": text_cfg.get("rope_scaling"),
        "query_pre_attn_scalar": text_cfg["query_pre_attn_scalar"],
        "layer_types": text_cfg.get("layer_types"),
        "sliding_window_pattern": text_cfg.get("sliding_window_pattern"),
        "sliding_window": text_cfg["sliding_window"],
        "rms_norm_eps": text_cfg["rms_norm_eps"],
        "hidden_activation": text_cfg["hidden_activation"],
        "tie_word_embeddings": text_cfg["tie_word_embeddings"],
        "dtype": dtype,
        "layers_per_shard": layers_per_shard,
        "num_shards": num_shards,
        "shard_layout": {
            str(s): {"layer_start": shard_layer_ranges[s][0],
                     "layer_end":   shard_layer_ranges[s][1],
                     "file": f"shard_{s}.bin"}
            for s in range(num_shards)
        },
        "banned_sampling_tokens": text_cfg.get("banned_sampling_tokens", []),
        "eos_tokens": text_cfg.get("eos_tokens", [1]),
        "shared_file": "shared.bin",
        "shared_dtype": dtype,
        "rope_cos_file": "rope_cos.bin",
        "rope_sin_file": "rope_sin.bin",
        "rope_cos_local_file": "rope_cos_local.bin",
        "rope_sin_local_file": "rope_sin_local.bin",
        "tensors": [],
    }

    def classify(name):
        L = parse_layer_idx(name)
        if L is None:
            return ("shared", None)
        return ("layer", L)

    # Group tensor names by destination
    shared_names = []
    per_shard_names = {s: [] for s in range(num_shards)}
    for name in sorted(tensor_refs.keys()):
        kind, L = classify(name)
        if kind == "shared":
            shared_names.append(name)
        else:
            per_shard_names[layer_to_shard[L]].append(name)

    # Write shared.bin
    print(f"  writing shared.bin ({len(shared_names)} tensors)")
    off = 0
    with open(SHARDS_DIR / "shared.bin", "wb") as f:
        for name in shared_names:
            t = fetch_tensor(tensor_refs, name)
            if t is None:
                continue
            written = write_fp16_tensor(f, t)
            manifest["tensors"].append({
                "name": name, "shape": list(t.shape), "dtype": "float16",
                "file": "shared.bin", "offset": off, "size": written,
            })
            off += written
    print(f"    {off / 1024 / 1024:.1f} MB")

    # Write each shard.bin
    for s in range(num_shards):
        names = per_shard_names[s]
        print(f"  writing shard_{s}.bin ({len(names)} tensors, layers {shard_layer_ranges[s][0]}-{shard_layer_ranges[s][1]})")
        off = 0
        with open(SHARDS_DIR / f"shard_{s}.bin", "wb") as f:
            for name in names:
                t = fetch_tensor(tensor_refs, name)
                if t is None:
                    continue
                written = write_fp16_tensor(f, t)
                manifest["tensors"].append({
                    "name": name, "shape": list(t.shape), "dtype": "float16",
                    "file": f"shard_{s}.bin", "offset": off, "size": written,
                })
                off += written
        print(f"    {off / 1024 / 1024:.1f} MB")

    # Write RoPE cos/sin caches — global (full_attention) + local (sliding_attention)
    rs = text_cfg.get("rope_scaling") or {}
    theta_full  = (rs.get("full_attention")    or {}).get("rope_theta", text_cfg["rope_theta"])
    theta_local = (rs.get("sliding_attention") or {}).get("rope_theta", text_cfg["rope_theta"])
    max_pos, head_dim = text_cfg["max_position_embeddings"], text_cfg["head_dim"]

    print(f"  computing RoPE caches (max_pos={max_pos}, head_dim={head_dim}, theta_full={theta_full}, theta_local={theta_local})")
    cos_full, sin_full = compute_rope_cache(max_pos, head_dim, base=theta_full)
    (SHARDS_DIR / "rope_cos.bin").write_bytes(cos_full.tobytes())
    (SHARDS_DIR / "rope_sin.bin").write_bytes(sin_full.tobytes())
    cos_loc, sin_loc = compute_rope_cache(max_pos, head_dim, base=theta_local)
    (SHARDS_DIR / "rope_cos_local.bin").write_bytes(cos_loc.tobytes())
    (SHARDS_DIR / "rope_sin_local.bin").write_bytes(sin_loc.tobytes())
    print(f"    rope (full):  {cos_full.nbytes / 1024:.1f} KB × 2")
    print(f"    rope (local): {cos_loc.nbytes / 1024:.1f} KB × 2")

    with open(SHARDS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json written ({len(manifest['tensors'])} tensor entries)")

    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-1b-it",
                    help="HuggingFace model ID (default: google/gemma-3-1b-it)")
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    token = read_hf_token()
    if not token:
        print(f"No HF token found at {HF_TOKEN_FILE} or env HF_TOKEN")
        sys.exit(1)

    print(f"=== Gemma → Synapse converter ===")
    print(f"  model:  {args.model}")
    print(f"  shards: {args.num_shards}")
    print(f"  dtype:  {args.dtype}")

    print(f"\n  downloading config + weights from HF")
    config, paths = download_model(args.model, token)
    text_cfg = extract_gemma_text_config(config)
    text_cfg["model_id"] = args.model
    print(f"\n  model arch:")
    for k, v in text_cfg.items():
        print(f"    {k:<28} {v}")

    # Download tokenizer files so the coordinator can encode/decode without
    # HF network access at runtime. Placed under shards/tokenizer/.
    tok_dir = SHARDS_DIR / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        try:
            src = hf_hub_download(args.model, fname, token=token)
            shutil.copy(src, tok_dir / fname)
        except Exception as e:
            print(f"  tokenizer: skipping {fname} ({e.__class__.__name__})")
    print(f"  tokenizer files → {tok_dir}")

    # Parse tokenizer config for banned tokens (<unused*>) so the
    # inference pipeline can zero them before sampling.
    banned_ids = []
    eos_ids = []
    tok_config_path = tok_dir / "tokenizer_config.json"
    if tok_config_path.exists():
        import json as _j
        tc = _j.load(open(tok_config_path))
        added = tc.get("added_tokens_decoder", {})
        banned_ids = sorted(int(k) for k, v in added.items()
                            if "<unused" in str(v.get("content", "")))
        eos_ids = sorted(int(k) for k, v in added.items()
                         if v.get("content") in ("</s>", "<eos>", "<end_of_turn>"))
        print(f"  banned sampling tokens: {len(banned_ids)}, eos tokens: {eos_ids}")
    text_cfg["banned_sampling_tokens"] = banned_ids
    text_cfg["eos_tokens"] = eos_ids

    tensor_refs = load_all_tensors(paths)
    # Filter to text-transformer tensors only (strip vision/audio for Gemma 4)
    text_tensors = {
        k: v for k, v in tensor_refs.items()
        if k.startswith("model.") or k.startswith("lm_head")
    }
    skipped = len(tensor_refs) - len(text_tensors)
    if skipped:
        print(f"\n  skipping {skipped} vision/audio tensors (text-only extraction)")

    print(f"\n  writing shards to {SHARDS_DIR}")
    build_shards(text_tensors, text_cfg, args.num_shards, args.dtype)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
