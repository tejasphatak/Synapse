#!/usr/bin/env python3
"""
Download and split transformer models for distributed browser inference.

Supports:
  - GPT-2 family: gpt2, gpt2-medium, gpt2-large, gpt2-xl
  - Phi-3: microsoft/Phi-3-mini-4k-instruct
  - Llama 3.2: meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-3B
  - Any HuggingFace causal LM

Quantization:
  - float32 (default, full precision)
  - float16 (half precision, 2x compression)
  - int8 (8-bit quantization, 4x compression)
  - int4 (4-bit quantization, 8x compression)

Usage:
  python split.py                                    # GPT-2 small, float32, 2 shards
  python split.py --model gpt2-medium --dtype float16
  python split.py --model gpt2-xl --dtype int8 --num-shards 4
  python split.py --model microsoft/Phi-3-mini-4k-instruct --dtype int4 --num-shards 4

Output:
  shards/shard_0.bin ... shard_N.bin
  shards/shared.bin
  shards/manifest.json
"""

import argparse
import json
import os
import sys
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoConfig
except ImportError:
    print("Error: 'transformers' package required. Install with:")
    print("  pip install transformers torch")
    sys.exit(1)


SHARDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shards")

# ─── Model Presets ────────────────────────────────────────────────

MODEL_PRESETS = {
    "gpt2": {
        "hf_name": "gpt2",
        "num_layers": 12, "hidden_size": 768, "num_heads": 12,
        "head_dim": 64, "vocab_size": 50257, "max_seq_len": 1024,
        "arch": "gpt2",
    },
    "gpt2-medium": {
        "hf_name": "gpt2-medium",
        "num_layers": 24, "hidden_size": 1024, "num_heads": 16,
        "head_dim": 64, "vocab_size": 50257, "max_seq_len": 1024,
        "arch": "gpt2",
    },
    "gpt2-large": {
        "hf_name": "gpt2-large",
        "num_layers": 36, "hidden_size": 1280, "num_heads": 20,
        "head_dim": 64, "vocab_size": 50257, "max_seq_len": 1024,
        "arch": "gpt2",
    },
    "gpt2-xl": {
        "hf_name": "gpt2-xl",
        "num_layers": 48, "hidden_size": 1600, "num_heads": 25,
        "head_dim": 64, "vocab_size": 50257, "max_seq_len": 1024,
        "arch": "gpt2",
    },
}

# ─── Quantization ─────────────────────────────────────────────────

def quantize_int8(tensor):
    """Quantize float32 tensor to int8 with per-tensor scale."""
    abs_max = np.max(np.abs(tensor))
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int8), np.float32(1.0)
    scale = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, np.float32(scale)


def quantize_int4(tensor):
    """Quantize float32 tensor to int4 (packed into int8, two values per byte).
    Uses per-group quantization with group_size=32 for better accuracy."""
    flat = tensor.flatten()
    group_size = 32

    # Pad to multiple of group_size
    pad_len = (group_size - len(flat) % group_size) % group_size
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])

    num_groups = len(flat) // group_size
    flat = flat.reshape(num_groups, group_size)

    # Per-group scale
    abs_max = np.max(np.abs(flat), axis=1, keepdims=True)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)
    scales = (abs_max / 7.0).astype(np.float32).flatten()

    # Quantize to [-7, 7]
    quantized = np.clip(np.round(flat / (abs_max / 7.0)), -7, 7).astype(np.int8)
    quantized = quantized.flatten()

    # Remove padding
    if pad_len > 0:
        quantized = quantized[:len(tensor.flatten())]

    # Pack two int4 values per byte: high nibble + low nibble
    flat_q = quantized
    if len(flat_q) % 2 != 0:
        flat_q = np.concatenate([flat_q, np.zeros(1, dtype=np.int8)])

    # Pack: (val_even & 0xF) | (val_odd << 4)
    even = flat_q[0::2].astype(np.uint8) & 0x0F
    odd = (flat_q[1::2].astype(np.uint8) & 0x0F) << 4
    packed = (even | odd).astype(np.uint8)

    return packed, scales


def quantize_tensor(tensor, dtype):
    """Quantize a tensor to the specified dtype. Returns (data_bytes, metadata)."""
    if dtype == "float32":
        data = np.ascontiguousarray(tensor, dtype=np.float32)
        return data.tobytes(), {"dtype": "float32", "quant": None}

    elif dtype == "float16":
        data = np.ascontiguousarray(tensor, dtype=np.float16)
        return data.tobytes(), {"dtype": "float16", "quant": None}

    elif dtype == "int8":
        quantized, scale = quantize_int8(tensor)
        data = np.ascontiguousarray(quantized)
        # Append scale as 4 bytes at the end
        raw = data.tobytes() + scale.tobytes()
        return raw, {
            "dtype": "int8",
            "quant": "per_tensor",
            "scale_offset": len(data.tobytes()),
            "scale_size": 4,
        }

    elif dtype == "int4":
        packed, scales = quantize_int4(tensor)
        packed_bytes = np.ascontiguousarray(packed).tobytes()
        scales_bytes = np.ascontiguousarray(scales).tobytes()
        raw = packed_bytes + scales_bytes
        return raw, {
            "dtype": "int4",
            "quant": "per_group",
            "group_size": 32,
            "packed_size": len(packed_bytes),
            "scales_offset": len(packed_bytes),
            "scales_size": len(scales_bytes),
            "original_numel": tensor.size,
        }

    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def bytes_per_param(dtype):
    """Bytes per parameter for a given dtype."""
    return {"float32": 4, "float16": 2, "int8": 1, "int4": 0.5}[dtype]

# ─── Model Loading ────────────────────────────────────────────────

def load_model(model_name):
    """Load a model from HuggingFace (or a preset name)."""
    hf_name = model_name
    if model_name in MODEL_PRESETS:
        hf_name = MODEL_PRESETS[model_name]["hf_name"]

    print(f"Downloading {hf_name} from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(hf_name, torch_dtype="auto")
    config = model.config
    print(f"Download complete. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, config


def detect_architecture(model, config):
    """Detect model architecture and return standardized config."""
    arch_type = config.model_type  # "gpt2", "llama", "phi3", etc.

    # Check preset first
    for preset_name, preset in MODEL_PRESETS.items():
        if preset["hf_name"] == config._name_or_path or preset_name == config._name_or_path:
            return {**preset, "arch": arch_type}

    # Auto-detect from config
    return {
        "hf_name": config._name_or_path,
        "num_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12)),
        "hidden_size": getattr(config, "hidden_size", getattr(config, "n_embd", 768)),
        "num_heads": getattr(config, "num_attention_heads", getattr(config, "n_head", 12)),
        "head_dim": getattr(config, "hidden_size", 768) // getattr(config, "num_attention_heads", 12),
        "vocab_size": getattr(config, "vocab_size", 50257),
        "max_seq_len": getattr(config, "max_position_embeddings", getattr(config, "n_positions", 2048)),
        "arch": arch_type,
    }

# ─── Parameter Categorization ────────────────────────────────────

# Patterns for shared weights (embeddings, output head, final norm)
SHARED_PATTERNS = {
    "gpt2": [
        "transformer.wte", "transformer.wpe", "transformer.ln_f", "lm_head",
    ],
    "llama": [
        "model.embed_tokens", "model.norm", "lm_head",
    ],
    "phi3": [
        "model.embed_tokens", "model.norm", "lm_head",
    ],
    "phi": [
        "model.embed_tokens", "model.final_layernorm", "lm_head",
    ],
    "mistral": [
        "model.embed_tokens", "model.norm", "lm_head",
    ],
    "qwen2": [
        "model.embed_tokens", "model.norm", "lm_head",
    ],
}

# Patterns for identifying layer index from parameter name
LAYER_PATTERNS = {
    "gpt2": "transformer.h.",
    "llama": "model.layers.",
    "phi3": "model.layers.",
    "phi": "model.layers.",
    "mistral": "model.layers.",
    "qwen2": "model.layers.",
}


def categorize_params(model, arch_info, num_shards):
    """Categorize model parameters into shared + per-shard groups."""
    arch = arch_info["arch"]
    num_layers = arch_info["num_layers"]
    layers_per_shard = num_layers // num_shards

    if num_layers % num_shards != 0:
        # Handle uneven splits — last shard gets the remainder
        print(f"  Warning: {num_layers} layers don't divide evenly into {num_shards} shards")
        print(f"  Shards 0-{num_shards-2} get {layers_per_shard} layers, last shard gets {num_layers - layers_per_shard * (num_shards - 1)}")

    shared_prefixes = SHARED_PATTERNS.get(arch, SHARED_PATTERNS["llama"])
    layer_prefix = LAYER_PATTERNS.get(arch, "model.layers.")

    shared_params = {}
    shard_params = {i: {} for i in range(num_shards)}

    state_dict = model.state_dict()

    for name, tensor in state_dict.items():
        np_tensor = tensor.cpu().float().numpy()

        # Check if shared
        if any(name.startswith(prefix) for prefix in shared_prefixes):
            shared_params[name] = np_tensor
            continue

        # Check if layer-specific
        if name.startswith(layer_prefix):
            # Extract layer number
            after_prefix = name[len(layer_prefix):]
            layer_idx = int(after_prefix.split(".")[0])

            # Assign to shard
            shard_id = min(layer_idx // layers_per_shard, num_shards - 1)
            shard_params[shard_id][name] = np_tensor
            continue

        # Anything else → shared
        shared_params[name] = np_tensor

    return shared_params, shard_params, layers_per_shard

# ─── Writing Shards ──────────────────────────────────────────────

def write_shard(tensors, filename, manifest_entries, dtype):
    """Write tensors to a binary file with optional quantization."""
    filepath = os.path.join(SHARDS_DIR, filename)
    offset = 0
    original_size = 0
    quantized_size = 0

    with open(filepath, "wb") as f:
        for name, tensor in tensors.items():
            original_size += tensor.size * 4  # original float32 size

            raw_bytes, quant_meta = quantize_tensor(tensor, dtype)
            f.write(raw_bytes)

            entry = {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": quant_meta["dtype"],
                "file": filename,
                "offset": offset,
                "size": len(raw_bytes),
            }
            # Add quantization metadata if applicable
            if quant_meta.get("quant"):
                entry["quant"] = quant_meta

            manifest_entries.append(entry)
            offset += len(raw_bytes)
            quantized_size += len(raw_bytes)

    file_size = os.path.getsize(filepath)
    ratio = original_size / quantized_size if quantized_size > 0 else 1
    print(f"  {filename}: {file_size / 1024 / 1024:.1f} MB ({len(tensors)} tensors, {ratio:.1f}x compression)")
    return manifest_entries

# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Split transformer models for Synapse distributed inference")
    parser.add_argument("--model", default="gpt2",
                        help="Model name or HuggingFace ID (default: gpt2)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "int8", "int4"],
                        help="Weight dtype / quantization (default: float32)")
    parser.add_argument("--num-shards", type=int, default=2,
                        help="Number of shards to split into (default: 2)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: model/shards/)")
    args = parser.parse_args()

    global SHARDS_DIR
    if args.output_dir:
        SHARDS_DIR = args.output_dir
    os.makedirs(SHARDS_DIR, exist_ok=True)

    # Step 1: Load model
    model, config = load_model(args.model)
    arch_info = detect_architecture(model, config)

    print(f"\nModel: {arch_info['hf_name']}")
    print(f"Architecture: {arch_info['arch']}")
    print(f"Layers: {arch_info['num_layers']}, Hidden: {arch_info['hidden_size']}, Heads: {arch_info['num_heads']}")
    print(f"Vocab: {arch_info['vocab_size']:,}, Max seq: {arch_info['max_seq_len']}")
    print(f"Quantization: {args.dtype}")
    print(f"Shards: {args.num_shards}")

    # Step 2: Categorize parameters
    print("\nCategorizing parameters...")
    shared_params, shard_params, layers_per_shard = categorize_params(
        model, arch_info, args.num_shards
    )
    print(f"  Shared: {len(shared_params)} tensors")
    for i in range(args.num_shards):
        layer_start = i * layers_per_shard
        layer_end = min((i + 1) * layers_per_shard - 1, arch_info["num_layers"] - 1)
        print(f"  Shard {i} (layers {layer_start}-{layer_end}): {len(shard_params[i])} tensors")

    # Step 3: Write binary files
    print(f"\nWriting shard files ({args.dtype})...")
    manifest_entries = []

    # Shared weights — always float16 minimum (embeddings need precision)
    shared_dtype = "float16" if args.dtype in ("int8", "int4") else args.dtype
    write_shard(shared_params, "shared.bin", manifest_entries, shared_dtype)

    for i in range(args.num_shards):
        write_shard(shard_params[i], f"shard_{i}.bin", manifest_entries, args.dtype)

    # Step 4: Build shard layout
    shard_layout = {}
    for i in range(args.num_shards):
        layer_start = i * layers_per_shard
        layer_end = min((i + 1) * layers_per_shard - 1, arch_info["num_layers"] - 1)
        shard_layout[str(i)] = {
            "layer_start": layer_start,
            "layer_end": layer_end,
            "file": f"shard_{i}.bin",
        }

    # Step 5: Write manifest
    manifest = {
        "model": arch_info["hf_name"],
        "arch": arch_info["arch"],
        "num_layers": arch_info["num_layers"],
        "hidden_size": arch_info["hidden_size"],
        "num_heads": arch_info["num_heads"],
        "head_dim": arch_info["head_dim"],
        "vocab_size": arch_info["vocab_size"],
        "max_seq_len": arch_info["max_seq_len"],
        "dtype": args.dtype,
        "layers_per_shard": layers_per_shard,
        "num_shards": args.num_shards,
        "shard_layout": shard_layout,
        "shared_file": "shared.bin",
        "shared_dtype": shared_dtype,
        "tensors": manifest_entries,
    }

    manifest_path = os.path.join(SHARDS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  manifest.json written ({len(manifest_entries)} tensor entries)")

    # Summary
    total_params = sum(t.size for t in shared_params.values())
    total_params += sum(t.size for s in shard_params.values() for t in s.values())
    fp32_size = total_params * 4
    quantized_size = total_params * bytes_per_param(args.dtype)

    print(f"\n{'='*50}")
    print(f"Model:        {arch_info['hf_name']}")
    print(f"Parameters:   {total_params:,}")
    print(f"FP32 size:    {fp32_size / 1024 / 1024:.1f} MB")
    print(f"{args.dtype} size:  {quantized_size / 1024 / 1024:.1f} MB ({fp32_size / quantized_size:.1f}x compression)")
    print(f"Per node:     ~{quantized_size / args.num_shards / 1024 / 1024:.1f} MB (shard) + shared")
    print(f"{'='*50}")

    # Estimate if this fits on mobile
    per_node_mb = quantized_size / args.num_shards / 1024 / 1024
    shared_mb = sum(t.size for t in shared_params.values()) * bytes_per_param(shared_dtype) / 1024 / 1024
    total_per_node = per_node_mb + shared_mb

    print(f"\nPer-node GPU memory estimate: ~{total_per_node:.0f} MB")
    if total_per_node < 500:
        print("  -> Fits comfortably on mobile (Pixel 10, S26 Ultra)")
    elif total_per_node < 1500:
        print("  -> Fits on mobile with care (keep other apps closed)")
    elif total_per_node < 2500:
        print("  -> Tight on mobile — desktop recommended")
    else:
        print("  -> Too large for mobile — use more shards or stronger quantization")

    print(f"\nDone! Shards ready in: {SHARDS_DIR}")


if __name__ == "__main__":
    main()
