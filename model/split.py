#!/usr/bin/env python3
"""
Download GPT-2 (117M) weights from HuggingFace and split them into two shards
for distributed inference across two browser nodes.

Shard 0: Transformer layers 0-5
Shard 1: Transformer layers 6-11
Shared:  Token embeddings + positional embeddings + final layernorm + lm_head

Output:
  shards/shard_0.bin     - layers 0-5 weights concatenated
  shards/shard_1.bin     - layers 6-11 weights concatenated
  shards/shared.bin      - shared weights (embeddings, final layernorm, lm_head)
  shards/manifest.json   - tensor names, shapes, offsets, dtypes
"""

import json
import struct
import os
import sys
import numpy as np

try:
    from transformers import GPT2LMHeadModel
except ImportError:
    print("Error: 'transformers' package required. Install with:")
    print("  pip install transformers torch")
    sys.exit(1)


SHARDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shards")
NUM_LAYERS = 12
LAYERS_PER_SHARD = 6  # 2 shards of 6 layers each


def download_model():
    """Download GPT-2 small (117M) from HuggingFace."""
    print("Downloading GPT-2 (117M) from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    print("Download complete.")
    return model


def categorize_params(model):
    """Categorize model parameters into shared, shard_0, and shard_1."""
    shared_params = {}
    shard_params = {0: {}, 1: {}}

    state_dict = model.state_dict()

    for name, tensor in state_dict.items():
        np_tensor = tensor.cpu().numpy().astype(np.float32)

        # Shared weights: embeddings, final layernorm, lm_head
        if any(name.startswith(prefix) for prefix in [
            "transformer.wte",      # token embeddings
            "transformer.wpe",      # positional embeddings
            "transformer.ln_f",     # final layer norm
            "lm_head",              # language model head
        ]):
            shared_params[name] = np_tensor

        # Layer-specific weights
        elif name.startswith("transformer.h."):
            # Extract layer number: "transformer.h.{N}.xxx"
            parts = name.split(".")
            layer_idx = int(parts[2])
            shard_id = 0 if layer_idx < LAYERS_PER_SHARD else 1
            shard_params[shard_id][name] = np_tensor

        else:
            # Any other weights go to shared
            shared_params[name] = np_tensor

    return shared_params, shard_params


def write_shard(tensors, filename, manifest_entries):
    """Write a dict of tensors to a single binary file and record manifest entries."""
    filepath = os.path.join(SHARDS_DIR, filename)
    offset = 0

    with open(filepath, "wb") as f:
        for name, tensor in tensors.items():
            # Ensure contiguous C-order float32
            data = np.ascontiguousarray(tensor, dtype=np.float32)
            raw_bytes = data.tobytes()
            f.write(raw_bytes)

            manifest_entries.append({
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "float32",
                "file": filename,
                "offset": offset,
                "size": len(raw_bytes),
            })

            offset += len(raw_bytes)

    file_size = os.path.getsize(filepath)
    print(f"  {filename}: {file_size / 1024 / 1024:.1f} MB ({len(tensors)} tensors)")
    return manifest_entries


def main():
    os.makedirs(SHARDS_DIR, exist_ok=True)

    # Step 1: Download model
    model = download_model()

    # Step 2: Categorize parameters
    print("\nCategorizing parameters...")
    shared_params, shard_params = categorize_params(model)
    print(f"  Shared: {len(shared_params)} tensors")
    print(f"  Shard 0 (layers 0-5): {len(shard_params[0])} tensors")
    print(f"  Shard 1 (layers 6-11): {len(shard_params[1])} tensors")

    # Step 3: Write binary shard files
    print("\nWriting shard files...")
    manifest_entries = []
    write_shard(shared_params, "shared.bin", manifest_entries)
    write_shard(shard_params[0], "shard_0.bin", manifest_entries)
    write_shard(shard_params[1], "shard_1.bin", manifest_entries)

    # Step 4: Write manifest
    manifest = {
        "model": "gpt2",
        "num_layers": NUM_LAYERS,
        "hidden_size": 768,
        "num_heads": 12,
        "head_dim": 64,
        "vocab_size": 50257,
        "max_seq_len": 1024,
        "layers_per_shard": LAYERS_PER_SHARD,
        "num_shards": 2,
        "shard_layout": {
            "0": {"layer_start": 0, "layer_end": 5, "file": "shard_0.bin"},
            "1": {"layer_start": 6, "layer_end": 11, "file": "shard_1.bin"},
        },
        "shared_file": "shared.bin",
        "tensors": manifest_entries,
    }

    manifest_path = os.path.join(SHARDS_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  manifest.json written ({len(manifest_entries)} tensor entries)")

    # Summary
    total_params = sum(t.size for t in shared_params.values())
    total_params += sum(t.size for s in shard_params.values() for t in s.values())
    print(f"\nTotal parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB in float32)")
    print("Done! Shards ready in:", SHARDS_DIR)


if __name__ == "__main__":
    main()
