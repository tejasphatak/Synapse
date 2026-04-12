# Synapse

**Distributed browser LLM inference over WebGPU.**

Synapse splits a transformer model into shards and runs them across multiple browser tabs or devices, coordinated over WebSockets. Each device contributes its GPU via the WebGPU API — no server-side GPU required.

## First Successful Test (April 12, 2026)

GPT-2 (117M parameters, float16) running distributed across two phones:

- **Pixel 10 Pro XL** (4GB GPU) — Shard 0 (layers 0-5)
- **Samsung Galaxy S26 Ultra** (2GB GPU) — Shard 1 (layers 6-11)

Autoregressive generation completed successfully with streaming token output. Output quality has some junk characters — expected at this stage with per-tensor float16 quantization and no KV cache. The pipeline works end-to-end: prompt tokenization, distributed forward pass across both devices, token sampling, and multi-token generation loop.

## Architecture

```
[Prompt UI] ←→ [Coordinator Server] ←→ [Node 1: Shard 0] → [Node 2: Shard 1] → [Output]
                     (WebSocket)          (WebGPU)              (WebGPU)
```

- **Coordinator** (`coordinator/`): Node.js WebSocket server that assigns shards, routes activations between nodes, and drives the autoregressive generation loop
- **Compute Nodes** (`node/`): Browser clients that load model shards, run transformer layers on GPU via WebGPU compute shaders (WGSL), and pass activations to the next node
- **Protocol** (`protocol/`): Shared message types and validation for all WebSocket communication
- **Model Splitter** (`model/split.py`): Python script to split HuggingFace models into shard binaries with shared tensors (embeddings, final layernorm, lm_head)

## Key Features

- **WebGPU compute shaders** for matmul, attention, layernorm, GELU, embedding, bias addition, head slicing, and head concatenation — all running on-device
- **Zero GPU-CPU round-trips** in the forward pass (dedicated WGSL kernels replace all CPU fallbacks)
- **IndexedDB caching** — model shards (~150MB per device for float16 GPT-2) are cached in the browser, no re-download on refresh
- **Inter-layer buffer cleanup** — temporary GPU buffers are freed between layers to fit within mobile GPU memory limits (tested on 2GB Samsung GPU)
- **Mobile-ready** — pull-to-refresh disabled, fullscreen mode, wake lock, reconnect-on-resume
- **Streaming token generation** — tokens stream to the prompt UI as they're generated with per-token latency display

## Quick Start

```bash
cd synapse-src
bash start.sh        # Downloads GPT-2 (float16, 2 shards) and starts coordinator
```

The coordinator serves:
- **Prompt UI**: `http://localhost:8080/`
- **Dashboard**: `http://localhost:8080/ui/dashboard.html`
- **Compute Node**: `http://localhost:8080/node/` (open in 2+ browser tabs/devices)

Open the compute node URL on two devices, wait for both to report "ready", then type a prompt in the Prompt UI.

## Model Support

Currently tested with GPT-2. The splitter supports any HuggingFace transformer with configurable dtype (float32, float16, int8) and shard count:

```bash
cd model
python split.py --model gpt2 --dtype float16 --num-shards 2
```

## Status

This is a proof-of-concept. It works. The next targets are:
- KV cache to avoid recomputing attention for all previous tokens
- Reducing activation transfer size between nodes
- Better quantization (per-channel or mixed precision)
- Support for larger models (GPT-2 Medium, Qwen2, etc.)

## License

MIT
