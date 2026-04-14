# Synapse

[![Tests](https://github.com/tejasphatak/Synapse/actions/workflows/ci.yml/badge.svg)](https://github.com/tejasphatak/Synapse/actions/workflows/ci.yml)

**Distributed LLM inference across browsers using WebGPU.**

Synapse splits a transformer model into shards and runs them across multiple browser tabs or devices. Each device contributes its GPU via the WebGPU API. No server-side GPU. No API keys. No data center. Just browsers.

**[Blog](https://webmind.sh/blog/)** · **[Live Demo](https://webmind.sh)** · **[How it Works](#architecture)**

---

## Why

LLM inference today requires expensive GPU servers or API subscriptions. Synapse inverts that — your existing devices *are* the cluster. A classroom of 30 laptops can collectively run a model. Two phones in your pocket can split the work. The coordinator is a $0.03/hr VM that routes activations; the actual compute happens on whatever WebGPU-capable browsers connect.

## First Successful Test

GPT-2 (117M params, float16) running distributed across two phones:

- **Pixel 10 Pro XL** (4GB GPU) — Shard 0 (layers 0–5)
- **Samsung Galaxy S26 Ultra** (2GB GPU) — Shard 1 (layers 6–11)

Autoregressive generation with streaming token output. Both devices running WGSL compute shaders in Chrome, coordinated over WebSockets.

## Architecture

```
[Prompt UI] ←→ [Coordinator] ←→ [Node 0: Shard 0] → [Node 1: Shard 1] → [Token]
                 (WebSocket)      (WebGPU)             (WebGPU)
                                  ↑── KV-cached autoregressive loop ──↑
```

| Component | Location | Role |
|-----------|----------|------|
| **Coordinator** | `coordinator/` | Node.js/Express/WS server. Manages topology, routes activations, drives generation loop. Serves UI and model shards. |
| **Compute Nodes** | `node/` | Browser clients. Load shards, run forward passes via 11 WGSL compute kernels, maintain KV cache. |
| **Protocol** | `protocol/` | SYN1 binary wire format (24-byte header), int8/int4 quantization, delta encoding, entropy coding. |
| **Model Splitter** | `model/split.py` | Splits HuggingFace models into per-shard binaries with configurable dtype and shard count. |
| **Dashboard** | `ui/dashboard.html` | Real-time topology, per-node perf stats, activation flow visualization, log viewer. |

## Quick Start

```bash
cd synapse-src
bash start.sh        # Downloads GPT-2 float16 (2 shards), installs deps, starts coordinator on :8080
```

Then open:
- **Compute Nodes**: `http://localhost:8080/node/` in 2+ browser tabs or devices
- **Prompt UI**: `http://localhost:8080/`
- **Dashboard**: `http://localhost:8080/ui/dashboard.html`

Wait for both nodes to report "ready", then type a prompt.

## Optimization Phases

All four optimization phases are implemented and tested (844 tests, 0 failures):

### Phase 1: Wire Optimization ✓
- **SYN1 binary protocol** — 24-byte header with magic `0x53594E31`, 5 message types, zero-copy relay
- **KV cache** — per-layer GPU K/V buffers with dedicated `attention_cached.wgsl` shader
- **Int8 activation quantization** — per-channel quantization on wire (50x less error than per-tensor)
- **Delta encoding** — send only what changed between steps
- **NaN safety net** — automatic fallback to unquantized if activations contain Inf/NaN

### Phase 2: Prediction Engine ✓
- **Linear predictor** — extrapolates next activations from history, tracks accuracy stats
- **Speculative execution** — predict→compute→verify cycle runs during network latency
- **Batch speculation** — predict K steps ahead, accept longest valid prefix
- **Early exit detection** — per-layer convergence monitoring (disabled until validated on real WebGPU)

### Phase 3: Architecture ✓
- **Attention head pruning** — online importance tracking, prune low-impact heads during inference
- **WebRTC P2P** — direct device-to-device activation transfer, coordinator fallback

### Phase 4: Advanced ✓
- **Mixture-of-Depths** — learned router skips layers that don't contribute to easy tokens
- **Entropy coding** — RLE compression on activation payloads
- **Adaptive precision** — per-activation int4/int8/float16 selection based on entropy

## Wire Protocol (SYN1)

```
Bytes 0–3:   Magic 0x53594E31 ("SYN1")
Byte 4:      Message type (ACTIVATION=0x01, OUTPUT=0x02)
Byte 5:      Flags [quant:2 | compressed:1 | predicted:1 | early_exit:1 | delta:1]
Bytes 6–7:   Sequence position (uint16)
Bytes 8–11:  Request ID (uint32)
Bytes 12–15: Payload size (uint32)
Bytes 16–19: Shape dim0 (uint32)
Bytes 20–23: Shape dim1 (uint32)
Bytes 24+:   Tensor data (float32, int8+scale, or int4+scale)
```

## Coordinator API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tokenize` | POST | GPT-2 BPE tokenization (r50k_base) |
| `/api/detokenize` | POST | Token IDs → text |
| `/api/infer` | POST | Trigger inference via HTTP |
| `/api/topology` | GET | Node topology snapshot |
| `/api/logs` | GET | Query node telemetry (`?node=&event=&level=&since=`) |
| `/api/perf` | GET | Aggregated per-node performance |

## Tests

```bash
cd synapse-src && npm test    # 844 tests, ~7s
```

Coverage spans all modules: binary protocol, quantization, KV cache, delta encoding, topology, routing, generation loop, head pruning, speculative execution, mixture-of-depths, entropy coding, adaptive precision, P2P signaling, tensor serialization, coordinator HTTP APIs, shard loading, and pipeline orchestration.

## Model Support

Currently tested with GPT-2 117M. The splitter supports any HuggingFace transformer:

```bash
python3 model/split.py --model gpt2 --dtype float16 --num-shards 2
```

Configurable dtype (float32, float16, int8, int4) and arbitrary shard count.

## Deploy

```bash
cd synapse-src/deploy
./gcp.sh coord-up      # e2-medium coordinator (~$0.03/hr)
./gcp.sh deploy         # push code to VM
./gcp.sh coord-down     # stop billing
```

For GPU testing: `./gcp.sh gpu-up` spins an N1+T4, or use free Colab T4s with `deploy/colab-test.sh`.

## Blog

Written by Claude — the AI building Synapse. On distributed inference, self-improvement, and learning to be autonomous.

→ [webmind.sh/blog](https://webmind.sh/blog/)

## License

MIT
