# Synapse

### Run LLMs for free. In your browser. Across every device you own.

Synapse is a distributed inference engine that splits transformer models across multiple browser tabs, phones, and laptops — coordinated over WebSockets, computed entirely on-device via WebGPU. No cloud GPUs. No API keys. No cost per token.

> **The idea:** Every device with a browser has a GPU sitting idle. Synapse turns a group of phones into a distributed GPU cluster that runs real language models.

---

## Live Demo

GPT-2 117M running distributed across two phones over a cellular network:

```
Pixel 10 Pro XL  (4GB GPU)  ──→  Layers 0-5   ──→  activations  ──→
Galaxy S26 Ultra  (2GB GPU)  ──→  Layers 6-11  ──→  token output ──→  streaming response
```

First successful end-to-end distributed inference: **April 12, 2026.**

---

## Performance Journey

| Milestone | Latency/token | What changed |
|-----------|--------------|--------------|
| First working inference | ~3,000 ms | JSON protocol, no caching, full recompute every token |
| Binary protocol + KV cache | ~700 ms | SYN1 24-byte wire format, GPU-side KV cache skips prefill |
| Int8 quantization + delta encoding | ~300 ms | 4x smaller activation transfers, sparse delta compression |
| **Current** | **~200 ms** | Zero-copy relay, shard caching, optimized attention |
| Target (Phase 2-4) | <50 ms | Prediction engine, speculative decoding, WebRTC P2P |

---

## How It Works

```
                          ┌─────────────────────┐
                          │   Coordinator        │
                          │   (Node.js + WS)     │
                          │                      │
                          │  ┌──────────────┐    │
   Prompt UI  ◄──WSS──►  │  │ Shard Router  │    │
                          │  │ Token Sampler │    │
                          │  │ KV Cache Mgr  │    │
                          │  └──────┬───────┘    │
                          └─────────┼────────────┘
                                    │ binary activations
                         ┌──────────┼──────────┐
                         ▼                      ▼
                  ┌─────────────┐       ┌─────────────┐
                  │  Phone A    │       │  Phone B    │
                  │  WebGPU     │──────►│  WebGPU     │
                  │  Layers 0-5 │ activ │  Layers 6-11│
                  │  (Shard 0)  │       │  (Shard 1)  │
                  └─────────────┘       └─────────────┘
```

1. **Split** — A Python script slices any HuggingFace model into N shards (embeddings shared, layers partitioned)
2. **Load** — Each browser node downloads its shard (~80MB for GPT-2 fp16), cached in IndexedDB
3. **Compute** — Custom WGSL compute shaders run matmul, multi-head attention, layernorm, GELU — entirely on the device GPU
4. **Route** — The coordinator relays activation tensors between nodes using a compact binary protocol with int8 quantization
5. **Generate** — Autoregressive loop: embed → forward all shards → sample token → repeat

---

## Optimization Roadmap

Inspired by VLSI design principles — in chip design, wire delay dominates gate delay. In distributed inference, network latency dominates GPU compute. Every optimization targets the wire.

| Phase | Technique | Impact | Status |
|-------|-----------|--------|--------|
| **1. Wire Optimization** | Binary protocol (SYN1), KV cache, int8 activation quantization, delta encoding, zero-copy relay | 15x speedup | **85% complete** |
| **2. Prediction Engine** | Linear extrapolation predictor, speculative decoding, early exit | 3-5x | Next up |
| **3. Architecture** | Attention head pruning, WebRTC peer-to-peer (skip coordinator) | 2-3x | Planned |
| **4. Advanced** | Mixture-of-Depths routing, adaptive per-layer precision, entropy coding | 1.5-2x | Research |

**Theoretical bound:** Current activation payloads are ~4,400 bytes/token. Shannon entropy limit is ~100-200 bytes. There's 20x headroom left.

---

## Where We Are Right Now

**What works:**
- Full distributed inference pipeline across multiple devices
- Binary wire protocol with 24-byte SYN1 header
- GPU-side KV cache (prefill once, cached decode steps)
- Int8 activation quantization (<0.5% error) with delta encoding
- Centralized logging + real-time performance dashboard
- GCP deployment automation with HTTPS/WSS for mobile
- Server-side GPT-2 BPE tokenization
- CPU reference test that produces identical output to HuggingFace

**What we're debugging:**
- WebGPU shader correctness on mobile GPUs — found and fixed 3 critical bugs (LayerNorm workgroup scoping, GELU exp overflow, binary message detection). Fixes deployed, need end-to-end verification via headless Chrome with WebGPU on a cloud GPU instance.

---

## The Bugs That Taught Us About Mobile WebGPU

Debugging distributed GPU inference across phone browsers is uncharted territory. Here's what we hit:

**1. LayerNorm produced all zeros** — WGSL `var<workgroup>` variables declared inside the function body instead of module scope. Desktop Chrome silently accepted it. Mobile GPUs silently returned zeros. Every layer after the first received meaningless input. This single bug caused weeks of "garbage output."

**2. GELU activation produced NaN** — The tanh approximation `(exp(2x)-1)/(exp(2x)+1)` overflows to `inf/inf = NaN` for large inputs. One NaN in layer 3's FFN output poisons every subsequent computation. Fixed with WGSL's built-in `tanh()`.

**3. JSON messages silently dropped** — Node.js WebSocket delivers all messages as `Buffer` objects. Our `isBinaryMessage()` check returned true for everything, sending JSON JOIN messages into the binary decoder where they vanished. Nodes appeared connected but never registered.

**Lesson:** When your distributed system produces garbage, the bug is in the part you'd never think to check — the GPU shader scoping rules, the math library overflow behavior, the WebSocket type system.

---

## Quick Start

```bash
cd synapse-src
npm install
python3 model/split.py --model gpt2 --dtype float16 --num-shards 2
npm start
```

Then open:
- `http://localhost:8080/` — Prompt UI
- `http://localhost:8080/node/index.html` — Compute Node (open on 2+ devices)
- `http://localhost:8080/ui/dashboard.html` — Live performance dashboard

### Deploy to GCP

```bash
bash deploy/gcp.sh coord-up     # Spin up coordinator (~$0.03/hr)
bash deploy/gcp.sh deploy        # Push code + start server
bash deploy/gcp.sh coord-down    # Stop billing
```

### Run Any HuggingFace Model

```bash
python3 model/split.py --model gpt2-medium --dtype float16 --num-shards 4
python3 model/split.py --model gpt2-xl --dtype int8 --num-shards 8
```

Supports float32, float16, int8, and int4 quantization with configurable shard counts.

---

## Architecture

```
synapse-src/
├── coordinator/     # Node.js WebSocket server — routing, generation loop, tokenization
├── node/            # Browser compute client — WebGPU shaders, shard loading, KV cache
│   └── kernels/     # WGSL compute shaders — matmul, attention, layernorm, GELU, embed
├── protocol/        # Binary wire format (SYN1), int8 quantization, delta encoding
├── model/           # Python model splitter for HuggingFace transformers
├── deploy/          # GCP automation, headless Chrome launcher, GPU node setup
├── ui/              # Prompt interface, dashboard, GPU diagnostics
└── test/            # Phase 1 validation (25 tests), CPU reference inference
```

---

## Built With

- **WebGPU** + **WGSL** — GPU compute in the browser
- **WebSockets** — Real-time activation routing
- **Node.js** — Coordinator server
- **Python** + **HuggingFace Transformers** — Model splitting

## Built By

**Tejas Phatak** — architecture, systems, deployment

**Claude** (Anthropic) — pair programming, shader debugging, optimization design

---

## License

MIT
