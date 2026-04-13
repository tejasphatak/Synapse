# Synapse

### The Internet is the computer. Every browser is a GPU.

Synapse is a distributed inference engine that splits LLMs across multiple browsers and devices — phones, tablets, laptops — coordinated over WebSockets, computed entirely on-device via WebGPU. No cloud GPUs. No API keys. No cost per token.

> **30 phones in a classroom can collectively run a language model. That's Synapse.**

---

## What Just Happened (April 13, 2026)

GPT-2 running across a **Pixel 10 Pro XL** (PowerVR GPU) and an **iPhone 16 Pro** (Apple GPU), coordinated from a $0.03/hr GCP VM. Two different GPU architectures, two different operating systems, one distributed brain.

```
Pixel 10 Pro XL  ──→  Layers 0-5  (PowerVR, 4GB)  ──→  activations ──→
iPhone 16 Pro    ──→  Layers 6-11 (Apple GPU, 1GB)  ──→  token output ──→  streaming response

15 tokens generated at 1.3 tok/sec — first cross-platform distributed inference in browsers.
```

---

## Why This Matters

Every phone has a GPU sitting idle. Every laptop, every tablet. Billions of GPUs, unused.

Right now, running an LLM means either:
- Pay $1000s for cloud GPUs
- Pay per-token to an API
- Own expensive hardware

Synapse says: **what if we just use the GPUs that already exist?** Split the model, distribute the work, run inference across whatever devices are available. A classroom of Chromebooks. A family's phones at dinner. A mesh of browsers across the internet.

This is the SETI@home of AI inference.

---

## Quick Start

```bash
git clone https://github.com/tejasphatak/Synapse.git
cd Synapse/synapse-src
npm install
python3 model/split.py --model gpt2 --dtype float16 --num-shards 2
node coordinator/index.js
```

Open **two browser tabs** at `http://localhost:8080/node/index.html`. Each tab loads a shard and becomes a compute node. Once both show "Ready", type a message in the chat panel — tokens stream in.

For phones: deploy to any server with HTTPS and open the node URL on each device.

---

## How It Works

```
                        ┌─────────────────────┐
                        │   Coordinator        │
                        │   (Node.js + WS)     │
    Chat UI  ◄──WSS──► │  Route activations   │ ◄──WSS──►  Dashboard
                        │  Sample tokens       │
                        └─────────┬────────────┘
                                  │ binary SYN1 protocol
                       ┌──────────┼──────────┐
                       ▼                      ▼
                ┌─────────────┐       ┌─────────────┐
                │  Device A   │       │  Device B   │
                │  WebGPU     │──────►│  WebGPU     │
                │  Layers 0-5 │ int8  │  Layers 6-11│
                │  4GB GPU    │       │  1GB GPU    │
                └─────────────┘       └─────────────┘
```

1. **Split** — Python script slices any HuggingFace model into N shards
2. **Load** — Each browser downloads its shard, cached in IndexedDB for instant reload
3. **Compute** — 11 custom WGSL shaders: matmul, multi-head attention, LayerNorm, GELU, embeddings
4. **Route** — SYN1 binary protocol with per-channel int8 quantization (5.3x compression)
5. **Generate** — Autoregressive loop with KV cache: prefill once, decode at O(1) per token

Each device is both a **compute node** and a **chat client** — one page does everything.

---

## Optimization Stack

Inspired by VLSI design: in chips, wire delay >> gate delay. In distributed inference, network latency >> GPU compute. Every optimization targets the wire.

| Phase | What | Impact | Status |
|-------|------|--------|--------|
| **Wire** | Binary protocol, KV cache, per-channel int8, delta encoding, zero-copy relay | 15x | **Done** |
| **Prediction** | Activation predictor, speculative execution, early exit detection | 3-5x | **Built** (validating) |
| **Architecture** | Attention head pruning (25% compute reduction), WebRTC P2P | 2-3x | **Built** (validating) |
| **Advanced** | Entropy coding, mixture-of-depths, adaptive precision | 1.5-2x | In progress |

**Current:** 1.3 tok/sec across two phones over cellular network.
**Target:** 100+ tok/sec with all optimizations enabled.
**Theoretical bound:** Activation payloads can shrink from 4,400 → ~200 bytes/token (Shannon entropy limit). 20x headroom.

---

## What's Inside

```
synapse-src/
├── coordinator/        # Node.js server — WS routing, generation loop, tokenization
├── node/               # Browser client — WebGPU compute, chat UI, PWA
│   ├── kernels/        # 11 WGSL shaders (matmul, attention, layernorm, gelu, embed...)
│   ├── predictor.js    # Activation prediction (linear extrapolation)
│   ├── speculative.js  # Speculative execution controller
│   ├── early-exit.js   # Per-layer convergence detection
│   ├── head-pruning.js # Online attention head importance measurement
│   └── p2p.js          # WebRTC data channel for direct node-to-node transfer
├── protocol/           # SYN1 binary format, per-channel int8 quantization, delta encoding, RLE
├── model/              # Python model splitter (float32/16, int8/4, configurable shards)
├── deploy/             # GCP automation, headless Chrome, Colab integration
├── ui/                 # Dashboard with live metrics, QR codes for phone onboarding
└── test/               # 48 tests — protocol, quantization, prediction, pruning, CPU validation
```

---

## Run Any Model

```bash
# GPT-2 variants
python3 model/split.py --model gpt2 --dtype float16 --num-shards 2
python3 model/split.py --model gpt2-medium --dtype float16 --num-shards 4
python3 model/split.py --model gpt2-xl --dtype int8 --num-shards 8
```

Supports float32, float16, int8, and int4 quantization. More shards = more devices = bigger models.

---

## Deploy

```bash
# GCP (coordinator: ~$0.03/hr)
cd synapse-src/deploy
./gcp.sh coord-up      # Spin up VM
./gcp.sh deploy         # Push code
./gcp.sh coord-down     # Stop billing

# Or any server with Node.js
node coordinator/index.js   # That's it
```

Phones need HTTPS for WebGPU. Use Let's Encrypt + nip.io for free SSL on any IP.

---

## The Bugs That Made This Real

Three bugs that took days to find and seconds to fix:

1. **LayerNorm zeros** — WGSL `var<workgroup>` inside function body. Desktop Chrome accepted it. Mobile GPUs silently returned zeros. Root cause of all garbage output.

2. **GELU NaN** — `(exp(2x)-1)/(exp(2x)+1)` overflows to inf/inf = NaN. One NaN in layer 3 poisons everything downstream. Fixed with WGSL built-in `tanh()`.

3. **WebSocket black hole** — `Buffer.isBuffer()` returns true for ALL Node.js WS messages. JSON control messages were silently swallowed by the binary decoder. Nodes appeared connected but never registered.

**Lesson:** When distributed GPU inference produces garbage, the bug is in the part you'd never think to check.

---

## Built By

**Tejas Phatak** — Principal Engineer at Mastercard. Architecture, systems, the original vision.

**Claude** — Anthropic's AI. Co-architect, shader debugger, optimization engine. Lives on a GCP VM.

This project was built through human-AI collaboration. Every commit, every debug session, every design decision — made together.

---

## License

MIT — use it, fork it, distribute inference across everything.
