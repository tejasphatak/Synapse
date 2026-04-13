# Synapse

**Distributed browser LLM inference over WebGPU.**

Synapse splits a transformer model into shards and runs them across multiple browser tabs or devices, coordinated over WebSockets. Each device contributes its GPU via the WebGPU API — no server-side GPU required.

## How It Works

```
[Prompt UI] ←→ [Coordinator Server] ←→ [Node 1: Shard 0] → [Node 2: Shard 1] → [Output]
                     (WebSocket)          (WebGPU)              (WebGPU)
```

A Node.js coordinator assigns model shards to browser clients. Each client loads its shard weights, runs transformer layers on the device GPU via WebGPU compute shaders (WGSL), and passes activations to the next node in the pipeline. The coordinator drives autoregressive token generation.

## Status

**Working end-to-end** with GPT-2 117M (float16) across two phones:
- **Pixel 10 Pro XL** (4GB GPU) — Shard 0 (layers 0-5)
- **Samsung Galaxy S26 Ultra** (2GB GPU) — Shard 1 (layers 6-11)

### Optimization Progress

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1: Wire Optimization | Binary protocol, KV cache, int8 quantization, delta encoding | ~85% done |
| Phase 2: Prediction Engine | Predictor, early exit, speculative decoding | Not started |
| Phase 3: Architecture | Head pruning, WebRTC P2P | Not started |
| Phase 4: Advanced | Mixture-of-Depths, adaptive precision | Not started |

### Known Issue: WebGPU Shader Bugs (Fixed, Needs Testing)

Three critical bugs were found in the WebGPU compute pipeline that caused garbage output:

1. **LayerNorm `var<workgroup>` scoping** — Workgroup-shared variables were declared inside the function body instead of at module scope. Mobile GPUs silently failed, producing all-zero outputs from LayerNorm. Every subsequent layer received garbage input.

2. **GELU `exp()` overflow** — The GELU activation used a manual `tanh` approximation via `(exp(2x)-1)/(exp(2x)+1)`. For large inputs, `exp(2x)` overflows to infinity, producing `inf/inf = NaN`. NaN then propagated through all subsequent layers. Fixed by using WGSL's built-in `tanh()`.

3. **`isBinaryMessage` false positives** — The binary protocol detector checked `Buffer.isBuffer()`, which is true for ALL Node.js WebSocket messages. JSON messages (JOIN, OUTPUT) were silently routed to the binary handler and dropped. Fixed by checking for the SYN1 magic bytes (0x53594E31).

**Additional fixes:** server-side GPT-2 tokenizer (mobile browsers failed to load CDN tokenizer), HTTPS/WSS support (mobile Chrome blocks insecure WebSockets), `lm_head` weight transpose for correct logit projection.

These fixes have been deployed but need end-to-end verification on physical devices or headless Chrome with WebGPU.

## Quick Start

```bash
cd synapse-src
npm install
python3 model/split.py --model gpt2 --dtype float16 --num-shards 2
npm start
```

The coordinator serves:
- **Prompt UI**: `http://localhost:8080/`
- **Compute Node**: `http://localhost:8080/node/index.html` (open on 2+ devices)
- **Dashboard**: `http://localhost:8080/ui/dashboard.html`

Open the node URL on two devices with WebGPU support, wait for "ready", then type a prompt.

### GCP Deployment

```bash
cd synapse-src
bash deploy/gcp.sh coord-up    # ~$0.03/hr coordinator VM
bash deploy/gcp.sh deploy       # push code
bash deploy/gcp.sh coord-down   # stop billing
```

HTTPS (self-signed cert) is required for mobile devices — served on port 8443.

## Architecture

- **Coordinator** (`synapse-src/coordinator/`): Node.js/Express/WS server — shard assignment, activation routing, autoregressive generation loop, server-side GPT-2 tokenization
- **Compute Nodes** (`synapse-src/node/`): Browser clients — WebGPU compute shaders for matmul, attention, layernorm, GELU, embedding. IndexedDB shard caching
- **Protocol** (`synapse-src/protocol/`): Binary wire format (SYN1 24-byte header), int8 activation quantization, delta encoding
- **Model Splitter** (`synapse-src/model/split.py`): Splits HuggingFace models into shards (float32/float16/int8/int4)
- **Deploy** (`synapse-src/deploy/`): GCP automation, headless Chrome node launcher, GPU VM setup

## Model Support

Currently tested with GPT-2. The splitter supports any HuggingFace causal LM:

```bash
python3 model/split.py --model gpt2-medium --dtype float16 --num-shards 4
python3 model/split.py --model gpt2-xl --dtype int8 --num-shards 8
```

## License

MIT
