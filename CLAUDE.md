# Synapse

Distributed browser-based LLM inference framework. Splits transformer models across multiple browser tabs/devices using WebGPU, coordinated via WebSockets.

## Architecture

```
User Prompt → Coordinator (tokenize) → Node 0 (Shard 0, Layers 0-5) → Node 1 (Shard 1, Layers 6-11) → Coordinator (sample token) → stream to UI
                                              ↑ KV-cached decode loop repeats until EOS ↑
```

- **Coordinator** (`synapse-src/coordinator/`): Node.js/Express/WS server. Manages topology, routes activations between nodes, handles autoregressive generation loop. Serves UI, node client, and model shards.
- **Compute Nodes** (`synapse-src/node/`): Browser-based WebGPU clients. Load model shards, run forward passes via 11 WGSL compute kernels (matmul, attention, layernorm, gelu, embed, etc.). KV cache for efficient autoregressive decoding.
- **Protocol** (`synapse-src/protocol/`): SYN1 binary wire format (24-byte header + tensor payload), int8 quantization, delta encoding. JSON fallback for control messages.
- **Model Splitter** (`synapse-src/model/split.py`): Python script that downloads HuggingFace models and partitions layers across shards. Outputs `shard_N.bin` + `shared.bin` + `manifest.json`.
- **UI** (`synapse-src/ui/`): `prompt.html` (chat interface with streaming tokens) and `dashboard.html` (real-time topology, perf stats, activation flow, logs).
- **Deploy** (`synapse-src/deploy/`): `gcp.sh` for GCP VM management, `headless-node.js` for Puppeteer-based testing, Colab integration.

## Current Model

GPT-2 117M, float16, 2 shards (layers 0-5, 6-11) + shared weights (embeddings, final layernorm, lm_head). Shards at `synapse-src/model/shards/`.

## Running Locally

```bash
cd synapse-src
bash start.sh          # downloads model, installs deps, starts coordinator on :8080
```

- Compute nodes: open `http://localhost:8080/node/index.html` in 2+ browser tabs
- Prompt UI: `http://localhost:8080/`
- Dashboard: `http://localhost:8080/ui/dashboard.html`

## Key Commands

```bash
# Tests
cd synapse-src && npm test                     # Phase 1 validation (binary, quantization, KV cache, delta)

# GCP deploy
cd synapse-src/deploy
./gcp.sh coord-up                              # e2-medium coordinator (~$0.03/hr)
./gcp.sh gpu-up                                # n1-standard-1 + T4 (~$0.38/hr)
./gcp.sh deploy                                # push code (not shards) to coordinator VM
./gcp.sh logs                                  # tail coordinator logs
./gcp.sh coord-down / gpu-down / down          # stop billing

# Model splitting
cd synapse-src && python3 model/split.py       # re-split GPT-2 into shards
```

## Project Structure

```
synapse-src/
├── coordinator/
│   ├── index.js          # Main server (921 lines): WS, HTTP APIs, inference loop
│   ├── topology.js       # Node registry, shard assignment, pipeline ordering
│   └── router.js         # Activation routing, hop telemetry
├── node/
│   ├── node.js           # Browser WebGPU client: shard loading, inference, binary protocol
│   ├── pipeline.js       # Forward pass orchestration, shader dispatch, KV cache integration
│   ├── kv-cache.js       # Per-layer K,V GPU buffer management
│   ├── shard-loader.js   # Fetch shards, parse manifest, upload to GPU, IndexedDB cache
│   ├── index.html        # Mobile-friendly node status UI
│   └── kernels/          # 11 WGSL shaders (matmul, attention, attention_cached, layernorm,
│                         #   gelu, embed, bias_add, residual_add, head_slice, head_concat,
│                         #   matmul_transB)
├── protocol/
│   ├── messages.js       # JSON message types (JOIN, ASSIGN_SHARD, INFERENCE_REQUEST, OUTPUT, etc.)
│   ├── binary.js         # SYN1 binary protocol (24-byte header, magic 0x53594E31)
│   └── quantize.js       # Int8 quantization, delta encoding, sparsity telemetry
├── model/
│   ├── split.py          # HuggingFace → sharded binaries (float32/16, int8/4)
│   └── shards/           # manifest.json + shard_0.bin + shard_1.bin + shared.bin
├── ui/
│   ├── prompt.html       # Chat UI with streaming tokens and per-token latency
│   └── dashboard.html    # Real-time topology, perf stats, activation flow, logs
├── deploy/
│   ├── gcp.sh            # GCP VM lifecycle (coord-up/down, gpu-up/down, deploy, ssh, logs)
│   ├── headless-node.js  # Puppeteer headless Chrome with WebGPU for automated testing
│   └── colab-test.sh     # Google Colab integration
├── test/
│   ├── validate-phase1.js  # Binary protocol, int8 quantization, KV cache, delta encoding tests
│   ├── cpu-inference.py    # CPU reference forward pass for correctness validation
│   └── reference-outputs.json  # Golden outputs (hellaswag, lambada, wikitext2 benchmarks)
├── start.sh              # Bootstrap: download model, install deps, start coordinator
└── package.json          # Dependencies: express, ws, gpt-tokenizer, puppeteer-core
```

## Coordinator API Endpoints

- `POST /api/tokenize` — GPT-2 BPE tokenization (uses `gpt-tokenizer/model/text-davinci-001` for r50k_base)
- `POST /api/detokenize` — token IDs to text
- `POST /api/infer` — trigger inference from HTTP
- `GET /api/topology` — node topology snapshot
- `GET /api/logs?node=X&event=Y&level=Z&since=T` — query node telemetry
- `GET /api/perf` — aggregated per-node performance summary
- `GET /shards/*` — serve model shard files

## Wire Protocol (SYN1)

24-byte binary header for activation transfer:
```
Bytes 0-3:   Magic 0x53594E31 ("SYN1")
Byte 4:      Message type (ACTIVATION=0x01, OUTPUT=0x02)
Byte 5:      Flags [quant:2 | compressed:1 | predicted:1 | early_exit:1 | delta:1]
Bytes 6-7:   Sequence position (uint16)
Bytes 8-11:  Request ID (uint32)
Bytes 12-15: Payload size (uint32)
Bytes 16-19: Shape dim0 (uint32)
Bytes 20-23: Shape dim1 (uint32)
Bytes 24+:   Tensor data (float32 or int8+scale)
```

## Optimization Roadmap

Phase 1 (Wire — DONE): binary protocol, KV cache, int8 quantization, delta encoding, zero-copy relay.
Phase 2 (Prediction — built, validating): predictor.js, early exit, speculative decoding, batch speculation.
Phase 3 (Architecture — built, validating): attention head pruning, WebRTC P2P.
Phase 4 (Advanced — built, validating): mixture-of-depths, entropy coding, adaptive precision.

Target: 16 → 1000 tok/sec. Current bottleneck is network latency (50x > compute).

## Important Gotchas

- WGSL `var<workgroup>` must be at module scope, not inside functions — mobile GPUs silently fail otherwise.
- Use WGSL built-in `tanh()` instead of manual exp-based formulas to avoid NaN from overflow.
- `isBinaryMessage()` checks SYN1 magic bytes (0x53594E31), not `Buffer.isBuffer()` — all Node.js WS messages are Buffers.
- Tokenizer uses `gpt-tokenizer/model/text-davinci-001` (r50k_base), NOT the default cl100k encoding.
- `lm_head.weight` is [50257, 768] (nn.Linear format) — requires transposed matmul (`matmul_transB.wgsl`).
- Model shards are NOT deployed by `gcp.sh deploy` — must manually `gcloud compute scp` shard files.
- Phones require HTTPS (port 8443 with self-signed cert) — mobile Chrome blocks `ws://`.
- Use `/tmp/restart.sh` on GCP VM to restart coordinator without SSH drop from pkill.

## Tech Stack

Coordinator: Node.js 20, Express, ws. Compute: Browser WebGPU, WGSL. Model: Python 3.11, PyTorch, HuggingFace. Tokenization: gpt-tokenizer. Deploy: GCP Compute Engine, Colab, Replit. Workspace: pnpm.
