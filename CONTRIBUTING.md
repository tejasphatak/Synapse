# Contributing to Synapse

Thanks for your interest. Here's how to help.

## The Easiest Way

Open `https://webmind.sh/node/index.html` on your phone or laptop. You're now contributing GPU time to distributed inference. That's it.

## Development Setup

```bash
git clone https://github.com/tejasphatak/Synapse.git
cd Synapse/synapse-src
npm install
python3 model/split.py --model gpt2 --dtype float16 --num-shards 2
node coordinator/index.js
```

Open two browser tabs at `http://localhost:8080/node/index.html` to start compute nodes.

## Running Tests

```bash
cd synapse-src
npm test
```

860 tests covering protocol, binary encoding, pipeline, shard loading, generation, topology, routing, head pruning, P2P, adaptive precision, entropy coding, mixture-of-depths, KV cache, tensor serialization, logging, coordinator HTTP APIs, and batch speculation.

All tests are pure JS (no WebGPU required). They run in Node.js with `node:test`.

## What We Need Help With

**High impact:**
- Real WebGPU validation of speculative execution (Phase 2) — needs a device with WebGPU support
- Testing on diverse hardware: different phones, tablets, GPUs, browsers
- Larger models: split and test GPT-2 Medium/XL or other HuggingFace models
- WebRTC P2P relay — reduce coordinator bottleneck for activation transfer

**Good first issues:**
- Try the node page on your device and report performance numbers
- Improve error messages when WebGPU isn't available
- Add model download progress to the node UI

**Documentation:**
- Architecture deep-dives for specific components
- Deployment guides for non-GCP platforms (AWS, Fly.io, Railway)

## Code Style

- No frameworks, no build step. Vanilla JS, ES modules, WGSL shaders.
- Tests use `node:test` and `node:assert`. No test framework dependencies.
- Keep it minimal. Three clear lines > one clever abstraction.

## Submitting Changes

1. Fork the repo and create a branch
2. Make your changes
3. Run `npm test` — all tests must pass
4. Open a PR with a clear description of what and why

## Architecture Overview

See the [CLAUDE.md](CLAUDE.md) project docs or the [blog](https://webmind.sh/blog/) for technical deep-dives. Key files:

- `coordinator/index.js` — WebSocket server, generation loop, HTTP APIs
- `node/pipeline.js` — WebGPU compute shader orchestration
- `node/node.js` — Browser client, shard loading, binary protocol
- `protocol/binary.js` — SYN1 wire format (24-byte header + tensor payload)
- `protocol/quantize.js` — Per-channel int8 quantization, delta encoding

## License

MIT. Your contributions will be licensed under the same terms.
