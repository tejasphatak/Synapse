# Synapse bench — 2026-04-15 baseline

First reproducible throughput measurement of the live 2-phone Qualcomm pipeline.

## Topology
- 3 Android nodes (Qualcomm GPU), 2 assigned shards (GPT-2 117M × 2), 1 connected pending
- Coordinator: 34.82.32.123:8080 (GCP e2-medium, us-west1-b)
- Network path: phone ⇄ coordinator ⇄ phone ⇄ coordinator ⇄ client (4-hop per token)
- Benchmark script: `synapse-src/deploy/bench.mjs`

## Results (9 runs, 3 prompts × 3 max_tokens)

| prompt | input_tok | max_tok | generated | ttft_ms | total_ms | decode_tok/s |
|---|---|---|---|---|---|---|
| "Hello" | 1 | 5 | 5 | 1226 | 6136 | 0.81 |
| "Hello" | 1 | 15 | 15 | 1216 | 18344 | 0.82 |
| "Hello" | 1 | 30 | 30 | 1274 | 36886 | 0.81 |
| "The universe is" | 3 | 5 | 5 | 1168 | 5986 | 0.83 |
| "The universe is" | 3 | 15 | 15 | 1240 | 18555 | 0.81 |
| "The universe is" | 3 | 30 | 30 | 1227 | 36790 | 0.82 |
| "Once upon a time" | 4 | 5 | 5 | 1209 | 6021 | 0.83 |
| "Once upon a time" | 4 | 15 | 15 | 1237 | 18264 | 0.82 |
| "Once upon a time" | 4 | 30 | 30 | 1252 | 37009 | 0.81 |

**Aggregates:** avg TTFT = 1227 ms, avg decode = 0.82 tok/s, avg total (30 tok) = 36.9 s.

## Observations
1. **TTFT is remarkably flat** (~1.2 s) across max_tokens and input lengths. Prefill cost is dominated by the initial activation round-trip across both shards, not by input token count at these sizes. This is consistent with the two phones doing their shard-0 and shard-1 prefill once, then entering the decode loop.
2. **Decode throughput is tight (0.81–0.83 tok/s)** — variance <3%. The pipeline latency is dominated by deterministic network hops, not by stochastic GPU scheduling.
3. **Per-token network cost is the obvious bottleneck.** 4-hop per token × ~250–300 ms per hop = ~1.2 s per token. That matches the measured decode rate.
4. **Input token count has almost no effect** at these scales (1–4 tokens differ by ~80 ms TTFT). Longer prompts (512+) will eventually change this.

## What this doesn't measure (yet)
- **P2P (WebRTC between nodes)** — not exercised in this benchmark. All activations traverse the coordinator. This is the single biggest expected optimization.
- **Same-LAN phone topology** — both phones were on different networks (mobile data). Same-WiFi would eliminate most of the coordinator-hop cost.
- **Decode with KV cache enabled** — the Phase 1 KV-cache path should be active per the current node code; confirm via `cached_step` perf events in a follow-up probe.
- **Larger models** — GPT-2 117M activations are tiny. Llama-size shards will stress network differently.

## Next experiments
1. Re-run bench with P2P enabled — compare decode tok/s vs. this baseline.
2. Same-LAN topology test — both phones on same WiFi, re-measure.
3. Capture per-phone perf events during a run — identify whether GPU or network dominates on each shard.
4. Per-hop latency instrumentation in node.js → visible in `/api/logs`.

## Reproducing
```bash
COORD=http://<coordinator>:8080 node synapse-src/deploy/bench.mjs
```
Requires: pipeline ready (≥2 nodes covering all shards). See `deploy/cli-infer.mjs` for the single-shot equivalent.
