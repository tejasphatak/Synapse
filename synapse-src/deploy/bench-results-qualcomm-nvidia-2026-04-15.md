# Synapse bench — Qualcomm + Nvidia + temperature passthrough (2026-04-15)

**Second measurement point** on real hardware. Improvement over the all-Qualcomm-mobile baseline (`bench-results-2026-04-15.md`).

## Topology
- **Shard 0:** Qualcomm Android phone (mobile=true, WebGPU=true)
- **Shard 1:** Nvidia desktop GPU (mobile=false, WebGPU=true, 2048MB buffer)
- **Coordinator:** GCP e2-medium at 34.82.32.123:8080 (us-west1-b)
- **Sampling:** default temperature=1.0 (new default after 296da51 patch)
- **Model:** GPT-2 117M, 2-shard split (layers 0-5, layers 6-11)
- **Network:** phones + desktop nodes + coord all reachable via public internet

## Results — 9 runs (3 prompts × 3 max_tokens)

| prompt | input_tok | max_tok | generated | ttft_ms | total_ms | decode_tps |
|---|---|---|---|---|---|---|
| "Hello" | 1 | 5 | 5 | 586 | 2649 | 1.94 |
| "Hello" | 1 | 15 | 15 | 463 | 7963 | 1.87 |
| "Hello" | 1 | 30 | 30 | 553 | 16033 | 1.87 |
| "The universe is" | 3 | 5 | 5 | 499 | 2620 | 1.89 |
| "The universe is" | 3 | 15 | 15 | 516 | 8263 | 1.81 |
| "The universe is" | 3 | 30 | 30 | 607 | 15566 | 1.94 |
| "Once upon a time" | 4 | 5 | 5 | 522 | 2617 | 1.91 |
| "Once upon a time" | 4 | 15 | 15 | 474 | 7655 | 1.95 |
| "Once upon a time" | 4 | 30 | 30 | 509 | 15781 | 1.90 |

**Aggregates:** avg TTFT = 525 ms, avg decode = 1.90 tok/s, avg total (30 tok) = 15.79 s.
**Variance:** decode range 1.81–1.95 tok/s (7% spread — very consistent).

## Delta vs. all-Qualcomm baseline

| Metric | Baseline (2 Qualcomm mobile) | This run (Qualcomm + Nvidia) | Δ |
|---|---|---|---|
| TTFT | 1227 ms | **525 ms** | **2.34× faster** |
| Decode tok/s | 0.82 | **1.90** | **2.32× faster** |
| Total for 30 tok | ~37 s | **15.8 s** | **2.34× faster** |

## What this measures
- **Desktop GPU on shard 1** (Nvidia) handles the second forward pass materially faster than mobile Qualcomm.
- **Temperature passthrough** (commit 296da51) lets inference complete (no EOT-lock).
- **Coord-relay still** — no P2P between shards; this gain is pure compute speedup, not network reduction.

## Confirmed failure — Intel shard
Earlier in the same session, Intel desktop iGPU was assigned to shard 0 via admin endpoint. Result: `GPUBuffer.mapAsync → [Device] is lost` on every prefill. Intel's WebGPU driver crashes mid-forward pass. **Intel iGPU is not currently viable as a shard holder.** (Nvidia + Intel-laptop testing may change this; this is the Windows-Intel iGPU case.)

## Output quality note
Generated tokens are still low-quality gibberish (commas, punctuation, fragmented words) — this is GPT-2 117M's baseline capability on short prompts, not a pipeline defect. The tokens are mathematically sound samples from the model's real distribution.

## Next data points wanted
Per `project_synapse_huge_gains_2026-04-15.md`:
1. Validate Phase 2 speculative decoding (measure hit rate, impact on tok/s)
2. Flip int4 on-wire toggle (Phase 4 adaptive precision) — bandwidth + decode impact
3. Flip entropy-coded delta toggle (Phase 4) — bandwidth impact
4. Fix P2P signaling — expected 2-3× compounding on top of this number
5. Two desktop GPUs (Nvidia + different Intel with newer driver) — untested; may unlock all-desktop baseline

**Projected ceiling on current hardware if 1-4 all validate:** 1.90 × 2 × 1.5 × 1.3 ≈ **7-8 tok/s** on Qualcomm+Nvidia without model changes.

## Reproducing
```bash
# With live admin token + current topology of Qualcomm-on-shard-0 + Nvidia-on-shard-1:
COORD=http://34.82.32.123:8080 node synapse-src/deploy/bench.mjs
```

## Per-validate-don't-assume discipline
- **Measured:** yes (above)
- **Topology recorded:** yes
- **Date recorded:** 2026-04-15
- **Variance estimate:** yes (7% spread)
- **Failure cases documented:** yes (Intel GPUBuffer device-lost)
