#!/usr/bin/env python3
"""
Gemma LIVE parity — compare a coordinator's sampled top-20 logits
against the numpy full-forward reference for the same input tokens.

This is the real-fleet check: once Gemma shards are deployed to a coord
and nodes are streaming tokens, this script (a) tokenises a prompt with
the coord, (b) fires an inference for N tokens, (c) fetches the sample_top5
log events that include raw pre-temperature logits, (d) runs our numpy
full-forward on the SAME token sequence, and (e) compares top-20 token
IDs and logit values.

Passes if:
  - top-5 live tokens == top-5 numpy tokens (index set match)
  - per-token logit values within `--tol` (FP16 WebGPU expected to drift
    ~1e-3 vs CPU numpy fp32; tol=0.05 default is permissive)

Usage:
  COORD=http://coord.ip:8080 python3 gemma_live_parity.py \\
    --prompt "The universe is" --tokens 4
"""

import argparse, json, os, sys, time
from pathlib import Path
import urllib.request
import urllib.error
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gemma_layer_parity import HF_TOKEN_FILE, load_manifest
from gemma_full_parity import ours_full_forward


def http_post(url, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def http_get(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The universe is")
    ap.add_argument("--tokens", type=int, default=1,
                    help="how many tokens to generate (logit check is per-token)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="absolute logit tolerance (default 0.05 — FP16 WebGPU vs FP32 numpy)")
    ap.add_argument("--coord", default=os.environ.get("COORD", "http://localhost:8080"))
    args = ap.parse_args()

    print(f"  coord:   {args.coord}")
    print(f"  prompt:  {args.prompt!r}")
    print(f"  tokens:  {args.tokens}")

    # 1) tokenise on coord
    tok = http_post(f"{args.coord}/api/tokenize", {"text": args.prompt})
    input_ids = np.array(tok["tokenIds"], dtype=np.int64)
    print(f"  input_ids: {input_ids.tolist()}")

    # 2) fire an inference (one-shot blocking HTTP path)
    t0 = time.time()
    result = http_post(f"{args.coord}/api/infer",
                       {"tokenIds": input_ids.tolist()})
    dt = time.time() - t0
    print(f"  /api/infer completed in {dt:.2f}s")
    print(f"  result: {json.dumps(result)[:200]}")

    # 3) fetch sample_top5 logs — /api/infer is fire-and-forget, so poll.
    # Coord emits a short "req-N" form and a full "req-N-<ts>" form; match
    # by short prefix so both shapes are accepted.
    req_id = result.get("requestId", "")
    short_id = req_id.split("-")
    short_id = "-".join(short_id[:2]) if len(short_id) >= 2 else req_id
    logs_url = f"{args.coord}/api/logs?event=sample_top5"
    entries = []
    deadline = time.time() + 90
    while time.time() < deadline:
        logs = http_get(logs_url)
        pool = logs if isinstance(logs, list) else logs.get("logs", [])
        matched = [
            l for l in pool
            if str((l.get("data", {}) if isinstance(l, dict) else {}).get("requestId", "")).startswith(short_id)
        ]
        if matched:
            entries = matched
            break
        time.sleep(2)
    if not entries:
        print("  ✗ no sample_top5 log events seen within 90s — did inference fire? check coord logs.")
        sys.exit(2)

    # 4) numpy full-forward on the input_ids
    manifest = load_manifest()
    from transformers import AutoConfig
    tok_file = HF_TOKEN_FILE
    hf_cfg = AutoConfig.from_pretrained(
        manifest["model"],
        token=tok_file.read_text().strip() if tok_file.exists() else None,
    )
    cfg = {
        "layer_types":            hf_cfg.layer_types,
        "rope_scaling":           hf_cfg.rope_scaling,
        "query_pre_attn_scalar":  hf_cfg.query_pre_attn_scalar,
    }
    print(f"\n  running numpy full-forward on {len(input_ids)} tokens...")
    logits = ours_full_forward(input_ids, manifest, cfg)  # [seq, vocab]
    ref_last = logits[-1]
    ref_order = np.argsort(-ref_last)
    ref_top20 = [(int(i), float(ref_last[i])) for i in ref_order[:20]]

    # 5) compare first (most recent) entry
    e = entries[0] if not isinstance(entries[0], dict) else entries[0]
    payload = e.get("data") or e.get("payload") or e
    live_top20 = payload.get("top20") if isinstance(payload, dict) else None
    if not live_top20:
        print(f"  ✗ log entry missing top20 field; got: {json.dumps(e)[:200]}")
        sys.exit(3)

    print("\n  ─── top-5 comparison ───")
    live5 = [int(x[0]) for x in live_top20[:5]]
    ref5 = [t[0] for t in ref_top20[:5]]
    print(f"    live: {live5}")
    print(f"    ref:  {ref5}")
    set_match = set(live5) == set(ref5)
    order_match = live5 == ref5

    print("\n  ─── top-20 logit cross-check ───")
    ref_map = dict(ref_top20)
    diffs = []
    for tok_id, live_logit in live_top20:
        r = ref_map.get(int(tok_id))
        if r is None:
            continue
        diffs.append(abs(float(live_logit) - r))
    max_diff = max(diffs) if diffs else float("inf")
    mean_diff = sum(diffs) / len(diffs) if diffs else float("inf")
    print(f"    shared-token count: {len(diffs)}/20")
    print(f"    max |Δlogit|:       {max_diff:.4f}")
    print(f"    mean |Δlogit|:      {mean_diff:.4f}")

    ok = set_match and max_diff <= args.tol
    print("")
    if ok:
        print(f"  ✓ LIVE PARITY: top-5 set matches, max |Δlogit|={max_diff:.4f} ≤ {args.tol}")
    else:
        reasons = []
        if not set_match: reasons.append("top-5 set mismatch")
        if max_diff > args.tol: reasons.append(f"logit drift {max_diff:.4f} > {args.tol}")
        print(f"  ✗ LIVE DIVERGENCE: {'; '.join(reasons)}")
    if not order_match:
        print(f"    (top-5 order differs — usually fine if set matches and tol is wide)")


if __name__ == "__main__":
    main()
