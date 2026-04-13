"""
Synapse Validation — GPT-2 Reference Scores & Synapse Comparison

GPT-2 (117M) benchmark scores are well-known. We don't re-run them.
Instead we:
  1. Store known reference scores
  2. Generate a small set of deterministic reference outputs (greedy decode)
     for token-by-token comparison against Synapse
  3. Provide a WebSocket client that runs the same prompts through Synapse
     and compares results

Usage:
  source ~/dev-env/bin/activate

  # Step 1: Generate reference outputs (runs GPT-2 locally, ~30 sec on CPU)
  python test/benchmark-reference.py --generate

  # Step 2: Compare against Synapse (requires coordinator running)
  python test/benchmark-reference.py --compare ws://localhost:8080
"""

import argparse
import json
import os
import sys
import time

# ─── Known GPT-2 (117M) Benchmark Scores ─────────────────────────
# Source: EleutherAI lm-evaluation-harness, HuggingFace model card

REFERENCE_SCORES = {
    "model": "gpt2-117M",
    "benchmarks": {
        "hellaswag": {"acc_norm": 28.9, "metric": "acc_norm", "unit": "%"},
        "lambada_openai": {"acc": 32.6, "metric": "acc", "unit": "%"},
        "wikitext2": {"perplexity": 29.41, "metric": "word_perplexity", "unit": "ppl"},
        "arc_easy": {"acc": 43.8, "metric": "acc", "unit": "%"},
        "piqa": {"acc": 62.9, "metric": "acc", "unit": "%"},
    },
    "note": "Synapse must reproduce these scores within tolerance when running the same model weights.",
    "tolerance": {
        "greedy_decode": "exact token match (deterministic)",
        "perplexity": "within 5% of reference",
        "accuracy": "within 2% of reference",
    },
}

# Test prompts for greedy decode comparison
TEST_PROMPTS = [
    "The meaning of life is",
    "In a galaxy far far away",
    "def fibonacci(n):",
    "The quick brown fox",
    "Scientists recently discovered that",
    "Once upon a time there was",
    "The capital of France is",
    "import numpy as np\n",
]

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference-outputs.json")


def generate_references():
    """Generate deterministic greedy decode outputs from HuggingFace GPT-2."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print("Loading GPT-2 (117M)...")
    t0 = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    results = {
        "reference_scores": REFERENCE_SCORES,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "greedy_decode": [],
        "kv_cache_proof": [],
    }

    # ─── Greedy decode references ─────────────────────────────────
    print("=== Greedy Decode (20 tokens each) ===")
    for prompt in TEST_PROMPTS:
        ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_len = ids.shape[1]

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=20, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)

        new_ids = out[0].tolist()[prompt_len:]
        text = tokenizer.decode(new_ids)
        print(f"  {prompt!r} -> {text!r}")

        results["greedy_decode"].append({
            "prompt": prompt,
            "prompt_ids": out[0].tolist()[:prompt_len],
            "generated_ids": new_ids,
            "generated_text": text,
        })

    # ─── KV cache mathematical proof ─────────────────────────────
    # Show that cached and non-cached produce identical top-1 tokens
    import numpy as np

    print("\n=== KV Cache Proof (full vs cached must be identical) ===")
    for prompt in TEST_PROMPTS[:3]:
        ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            # Full sequence
            out_full = model(ids, use_cache=False)
            logits_full = out_full.logits[0, -1, :].numpy()

            # Cached: prefill N-1, step 1
            out_pre = model(ids[:, :-1], use_cache=True)
            out_step = model(ids[:, -1:], past_key_values=out_pre.past_key_values)
            logits_cached = out_step.logits[0, -1, :].numpy()

        top1_full = int(np.argmax(logits_full))
        top1_cached = int(np.argmax(logits_cached))
        max_diff = float(np.max(np.abs(logits_full - logits_cached)))

        status = "PASS" if top1_full == top1_cached else "FAIL"
        print(f"  [{status}] {prompt!r}: top1={top1_full}, max_diff={max_diff:.2e}")

        results["kv_cache_proof"].append({
            "prompt": prompt,
            "top1_full": top1_full,
            "top1_cached": top1_cached,
            "match": top1_full == top1_cached,
            "max_logit_diff": max_diff,
        })

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")
    all_kv_match = all(r["match"] for r in results["kv_cache_proof"])
    print(f"KV cache proof: {'ALL PASS' if all_kv_match else 'SOME FAILED'}")


def compare_synapse(coordinator_url):
    """Connect to Synapse coordinator and run the same prompts, compare outputs."""
    import asyncio
    import websockets

    if not os.path.exists(OUTPUT_FILE):
        print(f"No reference file found. Run with --generate first.")
        sys.exit(1)

    with open(OUTPUT_FILE) as f:
        reference = json.load(f)

    async def run():
        uri = f"{coordinator_url}?type=prompt"
        print(f"Connecting to {uri}...")

        async with websockets.connect(uri) as ws:
            # Wait for pipeline ready
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") == "PIPELINE_READY":
                    print("Pipeline ready.\n")
                    break
                elif msg.get("type") == "TOPOLOGY_UPDATE":
                    nodes = msg.get("nodes", [])
                    ready = sum(1 for n in nodes if n.get("status") == "ready")
                    print(f"  Topology: {ready}/{len(nodes)} nodes ready")

            results = []
            for ref in reference["greedy_decode"]:
                prompt = ref["prompt"]
                prompt_ids = ref["prompt_ids"]
                expected_ids = ref["generated_ids"]

                # Send inference request
                req = {
                    "type": "PROMPT_INFER",
                    "tokenIds": prompt_ids,
                    "maxTokens": len(expected_ids),
                }
                await ws.send(json.dumps(req))

                # Collect tokens
                synapse_ids = []
                while True:
                    msg = json.loads(await ws.recv())
                    if msg["type"] == "TOKEN_GENERATED":
                        synapse_ids.append(msg["token"])
                    elif msg["type"] == "GENERATION_DONE":
                        break
                    elif msg["type"] == "INFER_ERROR":
                        print(f"  ERROR: {msg.get('error')}")
                        break

                # Compare
                match_count = sum(1 for a, b in zip(expected_ids, synapse_ids) if a == b)
                total = min(len(expected_ids), len(synapse_ids))
                match_rate = match_count / total if total > 0 else 0

                status = "PASS" if match_rate == 1.0 else ("WARN" if match_rate > 0.8 else "FAIL")
                print(f"  [{status}] {prompt!r}: {match_count}/{total} tokens match ({match_rate:.0%})")

                if match_rate < 1.0 and total > 0:
                    # Show first mismatch
                    for i, (e, s) in enumerate(zip(expected_ids, synapse_ids)):
                        if e != s:
                            print(f"         First mismatch at token {i}: expected={e}, got={s}")
                            break

                results.append({
                    "prompt": prompt,
                    "match_rate": match_rate,
                    "expected_len": len(expected_ids),
                    "synapse_len": len(synapse_ids),
                })

            # Summary
            print("\n" + "=" * 50)
            avg_match = sum(r["match_rate"] for r in results) / len(results)
            exact = sum(1 for r in results if r["match_rate"] == 1.0)
            print(f"Average match rate: {avg_match:.1%}")
            print(f"Exact matches: {exact}/{len(results)}")
            print(f"Verdict: {'PASS' if avg_match > 0.95 else 'FAIL'}")

    asyncio.run(run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synapse LLM Quality Benchmark")
    parser.add_argument("--generate", action="store_true",
                        help="Generate reference outputs from HuggingFace GPT-2")
    parser.add_argument("--compare", type=str, metavar="URL",
                        help="Compare Synapse output against reference (e.g., ws://localhost:8080)")
    args = parser.parse_args()

    if args.generate:
        generate_references()
    elif args.compare:
        try:
            import websockets
        except ImportError:
            print("Installing websockets...")
            os.system(f"{sys.executable} -m pip install websockets -q")
            import websockets
        compare_synapse(args.compare)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  python test/benchmark-reference.py --generate    # one-time, ~30s")
        print("  python test/benchmark-reference.py --compare ws://localhost:8080")
