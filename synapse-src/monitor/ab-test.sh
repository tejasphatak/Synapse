#!/bin/bash
# ab-test.sh — Run A/B comparison: speculation ON vs OFF
# Sends identical prompts to coordinator, captures perf data for both modes.
# Designed to run alongside real user traffic, not dominate the pipeline.
# Usage: ./ab-test.sh [num_runs] [prompt]
#        TEST_PROBABILITY=30 ./ab-test.sh   # only run 30% of the time
set -uo pipefail

# Probabilistic gate — skip most runs so we don't hog the coordinator
# Default: 20% chance of running (1 in 5 invocations)
TEST_PROBABILITY=${TEST_PROBABILITY:-20}
ROLL=$((RANDOM % 100))
if [ "$ROLL" -ge "$TEST_PROBABILITY" ]; then
  echo "Skipped (roll=$ROLL, threshold=$TEST_PROBABILITY%). Next time."
  exit 0
fi

COORDINATOR_URL="http://34.82.32.123:8080"
DATA_DIR="$HOME/Synapse/synapse-src/monitor/data"
RUNS=${1:-5}
PROMPT=${2:-"The future of distributed computing is"}
TIMESTAMP=$(date +%s)
RESULTS_FILE="$DATA_DIR/ab-results-$TIMESTAMP.json"

mkdir -p "$DATA_DIR"

echo "A/B Test: $RUNS runs each, prompt: '$PROMPT'"
echo "Results: $RESULTS_FILE"

# Check coordinator is up
STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$COORDINATOR_URL/api/topology" 2>/dev/null)
if [ "$STATUS" != "200" ]; then
  echo "ERROR: Coordinator not reachable (HTTP $STATUS)"
  exit 1
fi

# Check we have ready nodes
READY=$(curl -s "$COORDINATOR_URL/api/topology" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ready = [n for n in d.get('nodes', []) if n.get('status') == 'ready']
print(len(ready))
" 2>/dev/null || echo "0")

if [ "$READY" -lt 2 ]; then
  echo "ERROR: Need at least 2 ready nodes, got $READY"
  exit 1
fi

echo "Coordinator UP, $READY nodes ready"
echo ""

# Run the test via a Python script for cleaner JSON handling
python3 - "$COORDINATOR_URL" "$RUNS" "$PROMPT" "$RESULTS_FILE" << 'PYEOF'
import sys, json, time, urllib.request, urllib.error

coord_url = sys.argv[1]
num_runs = int(sys.argv[2])
prompt = sys.argv[3]
results_file = sys.argv[4]

def run_inference(prompt_text, max_tokens=20):
    """Send inference request and measure response time."""
    data = json.dumps({
        "prompt": prompt_text,
        "maxTokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{coord_url}/api/infer",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            elapsed = time.time() - start
            return {
                "ok": True,
                "elapsed_ms": round(elapsed * 1000, 1),
                "tokens": result.get("tokens", []),
                "num_tokens": len(result.get("tokens", [])),
            }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed_ms": round((time.time() - start) * 1000, 1)}

def get_perf_snapshot():
    """Get current perf data from coordinator."""
    try:
        req = urllib.request.Request(f"{coord_url}/api/perf")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except:
        return {}

def get_speculation_logs(since_ms):
    """Get speculation logs since timestamp."""
    try:
        req = urllib.request.Request(f"{coord_url}/api/logs?event=speculation_accepted&since={since_ms}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            accepted = len(json.loads(resp.read()).get("logs", []))
        req = urllib.request.Request(f"{coord_url}/api/logs?event=speculation_rejected&since={since_ms}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            rejected = len(json.loads(resp.read()).get("logs", []))
        return {"accepted": accepted, "rejected": rejected}
    except:
        return {"accepted": 0, "rejected": 0}

results = {
    "timestamp": int(time.time()),
    "prompt": prompt,
    "num_runs": num_runs,
    "runs": [],
}

for i in range(num_runs):
    print(f"Run {i+1}/{num_runs}...", end=" ", flush=True)
    since = int(time.time() * 1000)
    run_result = run_inference(prompt)
    spec_stats = get_speculation_logs(since)

    run_data = {
        "run": i + 1,
        **run_result,
        "speculation": spec_stats,
    }
    results["runs"].append(run_data)

    if run_result["ok"]:
        tok_s = run_result["num_tokens"] / (run_result["elapsed_ms"] / 1000) if run_result["elapsed_ms"] > 0 else 0
        print(f"{run_result['elapsed_ms']}ms, {run_result['num_tokens']} tokens, {tok_s:.1f} tok/s, spec: {spec_stats['accepted']}a/{spec_stats['rejected']}r")
    else:
        print(f"FAILED: {run_result.get('error', '?')}")

    time.sleep(1)  # brief pause between runs

# Compute summary
ok_runs = [r for r in results["runs"] if r.get("ok")]
if ok_runs:
    times = [r["elapsed_ms"] for r in ok_runs]
    tok_counts = [r["num_tokens"] for r in ok_runs]
    total_spec_a = sum(r["speculation"]["accepted"] for r in ok_runs)
    total_spec_r = sum(r["speculation"]["rejected"] for r in ok_runs)

    results["summary"] = {
        "successful_runs": len(ok_runs),
        "avg_elapsed_ms": round(sum(times) / len(times), 1),
        "min_elapsed_ms": round(min(times), 1),
        "max_elapsed_ms": round(max(times), 1),
        "avg_tokens": round(sum(tok_counts) / len(tok_counts), 1),
        "avg_tok_per_sec": round(sum(t / (e / 1000) for t, e in zip(tok_counts, times)) / len(ok_runs), 2),
        "total_spec_accepted": total_spec_a,
        "total_spec_rejected": total_spec_r,
        "spec_hit_rate": round(total_spec_a / (total_spec_a + total_spec_r), 4) if (total_spec_a + total_spec_r) > 0 else None,
    }
    print(f"\nSummary: {results['summary']['avg_tok_per_sec']} tok/s avg, spec hit rate: {results['summary']['spec_hit_rate']}")

with open(results_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_file}")
PYEOF
