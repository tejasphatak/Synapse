/**
 * Elastic N-shard auto-resplit.
 *
 * Watches the ready-device count and re-runs `split.py` with a new shard
 * count when the optimal shape changes. Emits a process restart after
 * success — systemd brings the coord back up with the new manifest, and
 * reconnecting clients re-self-test + re-assign via tryAssignShards.
 *
 * Design constraints (2026-04-15):
 *
 *   - Must not resplit during an active generation (in-flight requests
 *     would die). Coord tracks active generations; we gate on
 *     `activeGenerationCount()`.
 *   - Must not oscillate. Devices flap on mobile; use a cooldown so
 *     frequent 3↔4 shard swaps don't burn the fleet.
 *   - Must keep at least 2 shards (distributed-inference invariant).
 *     Must not exceed `num_layers / 2` (each shard needs ≥ 2 layers
 *     or pipeline hops dominate compute).
 *   - Hysteresis: only act when the delta from current is ≥ 1 shard.
 *     Micro-fluctuations in ready count shouldn't trigger resplits.
 */

/**
 * Pure: pick the optimal shard count given current fleet state.
 *
 * Uses ~70% utilization so there's slack for churn. A fleet of 6 ready
 * devices picks 4 shards (72% busy) instead of 6 (100% busy, one drop
 * breaks the pipeline). Returns the current value when no better
 * option exists.
 *
 * @param {object} args
 * @param {number} args.readyCount    — devices with status=ready
 * @param {number} args.currentShards — current num_shards
 * @param {number} args.numLayers     — total transformer layers (12 for gpt-2 small)
 * @param {number} [args.minShards=2] — never go below this
 * @param {number} [args.maxShards]   — optional hard cap
 * @returns {number} proposed num_shards
 */
export function chooseShardCount({
  readyCount,
  currentShards,
  numLayers,
  minShards = 2,
  maxShards,
}) {
  // At most one shard per 2 layers (drift + overhead argument).
  const layerCap = Math.floor(numLayers / 2);
  const hardCap = maxShards ? Math.min(maxShards, layerCap) : layerCap;

  // 70% utilization target: leave headroom for churn.
  let proposed = Math.max(minShards, Math.floor(readyCount * 0.7));
  proposed = Math.min(proposed, hardCap);

  // Must be at least minShards even if readyCount is tiny — otherwise
  // a 1-device fleet proposes 0 shards and breaks everything.
  proposed = Math.max(proposed, minShards);

  // Hysteresis: only move if the delta is worth it.
  if (Math.abs(proposed - currentShards) < 1) return currentShards;

  return proposed;
}

/**
 * Decide whether to resplit right now. Applies cooldown + in-flight gates.
 *
 * @param {object} args
 * @param {number} args.readyCount
 * @param {number} args.currentShards
 * @param {number} args.numLayers
 * @param {number} args.activeGenerations
 * @param {number} args.lastResplitAt       — timestamp (ms); 0 if never
 * @param {number} [args.cooldownMs=120000] — 2 min default
 * @param {number} [args.now=Date.now()]
 * @returns {{ shouldResplit: boolean, targetShards: number, reason: string }}
 */
export function decideResplit(args) {
  const {
    readyCount,
    currentShards,
    numLayers,
    activeGenerations,
    lastResplitAt,
    cooldownMs = 120_000,
    now = Date.now(),
    minShards = 2,
    maxShards,
  } = args;

  const target = chooseShardCount({
    readyCount, currentShards, numLayers, minShards, maxShards,
  });

  if (target === currentShards) {
    return { shouldResplit: false, targetShards: target, reason: "no-change" };
  }
  if (activeGenerations > 0) {
    return { shouldResplit: false, targetShards: target, reason: "in-flight" };
  }
  if (lastResplitAt && (now - lastResplitAt) < cooldownMs) {
    return { shouldResplit: false, targetShards: target, reason: "cooldown" };
  }
  if (readyCount < minShards) {
    return { shouldResplit: false, targetShards: target, reason: "insufficient-devices" };
  }
  return { shouldResplit: true, targetShards: target, reason: "ok" };
}
