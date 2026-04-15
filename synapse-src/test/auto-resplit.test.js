import { test } from "node:test";
import assert from "node:assert";
import { chooseShardCount, decideResplit } from "../coordinator/auto-resplit.js";

test("chooseShardCount — respects minShards floor", () => {
  // 1 device, currently 2 shards: stay at 2 (min).
  assert.strictEqual(
    chooseShardCount({ readyCount: 1, currentShards: 2, numLayers: 12 }),
    2,
  );
});

test("chooseShardCount — respects layerCap (num_layers / 2)", () => {
  // 100 devices, 12 layers: cap at 6 shards (12/2).
  assert.strictEqual(
    chooseShardCount({ readyCount: 100, currentShards: 2, numLayers: 12 }),
    6,
  );
});

test("chooseShardCount — targets 70% utilization", () => {
  // 6 ready → floor(6*0.7)=4 shards → each shard has 3 layers, 2 devices spare.
  assert.strictEqual(
    chooseShardCount({ readyCount: 6, currentShards: 2, numLayers: 12 }),
    4,
  );
});

test("chooseShardCount — hysteresis: no move if delta < 1", () => {
  // Small fluctuation around the same count → stay put.
  assert.strictEqual(
    chooseShardCount({ readyCount: 3, currentShards: 2, numLayers: 12 }),
    2, // floor(3*0.7)=2, same as current
  );
});

test("chooseShardCount — grow from 2 to 3 at 5 devices", () => {
  // 5 ready → floor(5*0.7)=3 shards.
  assert.strictEqual(
    chooseShardCount({ readyCount: 5, currentShards: 2, numLayers: 12 }),
    3,
  );
});

test("chooseShardCount — shrink from 4 to 2 when fleet drops", () => {
  // 2 ready → stay at minShards=2 (currently at 4, must shrink).
  assert.strictEqual(
    chooseShardCount({ readyCount: 2, currentShards: 4, numLayers: 12 }),
    2,
  );
});

test("chooseShardCount — maxShards hard cap", () => {
  // Cap at 3 even with many devices.
  assert.strictEqual(
    chooseShardCount({ readyCount: 20, currentShards: 2, numLayers: 12, maxShards: 3 }),
    3,
  );
});

test("decideResplit — blocks when in-flight generation", () => {
  const r = decideResplit({
    readyCount: 6, currentShards: 2, numLayers: 12,
    activeGenerations: 1, lastResplitAt: 0, now: 1_000_000,
  });
  assert.strictEqual(r.shouldResplit, false);
  assert.strictEqual(r.reason, "in-flight");
  assert.strictEqual(r.targetShards, 4); // target still computed for telemetry
});

test("decideResplit — blocks during cooldown", () => {
  const now = 1_000_000;
  const r = decideResplit({
    readyCount: 6, currentShards: 2, numLayers: 12,
    activeGenerations: 0, lastResplitAt: now - 30_000, // 30s ago, cooldown is 120s
    now,
  });
  assert.strictEqual(r.shouldResplit, false);
  assert.strictEqual(r.reason, "cooldown");
});

test("decideResplit — allows after cooldown expires", () => {
  const now = 1_000_000;
  const r = decideResplit({
    readyCount: 6, currentShards: 2, numLayers: 12,
    activeGenerations: 0, lastResplitAt: now - 200_000, // 200s ago, past cooldown
    now,
  });
  assert.strictEqual(r.shouldResplit, true);
  assert.strictEqual(r.reason, "ok");
  assert.strictEqual(r.targetShards, 4);
});

test("decideResplit — returns no-change when target == current", () => {
  const r = decideResplit({
    readyCount: 3, currentShards: 2, numLayers: 12,
    activeGenerations: 0, lastResplitAt: 0,
  });
  assert.strictEqual(r.shouldResplit, false);
  assert.strictEqual(r.reason, "no-change");
});

test("decideResplit — first call (lastResplitAt=0) skips cooldown check", () => {
  const r = decideResplit({
    readyCount: 6, currentShards: 2, numLayers: 12,
    activeGenerations: 0, lastResplitAt: 0, now: 500,
  });
  assert.strictEqual(r.shouldResplit, true);
  assert.strictEqual(r.reason, "ok");
});

test("decideResplit — shrink scenario (fleet dropped)", () => {
  const r = decideResplit({
    readyCount: 2, currentShards: 4, numLayers: 12,
    activeGenerations: 0, lastResplitAt: 0,
  });
  assert.strictEqual(r.shouldResplit, true);
  assert.strictEqual(r.targetShards, 2);
});
