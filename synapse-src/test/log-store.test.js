/**
 * LogStore Tests — Ring buffer, query filtering, perf aggregation
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { LogStore } from "../coordinator/log-store.js";

describe("LogStore", () => {
  describe("constructor", () => {
    it("initializes empty with default capacity", () => {
      const store = new LogStore();
      assert.equal(store.length, 0);
      assert.equal(store.maxEntries, 5000);
    });

    it("accepts custom capacity", () => {
      const store = new LogStore(100);
      assert.equal(store.maxEntries, 100);
    });
  });

  describe("add", () => {
    it("adds entries", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "info", event: "test", timestamp: 1000 });
      assert.equal(store.length, 1);
    });

    it("evicts oldest entries when capacity exceeded", () => {
      const store = new LogStore(3);
      store.add({ nodeId: "n0", event: "a", timestamp: 1 });
      store.add({ nodeId: "n0", event: "b", timestamp: 2 });
      store.add({ nodeId: "n0", event: "c", timestamp: 3 });
      store.add({ nodeId: "n0", event: "d", timestamp: 4 });

      assert.equal(store.length, 3);
      // Oldest entry ("a") should be evicted
      assert.equal(store.entries[0].event, "b");
      assert.equal(store.entries[2].event, "d");
    });

    it("evicts correctly when burst-adding past capacity", () => {
      const store = new LogStore(5);
      for (let i = 0; i < 20; i++) {
        store.add({ nodeId: "n0", event: `e${i}`, timestamp: i });
      }
      assert.equal(store.length, 5);
      assert.equal(store.entries[0].event, "e15");
      assert.equal(store.entries[4].event, "e19");
    });
  });

  describe("query", () => {
    function populatedStore() {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "prefill", timestamp: 1000, data: {} });
      store.add({ nodeId: "n1", level: "error", event: "shard_load", timestamp: 2000, data: {} });
      store.add({ nodeId: "n0", level: "perf", event: "cached_step", timestamp: 3000, data: {} });
      store.add({ nodeId: "n1", level: "perf", event: "prefill", timestamp: 4000, data: {} });
      store.add({ nodeId: "n0", level: "info", event: "connected", timestamp: 5000, data: {} });
      return store;
    }

    it("returns all entries with no filters", () => {
      const store = populatedStore();
      const result = store.query();
      assert.equal(result.count, 5);
      assert.equal(result.total, 5);
    });

    it("filters by node", () => {
      const store = populatedStore();
      const result = store.query({ node: "n0" });
      assert.equal(result.count, 3);
      assert.ok(result.logs.every(e => e.nodeId === "n0"));
    });

    it("filters by event", () => {
      const store = populatedStore();
      const result = store.query({ event: "prefill" });
      assert.equal(result.count, 2);
      assert.ok(result.logs.every(e => e.event === "prefill"));
    });

    it("filters by level", () => {
      const store = populatedStore();
      const result = store.query({ level: "perf" });
      assert.equal(result.count, 3);
    });

    it("filters by since timestamp", () => {
      const store = populatedStore();
      const result = store.query({ since: 3000 });
      assert.equal(result.count, 2); // timestamps 4000 and 5000
    });

    it("combines multiple filters", () => {
      const store = populatedStore();
      const result = store.query({ node: "n0", level: "perf" });
      assert.equal(result.count, 2);
    });

    it("respects n limit", () => {
      const store = populatedStore();
      const result = store.query({ n: 2 });
      assert.equal(result.count, 2);
      assert.equal(result.total, 5);
      // Should return the last 2 entries
      assert.equal(result.logs[0].timestamp, 4000);
      assert.equal(result.logs[1].timestamp, 5000);
    });

    it("n cannot exceed maxEntries", () => {
      const store = new LogStore(3);
      for (let i = 0; i < 3; i++) {
        store.add({ nodeId: "n0", event: "x", timestamp: i });
      }
      const result = store.query({ n: 99999 });
      assert.equal(result.count, 3);
    });

    it("returns empty for no matches", () => {
      const store = populatedStore();
      const result = store.query({ node: "nonexistent" });
      assert.equal(result.count, 0);
      assert.equal(result.total, 5);
    });

    it("defaults n to 200", () => {
      const store = new LogStore();
      for (let i = 0; i < 300; i++) {
        store.add({ nodeId: "n0", event: "x", timestamp: i });
      }
      const result = store.query();
      assert.equal(result.count, 200);
    });
  });

  describe("getPerfSummary", () => {
    it("returns empty object for no perf entries", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "info", event: "test", timestamp: 1000 });
      const perf = store.getPerfSummary();
      assert.deepEqual(perf, {});
    });

    it("aggregates single node single event", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "prefill", timestamp: 1000, data: { latencyMs: 50 } });
      store.add({ nodeId: "n0", level: "perf", event: "prefill", timestamp: 2000, data: { latencyMs: 100 } });
      store.add({ nodeId: "n0", level: "perf", event: "prefill", timestamp: 3000, data: { latencyMs: 150 } });

      const perf = store.getPerfSummary();
      assert.ok(perf.n0);
      assert.equal(perf.n0.nodeId, "n0");
      assert.equal(perf.n0.lastSeen, 3000);

      const ev = perf.n0.events.prefill;
      assert.equal(ev.count, 3);
      assert.equal(ev.totalMs, 300);
      assert.equal(ev.minMs, 50);
      assert.equal(ev.maxMs, 150);
      assert.equal(ev.avgMs, 100);
    });

    it("aggregates multiple nodes", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "prefill", timestamp: 1000, data: { latencyMs: 50 } });
      store.add({ nodeId: "n1", level: "perf", event: "prefill", timestamp: 2000, data: { latencyMs: 80 } });

      const perf = store.getPerfSummary();
      assert.ok(perf.n0);
      assert.ok(perf.n1);
      assert.equal(perf.n0.events.prefill.count, 1);
      assert.equal(perf.n1.events.prefill.count, 1);
    });

    it("aggregates multiple events per node", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "prefill", timestamp: 1000, data: { latencyMs: 50 } });
      store.add({ nodeId: "n0", level: "perf", event: "cached_step", timestamp: 2000, data: { latencyMs: 10 } });

      const perf = store.getPerfSummary();
      assert.ok(perf.n0.events.prefill);
      assert.ok(perf.n0.events.cached_step);
    });

    it("uses durationMs when latencyMs is missing", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "shard_loaded", timestamp: 1000, data: { durationMs: 500 } });

      const perf = store.getPerfSummary();
      assert.equal(perf.n0.events.shard_loaded.totalMs, 500);
      assert.equal(perf.n0.events.shard_loaded.avgMs, 500);
    });

    it("handles missing data gracefully (no latency fields)", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "heartbeat", timestamp: 1000, data: {} });

      const perf = store.getPerfSummary();
      const ev = perf.n0.events.heartbeat;
      assert.equal(ev.count, 1);
      assert.equal(ev.totalMs, 0);
      assert.equal(ev.minMs, 0); // Infinity should be fixed to 0
      assert.equal(ev.maxMs, 0);
      assert.equal(ev.avgMs, 0);
    });

    it("handles null data gracefully", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "test", timestamp: 1000 });

      const perf = store.getPerfSummary();
      assert.equal(perf.n0.events.test.count, 1);
      assert.equal(perf.n0.events.test.totalMs, 0);
    });

    it("ignores non-perf entries", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "info", event: "connected", timestamp: 1000 });
      store.add({ nodeId: "n0", level: "error", event: "crash", timestamp: 2000 });
      store.add({ nodeId: "n0", level: "perf", event: "prefill", timestamp: 3000, data: { latencyMs: 50 } });

      const perf = store.getPerfSummary();
      assert.equal(Object.keys(perf).length, 1);
      assert.equal(Object.keys(perf.n0.events).length, 1);
    });

    it("tracks lastSeen correctly across events", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "a", timestamp: 1000, data: {} });
      store.add({ nodeId: "n0", level: "perf", event: "b", timestamp: 5000, data: {} });
      store.add({ nodeId: "n0", level: "perf", event: "a", timestamp: 3000, data: {} });

      const perf = store.getPerfSummary();
      assert.equal(perf.n0.lastSeen, 5000);
    });

    it("avgMs rounds to 2 decimal places", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "x", timestamp: 1000, data: { latencyMs: 10 } });
      store.add({ nodeId: "n0", level: "perf", event: "x", timestamp: 2000, data: { latencyMs: 20 } });
      store.add({ nodeId: "n0", level: "perf", event: "x", timestamp: 3000, data: { latencyMs: 30 } });

      const perf = store.getPerfSummary();
      assert.equal(perf.n0.events.x.avgMs, 20); // exact
    });

    it("handles non-round averages", () => {
      const store = new LogStore();
      store.add({ nodeId: "n0", level: "perf", event: "x", timestamp: 1000, data: { latencyMs: 10 } });
      store.add({ nodeId: "n0", level: "perf", event: "x", timestamp: 2000, data: { latencyMs: 11 } });
      store.add({ nodeId: "n0", level: "perf", event: "x", timestamp: 3000, data: { latencyMs: 12 } });

      const perf = store.getPerfSummary();
      assert.equal(perf.n0.events.x.avgMs, 11); // 33/3 = 11 exact
    });
  });

  describe("ring buffer edge cases", () => {
    it("capacity of 1", () => {
      const store = new LogStore(1);
      store.add({ nodeId: "n0", event: "a", timestamp: 1 });
      store.add({ nodeId: "n0", event: "b", timestamp: 2 });
      assert.equal(store.length, 1);
      assert.equal(store.entries[0].event, "b");
    });

    it("query after eviction returns correct total", () => {
      const store = new LogStore(3);
      for (let i = 0; i < 10; i++) {
        store.add({ nodeId: "n0", event: `e${i}`, timestamp: i });
      }
      const result = store.query();
      assert.equal(result.total, 3);
      assert.equal(result.count, 3);
    });
  });
});
