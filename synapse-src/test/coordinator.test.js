/**
 * Coordinator Tests — Topology and Router
 *
 * Tests the coordinator's core logic: node management, pipeline ordering,
 * shard assignment, and activation routing.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { Topology } from "../coordinator/topology.js";
import { Router } from "../coordinator/router.js";

// Mock WebSocket
function mockWs(readyState = 1) {
  const sent = [];
  return {
    readyState,
    send(data) { sent.push(data); },
    sent,
  };
}

describe("Topology", () => {
  it("adds and retrieves nodes", () => {
    const topo = new Topology(2);
    const ws = mockWs();
    topo.addNode("node-0", ws, { gpu: "v100" });

    const node = topo.getNode("node-0");
    assert.equal(node.nodeId, "node-0");
    assert.equal(node.status, "connected");
    assert.equal(node.shardId, null);
    assert.deepEqual(node.capabilities, { gpu: "v100" });
  });

  it("removes nodes and cleans pipeline", () => {
    const topo = new Topology(2);
    topo.addNode("node-0", mockWs());
    topo.addNode("node-1", mockWs());
    topo.assignShard("node-0", 0, 0, 5);
    topo.assignShard("node-1", 1, 6, 11);

    assert.equal(topo.pipeline.length, 2);
    topo.removeNode("node-0");
    assert.equal(topo.pipeline.length, 1);
    assert.equal(topo.getNode("node-0"), null);
  });

  it("assigns shards and builds pipeline in layer order", () => {
    const topo = new Topology(2);
    topo.addNode("node-1", mockWs());
    topo.addNode("node-0", mockWs());

    // Assign in reverse order — pipeline should still sort by layerStart
    topo.assignShard("node-1", 1, 6, 11);
    topo.assignShard("node-0", 0, 0, 5);

    assert.deepEqual(topo.pipeline, ["node-0", "node-1"]);
    assert.equal(topo.getNode("node-0").status, "assigned");
  });

  it("getNextNode follows pipeline order", () => {
    const topo = new Topology(3);
    topo.addNode("a", mockWs());
    topo.addNode("b", mockWs());
    topo.addNode("c", mockWs());
    topo.assignShard("a", 0, 0, 3);
    topo.assignShard("b", 1, 4, 7);
    topo.assignShard("c", 2, 8, 11);

    assert.equal(topo.getNextNode("a").nodeId, "b");
    assert.equal(topo.getNextNode("b").nodeId, "c");
    assert.equal(topo.getNextNode("c"), null);
  });

  it("getFirstNode and isLastNode", () => {
    const topo = new Topology(2);
    topo.addNode("first", mockWs());
    topo.addNode("last", mockWs());
    topo.assignShard("first", 0, 0, 5);
    topo.assignShard("last", 1, 6, 11);

    assert.equal(topo.getFirstNode().nodeId, "first");
    assert.equal(topo.isLastNode("last"), true);
    assert.equal(topo.isLastNode("first"), false);
  });

  it("isPipelineReady requires all shards ready", () => {
    const topo = new Topology(2);
    topo.addNode("n0", mockWs());
    topo.addNode("n1", mockWs());
    topo.assignShard("n0", 0, 0, 5);
    topo.assignShard("n1", 1, 6, 11);

    assert.equal(topo.isPipelineReady(), false); // assigned but not ready
    topo.markReady("n0");
    assert.equal(topo.isPipelineReady(), false); // only one ready
    topo.markReady("n1");
    assert.equal(topo.isPipelineReady(), true);
  });

  it("isPipelineReady false with insufficient nodes", () => {
    const topo = new Topology(2);
    topo.addNode("n0", mockWs());
    topo.assignShard("n0", 0, 0, 5);
    topo.markReady("n0");
    assert.equal(topo.isPipelineReady(), false); // need 2 shards
  });

  it("getUnassignedNodes returns nodes without shards", () => {
    const topo = new Topology(2);
    topo.addNode("assigned", mockWs());
    topo.addNode("waiting", mockWs());
    topo.assignShard("assigned", 0, 0, 5);

    const unassigned = topo.getUnassignedNodes();
    assert.equal(unassigned.length, 1);
    assert.equal(unassigned[0].nodeId, "waiting");
  });

  it("toSnapshot produces serializable output", () => {
    const topo = new Topology(2);
    topo.addNode("n0", mockWs());
    topo.assignShard("n0", 0, 0, 5);
    topo.markReady("n0");

    const snap = topo.toSnapshot();
    assert.equal(snap.nodes.length, 1);
    assert.equal(snap.nodes[0].nodeId, "n0");
    assert.equal(snap.nodes[0].status, "ready");
    assert.deepEqual(snap.pipeline, ["n0"]);
    // Should not include ws object
    assert.equal(snap.nodes[0].ws, undefined);
  });

  it("getStaleNodes detects timed-out nodes", () => {
    const topo = new Topology(2);
    topo.addNode("fresh", mockWs());
    topo.addNode("stale", mockWs());

    // Backdate the stale node
    topo.getNode("stale").lastPing = Date.now() - 60000;

    const stale = topo.getStaleNodes(30000);
    assert.equal(stale.length, 1);
    assert.equal(stale[0].nodeId, "stale");
  });

  it("updatePing refreshes timestamp", () => {
    const topo = new Topology(2);
    topo.addNode("n0", mockWs());
    topo.getNode("n0").lastPing = Date.now() - 60000;

    topo.updatePing("n0");
    const stale = topo.getStaleNodes(30000);
    assert.equal(stale.length, 0);
  });
});

describe("Router", () => {
  function setupPipeline() {
    const topo = new Topology(2);
    const ws0 = mockWs();
    const ws1 = mockWs();
    topo.addNode("n0", ws0);
    topo.addNode("n1", ws1);
    topo.assignShard("n0", 0, 0, 5);
    topo.assignShard("n1", 1, 6, 11);
    return { topo, ws0, ws1 };
  }

  it("routes activation to next node in pipeline", () => {
    const { topo, ws0, ws1 } = setupPipeline();
    const router = new Router(topo);

    const msg = {
      type: "ACTIVATION",
      fromNode: "n0",
      requestId: 42,
      layer: 5,
      activations: [1, 2, 3],
    };

    const result = router.routeActivation(msg, ws0);
    assert.equal(result, true);

    // ws1 should have received the forwarded message
    assert.equal(ws1.sent.length, 1);
    const forwarded = JSON.parse(ws1.sent[0]);
    assert.equal(forwarded.toNode, "n1");
    assert.equal(forwarded.fromNode, "n0");
  });

  it("tracks request hops", () => {
    const { topo, ws0 } = setupPipeline();
    const router = new Router(topo);

    router.routeActivation({
      type: "ACTIVATION",
      fromNode: "n0",
      requestId: 1,
      layer: 5,
    }, ws0);

    const stats = router.getRequestStats(1);
    assert.equal(stats.hops.length, 1);
    assert.equal(stats.hops[0].from, "n0");
    assert.equal(stats.hops[0].to, "n1");
  });

  it("fails when no next node exists", () => {
    const { topo, ws0, ws1 } = setupPipeline();
    const router = new Router(topo);

    // n1 is the last node — routing from it should fail
    const result = router.routeActivation({
      type: "ACTIVATION",
      fromNode: "n1",
      requestId: 2,
      layer: 11,
    }, ws1);

    assert.equal(result, false);
    // Error sent back to sender
    assert.equal(ws1.sent.length, 1);
    const err = JSON.parse(ws1.sent[0]);
    assert.equal(err.type, "ERROR");
  });

  it("fails when next node ws is disconnected", () => {
    const topo = new Topology(2);
    const ws0 = mockWs();
    const ws1 = mockWs(3); // readyState 3 = CLOSED
    topo.addNode("n0", ws0);
    topo.addNode("n1", ws1);
    topo.assignShard("n0", 0, 0, 5);
    topo.assignShard("n1", 1, 6, 11);

    const router = new Router(topo);
    const result = router.routeActivation({
      type: "ACTIVATION",
      fromNode: "n0",
      requestId: 3,
      layer: 5,
    }, ws0);

    assert.equal(result, false);
    const err = JSON.parse(ws0.sent[0]);
    assert.equal(err.code, "NODE_UNAVAILABLE");
  });

  it("broadcastOutput sends to all open dashboard clients", () => {
    const { topo } = setupPipeline();
    const router = new Router(topo);

    const d1 = mockWs();
    const d2 = mockWs();
    const d3 = mockWs(3); // closed

    router.broadcastOutput(
      { type: "OUTPUT", requestId: 10, token: "hello" },
      [d1, d2, d3]
    );

    assert.equal(d1.sent.length, 1);
    assert.equal(d2.sent.length, 1);
    assert.equal(d3.sent.length, 0); // closed, skipped
  });

  it("broadcastOutput records completion latency", () => {
    const { topo, ws0 } = setupPipeline();
    const router = new Router(topo);

    // Start a request
    router.routeActivation({
      type: "ACTIVATION",
      fromNode: "n0",
      requestId: 20,
      layer: 5,
    }, ws0);

    // Complete it
    router.broadcastOutput(
      { type: "OUTPUT", requestId: 20, token: "done" },
      []
    );

    const stats = router.getRequestStats(20);
    assert.ok(stats.completedAt);
    assert.ok(stats.latencyMs >= 0);
  });

  it("cleanupOldRequests removes completed entries", () => {
    const { topo, ws0 } = setupPipeline();
    const router = new Router(topo);

    // Create and complete a request
    router.routeActivation({
      type: "ACTIVATION",
      fromNode: "n0",
      requestId: 30,
      layer: 5,
    }, ws0);
    router.broadcastOutput({ type: "OUTPUT", requestId: 30 }, []);

    // Backdate completion
    router.activeRequests.get(30).completedAt = Date.now() - 120000;

    router.cleanupOldRequests(60000);
    assert.equal(router.getRequestStats(30), null);
  });

  it("cleanupOldRequests keeps recent and in-flight requests", () => {
    const { topo, ws0 } = setupPipeline();
    const router = new Router(topo);

    // In-flight (no completedAt)
    router.routeActivation({
      type: "ACTIVATION",
      fromNode: "n0",
      requestId: 40,
      layer: 5,
    }, ws0);

    // Recently completed
    router.routeActivation({
      type: "ACTIVATION",
      fromNode: "n0",
      requestId: 41,
      layer: 5,
    }, ws0);
    router.broadcastOutput({ type: "OUTPUT", requestId: 41 }, []);

    router.cleanupOldRequests(60000);
    assert.ok(router.getRequestStats(40)); // in-flight kept
    assert.ok(router.getRequestStats(41)); // recent kept
  });
});
