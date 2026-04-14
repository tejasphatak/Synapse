import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { Topology } from "../coordinator/topology.js";
import { Router } from "../coordinator/router.js";

/** Minimal mock WebSocket */
function mockWs(open = true) {
  const ws = {
    readyState: open ? 1 : 3,
    sent: [],
    send(data) {
      this.sent.push(typeof data === "string" ? JSON.parse(data) : data);
    },
  };
  return ws;
}

describe("Router", () => {
  let topo, router, ws1, ws2, ws3;

  beforeEach(() => {
    topo = new Topology(3);
    ws1 = mockWs();
    ws2 = mockWs();
    ws3 = mockWs();
    topo.addNode("n1", ws1);
    topo.addNode("n2", ws2);
    topo.addNode("n3", ws3);
    topo.assignShard("n1", 0, 0, 3);
    topo.assignShard("n2", 1, 4, 7);
    topo.assignShard("n3", 2, 8, 11);
    router = new Router(topo);
  });

  describe("routeActivation", () => {
    it("forwards activation to next node in pipeline", () => {
      const msg = {
        type: "ACTIVATION",
        fromNode: "n1",
        requestId: "req-1",
        layer: 3,
        data: [1, 2, 3],
      };
      const ok = router.routeActivation(msg, ws1);
      assert.equal(ok, true);
      assert.equal(ws2.sent.length, 1);
      assert.equal(ws2.sent[0].toNode, "n2");
      assert.equal(ws2.sent[0].fromNode, "n1");
    });

    it("sends error when no next node (last in pipeline)", () => {
      const msg = {
        type: "ACTIVATION",
        fromNode: "n3",
        requestId: "req-1",
        layer: 11,
      };
      const ok = router.routeActivation(msg, ws3);
      assert.equal(ok, false);
      assert.equal(ws3.sent.length, 1);
      assert.equal(ws3.sent[0].type, "ERROR");
      assert.ok(ws3.sent[0].code === "ROUTE_FAILED");
    });

    it("sends error when next node is disconnected", () => {
      ws2.readyState = 3; // CLOSED
      const msg = {
        type: "ACTIVATION",
        fromNode: "n1",
        requestId: "req-2",
        layer: 3,
      };
      const ok = router.routeActivation(msg, ws1);
      assert.equal(ok, false);
      assert.equal(ws1.sent.length, 1);
      assert.equal(ws1.sent[0].code, "NODE_UNAVAILABLE");
    });

    it("tracks request hops", () => {
      const msg = {
        type: "ACTIVATION",
        fromNode: "n1",
        requestId: "req-3",
        layer: 3,
      };
      router.routeActivation(msg, ws1);

      const msg2 = {
        type: "ACTIVATION",
        fromNode: "n2",
        requestId: "req-3",
        layer: 7,
      };
      router.routeActivation(msg2, ws2);

      const stats = router.getRequestStats("req-3");
      assert.equal(stats.hops.length, 2);
      assert.equal(stats.hops[0].from, "n1");
      assert.equal(stats.hops[0].to, "n2");
      assert.equal(stats.hops[1].from, "n2");
      assert.equal(stats.hops[1].to, "n3");
    });
  });

  describe("broadcastOutput", () => {
    it("sends to all open dashboard clients", () => {
      const d1 = mockWs();
      const d2 = mockWs();
      const d3 = mockWs(false); // closed

      const msg = {
        type: "OUTPUT",
        requestId: "req-4",
        token: "hello",
      };
      router.broadcastOutput(msg, [d1, d2, d3]);

      assert.equal(d1.sent.length, 1);
      assert.equal(d2.sent.length, 1);
      assert.equal(d3.sent.length, 0); // skipped
    });

    it("records completion time for tracked requests", () => {
      // First create the tracking entry via routeActivation
      router.routeActivation(
        { type: "ACTIVATION", fromNode: "n1", requestId: "req-5", layer: 3 },
        ws1
      );

      router.broadcastOutput(
        { type: "OUTPUT", requestId: "req-5", token: "x" },
        []
      );

      const stats = router.getRequestStats("req-5");
      assert.ok(stats.completedAt > 0);
      assert.ok(stats.latencyMs >= 0);
    });
  });

  describe("getRequestStats", () => {
    it("returns null for unknown request", () => {
      assert.equal(router.getRequestStats("nonexistent"), null);
    });
  });

  describe("cleanupOldRequests", () => {
    it("removes completed requests older than maxAge", () => {
      // Create and complete a request
      router.routeActivation(
        { type: "ACTIVATION", fromNode: "n1", requestId: "old-1", layer: 3 },
        ws1
      );
      router.broadcastOutput(
        { type: "OUTPUT", requestId: "old-1", token: "x" },
        []
      );

      // Backdate it
      const stats = router.getRequestStats("old-1");
      stats.completedAt = Date.now() - 120000;

      router.cleanupOldRequests(60000);
      assert.equal(router.getRequestStats("old-1"), null);
    });

    it("keeps recent completed requests", () => {
      router.routeActivation(
        { type: "ACTIVATION", fromNode: "n1", requestId: "new-1", layer: 3 },
        ws1
      );
      router.broadcastOutput(
        { type: "OUTPUT", requestId: "new-1", token: "x" },
        []
      );

      router.cleanupOldRequests(60000);
      assert.ok(router.getRequestStats("new-1") !== null);
    });

    it("keeps in-flight requests (no completedAt)", () => {
      router.routeActivation(
        { type: "ACTIVATION", fromNode: "n1", requestId: "inflight", layer: 3 },
        ws1
      );

      router.cleanupOldRequests(0); // aggressive cleanup
      assert.ok(router.getRequestStats("inflight") !== null);
    });
  });
});
