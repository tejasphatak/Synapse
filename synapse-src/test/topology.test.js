import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { Topology } from "../coordinator/topology.js";

describe("Topology", () => {
  let topo;

  beforeEach(() => {
    topo = new Topology(2);
  });

  describe("addNode / removeNode", () => {
    it("registers a node", () => {
      topo.addNode("n1", { fake: "ws" }, { gpu: "t4" });
      const node = topo.getNode("n1");
      assert.equal(node.nodeId, "n1");
      assert.equal(node.status, "connected");
      assert.equal(node.shardId, null);
      assert.deepEqual(node.capabilities, { gpu: "t4" });
    });

    it("returns null for unknown node", () => {
      assert.equal(topo.getNode("ghost"), null);
    });

    it("removes a node and cleans pipeline", () => {
      topo.addNode("n1", {});
      topo.addNode("n2", {});
      topo.assignShard("n1", 0, 0, 5);
      topo.assignShard("n2", 1, 6, 11);
      assert.equal(topo.pipeline.length, 2);

      topo.removeNode("n1");
      assert.equal(topo.getNode("n1"), null);
      assert.deepEqual(topo.pipeline, ["n2"]);
    });
  });

  describe("assignShard", () => {
    it("assigns shard and updates node state", () => {
      topo.addNode("n1", {});
      const ok = topo.assignShard("n1", 0, 0, 5);
      assert.equal(ok, true);
      const node = topo.getNode("n1");
      assert.equal(node.shardId, 0);
      assert.equal(node.layerStart, 0);
      assert.equal(node.layerEnd, 5);
      assert.equal(node.status, "assigned");
    });

    it("returns false for unknown node", () => {
      assert.equal(topo.assignShard("ghost", 0, 0, 5), false);
    });

    it("builds pipeline in layer order regardless of add order", () => {
      topo.addNode("n2", {});
      topo.addNode("n1", {});
      topo.assignShard("n2", 1, 6, 11);
      topo.assignShard("n1", 0, 0, 5);
      assert.deepEqual(topo.pipeline, ["n1", "n2"]);
    });
  });

  describe("pipeline navigation", () => {
    beforeEach(() => {
      topo.addNode("n1", {});
      topo.addNode("n2", {});
      topo.addNode("n3", {});
      topo.assignShard("n1", 0, 0, 3);
      topo.assignShard("n2", 1, 4, 7);
      topo.assignShard("n3", 2, 8, 11);
    });

    it("getFirstNode returns lowest-layer node", () => {
      assert.equal(topo.getFirstNode().nodeId, "n1");
    });

    it("getNextNode walks the pipeline", () => {
      assert.equal(topo.getNextNode("n1").nodeId, "n2");
      assert.equal(topo.getNextNode("n2").nodeId, "n3");
      assert.equal(topo.getNextNode("n3"), null);
    });

    it("getNextNode returns null for unknown node", () => {
      assert.equal(topo.getNextNode("ghost"), null);
    });

    it("isLastNode identifies pipeline tail", () => {
      assert.equal(topo.isLastNode("n3"), true);
      assert.equal(topo.isLastNode("n1"), false);
      assert.equal(topo.isLastNode("n2"), false);
    });

    it("getFirstNode returns null on empty pipeline", () => {
      const empty = new Topology(2);
      assert.equal(empty.getFirstNode(), null);
    });

    it("isLastNode returns false on empty pipeline", () => {
      const empty = new Topology(2);
      assert.equal(empty.isLastNode("n1"), false);
    });
  });

  describe("markReady / isPipelineReady", () => {
    it("pipeline not ready until all nodes ready", () => {
      topo.addNode("n1", {});
      topo.addNode("n2", {});
      topo.assignShard("n1", 0, 0, 5);
      topo.assignShard("n2", 1, 6, 11);

      assert.equal(topo.isPipelineReady(), false);

      topo.markReady("n1");
      assert.equal(topo.isPipelineReady(), false);

      topo.markReady("n2");
      assert.equal(topo.isPipelineReady(), true);
    });

    it("not ready if fewer nodes than expected shards", () => {
      topo.addNode("n1", {});
      topo.assignShard("n1", 0, 0, 5);
      topo.markReady("n1");
      assert.equal(topo.isPipelineReady(), false);
    });

    it("markReady on unknown node is a no-op", () => {
      topo.markReady("ghost"); // should not throw
    });
  });

  describe("getUnassignedNodes", () => {
    it("returns nodes without shard assignments", () => {
      topo.addNode("n1", {});
      topo.addNode("n2", {});
      topo.addNode("n3", {});
      topo.assignShard("n1", 0, 0, 5);

      const unassigned = topo.getUnassignedNodes();
      assert.equal(unassigned.length, 2);
      const ids = unassigned.map((n) => n.nodeId).sort();
      assert.deepEqual(ids, ["n2", "n3"]);
    });
  });

  describe("updatePing / getStaleNodes", () => {
    it("detects stale nodes", () => {
      topo.addNode("n1", {});
      topo.addNode("n2", {});

      // Manually set n1's lastPing to the past
      const n1 = topo.getNode("n1");
      n1.lastPing = Date.now() - 60000;

      const stale = topo.getStaleNodes(30000);
      assert.equal(stale.length, 1);
      assert.equal(stale[0].nodeId, "n1");
    });

    it("updatePing refreshes timestamp", () => {
      topo.addNode("n1", {});
      const n1 = topo.getNode("n1");
      n1.lastPing = Date.now() - 60000;

      topo.updatePing("n1");
      const stale = topo.getStaleNodes(30000);
      assert.equal(stale.length, 0);
    });

    it("updatePing on unknown node is a no-op", () => {
      topo.updatePing("ghost"); // should not throw
    });
  });

  describe("toSnapshot", () => {
    it("produces serializable snapshot without ws references", () => {
      topo.addNode("n1", { send() {} }, { gpu: "t4" });
      topo.assignShard("n1", 0, 0, 5);
      topo.markReady("n1");

      const snap = topo.toSnapshot();
      assert.equal(snap.nodes.length, 1);
      assert.equal(snap.nodes[0].nodeId, "n1");
      assert.equal(snap.nodes[0].status, "ready");
      assert.equal(snap.nodes[0].ws, undefined);
      assert.deepEqual(snap.pipeline, ["n1"]);
    });

    it("preserves pipeline order in snapshot", () => {
      topo.addNode("n2", {});
      topo.addNode("n1", {});
      topo.assignShard("n2", 1, 6, 11);
      topo.assignShard("n1", 0, 0, 5);

      const snap = topo.toSnapshot();
      assert.deepEqual(snap.pipeline, ["n1", "n2"]);
    });
  });

  describe("edge cases", () => {
    it("re-assigning shard updates pipeline correctly", () => {
      topo.addNode("n1", {});
      topo.addNode("n2", {});
      topo.assignShard("n1", 0, 0, 5);
      topo.assignShard("n2", 1, 6, 11);
      assert.deepEqual(topo.pipeline, ["n1", "n2"]);

      // Reassign n1 to later layers — pipeline should reorder
      topo.assignShard("n1", 0, 12, 17);
      assert.deepEqual(topo.pipeline, ["n2", "n1"]);
    });

    it("handles 3-shard topology", () => {
      const t3 = new Topology(3);
      t3.addNode("a", {});
      t3.addNode("b", {});
      t3.addNode("c", {});
      t3.assignShard("c", 2, 8, 11);
      t3.assignShard("a", 0, 0, 3);
      t3.assignShard("b", 1, 4, 7);

      assert.deepEqual(t3.pipeline, ["a", "b", "c"]);
      assert.equal(t3.isPipelineReady(), false);

      t3.markReady("a");
      t3.markReady("b");
      t3.markReady("c");
      assert.equal(t3.isPipelineReady(), true);
    });
  });
});
