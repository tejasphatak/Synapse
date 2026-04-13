/**
 * Topology Manager — Tracks which nodes are connected and which shards they hold.
 */

export class Topology {
  constructor(expectedShardCount = 2) {
    // nodeId → { nodeId, ws, shardId, layerStart, layerEnd, status, lastPing, capabilities }
    this.nodes = new Map();
    // Ordered pipeline: array of nodeIds from first shard to last
    this.pipeline = [];
    this.expectedShardCount = expectedShardCount;
  }

  /**
   * Register a new node.
   */
  addNode(nodeId, ws, capabilities = {}) {
    this.nodes.set(nodeId, {
      nodeId,
      ws,
      shardId: null,
      layerStart: null,
      layerEnd: null,
      status: "connected",
      lastPing: Date.now(),
      capabilities,
    });
  }

  /**
   * Remove a node from the topology.
   */
  removeNode(nodeId) {
    this.nodes.delete(nodeId);
    this.pipeline = this.pipeline.filter((id) => id !== nodeId);
  }

  /**
   * Assign a shard to a node.
   */
  assignShard(nodeId, shardId, layerStart, layerEnd) {
    const node = this.nodes.get(nodeId);
    if (!node) return false;

    node.shardId = shardId;
    node.layerStart = layerStart;
    node.layerEnd = layerEnd;
    node.status = "assigned";

    this._rebuildPipeline();
    return true;
  }

  /**
   * Mark a node as ready (shard loaded, ready for inference).
   */
  markReady(nodeId) {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    node.status = "ready";
  }

  /**
   * Update a node's last ping time.
   */
  updatePing(nodeId) {
    const node = this.nodes.get(nodeId);
    if (node) node.lastPing = Date.now();
  }

  /**
   * Get the next node in the pipeline after the given node.
   * Returns null if this is the last node.
   */
  getNode(nodeId) {
    return this.nodes.get(nodeId) || null;
  }

  getNextNode(nodeId) {
    const idx = this.pipeline.indexOf(nodeId);
    if (idx === -1 || idx === this.pipeline.length - 1) return null;
    return this.nodes.get(this.pipeline[idx + 1]) || null;
  }

  /**
   * Get the first node in the pipeline.
   */
  getFirstNode() {
    if (this.pipeline.length === 0) return null;
    return this.nodes.get(this.pipeline[0]) || null;
  }

  /**
   * Check if a node is the last in the pipeline.
   */
  isLastNode(nodeId) {
    return (
      this.pipeline.length > 0 &&
      this.pipeline[this.pipeline.length - 1] === nodeId
    );
  }

  /**
   * Check if the pipeline is complete (all shards assigned and ready).
   */
  isPipelineReady() {
    if (this.pipeline.length < this.expectedShardCount) return false;
    return this.pipeline.every((id) => {
      const node = this.nodes.get(id);
      return node && node.status === "ready";
    });
  }

  /**
   * Get the number of connected nodes waiting for shard assignment.
   */
  getUnassignedNodes() {
    return [...this.nodes.values()].filter((n) => n.shardId === null);
  }

  /**
   * Build a serializable snapshot for TOPOLOGY_UPDATE messages.
   */
  toSnapshot() {
    const nodes = [...this.nodes.values()].map((n) => ({
      nodeId: n.nodeId,
      shardId: n.shardId,
      layerStart: n.layerStart,
      layerEnd: n.layerEnd,
      status: n.status,
      capabilities: n.capabilities,
    }));
    return { nodes, pipeline: [...this.pipeline] };
  }

  /**
   * Rebuild the pipeline order based on shard assignments (sorted by layerStart).
   */
  _rebuildPipeline() {
    const assigned = [...this.nodes.values()]
      .filter((n) => n.shardId !== null)
      .sort((a, b) => a.layerStart - b.layerStart);

    this.pipeline = assigned.map((n) => n.nodeId);
  }

  /**
   * Detect stale nodes (no ping in the given timeout).
   */
  getStaleNodes(timeoutMs = 30000) {
    const now = Date.now();
    return [...this.nodes.values()].filter(
      (n) => now - n.lastPing > timeoutMs
    );
  }
}
