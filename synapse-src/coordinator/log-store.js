/**
 * LogStore — Centralized ring buffer for node telemetry logs.
 *
 * Collects log entries from all nodes, provides filtered queries,
 * and aggregates per-node performance statistics. Decoupled from
 * HTTP so the logic is testable.
 */

export class LogStore {
  /**
   * @param {number} [maxEntries=5000] - Maximum entries before oldest are evicted
   */
  constructor(maxEntries = 5000) {
    this.maxEntries = maxEntries;
    this.entries = [];
  }

  /**
   * Add a log entry to the store.
   * @param {{ nodeId: string, level: string, event: string, data?: object, timestamp?: number }} entry
   */
  add(entry) {
    this.entries.push(entry);
    if (this.entries.length > this.maxEntries) {
      this.entries.splice(0, this.entries.length - this.maxEntries);
    }
  }

  /**
   * Query logs with optional filters.
   * @param {{ node?: string, event?: string, level?: string, since?: number, n?: number }} filters
   * @returns {{ count: number, total: number, logs: object[] }}
   */
  query({ node, event, level, since, n = 200 } = {}) {
    const limit = Math.min(n, this.maxEntries);
    let results = this.entries;

    if (since) results = results.filter(e => e.timestamp > since);
    if (node) results = results.filter(e => e.nodeId === node);
    if (event) results = results.filter(e => e.event === event);
    if (level) results = results.filter(e => e.level === level);

    results = results.slice(-limit);

    return { count: results.length, total: this.entries.length, logs: results };
  }

  /**
   * Aggregate per-node performance statistics from perf-level logs.
   * @returns {Object.<string, { nodeId: string, events: Object, lastSeen: number }>}
   */
  getPerfSummary() {
    const perfByNode = {};

    for (const entry of this.entries) {
      if (entry.level !== "perf") continue;

      if (!perfByNode[entry.nodeId]) {
        perfByNode[entry.nodeId] = { nodeId: entry.nodeId, events: {}, lastSeen: 0 };
      }

      const node = perfByNode[entry.nodeId];
      node.lastSeen = Math.max(node.lastSeen, entry.timestamp);

      if (!node.events[entry.event]) {
        node.events[entry.event] = { count: 0, totalMs: 0, minMs: Infinity, maxMs: 0 };
      }

      const ev = node.events[entry.event];
      ev.count++;
      const ms = entry.data?.latencyMs || entry.data?.durationMs || 0;
      ev.totalMs += ms;
      ev.minMs = Math.min(ev.minMs, ms);
      ev.maxMs = Math.max(ev.maxMs, ms);
    }

    // Calculate averages and fix Infinity
    for (const node of Object.values(perfByNode)) {
      for (const ev of Object.values(node.events)) {
        ev.avgMs = ev.count > 0 ? +(ev.totalMs / ev.count).toFixed(2) : 0;
        if (ev.minMs === Infinity) ev.minMs = 0;
      }
    }

    return perfByNode;
  }

  /**
   * Number of entries in the store.
   */
  get length() {
    return this.entries.length;
  }
}
