/**
 * Router — Routes activation packets between nodes in the pipeline.
 *
 * Supports both JSON and binary (zero-copy) relay modes.
 */

import {
  createActivationMessage,
  createOutputMessage,
  createErrorMessage,
} from "../protocol/messages.js";

export class Router {
  constructor(topology) {
    this.topology = topology;
    // requestId → { startTime, fromNode, status }
    this.activeRequests = new Map();
  }

  /**
   * Route an activation message to the next node in the pipeline.
   * If the sender is the last node, this should not be called (output is handled separately).
   */
  routeActivation(msg, senderWs) {
    const nextNode = this.topology.getNextNode(msg.fromNode);

    if (!nextNode) {
      // No next node — this shouldn't happen if protocol is followed
      const err = createErrorMessage(
        "ROUTE_FAILED",
        `No next node in pipeline after ${msg.fromNode}`
      );
      senderWs.send(JSON.stringify(err));
      return false;
    }

    if (!nextNode.ws || nextNode.ws.readyState !== 1) {
      const err = createErrorMessage(
        "NODE_UNAVAILABLE",
        `Next node ${nextNode.nodeId} is not connected`
      );
      senderWs.send(JSON.stringify(err));
      return false;
    }

    // Update tracking
    if (!this.activeRequests.has(msg.requestId)) {
      this.activeRequests.set(msg.requestId, {
        startTime: Date.now(),
        hops: [],
      });
    }
    this.activeRequests.get(msg.requestId).hops.push({
      from: msg.fromNode,
      to: nextNode.nodeId,
      layer: msg.layer,
      timestamp: Date.now(),
    });

    // Forward to next node — update the toNode field
    const forwarded = {
      ...msg,
      toNode: nextNode.nodeId,
    };

    nextNode.ws.send(JSON.stringify(forwarded));
    return true;
  }

  /**
   * Broadcast an output message to all dashboard/UI connections.
   */
  broadcastOutput(msg, dashboardClients) {
    const serialized = JSON.stringify(msg);
    for (const client of dashboardClients) {
      if (client.readyState === 1) {
        client.send(serialized);
      }
    }

    // Record completion
    const req = this.activeRequests.get(msg.requestId);
    if (req) {
      req.completedAt = Date.now();
      req.latencyMs = req.completedAt - req.startTime;
    }
  }

  /**
   * Get stats for a given request.
   */
  getRequestStats(requestId) {
    return this.activeRequests.get(requestId) || null;
  }

  /**
   * Clean up old request tracking entries (older than maxAge ms).
   */
  cleanupOldRequests(maxAgeMs = 60000) {
    const now = Date.now();
    for (const [id, req] of this.activeRequests) {
      if (req.completedAt && now - req.completedAt > maxAgeMs) {
        this.activeRequests.delete(id);
      }
    }
  }
}
