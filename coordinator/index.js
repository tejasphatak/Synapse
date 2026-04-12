/**
 * Synapse Coordinator Server
 *
 * WebSocket server that:
 * 1. Accepts connections from browser nodes
 * 2. Assigns model shards to nodes
 * 3. Routes activation packets between nodes
 * 4. Serves model shard files over HTTP
 * 5. Broadcasts topology updates to dashboards
 */

import { createServer } from "http";
import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { WebSocketServer } from "ws";
import { Topology } from "./topology.js";
import { Router } from "./router.js";
import {
  parseMessage,
  MessageType,
  createAssignShardMessage,
  createPongMessage,
  createTopologyUpdateMessage,
  createInferenceRequestMessage,
  createErrorMessage,
} from "../protocol/messages.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = join(__dirname, "..");
const SHARDS_DIR = join(ROOT_DIR, "model", "shards");
const PORT = parseInt(process.env.PORT || "8080", 10);

// ─── Shard Configuration ──────────────────────────────────────────

const SHARD_CONFIG = [
  { shardId: 0, layerStart: 0, layerEnd: 5, file: "shard_0.bin" },
  { shardId: 1, layerStart: 6, layerEnd: 11, file: "shard_1.bin" },
];

// ─── State ────────────────────────────────────────────────────────

const topology = new Topology();
const router = new Router(topology);
const dashboardClients = new Set();
const promptClients = new Map(); // ws → { requestCallbacks }
let requestCounter = 0;

// ─── HTTP Server (serves static files + shard binaries) ───────────

const MIME_TYPES = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".bin": "application/octet-stream",
  ".wgsl": "text/plain",
  ".py": "text/plain",
};

function getMimeType(path) {
  const ext = path.substring(path.lastIndexOf("."));
  return MIME_TYPES[ext] || "application/octet-stream";
}

const httpServer = createServer((req, res) => {
  // CORS headers for browser access
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  // API: get topology snapshot
  if (req.url === "/api/topology") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(topology.toSnapshot()));
    return;
  }

  // API: trigger inference
  if (req.url === "/api/infer" && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      try {
        const { tokenIds } = JSON.parse(body);
        const result = startInference(tokenIds);
        res.writeHead(result.ok ? 200 : 503, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: e.message }));
      }
    });
    return;
  }

  // Serve shard files: /shards/shard_0.bin, /shards/manifest.json, etc.
  if (req.url.startsWith("/shards/")) {
    const filename = req.url.slice("/shards/".length);
    const filepath = join(SHARDS_DIR, filename);
    if (existsSync(filepath)) {
      const data = readFileSync(filepath);
      res.writeHead(200, {
        "Content-Type": getMimeType(filename),
        "Content-Length": data.length,
      });
      res.end(data);
      return;
    }
  }

  // Serve static files from project root
  let filePath = req.url === "/" ? "/ui/prompt.html" : req.url;
  const fullPath = join(ROOT_DIR, filePath);

  if (existsSync(fullPath)) {
    try {
      const data = readFileSync(fullPath);
      res.writeHead(200, { "Content-Type": getMimeType(fullPath) });
      res.end(data);
      return;
    } catch {
      // fall through to 404
    }
  }

  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("Not Found");
});

// ─── WebSocket Server ─────────────────────────────────────────────

const wss = new WebSocketServer({ server: httpServer });

wss.on("connection", (ws, req) => {
  const clientType = new URL(req.url, `http://localhost:${PORT}`).searchParams.get("type");
  const isDashboard = clientType === "dashboard";

  if (isDashboard) {
    dashboardClients.add(ws);
    console.log(`[coordinator] Dashboard connected (${dashboardClients.size} total)`);

    // Send current topology
    const snap = topology.toSnapshot();
    ws.send(JSON.stringify(createTopologyUpdateMessage(snap.nodes, snap.pipeline)));

    ws.on("close", () => {
      dashboardClients.delete(ws);
      console.log(`[coordinator] Dashboard disconnected`);
    });
    return;
  }

  // Prompt client — sends inference requests, receives outputs
  const isPrompt = clientType === "prompt";
  if (isPrompt) {
    promptClients.set(ws, { pendingRequests: new Map() });
    console.log(`[coordinator] Prompt client connected (${promptClients.size} total)`);

    // Send current topology so prompt UI knows pipeline state
    const snap = topology.toSnapshot();
    ws.send(JSON.stringify(createTopologyUpdateMessage(snap.nodes, snap.pipeline)));

    ws.on("message", (raw) => {
      let msg;
      try { msg = JSON.parse(raw.toString()); } catch { return; }

      if (msg.type === "PROMPT_INFER") {
        // Prompt client wants to run inference
        const result = startInference(msg.tokenIds);
        if (result.ok) {
          // Track which prompt client is waiting for this request
          promptClients.get(ws)?.pendingRequests.set(result.requestId, true);
          ws.send(JSON.stringify({ type: "INFER_STARTED", requestId: result.requestId }));
        } else {
          ws.send(JSON.stringify({ type: "INFER_ERROR", error: result.error }));
        }
      }
    });

    ws.on("close", () => {
      promptClients.delete(ws);
      console.log(`[coordinator] Prompt client disconnected`);
    });
    return;
  }

  // This is a compute node
  let nodeId = null;

  ws.on("message", (raw) => {
    const { msg, error } = parseMessage(raw.toString());
    if (error) {
      ws.send(JSON.stringify(createErrorMessage("PARSE_ERROR", error)));
      return;
    }

    switch (msg.type) {
      case MessageType.JOIN:
        nodeId = msg.nodeId;
        handleJoin(ws, msg);
        break;

      case MessageType.PING:
        if (nodeId) topology.updatePing(nodeId);
        ws.send(JSON.stringify(createPongMessage()));
        break;

      case MessageType.ACTIVATION:
        handleActivation(ws, msg);
        break;

      case MessageType.OUTPUT:
        handleOutput(ws, msg);
        break;

      case "NODE_READY":
        handleNodeReady(msg);
        break;

      default:
        ws.send(
          JSON.stringify(createErrorMessage("UNKNOWN_TYPE", `Unhandled message type: ${msg.type}`))
        );
    }
  });

  ws.on("close", () => {
    if (nodeId) {
      console.log(`[coordinator] Node ${nodeId} disconnected`);
      topology.removeNode(nodeId);
      broadcastTopology();
    }
  });

  ws.on("error", (err) => {
    console.error(`[coordinator] WebSocket error for node ${nodeId}:`, err.message);
  });
});

// ─── Message Handlers ─────────────────────────────────────────────

function handleJoin(ws, msg) {
  const { nodeId, capabilities } = msg;
  console.log(`[coordinator] Node ${nodeId} joined (webgpu: ${capabilities.webgpu})`);

  topology.addNode(nodeId, ws, capabilities);
  tryAssignShards();
  broadcastTopology();
}

function handleActivation(ws, msg) {
  // Check if sender is the last node — should send OUTPUT instead
  if (topology.isLastNode(msg.fromNode)) {
    ws.send(
      JSON.stringify(
        createErrorMessage("PROTOCOL_ERROR", "Last node should send OUTPUT, not ACTIVATION")
      )
    );
    return;
  }

  const success = router.routeActivation(msg, ws);
  if (success) {
    // Notify dashboards about activation flow
    broadcastToDashboards({
      type: "ACTIVATION_ROUTED",
      requestId: msg.requestId,
      from: msg.fromNode,
      to: msg.toNode,
      layer: msg.layer,
      tensorSize: msg.tensor?.data?.length || 0,
      timestamp: Date.now(),
    });
  }
}

function handleNodeReady(msg) {
  const { nodeId, shardId } = msg;
  console.log(`[coordinator] Node ${nodeId} ready (shard ${shardId})`);
  topology.markReady(nodeId);
  broadcastTopology();

  if (topology.isPipelineReady()) {
    console.log(`[coordinator] Pipeline is READY — all shards loaded`);
    // Notify all prompt clients
    for (const [ws] of promptClients) {
      if (ws.readyState === 1) {
        ws.send(JSON.stringify({ type: "PIPELINE_READY" }));
      }
    }
  }
}

function handleOutput(ws, msg) {
  console.log(
    `[coordinator] Output received for request ${msg.requestId}: ${msg.tokens?.length} tokens`
  );

  // Broadcast to dashboards
  router.broadcastOutput(msg, dashboardClients);

  // Send to the prompt client that initiated this request
  for (const [promptWs, state] of promptClients) {
    if (state.pendingRequests.has(msg.requestId) && promptWs.readyState === 1) {
      promptWs.send(JSON.stringify(msg));
      state.pendingRequests.delete(msg.requestId);
    }
  }

  // Log stats
  const stats = router.getRequestStats(msg.requestId);
  if (stats) {
    console.log(`[coordinator] Request ${msg.requestId} completed in ${stats.latencyMs}ms`);
  }
}

// ─── Shard Assignment ─────────────────────────────────────────────

function tryAssignShards() {
  const unassigned = topology.getUnassignedNodes();
  const assignedShardIds = new Set(
    [...topology.nodes.values()]
      .filter((n) => n.shardId !== null)
      .map((n) => n.shardId)
  );

  for (const config of SHARD_CONFIG) {
    if (assignedShardIds.has(config.shardId)) continue;

    // Find an unassigned node for this shard
    const node = unassigned.shift();
    if (!node) break;

    topology.assignShard(node.nodeId, config.shardId, config.layerStart, config.layerEnd);

    const assignMsg = createAssignShardMessage(
      config.shardId,
      config.layerStart,
      config.layerEnd,
      `/shards/${config.file}`,
      "/shards/shared.bin"
    );

    node.ws.send(JSON.stringify(assignMsg));
    console.log(
      `[coordinator] Assigned shard ${config.shardId} (layers ${config.layerStart}-${config.layerEnd}) to node ${node.nodeId}`
    );
  }
}

// ─── Inference Trigger ────────────────────────────────────────────

function startInference(tokenIds) {
  if (!topology.isPipelineReady()) {
    return { ok: false, error: "Pipeline not ready — waiting for all nodes" };
  }

  const firstNode = topology.getFirstNode();
  if (!firstNode || firstNode.ws.readyState !== 1) {
    return { ok: false, error: "First pipeline node not available" };
  }

  const requestId = `req-${++requestCounter}-${Date.now()}`;
  const msg = createInferenceRequestMessage(requestId, tokenIds);

  firstNode.ws.send(JSON.stringify(msg));

  // Track the request
  router.activeRequests.set(requestId, {
    startTime: Date.now(),
    hops: [],
  });

  console.log(`[coordinator] Inference started: ${requestId} (${tokenIds.length} tokens)`);

  // Notify dashboards
  broadcastToDashboards({
    type: "INFERENCE_STARTED",
    requestId,
    tokenCount: tokenIds.length,
    timestamp: Date.now(),
  });

  return { ok: true, requestId };
}

// ─── Broadcasts ───────────────────────────────────────────────────

function broadcastTopology() {
  const snap = topology.toSnapshot();
  const msg = JSON.stringify(createTopologyUpdateMessage(snap.nodes, snap.pipeline));

  // Send to all compute nodes
  for (const node of topology.nodes.values()) {
    if (node.ws && node.ws.readyState === 1) {
      node.ws.send(msg);
    }
  }

  // Send to dashboards
  for (const client of dashboardClients) {
    if (client.readyState === 1) {
      client.send(msg);
    }
  }
}

function broadcastToDashboards(data) {
  const msg = JSON.stringify(data);
  for (const client of dashboardClients) {
    if (client.readyState === 1) {
      client.send(msg);
    }
  }
}

// ─── Heartbeat Check ──────────────────────────────────────────────

setInterval(() => {
  const stale = topology.getStaleNodes(30000);
  for (const node of stale) {
    console.log(`[coordinator] Node ${node.nodeId} stale — removing`);
    if (node.ws) node.ws.terminate();
    topology.removeNode(node.nodeId);
  }
  if (stale.length > 0) broadcastTopology();
  router.cleanupOldRequests();
}, 10000);

// ─── Start ────────────────────────────────────────────────────────

httpServer.listen(PORT, () => {
  console.log(`[coordinator] Synapse coordinator running on http://localhost:${PORT}`);
  console.log(`[coordinator] WebSocket: ws://localhost:${PORT}`);
  console.log(`[coordinator] Shard files: http://localhost:${PORT}/shards/`);
  console.log(`[coordinator] Dashboard:   http://localhost:${PORT}/ui/dashboard.html`);
  console.log(`[coordinator] Prompt UI:   http://localhost:${PORT}/`);
});
