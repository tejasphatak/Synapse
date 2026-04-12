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

// Auto-read shard config from manifest if available, else fall back to defaults
function loadShardConfig() {
  const manifestPath = join(SHARDS_DIR, "manifest.json");
  if (existsSync(manifestPath)) {
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    const layout = manifest.shard_layout;
    return Object.entries(layout).map(([id, cfg]) => ({
      shardId: parseInt(id),
      layerStart: cfg.layer_start,
      layerEnd: cfg.layer_end,
      file: cfg.file,
    })).sort((a, b) => a.shardId - b.shardId);
  }
  // Default: GPT-2 small (12 layers, 2 shards)
  return [
    { shardId: 0, layerStart: 0, layerEnd: 5, file: "shard_0.bin" },
    { shardId: 1, layerStart: 6, layerEnd: 11, file: "shard_1.bin" },
  ];
}

const SHARD_CONFIG = loadShardConfig();

// ─── State ────────────────────────────────────────────────────────

const topology = new Topology(SHARD_CONFIG.length);
const router = new Router(topology);
const dashboardClients = new Set();
const promptClients = new Map(); // ws → { requestCallbacks }
let requestCounter = 0;

// ─── Generation State ────────────────────────────────────────────
// generationId → { tokenIds, generatedTokens, maxTokens, promptWs, startTime }
const activeGenerations = new Map();

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
        const maxTokens = msg.maxTokens || 30;
        const genId = `gen-${++requestCounter}-${Date.now()}`;

        // Start a generation session
        activeGenerations.set(genId, {
          tokenIds: [...msg.tokenIds],
          generatedTokens: [],
          maxTokens,
          promptWs: ws,
          startTime: Date.now(),
        });

        const result = startInference(msg.tokenIds, genId);
        if (result.ok) {
          promptClients.get(ws)?.pendingRequests.set(genId, true);
          ws.send(JSON.stringify({ type: "INFER_STARTED", requestId: genId }));
        } else {
          activeGenerations.delete(genId);
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
      console.error(`[coordinator] Parse error from node ${nodeId}: ${error}`);
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

      case MessageType.NODE_READY:
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
  console.log(`[coordinator] Activation received from ${msg.fromNode} (layer ${msg.layer}, request ${msg.requestId})`);

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
  const genId = msg.requestId;
  const gen = activeGenerations.get(genId);

  console.log(
    `[coordinator] Output received for ${genId}: token ${msg.tokens?.[0]} (${gen ? gen.generatedTokens.length + 1 + "/" + gen.maxTokens : "no gen"})`
  );

  // Broadcast to dashboards
  router.broadcastOutput(msg, dashboardClients);

  if (!gen) {
    // Legacy single-token mode — send directly to prompt client
    for (const [promptWs, state] of promptClients) {
      if (state.pendingRequests.has(genId) && promptWs.readyState === 1) {
        promptWs.send(JSON.stringify(msg));
        state.pendingRequests.delete(genId);
      }
    }
    return;
  }

  // ─── Autoregressive loop ─────────────────────────────
  const newToken = msg.tokens[0];
  gen.generatedTokens.push(newToken);
  gen.tokenIds.push(newToken);

  // Stream the token to the prompt client immediately
  if (gen.promptWs && gen.promptWs.readyState === 1) {
    gen.promptWs.send(JSON.stringify({
      type: "TOKEN_GENERATED",
      requestId: genId,
      token: newToken,
      tokenIndex: gen.generatedTokens.length,
      totalGenerated: gen.generatedTokens.length,
      maxTokens: gen.maxTokens,
    }));
  }

  // Check if we should stop: max tokens, or EOS token (50256 for GPT-2)
  const isEOS = newToken === 50256;
  const isDone = gen.generatedTokens.length >= gen.maxTokens || isEOS;

  if (isDone) {
    const elapsed = Date.now() - gen.startTime;
    const tokPerSec = (gen.generatedTokens.length / (elapsed / 1000)).toFixed(1);
    console.log(
      `[coordinator] Generation ${genId} complete: ${gen.generatedTokens.length} tokens in ${elapsed}ms (${tokPerSec} tok/s)`
    );

    // Send completion message
    if (gen.promptWs && gen.promptWs.readyState === 1) {
      gen.promptWs.send(JSON.stringify({
        type: "GENERATION_DONE",
        requestId: genId,
        tokens: gen.generatedTokens,
        totalTokens: gen.generatedTokens.length,
        elapsedMs: elapsed,
        tokensPerSecond: parseFloat(tokPerSec),
      }));
    }

    activeGenerations.delete(genId);

    // Clean up pending request tracking
    for (const [, state] of promptClients) {
      state.pendingRequests.delete(genId);
    }
  } else {
    // Continue generating — send the full sequence back through the pipeline
    const result = startInference(gen.tokenIds, genId);
    if (!result.ok) {
      console.error(`[coordinator] Generation ${genId} failed to continue: ${result.error}`);
      if (gen.promptWs && gen.promptWs.readyState === 1) {
        gen.promptWs.send(JSON.stringify({
          type: "GENERATION_DONE",
          requestId: genId,
          tokens: gen.generatedTokens,
          totalTokens: gen.generatedTokens.length,
          elapsedMs: Date.now() - gen.startTime,
          error: result.error,
        }));
      }
      activeGenerations.delete(genId);
    }
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

function startInference(tokenIds, generationId) {
  if (!topology.isPipelineReady()) {
    return { ok: false, error: "Pipeline not ready — waiting for all nodes" };
  }

  const firstNode = topology.getFirstNode();
  if (!firstNode || firstNode.ws.readyState !== 1) {
    return { ok: false, error: "First pipeline node not available" };
  }

  const requestId = generationId || `req-${++requestCounter}-${Date.now()}`;
  const msg = createInferenceRequestMessage(requestId, tokenIds);

  firstNode.ws.send(JSON.stringify(msg));

  // Track the request
  router.activeRequests.set(requestId, {
    startTime: Date.now(),
    hops: [],
  });

  console.log(`[coordinator] Inference step: ${requestId} (${tokenIds.length} tokens)`);

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

httpServer.listen(PORT, "0.0.0.0", () => {
  console.log(`[coordinator] Synapse coordinator running on http://0.0.0.0:${PORT}`);
  console.log(`[coordinator] WebSocket: ws://0.0.0.0:${PORT}`);
  console.log(`[coordinator] Shard files: http://0.0.0.0:${PORT}/shards/`);
  console.log(`[coordinator] Dashboard:   http://0.0.0.0:${PORT}/ui/dashboard.html`);
  console.log(`[coordinator] Prompt UI:   http://0.0.0.0:${PORT}/`);
});
