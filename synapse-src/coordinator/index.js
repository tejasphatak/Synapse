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
import { createServer as createHttpsServer } from "https";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { encode as gptEncode, decode as gptDecode } from "gpt-tokenizer/model/text-davinci-001";
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
import {
  isBinaryMessage,
  decodeBinaryMessage,
  decodeOutputTokens,
  peekMessageType,
  peekRequestId,
  BinaryMsgType,
  uint32ToRequestId,
  registerRequestId,
} from "../protocol/binary.js";

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
let binaryRequestCounter = 0; // uint32 IDs for binary protocol

// ─── Generation State ────────────────────────────────────────────
// generationId → { tokenIds, generatedTokens, maxTokens, promptWs, startTime }
const activeGenerations = new Map();

// ─── Centralized Log Store ───────────────────────────────────────
// Ring buffer of log entries from all nodes, queryable via /api/logs
const LOG_MAX = 5000;
const logStore = [];

function addLog(entry) {
  logStore.push(entry);
  if (logStore.length > LOG_MAX) logStore.splice(0, logStore.length - LOG_MAX);
}

// ─── HTTP Server (serves static files + shard binaries) ───────────

const MIME_TYPES = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".bin": "application/octet-stream",
  ".wgsl": "text/plain",
  ".py": "text/plain",
  ".ipynb": "application/json",
};

function getMimeType(path) {
  const ext = path.substring(path.lastIndexOf("."));
  return MIME_TYPES[ext] || "application/octet-stream";
}

// ─── SSL Configuration ───────────────────────────────────────────
const SSL_CERT = join(ROOT_DIR, "..", "certs", "cert.pem");
const SSL_KEY = join(ROOT_DIR, "..", "certs", "key.pem");
const SSL_PORT = parseInt(process.env.SSL_PORT || "8443", 10);
const hasSSL = existsSync(SSL_CERT) && existsSync(SSL_KEY);

function requestHandler(req, res) {
  // CORS headers for browser access
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  // API: tokenize text using GPT-2 BPE
  if (req.url === "/api/tokenize" && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => { body += chunk; });
    req.on("end", () => {
      try {
        const { text } = JSON.parse(body);
        const tokenIds = gptEncode(text);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ tokenIds }));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // API: detokenize token IDs back to text
  if (req.url === "/api/detokenize" && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => { body += chunk; });
    req.on("end", () => {
      try {
        const { tokenIds } = JSON.parse(body);
        const text = gptDecode(tokenIds);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ text }));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: e.message }));
      }
    });
    return;
  }

  // API: receive diagnostic results from browser
  if (req.url === "/api/diag" && req.method === "POST") {
    let body = "";
    req.on("data", (chunk) => { body += chunk; });
    req.on("end", () => {
      try { writeFileSync("/tmp/diag-results.txt", body); } catch {}
      console.log("[coordinator] Diag results:\n" + body);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
    });
    return;
  }

  // API: get topology snapshot
  if (req.url === "/api/topology") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(topology.toSnapshot()));
    return;
  }

  // API: query collected logs from all nodes
  // GET /api/logs                     → last 200 entries
  // GET /api/logs?n=500               → last 500 entries
  // GET /api/logs?node=node-abc123    → filter by nodeId
  // GET /api/logs?event=layer_forward → filter by event type
  // GET /api/logs?level=perf          → filter by level
  // GET /api/logs?since=1713000000000 → entries after timestamp
  if (req.url.startsWith("/api/logs") && req.method === "GET") {
    const params = new URL(req.url, `http://localhost:${PORT}`).searchParams;
    const n = Math.min(parseInt(params.get("n") || "200", 10), LOG_MAX);
    const nodeFilter = params.get("node");
    const eventFilter = params.get("event");
    const levelFilter = params.get("level");
    const since = parseInt(params.get("since") || "0", 10);

    let results = logStore;
    if (since) results = results.filter(e => e.timestamp > since);
    if (nodeFilter) results = results.filter(e => e.nodeId === nodeFilter);
    if (eventFilter) results = results.filter(e => e.event === eventFilter);
    if (levelFilter) results = results.filter(e => e.level === levelFilter);
    results = results.slice(-n);

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ count: results.length, total: logStore.length, logs: results }));
    return;
  }

  // API: get live performance summary (aggregated per-node stats)
  if (req.url === "/api/perf" && req.method === "GET") {
    const perfByNode = {};
    for (const entry of logStore) {
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
    // Calculate averages
    for (const node of Object.values(perfByNode)) {
      for (const ev of Object.values(node.events)) {
        ev.avgMs = ev.count > 0 ? +(ev.totalMs / ev.count).toFixed(2) : 0;
        if (ev.minMs === Infinity) ev.minMs = 0;
      }
    }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(perfByNode));
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

  // Serve static files from project root (strip query string)
  const urlPath = req.url.split("?")[0];
  let filePath = urlPath === "/" ? "/ui/home.html" : urlPath === "/chat" ? "/ui/prompt.html" : urlPath;
  const fullPath = join(ROOT_DIR, filePath);

  if (existsSync(fullPath)) {
    try {
      const data = readFileSync(fullPath);
      const headers = { "Content-Type": getMimeType(fullPath) };
      if (fullPath.endsWith(".wgsl") || fullPath.endsWith(".js") || fullPath.endsWith(".html")) {
        headers["Cache-Control"] = "no-cache, no-store, must-revalidate";
      }
      res.writeHead(200, headers);
      res.end(data);
      return;
    } catch {
      // fall through to 404
    }
  }

  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("Not Found");
}

const httpServer = createServer(requestHandler);

// ─── HTTPS Server (if certs available) ───────────────────────────
let httpsServer = null;
if (hasSSL) {
  httpsServer = createHttpsServer({
    cert: readFileSync(SSL_CERT),
    key: readFileSync(SSL_KEY),
  }, requestHandler);
}

// ─── WebSocket Server ─────────────────────────────────────────────
// Attach WS to both HTTP and HTTPS servers

const wss = new WebSocketServer({ server: httpServer });
let wssSecure = null;

function setupWss(wsServer) {
  wsServer.on("connection", wsConnectionHandler);
}

function wsConnectionHandler(ws, req) {
  const clientType = new URL(req.url, `http://localhost:${PORT}`).searchParams.get("type");
  console.log(`[coordinator] WS connection: type=${clientType} url=${req.url}`);
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
          prefillDone: false,
          promptLen: msg.tokenIds.length,
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
    // ─── Binary protocol path ─────────────────────────────
    if (isBinaryMessage(raw)) {
      try {
        const msgType = peekMessageType(raw);

        if (msgType === BinaryMsgType.ACTIVATION) {
          // Zero-copy relay: forward the raw binary buffer to the next node
          handleBinaryActivation(ws, raw, nodeId);
        } else if (msgType === BinaryMsgType.OUTPUT) {
          handleBinaryOutput(ws, raw);
        }
      } catch (err) {
        console.error(`[coordinator] Binary parse error from node ${nodeId}:`, err.message);
      }
      return;
    }

    // ─── JSON protocol path ───────────────────────────────
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

      case MessageType.NODE_LOG:
        handleNodeLog(msg);
        break;

      case "P2P_SIGNAL":
        // Relay WebRTC signaling between nodes
        if (msg.to) {
          const targetNode = topology.getNode(msg.to);
          if (targetNode && targetNode.ws && targetNode.ws.readyState === 1) {
            targetNode.ws.send(JSON.stringify(msg));
          }
        }
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
}

// Wire up WebSocket handlers
setupWss(wss);
if (httpsServer) {
  wssSecure = new WebSocketServer({ server: httpsServer });
  setupWss(wssSecure);
}

// ─── Message Handlers ─────────────────────────────────────────────

function handleJoin(ws, msg) {
  const { nodeId, capabilities } = msg;
  const protoV2 = capabilities.protocolV2 ? " [proto_v2]" : "";
  console.log(`[coordinator] Node ${nodeId} joined (webgpu: ${capabilities.webgpu}${protoV2})`);

  topology.addNode(nodeId, ws, capabilities);
  tryAssignShards();
  broadcastTopology();
}

/**
 * Handle binary ACTIVATION: zero-copy relay to next node.
 * We only peek at the header for routing — never decode the tensor payload.
 */
function handleBinaryActivation(senderWs, rawBuffer, senderNodeId) {
  if (!senderNodeId) return;

  const nextNode = topology.getNextNode(senderNodeId);
  if (!nextNode || !nextNode.ws || nextNode.ws.readyState !== 1) {
    console.error(`[coordinator] Binary relay failed: no next node after ${senderNodeId}`);
    return;
  }

  // Forward the raw binary buffer — zero-copy relay
  nextNode.ws.send(rawBuffer);

  const numericId = peekRequestId(rawBuffer);
  const requestId = uint32ToRequestId(numericId);

  // Track hop for telemetry
  if (!router.activeRequests.has(requestId)) {
    router.activeRequests.set(requestId, { startTime: Date.now(), hops: [] });
  }
  router.activeRequests.get(requestId).hops.push({
    from: senderNodeId,
    to: nextNode.nodeId,
    timestamp: Date.now(),
    binary: true,
  });

  broadcastToDashboards({
    type: "ACTIVATION_ROUTED",
    requestId,
    from: senderNodeId,
    to: nextNode.nodeId,
    binary: true,
    timestamp: Date.now(),
  });
}

/**
 * Handle binary OUTPUT: decode tokens, feed back into autoregressive loop.
 */
function handleBinaryOutput(ws, rawBuffer) {
  const decoded = decodeBinaryMessage(rawBuffer);
  const requestId = uint32ToRequestId(decoded.requestId);
  const tokens = decodeOutputTokens(decoded.payload);

  // Delegate to the existing JSON handler with a synthetic message
  handleOutput(ws, { requestId, tokens, timestamp: Date.now() });
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

function handleNodeLog(msg) {
  addLog({
    nodeId: msg.nodeId,
    level: msg.level,
    event: msg.event,
    data: msg.data || {},
    timestamp: msg.timestamp || Date.now(),
  });

  // Forward perf-level logs to dashboards for live monitoring
  if (msg.level === "perf" || msg.level === "error") {
    broadcastToDashboards({
      type: "NODE_LOG",
      nodeId: msg.nodeId,
      level: msg.level,
      event: msg.event,
      data: msg.data,
      timestamp: msg.timestamp,
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
  let genId = msg.requestId;
  let gen = activeGenerations.get(genId);

  // If the requestId came from a node that didn't have the string mapping
  // (e.g., last node received binary activation with numeric ID only),
  // try to resolve via the registered binary ID mapping.
  if (!gen && genId.startsWith("req-")) {
    const numericId = parseInt(genId.slice(4), 10);
    const resolved = uint32ToRequestId(numericId);
    if (resolved !== genId) {
      genId = resolved;
      gen = activeGenerations.get(genId);
    }
  }

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

    // Log generation completion for telemetry
    addLog({
      nodeId: "coordinator",
      level: "perf",
      event: "generation_complete",
      data: {
        requestId: genId,
        totalTokens: gen.generatedTokens.length,
        elapsedMs: elapsed,
        tokensPerSecond: parseFloat(tokPerSec),
        promptLen: gen.promptLen,
      },
      timestamp: Date.now(),
    });

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

    // Free KV caches on all pipeline nodes
    broadcastKVReset(genId);
    activeGenerations.delete(genId);

    // Clean up pending request tracking
    for (const [, state] of promptClients) {
      state.pendingRequests.delete(genId);
    }
  } else {
    // Mark prefill as done after first output
    gen.prefillDone = true;

    // Continue generating — use KV-cached single-token step
    const seqPos = gen.tokenIds.length - 1; // position of the new token
    const result = continueGeneration(genId, newToken, seqPos);
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

  // Assign a uint32 binary request ID and register the mapping
  const binaryReqId = ++binaryRequestCounter;
  registerRequestId(requestId, binaryReqId);

  // Store binary request ID in generation state for reuse in continueGeneration
  const gen = activeGenerations.get(requestId);
  if (gen) gen._binaryReqId = binaryReqId;

  const msg = createInferenceRequestMessage(requestId, tokenIds);
  msg.binaryRequestId = binaryReqId; // nodes use this for binary wire format

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

/**
 * Continue an autoregressive generation with a single new token (KV-cached path).
 * Sends INFERENCE_STEP instead of INFERENCE_REQUEST with full sequence.
 */
function continueGeneration(genId, tokenId, seqPos) {
  if (!topology.isPipelineReady()) {
    return { ok: false, error: "Pipeline not ready" };
  }

  const firstNode = topology.getFirstNode();
  if (!firstNode || firstNode.ws.readyState !== 1) {
    return { ok: false, error: "First pipeline node not available" };
  }

  // Reuse the existing binary request ID mapping
  const gen = activeGenerations.get(genId);
  const binaryReqId = gen?._binaryReqId;

  const msg = {
    type: MessageType.INFERENCE_STEP,
    requestId: genId,
    tokenId,
    seqPos,
    binaryRequestId: binaryReqId || null,
    timestamp: Date.now(),
  };

  firstNode.ws.send(JSON.stringify(msg));

  console.log(`[coordinator] Inference step: ${genId} token=${tokenId} pos=${seqPos}`);

  broadcastToDashboards({
    type: "INFERENCE_STEP",
    requestId: genId,
    tokenId,
    seqPos,
    timestamp: Date.now(),
  });

  return { ok: true };
}

/**
 * Broadcast KV_RESET to all pipeline nodes when a generation completes.
 */
function broadcastKVReset(requestId) {
  const msg = JSON.stringify({
    type: MessageType.KV_RESET,
    requestId,
    timestamp: Date.now(),
  });

  for (const node of topology.nodes.values()) {
    if (node.ws && node.ws.readyState === 1) {
      node.ws.send(msg);
    }
  }
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

if (httpsServer) {
  httpsServer.listen(SSL_PORT, "0.0.0.0", () => {
    console.log(`[coordinator] HTTPS running on https://0.0.0.0:${SSL_PORT}`);
    console.log(`[coordinator] Secure WebSocket: wss://0.0.0.0:${SSL_PORT}`);
  });
}
