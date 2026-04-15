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
import { readFileSync, writeFileSync, existsSync, statSync } from "fs";
import { join, dirname, resolve } from "path";
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
  peekFlags,
  BinaryMsgType,
  Flags,
  uint32ToRequestId,
  registerRequestId,
} from "../protocol/binary.js";
import { GenerationManager } from "./generation.js";
import { LogStore } from "./log-store.js";

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

// Shard URL version: uses manifest file mtime so each re-split generates a
// distinct URL query, bypassing browser HTTP cache that may hold stale
// responses (with old content + Content-Length) for a given shard path.
// Cache-Control: no-store prevents future caching but does NOT invalidate
// existing cached responses. URL versioning is the only reliable bust.
function shardVersion() {
  try {
    const mp = join(SHARDS_DIR, "manifest.json");
    return String(statSync(mp).mtimeMs | 0);
  } catch { return String(Date.now()); }
}
const SHARD_VERSION = shardVersion();
console.log(`[coordinator] Shard version: ${SHARD_VERSION}`);

// ─── State ────────────────────────────────────────────────────────

const topology = new Topology(SHARD_CONFIG.length);
const router = new Router(topology);
const dashboardClients = new Set();
const promptClients = new Map(); // ws → { requestCallbacks }
let requestCounter = 0;
let binaryRequestCounter = 0; // uint32 IDs for binary protocol

// ─── Generation State ────────────────────────────────────────────
const GENERATION_TIMEOUT_MS = 60000;
const generations = new GenerationManager(GENERATION_TIMEOUT_MS);

// ─── Centralized Log Store ───────────────────────────────────────
const LOG_MAX = 5000;
const logStore = new LogStore(LOG_MAX);

function addLog(entry) {
  logStore.add(entry);
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

  // API: public health dashboard — safe, no credentials or PII
  if (req.url === "/api/health") {
    const uptime = process.uptime();
    const nodes = [...topology.nodes.values()];
    const readyNodes = nodes.filter(n => n.status === "ready");
    const totalInferences = generations.size;
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      status: readyNodes.length >= SHARD_CONFIG.length ? "operational" : nodes.length > 0 ? "degraded" : "waiting_for_nodes",
      version: "0.1.0",
      uptime_seconds: Math.floor(uptime),
      nodes: {
        connected: nodes.length,
        ready: readyNodes.length,
        required: SHARD_CONFIG.length,
        shards_covered: [...new Set(readyNodes.map(n => n.shardId))].length,
      },
      model: { name: "GPT-2 117M", dtype: "float16", shards: SHARD_CONFIG.length },
      pipeline_ready: topology.pipeline.length >= SHARD_CONFIG.length,
      active_generations: totalInferences,
      logs_collected: logStore.length,
      phase: "Phases 1-4 complete — validating with real WebGPU",
      timestamp: Date.now(),
    }));
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
    const result = logStore.query({
      n: parseInt(params.get("n") || "200", 10),
      node: params.get("node") || undefined,
      event: params.get("event") || undefined,
      level: params.get("level") || undefined,
      since: parseInt(params.get("since") || "0", 10) || undefined,
    });

    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(result));
    return;
  }

  // API: get live performance summary (aggregated per-node stats)
  if (req.url === "/api/perf" && req.method === "GET") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(logStore.getPerfSummary()));
    return;
  }

  // API: admin shard assignment — explicitly assign a shard to a specific
  // (currently connected, unassigned) node. Useful when a late-arriving
  // higher-capability node should take over before a natural disconnect.
  //
  // Auth: require X-Admin-Token header to match NEX_ADMIN_TOKEN env (if set).
  // If NEX_ADMIN_TOKEN is unset, endpoint is disabled — fail closed.
  //
  // Body: {"nodeId":"node-abc...","shardId":0}  or  {"nodeId":"...","unassign":true}
  if (req.url === "/api/assign" && req.method === "POST") {
    const expectedToken = process.env.NEX_ADMIN_TOKEN;
    if (!expectedToken) {
      res.writeHead(503, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "admin endpoint disabled (no NEX_ADMIN_TOKEN)" }));
      return;
    }
    const tokenHeader = req.headers["x-admin-token"];
    if (tokenHeader !== expectedToken) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "unauthorized" }));
      return;
    }
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      try {
        const { nodeId, shardId, unassign } = JSON.parse(body);
        if (!nodeId) throw new Error("nodeId required");
        const node = topology.getNode(nodeId);
        if (!node) throw new Error(`node not found: ${nodeId}`);

        if (unassign) {
          // Clear this node's shard assignment. Node keeps connection; future
          // tryAssignShards may re-assign. Pipeline rebuilds to exclude it.
          node.shardId = null;
          node.layerStart = null;
          node.layerEnd = null;
          node.status = "connected";
          topology._rebuildPipeline();
          broadcastTopology();
          console.log(`[coordinator] admin: unassigned ${nodeId}`);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true, action: "unassign", nodeId, topology: topology.toSnapshot() }));
          return;
        }

        if (typeof shardId !== "number") throw new Error("shardId (number) required unless unassign:true");
        const config = SHARD_CONFIG.find((c) => c.shardId === shardId);
        if (!config) throw new Error(`unknown shardId: ${shardId}`);
        if (node.shardId !== null && node.shardId !== shardId) {
          throw new Error(`node already holds shard ${node.shardId}; unassign first`);
        }

        const ok = topology.assignShard(nodeId, shardId, config.layerStart, config.layerEnd);
        if (!ok) throw new Error("assignShard failed");

        const assignMsg = createAssignShardMessage(
          config.shardId,
          config.layerStart,
          config.layerEnd,
          `/shards/${config.file}?v=${SHARD_VERSION}`,
          `/shards/shared.bin?v=${SHARD_VERSION}`,
        );
        node.ws.send(JSON.stringify(assignMsg));
        broadcastTopology();
        console.log(
          `[coordinator] admin: assigned shard ${shardId} (layers ${config.layerStart}-${config.layerEnd}) to ${nodeId}`,
        );

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, action: "assign", nodeId, shardId, topology: topology.toSnapshot() }));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: e.message }));
      }
    });
    return;
  }

  // API: admin-triggered client reload. Broadcasts CLIENT_RELOAD to all
  // connected nodes so they pick up new node.js/pipeline.js code without
  // manual browser refresh. Auth: NEX_ADMIN_TOKEN same as /api/assign.
  //
  // Body (optional): {"delayMs":1500, "reason":"post-deploy", "targetShardId":0}
  // Default: reload ALL nodes after 1000ms with reason "admin-triggered".
  if (req.url === "/api/reload" && req.method === "POST") {
    const expectedToken = process.env.NEX_ADMIN_TOKEN;
    if (!expectedToken) {
      res.writeHead(503, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "admin endpoint disabled (no NEX_ADMIN_TOKEN)" }));
      return;
    }
    if (req.headers["x-admin-token"] !== expectedToken) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "unauthorized" }));
      return;
    }
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      try {
        const opts = body ? JSON.parse(body) : {};
        const delayMs = opts.delayMs ?? 1000;
        const reason = opts.reason ?? "admin-triggered";
        const targetShardId = opts.targetShardId; // undefined = all nodes
        const msg = JSON.stringify({ type: "CLIENT_RELOAD", delayMs, reason });
        let sent = 0;
        for (const [nodeId, node] of topology.nodes) {
          if (targetShardId !== undefined && node.shardId !== targetShardId) continue;
          if (node.ws?.readyState === 1) {
            node.ws.send(msg);
            sent++;
          }
        }
        console.log(`[coordinator] admin: CLIENT_RELOAD broadcast to ${sent} node(s) (delay=${delayMs}ms, reason="${reason}")`);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, action: "reload", nodesNotified: sent, delayMs, reason }));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: e.message }));
      }
    });
    return;
  }

  // API: admin-triggered HOT reload. Broadcasts HOT_RELOAD to browser nodes.
  // Browser dynamically re-imports node.js + instance-swaps while keeping the
  // WebGPU device + shard buffers alive. No page reload, no user tap required.
  // Body: {"reason":"deploy-xyz", "targetShardId":0}
  if (req.url === "/api/hot-reload" && req.method === "POST") {
    const expectedToken = process.env.NEX_ADMIN_TOKEN;
    if (!expectedToken) {
      res.writeHead(503, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "admin endpoint disabled (no NEX_ADMIN_TOKEN)" }));
      return;
    }
    if (req.headers["x-admin-token"] !== expectedToken) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: false, error: "unauthorized" }));
      return;
    }
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      try {
        const opts = body ? JSON.parse(body) : {};
        const reason = opts.reason ?? "admin-triggered";
        const targetShardId = opts.targetShardId;
        const msg = JSON.stringify({ type: "HOT_RELOAD", reason, ts: Date.now() });
        let sent = 0;
        for (const [, node] of topology.nodes) {
          if (targetShardId !== undefined && node.shardId !== targetShardId) continue;
          if (node.ws?.readyState === 1) {
            node.ws.send(msg);
            sent++;
          }
        }
        console.log(`[coordinator] admin: HOT_RELOAD broadcast to ${sent} node(s) (reason="${reason}")`);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, action: "hot-reload", nodesNotified: sent, reason }));
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: false, error: e.message }));
      }
    });
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
    const filename = decodeURIComponent(req.url.slice("/shards/".length).split("?")[0]);
    const filepath = resolve(SHARDS_DIR, filename);
    // Path traversal guard: resolved path must stay within SHARDS_DIR
    if (filepath.startsWith(SHARDS_DIR + "/") && existsSync(filepath)) {
      const data = readFileSync(filepath);
      res.writeHead(200, {
        "Content-Type": getMimeType(filename),
        "Content-Length": data.length,
        // Shard contents change with split config (num_shards, dtype). Without
        // no-store the browser HTTP cache serves stale/truncated older-config
        // bytes, causing opaque "exceeds source buffer" errors. IndexedDB
        // layer in shard-loader.js does versioned caching properly; HTTP
        // layer shouldn't double-cache.
        "Cache-Control": "no-store",
      });
      res.end(data);
      return;
    }
  }

  // Serve static files from project root (strip query string)
  const urlPath = decodeURIComponent(req.url.split("?")[0]);
  let filePath = urlPath === "/" ? "/ui/home.html" : urlPath === "/chat" ? "/ui/prompt.html" : urlPath;
  const fullPath = resolve(ROOT_DIR, filePath.startsWith("/") ? filePath.slice(1) : filePath);

  // Path traversal guard: resolved path must stay within ROOT_DIR
  if ((fullPath === ROOT_DIR || fullPath.startsWith(ROOT_DIR + "/")) && existsSync(fullPath)) {
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
        const temperature = msg.temperature; // may be undefined; defaults downstream
        const genId = `gen-${++requestCounter}-${Date.now()}`;

        const gen = generations.create(genId, msg.tokenIds, maxTokens);
        gen.promptWs = ws;

        const result = startInference(msg.tokenIds, genId, { temperature });
        if (result.ok) {
          promptClients.get(ws)?.pendingRequests.set(genId, true);
          ws.send(JSON.stringify({ type: "INFER_STARTED", requestId: genId }));
        } else {
          generations.remove(genId);
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
        } else if (msgType === BinaryMsgType.EARLY_EXIT) {
          // Node detected convergence — skip remaining pipeline, treat as output
          handleBinaryEarlyExit(ws, raw, nodeId);
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
      // Refill any shard slot the disconnected node was holding from the
      // unassigned pool, otherwise new connects sit idle as "waiting".
      tryAssignShards();
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

  // Self-test telemetry: if the device reported failures, log them to a
  // pending-fix registry. The self-healing loop reads this registry and
  // proposes kernel patches. Append-only; each record timestamped.
  const st = capabilities.selfTest;
  if (st && st.pass === false && Array.isArray(st.failures) && st.failures.length > 0) {
    const record = {
      ts: Date.now(),
      nodeId,
      gpuVendor: capabilities.gpuVendor || "unknown",
      userAgent: (capabilities.userAgent || "").slice(0, 120),
      mobile: !!capabilities.mobile,
      failures: st.failures,
    };
    try {
      const logPath = join(ROOT_DIR, "logs", "pending_self_test_fixes.jsonl");
      const logDir = join(ROOT_DIR, "logs");
      if (!existsSync(logDir)) {
        // best-effort mkdir
        try { require("fs").mkdirSync(logDir, { recursive: true }); } catch (_) {}
      }
      writeFileSync(logPath, JSON.stringify(record) + "\n", { flag: "a" });
    } catch (e) {
      console.warn(`[coordinator] failed to write pending_fix record: ${e.message}`);
    }
    addLog({
      nodeId,
      level: "warn",
      event: "self_test_failed",
      data: { vendor: capabilities.gpuVendor, failures: st.failures },
      timestamp: Date.now(),
    });
    console.warn(`[coordinator] Node ${nodeId} (${capabilities.gpuVendor}) failed self-test: ${st.failures.map(f => f.kernel).join(", ")}`);
    // Policy: failing devices are STILL added to topology but won't get shards
    // assigned by tryAssignShards. This gives visibility without breaking the
    // inference pool. Future: auto-exclude once auto-fix loop is in place.
  } else if (st && st.pass === true) {
    addLog({
      nodeId,
      level: "info",
      event: "self_test_passed",
      data: { vendor: capabilities.gpuVendor },
      timestamp: Date.now(),
    });
  }

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

  const flags = peekFlags(rawBuffer);
  const predicted = !!(flags & Flags.PREDICTED);

  // Track hop for telemetry
  if (!router.activeRequests.has(requestId)) {
    router.activeRequests.set(requestId, { startTime: Date.now(), hops: [] });
  }
  router.activeRequests.get(requestId).hops.push({
    from: senderNodeId,
    to: nextNode.nodeId,
    timestamp: Date.now(),
    binary: true,
    predicted,
  });

  broadcastToDashboards({
    type: "ACTIVATION_ROUTED",
    requestId,
    from: senderNodeId,
    to: nextNode.nodeId,
    binary: true,
    predicted,
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

/**
 * Handle binary EARLY_EXIT: a node detected convergence and is short-circuiting
 * the pipeline. Treat the activation as final output — skip remaining nodes.
 */
function handleBinaryEarlyExit(ws, rawBuffer, senderNodeId) {
  const decoded = decodeBinaryMessage(rawBuffer);
  const requestId = uint32ToRequestId(decoded.requestId);
  const tokens = decodeOutputTokens(decoded.payload);

  logStore.add({
    nodeId: senderNodeId,
    event: "early_exit",
    level: "info",
    data: { requestId, layersSaved: "unknown" },
    timestamp: Date.now(),
  });

  broadcastToDashboards({
    type: "EARLY_EXIT",
    requestId,
    nodeId: senderNodeId,
    timestamp: Date.now(),
  });

  // Feed into generation loop as if the full pipeline completed
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
  let gen = generations.get(genId);

  // If the requestId came from a node that didn't have the string mapping
  // (e.g., last node received binary activation with numeric ID only),
  // try to resolve via the registered binary ID mapping.
  if (!gen && genId.startsWith("req-")) {
    const numericId = parseInt(genId.slice(4), 10);
    const resolved = uint32ToRequestId(numericId);
    if (resolved !== genId) {
      genId = resolved;
      gen = generations.get(genId);
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
  const { done, reason, seqPos } = gen.addToken(newToken);

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

  if (done) {
    const stats = gen.getStats();
    console.log(
      `[coordinator] Generation ${genId} complete: ${stats.totalTokens} tokens in ${stats.elapsedMs}ms (${stats.tokensPerSecond} tok/s)`
    );

    addLog({
      nodeId: "coordinator",
      level: "perf",
      event: "generation_complete",
      data: {
        requestId: genId,
        totalTokens: stats.totalTokens,
        elapsedMs: stats.elapsedMs,
        tokensPerSecond: stats.tokensPerSecond,
        promptLen: stats.promptLen,
      },
      timestamp: Date.now(),
    });

    if (gen.promptWs && gen.promptWs.readyState === 1) {
      gen.promptWs.send(JSON.stringify({
        type: "GENERATION_DONE",
        requestId: genId,
        tokens: gen.generatedTokens,
        totalTokens: stats.totalTokens,
        elapsedMs: stats.elapsedMs,
        tokensPerSecond: stats.tokensPerSecond,
      }));
    }

    broadcastKVReset(genId);
    generations.remove(genId);

    for (const [, state] of promptClients) {
      state.pendingRequests.delete(genId);
    }
  } else {
    // Continue generating — use KV-cached single-token step
    const result = continueGeneration(genId, newToken, seqPos);
    if (!result.ok) {
      console.error(`[coordinator] Generation ${genId} failed to continue: ${result.error}`);
      if (gen.promptWs && gen.promptWs.readyState === 1) {
        const stats = gen.getStats();
        gen.promptWs.send(JSON.stringify({
          type: "GENERATION_DONE",
          requestId: genId,
          tokens: gen.generatedTokens,
          totalTokens: stats.totalTokens,
          elapsedMs: stats.elapsedMs,
          error: result.error,
        }));
      }
      generations.remove(genId);
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
      `/shards/${config.file}?v=${SHARD_VERSION}`,
      `/shards/shared.bin?v=${SHARD_VERSION}`
    );

    node.ws.send(JSON.stringify(assignMsg));
    console.log(
      `[coordinator] Assigned shard ${config.shardId} (layers ${config.layerStart}-${config.layerEnd}) to node ${node.nodeId}`
    );
  }
}

// ─── Inference Trigger ────────────────────────────────────────────

function startInference(tokenIds, generationId, options = {}) {
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
  const gen = generations.get(requestId);
  if (gen) gen._binaryReqId = binaryReqId;

  const msg = createInferenceRequestMessage(requestId, tokenIds, {
    temperature: options.temperature,
  });
  msg.binaryRequestId = binaryReqId; // nodes use this for binary wire format

  // Stash temperature on the generation so continueGeneration can reuse it for decode steps
  if (gen && options.temperature !== undefined) gen._temperature = options.temperature;

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
  const gen = generations.get(genId);
  const binaryReqId = gen?._binaryReqId;

  const msg = {
    type: MessageType.INFERENCE_STEP,
    requestId: genId,
    tokenId,
    seqPos,
    binaryRequestId: binaryReqId || null,
    timestamp: Date.now(),
  };
  if (gen && gen._temperature !== undefined) msg.temperature = gen._temperature;

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

// Coord-initiated PING: every 10s, poke every connected node so idle/backgrounded
// tabs keep their WebSocket warm (mobile browsers throttle TX but still RX).
setInterval(() => {
  const pingMsg = JSON.stringify({ type: "PING" });
  for (const node of topology.nodes.values()) {
    if (node.ws && node.ws.readyState === 1) {
      try { node.ws.send(pingMsg); } catch (_) { /* ignore */ }
    }
  }
}, 10000);

// Coord self-audit: every 60s, if any nodes are connected but unassigned,
// call tryAssignShards() as a backstop for missed assign opportunities.
// This is the paranoid belt-and-suspenders fix for the "waiting for assignment"
// bug — even if some disconnect path misses calling tryAssignShards, this
// sweep catches it within a minute.
setInterval(() => {
  const unassignedCount = topology.getUnassignedNodes().length;
  if (unassignedCount > 0) {
    const beforeReady = [...topology.nodes.values()].filter(n => n.shardId !== null).length;
    tryAssignShards();
    const afterReady = [...topology.nodes.values()].filter(n => n.shardId !== null).length;
    if (afterReady > beforeReady) {
      console.log(`[coordinator] self-audit: assigned ${afterReady - beforeReady} shard(s) to previously-unassigned nodes`);
      broadcastTopology();
    }
  }
}, 60000);

setInterval(() => {
  const stale = topology.getStaleNodes(60000);
  for (const node of stale) {
    console.log(`[coordinator] Node ${node.nodeId} stale — removing`);
    if (node.ws) node.ws.terminate();
    topology.removeNode(node.nodeId);
  }
  if (stale.length > 0) {
    tryAssignShards(); // refill any shard slots vacated by the sweep
    broadcastTopology();
  }
  router.cleanupOldRequests();

  // Sweep timed-out generations — prevents memory leaks from orphaned requests
  const now = Date.now();
  const timedOut = generations.sweepTimedOut(now);
  for (const gen of timedOut) {
    const stats = gen.getStats(now);
    console.warn(`[coordinator] Generation ${gen.id} timed out after ${Math.round(stats.elapsedMs / 1000)}s — cleaning up`);

    if (gen.promptWs && gen.promptWs.readyState === 1) {
      gen.promptWs.send(JSON.stringify({
        type: "GENERATION_DONE",
        requestId: gen.id,
        tokens: gen.generatedTokens,
        totalTokens: stats.totalTokens,
        elapsedMs: stats.elapsedMs,
        error: "Generation timed out — a node may have disconnected",
      }));
    }

    addLog({
      nodeId: "coordinator",
      level: "error",
      event: "generation_timeout",
      data: { requestId: gen.id, generatedTokens: stats.totalTokens, elapsedMs: stats.elapsedMs },
      timestamp: now,
    });

    broadcastKVReset(gen.id);

    for (const [, state] of promptClients) {
      state.pendingRequests.delete(gen.id);
    }
  }
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
