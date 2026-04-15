#!/usr/bin/env node
/**
 * cli-infer.mjs — command-line Synapse inference client.
 *
 * Connects to a Synapse coordinator as a prompt client over WebSocket,
 * tokenizes a prompt, fires PROMPT_INFER, streams TOKEN_GENERATED messages
 * and prints decoded text as it arrives. Prints final perf stats.
 *
 * Use this to:
 *   - Validate a live coordinator + node-pool end-to-end from the command line
 *   - Measure tok/sec on a specific topology without the browser UI
 *   - Script demos / CI smoke tests
 *
 * Usage:
 *   node deploy/cli-infer.mjs                            # default prompt, 20 tokens
 *   node deploy/cli-infer.mjs "The universe is" 30
 *   COORD=http://<ip>:8080 node deploy/cli-infer.mjs "..."
 *
 * Requires: a running coordinator at $COORD (default http://localhost:8080)
 * with pipeline ready (≥2 nodes covering all shards). The `ws` package is
 * already a dependency of the coordinator, so no new deps.
 */

import { createRequire } from "node:module";

// Resolve ws from repo-local node_modules so this script runs from any cwd.
const require = createRequire(import.meta.url);
let WebSocket;
try {
  WebSocket = require("ws");
} catch {
  // Fall back to trying the synapse-src path
  const fallback = createRequire(
    new URL("../node_modules/ws/package.json", import.meta.url),
  );
  WebSocket = fallback("ws");
}

const COORD = process.env.COORD || "http://localhost:8080";
const WS_URL = COORD.replace(/^http/, "ws") + "/?type=prompt";
const TIMEOUT_MS = parseInt(process.env.TIMEOUT_MS || "180000", 10);

// Parse positional prompt + max_tokens + optional --temperature flag
const args = process.argv.slice(2);
let TEMPERATURE;
const positional = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--temperature" || args[i] === "-t") {
    TEMPERATURE = parseFloat(args[++i]);
  } else if (args[i].startsWith("--temperature=")) {
    TEMPERATURE = parseFloat(args[i].split("=")[1]);
  } else {
    positional.push(args[i]);
  }
}
const PROMPT = positional[0] || "The universe is";
const MAX_TOKENS = parseInt(positional[1] || "20", 10);

async function http(path, body) {
  const r = await fetch(`${COORD}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

const t_start = Date.now();

const tok = await http("/api/tokenize", { text: PROMPT });
console.log(`PROMPT: "${PROMPT}"  →  ${tok.tokenIds.length} input tokens`);

const ws = new WebSocket(WS_URL);
let fired = false;
let t_infer_start = null;
let t_first_token = null;

ws.on("open", () => {
  console.log(`CONNECTED: ${WS_URL}`);
});

ws.on("message", async (data) => {
  const msg = JSON.parse(data.toString());

  if (msg.type === "TOPOLOGY_UPDATE") {
    const nodes = msg.nodes || [];
    const ready = nodes.filter((n) => n.status === "ready");
    const pipelineSlots = msg.pipeline?.length ?? 0;
    console.log(
      `TOPOLOGY: ${nodes.length} nodes, ${ready.length} ready, ${pipelineSlots} pipeline slots`,
    );

    // Fire when pipeline is ready (also handles the "already-ready on connect" case
    // where PIPELINE_READY event is not re-emitted)
    if (!fired && pipelineSlots >= 2) {
      fired = true;
      const summary = ready
        .map((n) => `${n.nodeId.slice(5, 9)}:shard${n.shardId}:${n.capabilities?.gpuVendor || "?"}`)
        .join(", ");
      console.log(`NODES: ${summary}`);
      console.log(`\nOUTPUT:`);
      t_infer_start = Date.now();
      const request = {
        type: "PROMPT_INFER",
        tokenIds: tok.tokenIds,
        maxTokens: MAX_TOKENS,
      };
      if (TEMPERATURE !== undefined && !Number.isNaN(TEMPERATURE)) {
        request.temperature = TEMPERATURE;
      }
      ws.send(JSON.stringify(request));
    }
    return;
  }

  if (msg.type === "INFER_STARTED") {
    return;
  }

  if (msg.type === "TOKEN_GENERATED") {
    if (!t_first_token) t_first_token = Date.now();
    const detok = await http("/api/detokenize", { tokenIds: [msg.token] }).catch(() => null);
    process.stdout.write(detok?.text ?? `[${msg.token}]`);
    return;
  }

  if (msg.type === "GENERATION_DONE") {
    const total = Date.now() - t_infer_start;
    const ttft = t_first_token ? t_first_token - t_infer_start : null;
    const decodeMs = t_first_token ? Date.now() - t_first_token : 0;
    const tps = msg.totalTokens > 1 ? ((msg.totalTokens - 1) * 1000) / decodeMs : 0;

    console.log(`\n\nSTATS:`);
    console.log(`  generated:       ${msg.totalTokens} tokens`);
    console.log(`  time to first:   ${ttft ?? "?"} ms`);
    console.log(`  total:           ${total} ms`);
    console.log(`  decode tok/sec:  ${tps.toFixed(2)}`);
    console.log(`  reported:        ${(msg.tokensPerSecond ?? 0).toFixed?.(2) ?? msg.tokensPerSecond}`);

    const full = await http("/api/detokenize", { tokenIds: msg.tokens }).catch(() => null);
    console.log(`\nFULL TEXT: "${PROMPT}${full?.text ?? ""}"`);

    ws.close();
    process.exit(0);
  }

  if (msg.type === "INFER_ERROR") {
    console.error(`\nERROR: ${msg.error}`);
    ws.close();
    process.exit(1);
  }
});

ws.on("error", (e) => {
  console.error(`WS ERROR: ${e.message}`);
  process.exit(1);
});

setTimeout(() => {
  console.error(`\nTIMEOUT after ${TIMEOUT_MS}ms`);
  ws.close();
  process.exit(2);
}, TIMEOUT_MS);
