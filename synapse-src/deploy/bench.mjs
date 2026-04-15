#!/usr/bin/env node
/**
 * bench.mjs — Synapse throughput benchmark.
 *
 * Runs cli-infer against a live coordinator multiple times with varied
 * max_tokens, collects TTFT + decode-tok/sec + total latency, prints a
 * CSV summary. Designed to quickly baseline a topology.
 *
 * Usage:
 *   node deploy/bench.mjs                                  # default 3 runs x 3 lengths
 *   COORD=http://<ip>:8080 node deploy/bench.mjs
 *
 * Output: CSV rows to stdout + human summary at end.
 */

import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
let WebSocket;
try {
  WebSocket = require("ws");
} catch {
  const fb = createRequire(new URL("../node_modules/ws/package.json", import.meta.url));
  WebSocket = fb("ws");
}

const COORD = process.env.COORD || "http://34.82.32.123:8080";
const WS_URL = COORD.replace(/^http/, "ws") + "/?type=prompt";

const PROMPTS = ["Hello", "The universe is", "Once upon a time"];
const MAX_TOKENS_SET = [5, 15, 30];

async function http(path, body) {
  const r = await fetch(`${COORD}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

function runOne(prompt, maxTokens) {
  return new Promise(async (resolve) => {
    const tok = await http("/api/tokenize", { text: prompt });
    const ws = new WebSocket(WS_URL);
    let fired = false;
    let t_infer = null;
    let t_first = null;
    const genTokens = [];

    const timer = setTimeout(() => {
      ws.close();
      resolve({ prompt, maxTokens, ok: false, reason: "timeout" });
    }, 120000);

    ws.on("message", async (data) => {
      const msg = JSON.parse(data.toString());
      if (!fired && msg.type === "TOPOLOGY_UPDATE" && (msg.pipeline?.length ?? 0) >= 2) {
        fired = true;
        t_infer = Date.now();
        ws.send(JSON.stringify({ type: "PROMPT_INFER", tokenIds: tok.tokenIds, maxTokens }));
      }
      if (msg.type === "TOKEN_GENERATED") {
        if (!t_first) t_first = Date.now();
        genTokens.push(msg.token);
      }
      if (msg.type === "GENERATION_DONE") {
        clearTimeout(timer);
        const total = Date.now() - t_infer;
        const ttft = t_first - t_infer;
        const decodeMs = Date.now() - t_first;
        const decodeTps = genTokens.length > 1 ? ((genTokens.length - 1) * 1000) / decodeMs : 0;
        ws.close();
        resolve({
          prompt,
          inputTokens: tok.tokenIds.length,
          maxTokens,
          generated: genTokens.length,
          ttft_ms: ttft,
          total_ms: total,
          decode_tps: decodeTps,
          reported_tps: msg.tokensPerSecond ?? 0,
          ok: true,
        });
      }
      if (msg.type === "INFER_ERROR") {
        clearTimeout(timer);
        ws.close();
        resolve({ prompt, maxTokens, ok: false, reason: msg.error });
      }
    });
    ws.on("error", (e) => {
      clearTimeout(timer);
      resolve({ prompt, maxTokens, ok: false, reason: `ws: ${e.message}` });
    });
  });
}

console.log("prompt,input_tokens,max_tokens,generated,ttft_ms,total_ms,decode_tps,reported_tps,ok");
const results = [];
for (const p of PROMPTS) {
  for (const mt of MAX_TOKENS_SET) {
    process.stderr.write(`[bench] ${p.slice(0, 20)}  max=${mt}  `);
    const r = await runOne(p, mt);
    results.push(r);
    if (r.ok) {
      process.stderr.write(
        `ttft=${r.ttft_ms}ms total=${r.total_ms}ms decode=${r.decode_tps.toFixed(2)}tok/s\n`,
      );
      console.log(
        `"${p}",${r.inputTokens},${r.maxTokens},${r.generated},${r.ttft_ms},${r.total_ms},${r.decode_tps.toFixed(2)},${r.reported_tps.toFixed?.(2) ?? r.reported_tps},1`,
      );
    } else {
      process.stderr.write(`FAIL: ${r.reason}\n`);
      console.log(`"${p}",,${mt},,,,,,0`);
    }
  }
}

// Summary to stderr so CSV on stdout is clean
const ok = results.filter((r) => r.ok);
if (ok.length > 0) {
  const avg = (key) => ok.reduce((s, r) => s + r[key], 0) / ok.length;
  process.stderr.write(
    `\n[bench] summary: n=${ok.length}  avg_ttft=${avg("ttft_ms").toFixed(0)}ms  avg_decode=${avg("decode_tps").toFixed(2)}tok/s  avg_total=${avg("total_ms").toFixed(0)}ms\n`,
  );
}
process.exit(0);
