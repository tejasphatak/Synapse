/**
 * Coordinator HTTP API Integration Tests
 *
 * Spawns the coordinator on a random port and tests all HTTP endpoints.
 * No refactoring needed — tests the actual server as-is.
 */

import { describe, it, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const COORDINATOR = join(__dirname, "..", "coordinator", "index.js");

// Pick a random high port to avoid conflicts
const PORT = 19000 + Math.floor(Math.random() * 1000);
const BASE = `http://127.0.0.1:${PORT}`;

let proc;

async function waitForServer(url, timeoutMs = 10000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      // not ready yet
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`Server did not start within ${timeoutMs}ms`);
}

describe("Coordinator HTTP API", () => {
  before(async () => {
    proc = spawn("node", [COORDINATOR], {
      env: { ...process.env, PORT: String(PORT) },
      stdio: ["ignore", "pipe", "pipe"],
    });

    // Capture stderr for debugging if something goes wrong
    let stderr = "";
    proc.stderr.on("data", (d) => { stderr += d.toString(); });
    proc.on("error", (err) => {
      console.error("Coordinator process error:", err);
    });

    try {
      await waitForServer(`${BASE}/api/health`, 10000);
    } catch (e) {
      console.error("Coordinator stderr:", stderr);
      throw e;
    }
  });

  after(() => {
    if (proc) {
      proc.kill("SIGTERM");
      proc = null;
    }
  });

  // ─── /api/health ──────────────────────────────────────────────

  describe("GET /api/health", () => {
    it("returns health status", async () => {
      const res = await fetch(`${BASE}/api/health`);
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.equal(body.status, "waiting_for_nodes");
      assert.equal(body.version, "0.1.0");
      assert.equal(typeof body.uptime_seconds, "number");
      assert.equal(body.nodes.connected, 0);
      assert.equal(body.nodes.ready, 0);
      assert.equal(body.pipeline_ready, false);
      assert.equal(body.active_generations, 0);
      assert.equal(typeof body.timestamp, "number");
    });

    it("reports correct model info", async () => {
      const res = await fetch(`${BASE}/api/health`);
      const body = await res.json();
      assert.equal(body.model.name, "GPT-2 117M");
      assert.equal(body.model.dtype, "float16");
      assert.ok(body.model.shards >= 2);
    });
  });

  // ─── /api/topology ────────────────────────────────────────────

  describe("GET /api/topology", () => {
    it("returns empty topology", async () => {
      const res = await fetch(`${BASE}/api/topology`);
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.ok(Array.isArray(body.nodes));
      assert.ok(Array.isArray(body.pipeline));
      assert.equal(body.nodes.length, 0);
      assert.equal(body.pipeline.length, 0);
    });
  });

  // ─── /api/tokenize ────────────────────────────────────────────

  describe("POST /api/tokenize", () => {
    it("tokenizes text with GPT-2 BPE", async () => {
      const res = await fetch(`${BASE}/api/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "Hello world" }),
      });
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.ok(Array.isArray(body.tokenIds));
      assert.ok(body.tokenIds.length > 0);
      // GPT-2 BPE: "Hello" = [15496], " world" = [995]
      assert.deepEqual(body.tokenIds, [15496, 995]);
    });

    it("tokenizes empty string", async () => {
      const res = await fetch(`${BASE}/api/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: "" }),
      });
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.ok(Array.isArray(body.tokenIds));
      assert.equal(body.tokenIds.length, 0);
    });

    it("returns 400 on invalid JSON", async () => {
      const res = await fetch(`${BASE}/api/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "not json",
      });
      assert.equal(res.status, 400);
      const body = await res.json();
      assert.ok(body.error);
    });
  });

  // ─── /api/detokenize ──────────────────────────────────────────

  describe("POST /api/detokenize", () => {
    it("detokenizes token IDs to text", async () => {
      const res = await fetch(`${BASE}/api/detokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tokenIds: [15496, 995] }),
      });
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.equal(body.text, "Hello world");
    });

    it("round-trips tokenize → detokenize", async () => {
      const text = "The quick brown fox jumps over the lazy dog";
      const tokRes = await fetch(`${BASE}/api/tokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const { tokenIds } = await tokRes.json();

      const detokRes = await fetch(`${BASE}/api/detokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tokenIds }),
      });
      const result = await detokRes.json();
      assert.equal(result.text, text);
    });

    it("returns 400 on invalid JSON", async () => {
      const res = await fetch(`${BASE}/api/detokenize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "broken",
      });
      assert.equal(res.status, 400);
    });
  });

  // ─── /api/logs ────────────────────────────────────────────────

  describe("GET /api/logs", () => {
    it("returns log query result", async () => {
      const res = await fetch(`${BASE}/api/logs`);
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.equal(typeof body.count, "number");
      assert.equal(typeof body.total, "number");
      assert.ok(Array.isArray(body.logs));
      assert.equal(body.count, 0);
    });

    it("accepts query parameters", async () => {
      const res = await fetch(`${BASE}/api/logs?n=10&level=perf&event=fwd`);
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.equal(typeof body.count, "number");
      assert.ok(Array.isArray(body.logs));
    });
  });

  // ─── /api/perf ────────────────────────────────────────────────

  describe("GET /api/perf", () => {
    it("returns performance summary", async () => {
      const res = await fetch(`${BASE}/api/perf`);
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.equal(typeof body, "object");
    });
  });

  // ─── /api/infer ───────────────────────────────────────────────

  describe("POST /api/infer", () => {
    it("returns 503 when pipeline not ready", async () => {
      const res = await fetch(`${BASE}/api/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tokenIds: [15496, 995] }),
      });
      assert.equal(res.status, 503);
      const body = await res.json();
      assert.equal(body.ok, false);
      assert.match(body.error, /not ready/i);
    });

    it("returns 400 on invalid JSON", async () => {
      const res = await fetch(`${BASE}/api/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{bad",
      });
      assert.equal(res.status, 400);
      const body = await res.json();
      assert.equal(body.ok, false);
    });
  });

  // ─── Static file serving ──────────────────────────────────────

  describe("Static files", () => {
    it("serves home page at /", async () => {
      const res = await fetch(`${BASE}/`);
      assert.equal(res.status, 200);
      assert.match(res.headers.get("content-type"), /text\/html/);
    });

    it("serves dashboard", async () => {
      const res = await fetch(`${BASE}/ui/dashboard.html`);
      assert.equal(res.status, 200);
      assert.match(res.headers.get("content-type"), /text\/html/);
    });

    it("serves node client HTML", async () => {
      const res = await fetch(`${BASE}/node/index.html`);
      assert.equal(res.status, 200);
      assert.match(res.headers.get("content-type"), /text\/html/);
    });

    it("serves manifest.json from shards", async () => {
      const res = await fetch(`${BASE}/shards/manifest.json`);
      assert.equal(res.status, 200);
      const body = await res.json();
      assert.ok(body.shard_layout || body.model_type);
    });

    it("returns 404 for nonexistent path", async () => {
      const res = await fetch(`${BASE}/nonexistent.html`);
      assert.equal(res.status, 404);
    });
  });

  // ─── Path traversal protection ────────────────────────────────

  describe("Path traversal protection", () => {
    it("blocks shard path traversal", async () => {
      const res = await fetch(`${BASE}/shards/../../../etc/passwd`);
      assert.equal(res.status, 404);
    });

    it("blocks encoded shard path traversal", async () => {
      const res = await fetch(`${BASE}/shards/..%2F..%2F..%2Fetc%2Fpasswd`);
      assert.equal(res.status, 404);
    });

    it("blocks static file path traversal", async () => {
      const res = await fetch(`${BASE}/../../../etc/passwd`);
      assert.equal(res.status, 404);
    });
  });

  // ─── CORS ─────────────────────────────────────────────────────

  describe("CORS headers", () => {
    it("includes CORS headers on API responses", async () => {
      const res = await fetch(`${BASE}/api/health`);
      assert.equal(res.headers.get("access-control-allow-origin"), "*");
    });

    it("handles OPTIONS preflight", async () => {
      const res = await fetch(`${BASE}/api/tokenize`, {
        method: "OPTIONS",
      });
      assert.equal(res.status, 204);
      assert.equal(res.headers.get("access-control-allow-origin"), "*");
      assert.match(res.headers.get("access-control-allow-methods"), /POST/);
    });
  });

  // ─── Cache headers ────────────────────────────────────────────

  describe("Cache headers", () => {
    it("sets no-cache on JS files", async () => {
      const res = await fetch(`${BASE}/node/node.js`);
      assert.equal(res.status, 200);
      assert.match(res.headers.get("cache-control"), /no-cache/);
    });

    it("sets no-cache on HTML files", async () => {
      const res = await fetch(`${BASE}/node/index.html`);
      assert.equal(res.status, 200);
      assert.match(res.headers.get("cache-control"), /no-cache/);
    });
  });
});
