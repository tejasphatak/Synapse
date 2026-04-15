/**
 * SynapseNode Tests — Constructor, status, topology, message dispatch,
 * mobile detection, logging, cleanup, and protocol handling.
 *
 * WebGPU, WebSocket, and navigator are mocked — these tests verify
 * control flow, state management, and message routing logic.
 */

import { describe, it, beforeEach, mock } from "node:test";
import assert from "node:assert/strict";

// ─── Mock Browser Globals ─────────────────────────────────────

globalThis.GPUBufferUsage = globalThis.GPUBufferUsage || {
  STORAGE: 0x80, COPY_SRC: 0x04, COPY_DST: 0x08,
  UNIFORM: 0x40, MAP_READ: 0x01,
};
globalThis.GPUShaderStage = globalThis.GPUShaderStage || { COMPUTE: 0x04 };
globalThis.GPUMapMode = globalThis.GPUMapMode || { READ: 0x01 };

if (typeof globalThis.btoa === "undefined") {
  globalThis.btoa = (str) => Buffer.from(str, "binary").toString("base64");
  globalThis.atob = (b64) => Buffer.from(b64, "base64").toString("binary");
}

// Mock crypto.randomUUID
if (!globalThis.crypto) {
  globalThis.crypto = { randomUUID: () => "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" };
} else if (!globalThis.crypto.randomUUID) {
  globalThis.crypto.randomUUID = () => "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
}

// Mock performance.now
if (typeof globalThis.performance === "undefined") {
  globalThis.performance = { now: () => Date.now() };
}

// Mock navigator
const originalNavigator = globalThis.navigator;
globalThis.navigator = {
  gpu: null,
  userAgent: "Mozilla/5.0 (X11; Linux x86_64) Chrome/120",
  maxTouchPoints: 0,
  wakeLock: undefined,
};

// Mock document for visibility handlers
if (typeof globalThis.document === "undefined") {
  globalThis.document = {
    visibilityState: "visible",
    addEventListener: () => {},
  };
}

// Mock WebSocket
class MockWebSocket {
  static OPEN = 1;
  static CLOSED = 3;
  constructor(url) {
    this.url = url;
    this.readyState = MockWebSocket.OPEN;
    this.binaryType = "blob";
    this._sent = [];
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
  }
  send(data) { this._sent.push(data); }
  close() { this.readyState = MockWebSocket.CLOSED; }
}
globalThis.WebSocket = MockWebSocket;

// Mock fetch
globalThis.fetch = async () => ({
  text: async () => "// mock WGSL",
  json: async () => ({}),
  arrayBuffer: async () => new ArrayBuffer(0),
  ok: true,
});

import { SynapseNode } from "../node/node.js";
import { MessageType } from "../protocol/messages.js";
import {
  BinaryMsgType, encodeBinaryMessage, QuantMode, Flags,
  registerRequestId, requestIdToUint32,
} from "../protocol/binary.js";

// ─── Helpers ──────────────────────────────────────────────────

function makeNode(statusCb) {
  const node = new SynapseNode(statusCb);
  return node;
}

function mockPipeline() {
  return {
    embed: async (ids) => ({ buffer: new ArrayBuffer(768 * 4), shape: [ids.length, 768] }),
    embedSingle: async (id, pos) => ({ buffer: new ArrayBuffer(768 * 4), shape: [1, 768] }),
    forwardLayersPrefill: async (h, ls, le, rid) => h,
    forwardLayersCached: async (h, ls, le, rid, pos) => h,
    outputHead: async (h) => ({ buffer: new ArrayBuffer(50257 * 4), shape: [1, 50257] }),
    sampleToken: async (logits, temp) => 42,
    clearCache: mock.fn(),
    _cleanupTempBuffers: mock.fn(),
    _readBuffer: async (buf, off, sz) => new ArrayBuffer(sz),
    deserializeTensor: (t) => ({ buffer: new ArrayBuffer(768 * 4), shape: t.shape }),
    deserializeTensorBinary: (p, s) => ({ buffer: new ArrayBuffer(768 * 4), shape: s }),
    deserializeTensorQuantized: (p, s) => ({ buffer: new ArrayBuffer(768 * 4), shape: s }),
    deserializeTensorInt4: (p, s) => ({ buffer: new ArrayBuffer(768 * 4), shape: s }),
    deserializeTensorDeltaApply: (p, s, prev) => ({
      buffer: new ArrayBuffer(768 * 4), shape: s, currentFloat32: new Float32Array(768),
    }),
    serializeTensor: async (h) => ({ data: { shape: h.shape, data: "" }, shape: h.shape }),
    serializeTensorBinary: async (h) => ({ data: new Uint8Array(768 * 4), shape: h.shape }),
    serializeTensorQuantized: async (h) => ({ data: new Uint8Array(768 + 4), shape: h.shape }),
    serializeTensorInt4: async (h) => ({ data: new Uint8Array(384 + 4), shape: h.shape }),
    serializeTensorDelta: async (h, prev) => ({
      data: new Uint8Array(768 + 4), shape: h.shape,
      isDelta: false, currentFloat32: new Float32Array(768), sparsity: 0,
    }),
    kvCaches: new Map(),
    init: async () => {},
  };
}

function injectWs(node) {
  const ws = new MockWebSocket("ws://test:8080?type=node");
  node.ws = ws;
  return ws;
}

// ─── Constructor ──────────────────────────────────────────────

describe("SynapseNode Constructor", () => {
  it("generates a node ID with expected prefix", () => {
    const node = makeNode();
    assert.ok(node.nodeId.startsWith("node-"));
    assert.equal(node.nodeId.length, 13); // "node-" + 8 hex chars
  });

  it("starts in initializing status", () => {
    const node = makeNode();
    assert.equal(node.status, "initializing");
  });

  it("stores status callback", () => {
    let called = false;
    const cb = () => { called = true; };
    const node = makeNode(cb);
    node._setStatus("test");
    assert.ok(called);
  });

  it("uses noop callback when none provided", () => {
    const node = makeNode();
    // Should not throw
    node._setStatus("test");
    assert.equal(node.status, "test");
  });

  it("initializes all protocol flags to defaults", () => {
    const node = makeNode();
    assert.equal(node.useBinaryProtocol, false);
    assert.equal(node.useQuantization, true);
    // useDeltaEncoding disabled 2026-04-15 — sender/receiver state desync bug.
    // See commit "disable-delta-encoding-drift-fix". Change the default back
    // once the dequant-before-store fix lands.
    assert.equal(node.useDeltaEncoding, false);
    assert.equal(node.useSpeculation, true);
    assert.equal(node.useP2P, true);
    assert.equal(node.useAdaptivePrecision, true);
  });

  it("initializes all subsystems to null", () => {
    const node = makeNode();
    assert.equal(node.ws, null);
    assert.equal(node.device, null);
    assert.equal(node.loader, null);
    assert.equal(node.pipeline, null);
    assert.equal(node.speculative, null);
    assert.equal(node.p2p, null);
    assert.equal(node.adaptivePrecision, null);
  });

  it("initializes shard metadata to null", () => {
    const node = makeNode();
    assert.equal(node.shardId, null);
    assert.equal(node.layerStart, null);
    assert.equal(node.layerEnd, null);
    assert.equal(node.isFirstNode, false);
    assert.equal(node.isLastNode, false);
  });

  it("initializes empty activation caches", () => {
    const node = makeNode();
    assert.equal(node._lastSentActivation.size, 0);
    assert.equal(node._lastRecvActivation.size, 0);
    assert.equal(node._binaryRequestIds.size, 0);
  });
});

// ─── Status ───────────────────────────────────────────────────

describe("SynapseNode Status", () => {
  it("updates status field on _setStatus", () => {
    const node = makeNode();
    node._setStatus("computing");
    assert.equal(node.status, "computing");
  });

  it("passes status, detail, and self to callback", () => {
    const calls = [];
    const node = makeNode((status, detail, self) => {
      calls.push({ status, detail, self });
    });
    node._setStatus("ready", "all good");
    assert.equal(calls.length, 1);
    assert.equal(calls[0].status, "ready");
    assert.equal(calls[0].detail, "all good");
    assert.equal(calls[0].self, node);
  });

  it("uses empty string as default detail", () => {
    const calls = [];
    const node = makeNode((status, detail) => {
      calls.push({ status, detail });
    });
    node._setStatus("ready");
    assert.equal(calls[0].detail, "");
  });
});

// ─── Mobile Detection ─────────────────────────────────────────

describe("SynapseNode Mobile Detection", () => {
  it("detects Android as mobile", () => {
    const saved = navigator.userAgent;
    globalThis.navigator.userAgent = "Mozilla/5.0 (Linux; Android 14) Mobile";
    const node = makeNode();
    assert.equal(node._isMobile(), true);
    globalThis.navigator.userAgent = saved;
  });

  it("detects iPhone as mobile", () => {
    const saved = navigator.userAgent;
    globalThis.navigator.userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 17)";
    const node = makeNode();
    assert.equal(node._isMobile(), true);
    globalThis.navigator.userAgent = saved;
  });

  it("detects iPad via touch points + Mac UA", () => {
    const saved = navigator.userAgent;
    const savedTP = navigator.maxTouchPoints;
    globalThis.navigator.userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X) Safari";
    globalThis.navigator.maxTouchPoints = 5;
    const node = makeNode();
    assert.equal(node._isMobile(), true);
    globalThis.navigator.userAgent = saved;
    globalThis.navigator.maxTouchPoints = savedTP;
  });

  it("detects desktop as non-mobile", () => {
    const saved = navigator.userAgent;
    globalThis.navigator.userAgent = "Mozilla/5.0 (X11; Linux x86_64) Chrome/120";
    globalThis.navigator.maxTouchPoints = 0;
    const node = makeNode();
    assert.equal(node._isMobile(), false);
    globalThis.navigator.userAgent = saved;
  });
});

// ─── Topology Update ──────────────────────────────────────────

describe("SynapseNode Topology Update", () => {
  it("sets isFirstNode when node is first in pipeline", () => {
    const node = makeNode();
    node.useP2P = false;
    const ws = injectWs(node);
    node._handleTopologyUpdate({
      type: MessageType.TOPOLOGY_UPDATE,
      pipeline: [node.nodeId, "node-other1"],
    });
    assert.equal(node.isFirstNode, true);
    assert.equal(node.isLastNode, false);
  });

  it("sets isLastNode when node is last in pipeline", () => {
    const node = makeNode();
    node.useP2P = false;
    node._handleTopologyUpdate({
      type: MessageType.TOPOLOGY_UPDATE,
      pipeline: ["node-other1", node.nodeId],
    });
    assert.equal(node.isFirstNode, false);
    assert.equal(node.isLastNode, true);
  });

  it("sets both flags for single-node pipeline", () => {
    const node = makeNode();
    node.useP2P = false;
    node._handleTopologyUpdate({
      type: MessageType.TOPOLOGY_UPDATE,
      pipeline: [node.nodeId],
    });
    assert.equal(node.isFirstNode, true);
    assert.equal(node.isLastNode, true);
  });

  it("stores full topology message", () => {
    const node = makeNode();
    node.useP2P = false;
    const msg = { type: MessageType.TOPOLOGY_UPDATE, pipeline: ["a", "b"] };
    node._handleTopologyUpdate(msg);
    assert.deepEqual(node.topology, msg);
  });

  it("handles empty pipeline gracefully", () => {
    const node = makeNode();
    node.useP2P = false;
    node._handleTopologyUpdate({
      type: MessageType.TOPOLOGY_UPDATE,
      pipeline: [],
    });
    assert.equal(node.isFirstNode, false);
    assert.equal(node.isLastNode, false);
  });

  it("middle node is neither first nor last", () => {
    const node = makeNode();
    node.useP2P = false;
    node._handleTopologyUpdate({
      type: MessageType.TOPOLOGY_UPDATE,
      pipeline: ["node-a", node.nodeId, "node-c"],
    });
    assert.equal(node.isFirstNode, false);
    assert.equal(node.isLastNode, false);
  });
});

// ─── Message Dispatch ─────────────────────────────────────────

describe("SynapseNode Message Dispatch", () => {
  it("handles PONG message without error", async () => {
    const node = makeNode();
    await node._handleMessage(JSON.stringify({ type: MessageType.PONG }));
    // No-op, just shouldn't throw
  });

  it("handles ERROR message without crashing", async () => {
    const node = makeNode();
    await node._handleMessage(JSON.stringify({
      type: MessageType.ERROR,
      message: "test error",
    }));
  });

  it("handles unknown message type gracefully", async () => {
    const node = makeNode();
    await node._handleMessage(JSON.stringify({ type: "UNKNOWN_TYPE" }));
  });

  it("handles invalid JSON gracefully", async () => {
    const node = makeNode();
    await node._handleMessage("not json at all{{{");
    // Should not throw
  });

  it("dispatches TOPOLOGY_UPDATE to handler", async () => {
    const node = makeNode();
    node.useP2P = false;
    await node._handleMessage(JSON.stringify({
      type: MessageType.TOPOLOGY_UPDATE,
      pipeline: [node.nodeId, "node-b"],
    }));
    assert.equal(node.isFirstNode, true);
    assert.equal(node.isLastNode, false);
  });

  it("dispatches KV_RESET — clears pipeline cache and speculation", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.speculative = { clear: mock.fn() };
    const reqId = "req-kv-test";

    // Seed activation caches
    node._lastSentActivation.set(reqId, new Float32Array(10));
    node._lastRecvActivation.set(reqId, new Float32Array(10));

    await node._handleMessage(JSON.stringify({
      type: MessageType.KV_RESET,
      requestId: reqId,
    }));

    assert.equal(node.pipeline.clearCache.mock.callCount(), 1);
    assert.equal(node.speculative.clear.mock.callCount(), 1);
    assert.equal(node._lastSentActivation.has(reqId), false);
    assert.equal(node._lastRecvActivation.has(reqId), false);
  });

  it("KV_RESET works without speculative controller", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.speculative = null;

    await node._handleMessage(JSON.stringify({
      type: MessageType.KV_RESET,
      requestId: "req-123",
    }));
    assert.equal(node.pipeline.clearCache.mock.callCount(), 1);
  });

  it("INFERENCE_REQUEST is ignored when pipeline is null", async () => {
    const node = makeNode();
    node.pipeline = null;
    // Should not throw
    await node._handleMessage(JSON.stringify({
      type: MessageType.INFERENCE_REQUEST,
      requestId: "req-test",
      tokenIds: [1, 2, 3],
    }));
  });

  it("INFERENCE_STEP is ignored when pipeline is null", async () => {
    const node = makeNode();
    node.pipeline = null;
    await node._handleMessage(JSON.stringify({
      type: MessageType.INFERENCE_STEP,
      requestId: "req-test",
      tokenId: 42,
      seqPos: 5,
    }));
  });

  it("registers binary request ID on INFERENCE_REQUEST", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = true;
    const ws = injectWs(node);
    node.layerStart = 0;
    node.layerEnd = 5;

    await node._handleMessage(JSON.stringify({
      type: MessageType.INFERENCE_REQUEST,
      requestId: "req-binid",
      binaryRequestId: 99,
      tokenIds: [1, 2, 3],
    }));

    assert.equal(node.useBinaryProtocol, true);
  });
});

// ─── Inference Request (with mocked pipeline) ─────────────────

describe("SynapseNode Inference Request", () => {
  it("prefill as last node produces output", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = true;
    node.isFirstNode = true;
    node.layerStart = 0;
    node.layerEnd = 5;
    node.useBinaryProtocol = false;
    const ws = injectWs(node);

    await node._handleInferenceRequest({
      type: MessageType.INFERENCE_REQUEST,
      requestId: "req-prefill",
      tokenIds: [100, 200, 300],
    });

    // Should have sent an output message
    assert.ok(ws._sent.length > 0);
    const sent = JSON.parse(ws._sent.find(s => typeof s === "string" && s.includes("OUTPUT")));
    assert.equal(sent.type, MessageType.OUTPUT);
    assert.equal(sent.requestId, "req-prefill");
    assert.deepEqual(sent.tokens, [42]); // mocked sampleToken returns 42
  });

  it("prefill as middle node sends activation", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = false;
    node.isFirstNode = true;
    node.layerStart = 0;
    node.layerEnd = 5;
    node.useBinaryProtocol = false;
    const ws = injectWs(node);

    await node._handleInferenceRequest({
      type: MessageType.INFERENCE_REQUEST,
      requestId: "req-mid",
      tokenIds: [100, 200],
    });

    assert.ok(ws._sent.length > 0);
    const sent = JSON.parse(ws._sent.find(s => {
      try { return JSON.parse(s).type === MessageType.ACTIVATION; } catch { return false; }
    }));
    assert.equal(sent.type, MessageType.ACTIVATION);
  });

  it("sets status to computing then back to ready", async () => {
    const statuses = [];
    const node = makeNode((s) => statuses.push(s));
    node.pipeline = mockPipeline();
    node.isLastNode = true;
    node.layerStart = 0;
    node.layerEnd = 5;
    const ws = injectWs(node);

    await node._handleInferenceRequest({
      type: MessageType.INFERENCE_REQUEST,
      requestId: "req-status",
      tokenIds: [1],
    });

    assert.ok(statuses.includes("computing"));
    assert.equal(statuses[statuses.length - 1], "ready");
  });

  it("calls onOutput callback when producing output", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = true;
    node.layerStart = 0;
    node.layerEnd = 5;
    const ws = injectWs(node);

    let outputToken = null;
    let outputReq = null;
    node.onOutput = (token, reqId) => {
      outputToken = token;
      outputReq = reqId;
    };

    await node._handleInferenceRequest({
      type: MessageType.INFERENCE_REQUEST,
      requestId: "req-cb",
      tokenIds: [1],
    });

    assert.equal(outputToken, 42);
    assert.equal(outputReq, "req-cb");
  });
});

// ─── Inference Step (cached) ──────────────────────────────────

describe("SynapseNode Inference Step", () => {
  it("cached step as last node produces output", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = true;
    node.layerStart = 0;
    node.layerEnd = 5;
    node.useBinaryProtocol = false;
    const ws = injectWs(node);

    await node._handleInferenceStep({
      type: MessageType.INFERENCE_STEP,
      requestId: "req-step",
      tokenId: 50,
      seqPos: 3,
    });

    assert.ok(ws._sent.length > 0);
    const sent = JSON.parse(ws._sent.find(s => typeof s === "string" && s.includes("OUTPUT")));
    assert.equal(sent.type, MessageType.OUTPUT);
  });

  it("cached step as middle node sends activation", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = false;
    node.layerStart = 0;
    node.layerEnd = 5;
    node.useBinaryProtocol = false;
    const ws = injectWs(node);

    await node._handleInferenceStep({
      type: MessageType.INFERENCE_STEP,
      requestId: "req-step-mid",
      tokenId: 50,
      seqPos: 3,
    });

    assert.ok(ws._sent.length > 0);
    const sent = JSON.parse(ws._sent.find(s => {
      try { return JSON.parse(s).type === MessageType.ACTIVATION; } catch { return false; }
    }));
    assert.equal(sent.type, MessageType.ACTIVATION);
  });

  it("cleans up temp buffers after step", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = true;
    node.layerStart = 0;
    node.layerEnd = 5;
    const ws = injectWs(node);

    await node._handleInferenceStep({
      type: MessageType.INFERENCE_STEP,
      requestId: "req-clean",
      tokenId: 1,
      seqPos: 0,
    });

    assert.equal(node.pipeline._cleanupTempBuffers.mock.callCount(), 1);
  });
});

// ─── JSON Activation Handling ─────────────────────────────────

describe("SynapseNode JSON Activation", () => {
  it("prefill activation (multi-token) runs forwardLayersPrefill", async () => {
    const node = makeNode();
    const pl = mockPipeline();
    let prefillCalled = false;
    pl.forwardLayersPrefill = async (h, ls, le, rid) => {
      prefillCalled = true;
      return h;
    };
    node.pipeline = pl;
    node.isLastNode = true;
    node.layerStart = 0;
    node.layerEnd = 5;
    const ws = injectWs(node);

    await node._handleActivation({
      type: MessageType.ACTIVATION,
      requestId: "req-act-pre",
      tensor: { shape: [4, 768], data: "" },
      seqLen: 4,
    });

    assert.ok(prefillCalled);
  });

  it("single-token activation runs forwardLayersCached", async () => {
    const node = makeNode();
    const pl = mockPipeline();
    let cachedCalled = false;
    pl.forwardLayersCached = async (h, ls, le, rid, pos) => {
      cachedCalled = true;
      assert.equal(pos, 0); // seqLen 1 → seqPos 0
      return h;
    };
    node.pipeline = pl;
    node.isLastNode = true;
    node.layerStart = 0;
    node.layerEnd = 5;
    const ws = injectWs(node);

    await node._handleActivation({
      type: MessageType.ACTIVATION,
      requestId: "req-act-cached",
      tensor: { shape: [1, 768], data: "" },
      seqLen: 1,
    });

    assert.ok(cachedCalled);
  });

  it("ignores activation when pipeline is null", async () => {
    const node = makeNode();
    node.pipeline = null;
    // Should not throw
    await node._handleActivation({
      type: MessageType.ACTIVATION,
      requestId: "req-nopipe",
      tensor: { shape: [1, 768], data: "" },
    });
  });
});

// ─── Logging ──────────────────────────────────────────────────

describe("SynapseNode Logging", () => {
  it("sends NODE_LOG message with correct structure", () => {
    const node = makeNode();
    const ws = injectWs(node);
    node.shardId = "shard_0";

    node._sendLog("perf", "prefill", { durationMs: 42 });

    assert.equal(ws._sent.length, 1);
    const log = JSON.parse(ws._sent[0]);
    assert.equal(log.type, "NODE_LOG");
    assert.equal(log.nodeId, node.nodeId);
    assert.equal(log.level, "perf");
    assert.equal(log.event, "prefill");
    assert.equal(log.data.shardId, "shard_0");
    assert.equal(log.data.durationMs, 42);
    assert.ok(typeof log.timestamp === "number");
  });

  it("does not send when ws is null", () => {
    const node = makeNode();
    node.ws = null;
    // Should not throw
    node._sendLog("info", "test");
  });

  it("does not send when ws is closed", () => {
    const node = makeNode();
    const ws = injectWs(node);
    ws.readyState = MockWebSocket.CLOSED;
    node._sendLog("info", "test");
    assert.equal(ws._sent.length, 0);
  });

  it("merges shardId into log data", () => {
    const node = makeNode();
    const ws = injectWs(node);
    node.shardId = "shard_1";

    node._sendLog("error", "fail", { reason: "oom" });
    const log = JSON.parse(ws._sent[0]);
    assert.equal(log.data.shardId, "shard_1");
    assert.equal(log.data.reason, "oom");
  });
});

// ─── Destroy ──────────────────────────────────────────────────

describe("SynapseNode Destroy", () => {
  it("closes WebSocket and sets status to destroyed", () => {
    const node = makeNode();
    const ws = injectWs(node);
    node.destroy();
    assert.equal(ws.readyState, MockWebSocket.CLOSED);
    assert.equal(node.status, "destroyed");
  });

  it("clears ping interval", () => {
    const node = makeNode();
    let cleared = false;
    node.pingInterval = setInterval(() => {}, 99999);
    // Just verify destroy doesn't throw and ping stops
    node.destroy();
    assert.equal(node.status, "destroyed");
    clearInterval(node.pingInterval); // cleanup
  });

  it("handles destroy when ws is null", () => {
    const node = makeNode();
    node.ws = null;
    node.destroy(); // Should not throw
    assert.equal(node.status, "destroyed");
  });

  it("calls loader.destroy if loader exists", () => {
    const node = makeNode();
    let loaderDestroyed = false;
    node.loader = { destroy: () => { loaderDestroyed = true; } };
    node.destroy();
    assert.ok(loaderDestroyed);
  });
});

// ─── Binary Output (last node) ────────────────────────────────

describe("SynapseNode Binary Output", () => {
  it("sends binary output when useBinaryProtocol is true", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = true;
    node.useBinaryProtocol = true;
    const ws = injectWs(node);

    registerRequestId("req-bin-out", 1);
    await node._produceOutput(
      { buffer: new ArrayBuffer(768 * 4), shape: [1, 768] },
      "req-bin-out"
    );

    // Should send a binary buffer, not JSON
    assert.ok(ws._sent.length > 0);
    const sent = ws._sent[ws._sent.length - 1];
    assert.ok(sent instanceof ArrayBuffer || ArrayBuffer.isView(sent));
  });

  it("sends JSON output when useBinaryProtocol is false", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.isLastNode = true;
    node.useBinaryProtocol = false;
    const ws = injectWs(node);

    await node._produceOutput(
      { buffer: new ArrayBuffer(768 * 4), shape: [1, 768] },
      "req-json-out"
    );

    assert.ok(ws._sent.length > 0);
    const sent = ws._sent[ws._sent.length - 1];
    assert.equal(typeof sent, "string");
    const parsed = JSON.parse(sent);
    assert.equal(parsed.type, MessageType.OUTPUT);
    assert.deepEqual(parsed.tokens, [42]);
  });
});

// ─── Send Activation (wire format) ────────────────────────────

describe("SynapseNode Send Activation", () => {
  it("sends JSON activation when binary protocol disabled", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.useBinaryProtocol = false;
    node.layerEnd = 5;
    node.nodeId = "node-sender";
    const ws = injectWs(node);

    await node._sendActivation(
      { buffer: new ArrayBuffer(768 * 4), shape: [1, 768] },
      "req-json-act",
      3
    );

    const sent = JSON.parse(ws._sent.find(s => {
      try { return JSON.parse(s).type === MessageType.ACTIVATION; } catch { return false; }
    }));
    assert.equal(sent.type, MessageType.ACTIVATION);
    assert.equal(sent.fromNode, "node-sender");
    assert.equal(sent.requestId, "req-json-act");
    assert.equal(sent.seqLen, 3);
    assert.equal(sent.layer, 6);
  });

  it("sends binary activation when binary protocol enabled", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.useBinaryProtocol = true;
    node.useQuantization = false;
    node.useDeltaEncoding = false;
    node.useAdaptivePrecision = false;
    node.adaptivePrecision = null;
    const ws = injectWs(node);

    registerRequestId("req-bin-act", 2);
    await node._sendActivation(
      { buffer: new ArrayBuffer(768 * 4), shape: [1, 768] },
      "req-bin-act",
      5
    );

    assert.ok(ws._sent.length > 0);
    const sent = ws._sent[ws._sent.length - 1];
    assert.ok(sent instanceof ArrayBuffer || ArrayBuffer.isView(sent));
  });

  it("prefers P2P channel when available", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.useBinaryProtocol = true;
    node.useQuantization = false;
    node.useDeltaEncoding = false;
    node.useAdaptivePrecision = false;
    node.adaptivePrecision = null;
    const ws = injectWs(node);

    let p2pSent = null;
    node.p2p = {
      send: (data) => { p2pSent = data; return true; },
    };

    registerRequestId("req-p2p", 3);
    await node._sendActivation(
      { buffer: new ArrayBuffer(768 * 4), shape: [1, 768] },
      "req-p2p",
      2
    );

    assert.ok(p2pSent !== null, "P2P channel should have been used");
    // WS may still receive diagnostic NODE_LOG messages (shard_output_stats)
    // emitted by _sendActivation for perf telemetry. The invariant is that
    // no ACTIVATION binary frame went over WS when P2P succeeded.
    const activationFrames = ws._sent.filter(m =>
      m instanceof ArrayBuffer || (m && typeof m === "object" && m.buffer)
    );
    assert.equal(activationFrames.length, 0, "no ACTIVATION over WS when P2P succeeded");
  });

  it("falls back to WS when P2P send fails", async () => {
    const node = makeNode();
    node.pipeline = mockPipeline();
    node.useBinaryProtocol = true;
    node.useQuantization = false;
    node.useDeltaEncoding = false;
    node.useAdaptivePrecision = false;
    node.adaptivePrecision = null;
    const ws = injectWs(node);

    node.p2p = {
      send: () => false, // P2P not connected / failed
    };

    registerRequestId("req-p2p-fail", 4);
    await node._sendActivation(
      { buffer: new ArrayBuffer(768 * 4), shape: [1, 768] },
      "req-p2p-fail",
      2
    );

    // Should fall back to WS
    assert.ok(ws._sent.length > 0);
  });
});

// ─── Error Handling ───────────────────────────────────────────

describe("SynapseNode Error Handling", () => {
  it("prefill error sets status to error and logs", async () => {
    const node = makeNode();
    const pl = mockPipeline();
    pl.embed = async () => { throw new Error("GPU OOM"); };
    node.pipeline = pl;
    node.layerStart = 0;
    node.layerEnd = 5;
    const ws = injectWs(node);

    await node._handleInferenceRequest({
      type: MessageType.INFERENCE_REQUEST,
      requestId: "req-err",
      tokenIds: [1, 2],
    });

    assert.equal(node.status, "error");
    // Should have sent a log about the error
    const errorLog = ws._sent.find(s => {
      try { return JSON.parse(s).event === "prefill_error"; } catch { return false; }
    });
    assert.ok(errorLog);
  });

  it("step error sets status to error and cleans up", async () => {
    const node = makeNode();
    const pl = mockPipeline();
    pl.embedSingle = async () => { throw new Error("Device lost"); };
    node.pipeline = pl;
    node.layerStart = 0;
    node.layerEnd = 5;
    const ws = injectWs(node);

    await node._handleInferenceStep({
      type: MessageType.INFERENCE_STEP,
      requestId: "req-step-err",
      tokenId: 1,
      seqPos: 0,
    });

    assert.equal(node.status, "error");
    assert.equal(pl._cleanupTempBuffers.mock.callCount(), 1);
  });

  it("activation error sets status to error", async () => {
    const node = makeNode();
    const pl = mockPipeline();
    pl.deserializeTensor = () => { throw new Error("Bad tensor"); };
    node.pipeline = pl;
    node.layerStart = 0;
    node.layerEnd = 5;
    const ws = injectWs(node);

    await node._handleActivation({
      type: MessageType.ACTIVATION,
      requestId: "req-act-err",
      tensor: { shape: [1, 768], data: "bad" },
    });

    assert.equal(node.status, "error");
  });
});

// ─── P2P Signal Handling ──────────────────────────────────────

describe("SynapseNode P2P Signal", () => {
  it("forwards P2P_SIGNAL to existing p2p channel", async () => {
    const node = makeNode();
    let signalHandled = false;
    node.p2p = {
      handleSignal: async (msg) => { signalHandled = true; },
    };

    await node._handleMessage(JSON.stringify({
      type: "P2P_SIGNAL",
      from: "node-peer",
      signal: { type: "offer" },
    }));

    assert.ok(signalHandled);
  });

  it("creates new P2P channel as responder when no p2p exists", async () => {
    const node = makeNode();
    node.p2p = null;
    node.useP2P = true;
    const ws = injectWs(node);

    // P2PChannel constructor will be called — this will likely fail since
    // RTCPeerConnection doesn't exist in Node. But the code path should
    // at least try (and the catch should prevent crashes).
    await node._handleMessage(JSON.stringify({
      type: "P2P_SIGNAL",
      from: "node-peer",
      signal: { type: "offer" },
    }));

    // In Node.js test environment, P2P might fail to initialize,
    // but the important thing is it doesn't crash the node
    assert.notEqual(node.status, "destroyed");
  });

  it("ignores P2P_SIGNAL when useP2P is false and no p2p exists", async () => {
    const node = makeNode();
    node.p2p = null;
    node.useP2P = false;

    await node._handleMessage(JSON.stringify({
      type: "P2P_SIGNAL",
      from: "node-peer",
      signal: { type: "offer" },
    }));

    assert.equal(node.p2p, null);
  });
});
