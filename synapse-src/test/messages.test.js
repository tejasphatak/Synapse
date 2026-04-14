import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  MessageType,
  PROTOCOL_V2,
  createJoinMessage,
  createAssignShardMessage,
  createActivationMessage,
  createOutputMessage,
  createTopologyUpdateMessage,
  createInferenceRequestMessage,
  createPingMessage,
  createPongMessage,
  createNodeReadyMessage,
  createInferenceStepMessage,
  createKVResetMessage,
  createNodeLogMessage,
  createErrorMessage,
  validateMessage,
  parseMessage,
} from "../protocol/messages.js";

// ─── MessageType enum ────────────────────────────────────────────

describe("MessageType", () => {
  it("has all expected types", () => {
    const expected = [
      "JOIN", "PING", "ASSIGN_SHARD", "PONG", "TOPOLOGY_UPDATE",
      "INFERENCE_REQUEST", "ACTIVATION", "NODE_READY", "OUTPUT",
      "INFERENCE_STEP", "KV_RESET", "NODE_LOG", "ERROR",
    ];
    for (const t of expected) {
      assert.equal(MessageType[t], t);
    }
  });

  it("PROTOCOL_V2 is proto_v2", () => {
    assert.equal(PROTOCOL_V2, "proto_v2");
  });
});

// ─── Message Constructors ────────────────────────────────────────

describe("createJoinMessage", () => {
  it("creates JOIN with defaults", () => {
    const msg = createJoinMessage("node-1");
    assert.equal(msg.type, "JOIN");
    assert.equal(msg.nodeId, "node-1");
    assert.equal(msg.capabilities.webgpu, false);
    assert.equal(msg.capabilities.maxLayers, 6);
    assert.equal(typeof msg.timestamp, "number");
  });

  it("merges custom capabilities", () => {
    const msg = createJoinMessage("node-2", { webgpu: true, maxLayers: 12, fp16: true });
    assert.equal(msg.capabilities.webgpu, true);
    assert.equal(msg.capabilities.maxLayers, 12);
    assert.equal(msg.capabilities.fp16, true);
  });
});

describe("createAssignShardMessage", () => {
  it("creates ASSIGN_SHARD with all fields", () => {
    const msg = createAssignShardMessage(0, 0, 5, "/shards/shard_0.bin", "/shards/shared.bin");
    assert.equal(msg.type, "ASSIGN_SHARD");
    assert.equal(msg.shardId, 0);
    assert.equal(msg.layerStart, 0);
    assert.equal(msg.layerEnd, 5);
    assert.equal(msg.shardUrl, "/shards/shard_0.bin");
    assert.equal(msg.sharedUrl, "/shards/shared.bin");
  });
});

describe("createActivationMessage", () => {
  it("creates ACTIVATION with tensor metadata", () => {
    const msg = createActivationMessage("node-0", "node-1", 5, 42, "AAAA", [1, 768]);
    assert.equal(msg.type, "ACTIVATION");
    assert.equal(msg.fromNode, "node-0");
    assert.equal(msg.toNode, "node-1");
    assert.equal(msg.layer, 5);
    assert.equal(msg.requestId, 42);
    assert.deepEqual(msg.tensor.shape, [1, 768]);
    assert.equal(msg.tensor.dtype, "float32");
    assert.equal(msg.tensor.data, "AAAA");
    assert.equal(typeof msg.timestamp, "number");
  });
});

describe("createOutputMessage", () => {
  it("creates OUTPUT with tokens and text", () => {
    const msg = createOutputMessage(99, [464, 1917], "The world");
    assert.equal(msg.type, "OUTPUT");
    assert.equal(msg.requestId, 99);
    assert.deepEqual(msg.tokens, [464, 1917]);
    assert.equal(msg.text, "The world");
  });
});

describe("createTopologyUpdateMessage", () => {
  it("creates TOPOLOGY_UPDATE", () => {
    const nodes = [{ nodeId: "n1", shardId: 0 }];
    const pipeline = ["n1"];
    const msg = createTopologyUpdateMessage(nodes, pipeline);
    assert.equal(msg.type, "TOPOLOGY_UPDATE");
    assert.deepEqual(msg.nodes, nodes);
    assert.deepEqual(msg.pipeline, pipeline);
  });
});

describe("createInferenceRequestMessage", () => {
  it("creates INFERENCE_REQUEST", () => {
    const msg = createInferenceRequestMessage(7, [1, 2, 3]);
    assert.equal(msg.type, "INFERENCE_REQUEST");
    assert.equal(msg.requestId, 7);
    assert.deepEqual(msg.tokenIds, [1, 2, 3]);
  });
});

describe("createPingMessage / createPongMessage", () => {
  it("creates PING", () => {
    const msg = createPingMessage();
    assert.equal(msg.type, "PING");
    assert.equal(typeof msg.timestamp, "number");
  });

  it("creates PONG", () => {
    const msg = createPongMessage();
    assert.equal(msg.type, "PONG");
    assert.equal(typeof msg.timestamp, "number");
  });
});

describe("createNodeReadyMessage", () => {
  it("creates NODE_READY", () => {
    const msg = createNodeReadyMessage("node-3", 1);
    assert.equal(msg.type, "NODE_READY");
    assert.equal(msg.nodeId, "node-3");
    assert.equal(msg.shardId, 1);
  });
});

describe("createInferenceStepMessage", () => {
  it("creates INFERENCE_STEP with seqPos", () => {
    const msg = createInferenceStepMessage(10, 464, 5);
    assert.equal(msg.type, "INFERENCE_STEP");
    assert.equal(msg.requestId, 10);
    assert.equal(msg.tokenId, 464);
    assert.equal(msg.seqPos, 5);
    assert.equal(msg.binaryRequestId, null);
  });

  it("includes binaryRequestId when provided", () => {
    const msg = createInferenceStepMessage(10, 464, 5, 0xFF01);
    assert.equal(msg.binaryRequestId, 0xFF01);
  });
});

describe("createKVResetMessage", () => {
  it("creates KV_RESET", () => {
    const msg = createKVResetMessage(42);
    assert.equal(msg.type, "KV_RESET");
    assert.equal(msg.requestId, 42);
  });
});

describe("createNodeLogMessage", () => {
  it("creates NODE_LOG with data", () => {
    const msg = createNodeLogMessage("node-1", "perf", "layer_forward", { latencyMs: 12.5 });
    assert.equal(msg.type, "NODE_LOG");
    assert.equal(msg.nodeId, "node-1");
    assert.equal(msg.level, "perf");
    assert.equal(msg.event, "layer_forward");
    assert.deepEqual(msg.data, { latencyMs: 12.5 });
  });

  it("defaults data to empty object", () => {
    const msg = createNodeLogMessage("node-1", "info", "shard_load");
    assert.deepEqual(msg.data, {});
  });
});

describe("createErrorMessage", () => {
  it("creates ERROR with code and message", () => {
    const msg = createErrorMessage("SHARD_NOT_FOUND", "Shard 3 does not exist");
    assert.equal(msg.type, "ERROR");
    assert.equal(msg.code, "SHARD_NOT_FOUND");
    assert.equal(msg.message, "Shard 3 does not exist");
    assert.equal(msg.details, null);
  });

  it("includes details when provided", () => {
    const msg = createErrorMessage("GPU_OOM", "Out of memory", { available: 512, required: 1024 });
    assert.deepEqual(msg.details, { available: 512, required: 1024 });
  });
});

// ─── validateMessage ─────────────────────────────────────────────

describe("validateMessage", () => {
  it("rejects null", () => {
    const r = validateMessage(null);
    assert.equal(r.valid, false);
    assert.match(r.error, /non-null object/);
  });

  it("rejects non-object", () => {
    assert.equal(validateMessage("hello").valid, false);
    assert.equal(validateMessage(42).valid, false);
  });

  it("rejects unknown type", () => {
    const r = validateMessage({ type: "EXPLODE" });
    assert.equal(r.valid, false);
    assert.match(r.error, /Unknown message type/);
  });

  it("rejects missing type", () => {
    const r = validateMessage({ nodeId: "n1" });
    assert.equal(r.valid, false);
  });

  it("validates a well-formed JOIN", () => {
    const msg = createJoinMessage("node-1", { webgpu: true });
    assert.deepEqual(validateMessage(msg), { valid: true, error: null });
  });

  it("rejects JOIN missing nodeId", () => {
    const r = validateMessage({ type: "JOIN", capabilities: {} });
    assert.equal(r.valid, false);
    assert.match(r.error, /nodeId/);
  });

  it("rejects JOIN missing capabilities", () => {
    const r = validateMessage({ type: "JOIN", nodeId: "n1" });
    assert.equal(r.valid, false);
    assert.match(r.error, /capabilities/);
  });

  it("validates PING with no required fields", () => {
    assert.deepEqual(validateMessage(createPingMessage()), { valid: true, error: null });
  });

  it("validates PONG with no required fields", () => {
    assert.deepEqual(validateMessage(createPongMessage()), { valid: true, error: null });
  });

  it("rejects ASSIGN_SHARD missing shardUrl", () => {
    const r = validateMessage({ type: "ASSIGN_SHARD", shardId: 0, layerStart: 0, layerEnd: 5, sharedUrl: "/s" });
    assert.equal(r.valid, false);
    assert.match(r.error, /shardUrl/);
  });

  it("rejects INFERENCE_STEP missing seqPos", () => {
    const r = validateMessage({ type: "INFERENCE_STEP", requestId: 1, tokenId: 42 });
    assert.equal(r.valid, false);
    assert.match(r.error, /seqPos/);
  });

  // Type-specific: ACTIVATION tensor validation
  it("rejects ACTIVATION with missing tensor shape", () => {
    const r = validateMessage({
      type: "ACTIVATION", fromNode: "a", layer: 0, requestId: 1,
      tensor: { data: "AAAA" },
    });
    assert.equal(r.valid, false);
    assert.match(r.error, /shape array/);
  });

  it("rejects ACTIVATION with non-array shape", () => {
    const r = validateMessage({
      type: "ACTIVATION", fromNode: "a", layer: 0, requestId: 1,
      tensor: { shape: "1x768", data: "AAAA" },
    });
    assert.equal(r.valid, false);
    assert.match(r.error, /shape array/);
  });

  it("rejects ACTIVATION with non-string tensor data", () => {
    const r = validateMessage({
      type: "ACTIVATION", fromNode: "a", layer: 0, requestId: 1,
      tensor: { shape: [1, 768], data: 12345 },
    });
    assert.equal(r.valid, false);
    assert.match(r.error, /base64 string/);
  });

  it("validates well-formed ACTIVATION", () => {
    const msg = createActivationMessage("n0", "n1", 5, 1, "AAAA", [1, 768]);
    assert.deepEqual(validateMessage(msg), { valid: true, error: null });
  });

  // Type-specific: OUTPUT tokens validation
  it("rejects OUTPUT with non-array tokens", () => {
    const r = validateMessage({ type: "OUTPUT", requestId: 1, tokens: "464,1917" });
    assert.equal(r.valid, false);
    assert.match(r.error, /tokens must be an array/);
  });

  it("validates well-formed OUTPUT", () => {
    const msg = createOutputMessage(1, [464], "The");
    assert.deepEqual(validateMessage(msg), { valid: true, error: null });
  });

  // All constructors produce valid messages
  it("all constructors produce valid messages", () => {
    const messages = [
      createJoinMessage("n1"),
      createAssignShardMessage(0, 0, 5, "/s0", "/shared"),
      createActivationMessage("n0", "n1", 5, 1, "AAAA", [1, 768]),
      createOutputMessage(1, [464], "The"),
      createTopologyUpdateMessage([{ nodeId: "n1" }], ["n1"]),
      createInferenceRequestMessage(1, [1, 2, 3]),
      createPingMessage(),
      createPongMessage(),
      createNodeReadyMessage("n1", 0),
      createInferenceStepMessage(1, 464, 5),
      createKVResetMessage(1),
      createNodeLogMessage("n1", "perf", "fwd", { ms: 10 }),
      createErrorMessage("ERR", "bad"),
    ];
    for (const msg of messages) {
      const r = validateMessage(msg);
      assert.equal(r.valid, true, `${msg.type} should be valid: ${r.error}`);
    }
  });
});

// ─── parseMessage ────────────────────────────────────────────────

describe("parseMessage", () => {
  it("parses valid JSON into validated message", () => {
    const raw = JSON.stringify(createPingMessage());
    const { msg, error } = parseMessage(raw);
    assert.equal(error, null);
    assert.equal(msg.type, "PING");
  });

  it("rejects invalid JSON", () => {
    const { msg, error } = parseMessage("{not json}");
    assert.equal(msg, null);
    assert.match(error, /Invalid JSON/);
  });

  it("rejects valid JSON with bad message structure", () => {
    const { msg, error } = parseMessage(JSON.stringify({ foo: "bar" }));
    assert.equal(msg, null);
    assert.ok(error);
  });

  it("rejects valid JSON with missing required fields", () => {
    const { msg, error } = parseMessage(JSON.stringify({ type: "JOIN" }));
    assert.equal(msg, null);
    assert.match(error, /nodeId/);
  });

  it("parses complex message (ACTIVATION)", () => {
    const original = createActivationMessage("n0", "n1", 3, 99, "base64data", [1, 768]);
    const { msg, error } = parseMessage(JSON.stringify(original));
    assert.equal(error, null);
    assert.equal(msg.type, "ACTIVATION");
    assert.equal(msg.requestId, 99);
    assert.deepEqual(msg.tensor.shape, [1, 768]);
  });

  it("rejects empty string", () => {
    const { msg, error } = parseMessage("");
    assert.equal(msg, null);
    assert.match(error, /Invalid JSON/);
  });
});
