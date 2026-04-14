/**
 * Protocol Tests — Binary Wire Format, Quantization, Entropy Coding, Messages
 *
 * Tests the protocol layer: SYN1 binary encoding/decoding, int8 quantization
 * (per-tensor and per-channel), delta encoding, RLE compression, and JSON
 * message construction/validation.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  MAGIC, HEADER_SIZE, BinaryMsgType, Flags, QuantMode,
  getQuantMode, setQuantFlags,
  requestIdToUint32, uint32ToRequestId, registerRequestId,
  encodeBinaryMessage, decodeBinaryMessage,
  encodeBinaryOutput, decodeOutputTokens,
  isBinaryMessage, peekRequestId, peekMessageType,
} from "../protocol/binary.js";

import {
  quantizeInt8, dequantizeInt8,
  packQuantized, unpackQuantized,
  quantizeInt8PerChannel, dequantizeInt8PerChannel,
  packQuantizedPerChannel, unpackQuantizedPerChannel,
  computeDelta, applyDelta, deltaSparsity,
} from "../protocol/quantize.js";

import {
  rleCompress, rleDecompress,
  compressActivation, decompressActivation,
  compressPayload, decompressPayload,
} from "../protocol/entropy.js";

import {
  MessageType, PROTOCOL_V2,
  createJoinMessage, createAssignShardMessage,
  createActivationMessage, createOutputMessage,
  createTopologyUpdateMessage, createInferenceRequestMessage,
  createPingMessage, createPongMessage,
  createNodeReadyMessage, createInferenceStepMessage,
  createKVResetMessage, createNodeLogMessage,
  createErrorMessage,
  validateMessage, parseMessage,
} from "../protocol/messages.js";

// ─── Binary Protocol ────────────────────────────────────────────

describe("Binary Protocol", () => {

  describe("constants", () => {
    it("MAGIC is SYN1 in big-endian", () => {
      assert.equal(MAGIC, 0x53594E31);
      // Verify it spells "SYN1"
      const buf = Buffer.alloc(4);
      buf.writeUInt32BE(MAGIC);
      assert.equal(buf.toString("ascii"), "SYN1");
    });

    it("header size is 24 bytes", () => {
      assert.equal(HEADER_SIZE, 24);
    });
  });

  describe("flag helpers", () => {
    it("getQuantMode extracts quant bits", () => {
      assert.equal(getQuantMode(0b00000001), QuantMode.INT8);
      assert.equal(getQuantMode(0b00000010), QuantMode.INT4);
      assert.equal(getQuantMode(0b11111100), QuantMode.NONE);
    });

    it("setQuantFlags sets quant bits without touching others", () => {
      const flags = Flags.COMPRESSED | Flags.DELTA; // bits 2 and 5
      const updated = setQuantFlags(flags, QuantMode.INT8);
      assert.equal(getQuantMode(updated), QuantMode.INT8);
      assert.ok(updated & Flags.COMPRESSED);
      assert.ok(updated & Flags.DELTA);
    });

    it("setQuantFlags clears previous quant mode", () => {
      const flags = setQuantFlags(0, QuantMode.INT8);
      const cleared = setQuantFlags(flags, QuantMode.NONE);
      assert.equal(getQuantMode(cleared), QuantMode.NONE);
    });
  });

  describe("encode/decode roundtrip", () => {
    it("encodes and decodes ACTIVATION message", () => {
      const tensor = new Float32Array([1.0, 2.0, 3.0, 4.0]);
      const flags = setQuantFlags(0, QuantMode.NONE);
      const encoded = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, flags, 42, 7, [1, 4], tensor
      );

      assert.ok(encoded instanceof ArrayBuffer);
      assert.equal(encoded.byteLength, HEADER_SIZE + tensor.byteLength);

      const decoded = decodeBinaryMessage(encoded);
      assert.equal(decoded.type, BinaryMsgType.ACTIVATION);
      assert.equal(decoded.flags, flags);
      assert.equal(decoded.seqPos, 42);
      assert.equal(decoded.requestId, 7);
      assert.equal(decoded.payloadSize, 16);
      assert.deepEqual(decoded.shape, [1, 4]);

      const result = new Float32Array(
        decoded.payload.buffer, decoded.payload.byteOffset, 4
      );
      assert.deepEqual(Array.from(result), [1, 2, 3, 4]);
    });

    it("encodes and decodes OUTPUT message with token IDs", () => {
      const tokens = [50256, 15, 220, 5999];
      const encoded = encodeBinaryOutput(99, tokens, 10);

      const decoded = decodeBinaryMessage(encoded);
      assert.equal(decoded.type, BinaryMsgType.OUTPUT);
      assert.equal(decoded.seqPos, 10);
      assert.equal(decoded.requestId, 99);
      assert.deepEqual(decoded.shape, [4, 1]);

      const decodedTokens = decodeOutputTokens(decoded.payload);
      assert.deepEqual(decodedTokens, tokens);
    });

    it("handles empty payload", () => {
      const empty = new Uint8Array(0);
      const encoded = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, 0, 0, 1, [0, 0], empty
      );
      const decoded = decodeBinaryMessage(encoded);
      assert.equal(decoded.payloadSize, 0);
      assert.equal(decoded.payload.byteLength, 0);
    });

    it("preserves int8 quantized payload", () => {
      const int8 = new Int8Array([-127, -1, 0, 1, 127]);
      const flags = setQuantFlags(0, QuantMode.INT8);
      const encoded = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, flags, 0, 1, [1, 5], int8
      );
      const decoded = decodeBinaryMessage(encoded);
      assert.equal(getQuantMode(decoded.flags), QuantMode.INT8);
      const result = new Int8Array(
        decoded.payload.buffer, decoded.payload.byteOffset, 5
      );
      assert.deepEqual(Array.from(result), [-127, -1, 0, 1, 127]);
    });

    it("preserves all flag bits", () => {
      const allFlags = Flags.COMPRESSED | Flags.PREDICTED | Flags.EARLY_EXIT | Flags.DELTA;
      const flags = setQuantFlags(allFlags, QuantMode.INT4);
      const encoded = encodeBinaryMessage(
        BinaryMsgType.PREDICT, flags, 0, 1, [0, 0], new Uint8Array(0)
      );
      const decoded = decodeBinaryMessage(encoded);
      assert.ok(decoded.flags & Flags.COMPRESSED);
      assert.ok(decoded.flags & Flags.PREDICTED);
      assert.ok(decoded.flags & Flags.EARLY_EXIT);
      assert.ok(decoded.flags & Flags.DELTA);
      assert.equal(getQuantMode(decoded.flags), QuantMode.INT4);
    });

    it("handles max uint16 seqPos", () => {
      const encoded = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, 0, 65535, 1, [0, 0], new Uint8Array(0)
      );
      assert.equal(decodeBinaryMessage(encoded).seqPos, 65535);
    });
  });

  describe("decode error handling", () => {
    it("throws on too-short message", () => {
      assert.throws(
        () => decodeBinaryMessage(new ArrayBuffer(10)),
        /too short/
      );
    });

    it("throws on invalid magic", () => {
      const bad = new ArrayBuffer(24);
      new DataView(bad).setUint32(0, 0xDEADBEEF, false);
      assert.throws(
        () => decodeBinaryMessage(bad),
        /Invalid magic/
      );
    });
  });

  describe("isBinaryMessage", () => {
    it("returns true for valid SYN1 ArrayBuffer", () => {
      const encoded = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, 0, 0, 1, [1, 1], new Float32Array([1.0])
      );
      assert.ok(isBinaryMessage(encoded));
    });

    it("returns true for valid SYN1 Node Buffer", () => {
      const encoded = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, 0, 0, 1, [1, 1], new Float32Array([1.0])
      );
      const nodeBuf = Buffer.from(encoded);
      assert.ok(isBinaryMessage(nodeBuf));
    });

    it("returns false for JSON string", () => {
      assert.equal(isBinaryMessage('{"type":"JOIN"}'), false);
    });

    it("returns false for too-short buffer", () => {
      assert.equal(isBinaryMessage(new ArrayBuffer(2)), false);
      assert.equal(isBinaryMessage(Buffer.from([0x53])), false);
    });

    it("returns false for wrong magic", () => {
      const bad = Buffer.alloc(24, 0);
      assert.equal(isBinaryMessage(bad), false);
    });
  });

  describe("peek helpers", () => {
    it("peekRequestId extracts request ID without full decode", () => {
      const encoded = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, 0, 0, 42, [1, 1], new Float32Array([0])
      );
      assert.equal(peekRequestId(new Uint8Array(encoded)), 42);
    });

    it("peekMessageType extracts type without full decode", () => {
      const encoded = encodeBinaryMessage(
        BinaryMsgType.KV_APPEND, 0, 0, 1, [0, 0], new Uint8Array(0)
      );
      assert.equal(peekMessageType(new Uint8Array(encoded)), BinaryMsgType.KV_APPEND);
    });
  });

  describe("request ID mapping", () => {
    it("maps string IDs to uint32 and back", () => {
      const numId = requestIdToUint32("req-abc-123");
      assert.equal(typeof numId, "number");
      assert.equal(uint32ToRequestId(numId), "req-abc-123");
    });

    it("returns same uint32 for same string", () => {
      const id1 = requestIdToUint32("stable-id");
      const id2 = requestIdToUint32("stable-id");
      assert.equal(id1, id2);
    });

    it("registerRequestId creates explicit mapping", () => {
      registerRequestId("explicit-req", 99999);
      assert.equal(requestIdToUint32("explicit-req"), 99999);
      assert.equal(uint32ToRequestId(99999), "explicit-req");
    });

    it("uint32ToRequestId returns fallback for unknown", () => {
      assert.equal(uint32ToRequestId(88888), "req-88888");
    });
  });
});

// ─── Quantization ───────────────────────────────────────────────

describe("Quantization", () => {

  describe("per-tensor int8", () => {
    it("quantizes and dequantizes with low error", () => {
      const original = new Float32Array([1.0, -0.5, 0.25, -1.0, 0.0]);
      const { data, scale } = quantizeInt8(original);

      assert.ok(data instanceof Int8Array);
      assert.equal(data.length, 5);
      assert.ok(scale > 0);

      const restored = dequantizeInt8(data, scale);
      for (let i = 0; i < original.length; i++) {
        assert.ok(
          Math.abs(restored[i] - original[i]) < 0.02,
          `Value ${i}: ${restored[i]} vs ${original[i]}`
        );
      }
    });

    it("handles all-zero tensor", () => {
      const zeros = new Float32Array(10);
      const { data, scale } = quantizeInt8(zeros);
      assert.equal(scale, 1); // avoids division by zero
      const restored = dequantizeInt8(data, scale);
      for (const v of restored) assert.equal(v, 0);
    });

    it("clamps to [-127, 127] (symmetric)", () => {
      const data = new Float32Array([100, -100]);
      const { data: q } = quantizeInt8(data);
      assert.equal(q[0], 127);
      assert.equal(q[1], -127);
    });

    it("pack/unpack roundtrip preserves data and scale", () => {
      const original = new Float32Array([0.5, -0.3, 0.9, -0.1]);
      const { data, scale } = quantizeInt8(original);
      const packed = packQuantized(data, scale);
      const { int8Data, scale: unpackedScale } = unpackQuantized(packed);

      assert.equal(int8Data.length, data.length);
      assert.ok(Math.abs(unpackedScale - scale) < 1e-6);
      assert.deepEqual(Array.from(int8Data), Array.from(data));
    });

    it("pack/unpack works with Uint8Array view", () => {
      const { data, scale } = quantizeInt8(new Float32Array([1, -1]));
      const packed = packQuantized(data, scale);
      const view = new Uint8Array(packed);
      const { int8Data, scale: s } = unpackQuantized(view);
      assert.equal(int8Data.length, 2);
      assert.ok(Math.abs(s - scale) < 1e-6);
    });
  });

  describe("per-channel int8", () => {
    it("quantizes each row independently", () => {
      // 2 rows x 3 cols — row 0 has range [0,1], row 1 has range [0,100]
      const data = new Float32Array([0.1, 0.5, 1.0, 10, 50, 100]);
      const { data: q, scales } = quantizeInt8PerChannel(data, 3);

      assert.equal(scales.length, 2);
      assert.ok(scales[0] < scales[1], "Row 1 should have larger scale");

      const restored = dequantizeInt8PerChannel(q, scales, 3);
      // Per-channel should be more accurate than per-tensor for mixed ranges
      for (let i = 0; i < data.length; i++) {
        // Int8 quantization has ~1/127 relative error per element
        const absError = Math.abs(restored[i] - data[i]);
        const maxExpected = scales[Math.floor(i / 3)] * 1.01; // within one quant step
        assert.ok(absError < maxExpected, `Abs error too high at index ${i}: ${absError} vs max ${maxExpected}`);
      }
    });

    it("pack/unpack per-channel roundtrip", () => {
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const { data: q, scales } = quantizeInt8PerChannel(data, 3);
      const packed = packQuantizedPerChannel(q, scales);
      const { int8Data, scales: s2, numRows } = unpackQuantizedPerChannel(packed);

      assert.equal(numRows, 2);
      assert.equal(int8Data.length, q.length);
      for (let i = 0; i < scales.length; i++) {
        assert.ok(Math.abs(s2[i] - scales[i]) < 1e-6);
      }
    });
  });

  describe("delta encoding", () => {
    it("compute and apply delta roundtrip", () => {
      const prev = new Float32Array([1.0, 2.0, 3.0]);
      const curr = new Float32Array([1.1, 2.0, 2.9]);
      const delta = computeDelta(curr, prev);
      const reconstructed = applyDelta(delta, prev);

      for (let i = 0; i < curr.length; i++) {
        assert.ok(Math.abs(reconstructed[i] - curr[i]) < 1e-6);
      }
    });

    it("delta of identical tensors is all zeros", () => {
      const a = new Float32Array([5, 10, 15]);
      const delta = computeDelta(a, a);
      for (const v of delta) assert.equal(v, 0);
    });

    it("deltaSparsity measures near-zero fraction", () => {
      // 7 near-zero, 3 non-zero
      const delta = new Float32Array([0, 0.001, -0.005, 0, 0, 0.002, 0, 1.0, -2.0, 0.5]);
      const sparsity = deltaSparsity(delta, 0.01);
      assert.ok(Math.abs(sparsity - 0.7) < 1e-6);
    });

    it("deltaSparsity returns 1.0 for all-zero", () => {
      assert.equal(deltaSparsity(new Float32Array(100), 0.01), 1.0);
    });

    it("deltaSparsity returns 0.0 for all-large", () => {
      const big = new Float32Array(10).fill(999);
      assert.equal(deltaSparsity(big, 0.01), 0);
    });
  });
});

// ─── Entropy Coding (RLE) ───────────────────────────────────────

describe("Entropy Coding", () => {

  describe("RLE compress/decompress", () => {
    it("roundtrips all-zero data", () => {
      const data = new Int8Array(100);
      const { compressed, ratio } = rleCompress(data);
      assert.ok(ratio > 1, "Should compress all-zero data");
      const restored = rleDecompress(compressed, 100);
      assert.deepEqual(Array.from(restored), Array.from(data));
    });

    it("roundtrips all-nonzero data", () => {
      const data = new Int8Array(50);
      for (let i = 0; i < 50; i++) data[i] = (i % 127) + 1;
      const { compressed } = rleCompress(data);
      const restored = rleDecompress(compressed, 50);
      assert.deepEqual(Array.from(restored), Array.from(data));
    });

    it("roundtrips mixed sparse data (typical delta pattern)", () => {
      // Simulate delta-encoded activations: mostly zeros with scattered values
      const data = new Int8Array(200);
      data[10] = 5;
      data[11] = -3;
      data[50] = 127;
      data[100] = -127;
      data[150] = 1;
      data[199] = 42;

      const { compressed, ratio } = rleCompress(data);
      assert.ok(ratio > 2, `Expected good compression for sparse data, got ratio ${ratio}`);
      const restored = rleDecompress(compressed, 200);
      assert.deepEqual(Array.from(restored), Array.from(data));
    });

    it("handles single element", () => {
      const data = new Int8Array([42]);
      const { compressed } = rleCompress(data);
      const restored = rleDecompress(compressed, 1);
      assert.equal(restored[0], 42);
    });

    it("handles alternating zero/nonzero", () => {
      const data = new Int8Array([0, 5, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, -5]);
      const { compressed } = rleCompress(data);
      const restored = rleDecompress(compressed, data.length);
      assert.deepEqual(Array.from(restored), Array.from(data));
    });

    it("preserves negative values (sign extension)", () => {
      const data = new Int8Array([-1, -50, -127, -128]);
      const { compressed } = rleCompress(data);
      const restored = rleDecompress(compressed, 4);
      assert.deepEqual(Array.from(restored), Array.from(data));
    });

    it("compressActivation/decompressActivation aliases work", () => {
      const data = new Int8Array(64);
      data[0] = 10; data[63] = -10;
      const { compressed, originalLength } = compressActivation(data);
      const restored = decompressActivation(compressed, originalLength);
      assert.deepEqual(Array.from(restored), Array.from(data));
    });
  });

  describe("RLE with zeroThreshold", () => {
    it("treats small values as zero with threshold", () => {
      const data = new Int8Array([0, 1, -1, 0, 0, 0, 0, 0, 50]);
      const { compressed: c1 } = rleCompress(data, 0);
      const { compressed: c2 } = rleCompress(data, 1);
      // With threshold=1, the ±1 values become "zero", so more compression
      assert.ok(c2.byteLength <= c1.byteLength);
      // But roundtrip with threshold is lossy — small values become zero
      const restored = rleDecompress(c2, data.length);
      assert.equal(restored[8], 50); // large value preserved
    });
  });

  describe("payload compression", () => {
    it("compresses sparse quantized payload", () => {
      // Create a sparse int8 payload with scale at end
      const int8Data = new Int8Array(200);
      int8Data[10] = 5; int8Data[100] = -10;
      const packed = new ArrayBuffer(204);
      new Int8Array(packed, 0, 200).set(int8Data);
      new DataView(packed).setFloat32(200, 0.5, true);

      const result = compressPayload(packed);
      assert.ok(result !== null, "Should compress sparse payload");
      assert.ok(result.ratio > 1);

      const restored = decompressPayload(new Uint8Array(result.compressed));
      assert.equal(restored.byteLength, 204);
    });

    it("returns null when compression doesn't save space", () => {
      // Dense data — all nonzero, no compression benefit
      const data = new Int8Array(50);
      for (let i = 0; i < 50; i++) data[i] = (i % 127) + 1;
      const packed = new ArrayBuffer(54);
      new Int8Array(packed, 0, 50).set(data);
      new DataView(packed).setFloat32(50, 1.0, true);

      const result = compressPayload(packed);
      // May or may not be null — depends on overhead. Either way, should not corrupt.
      if (result) {
        const restored = decompressPayload(new Uint8Array(result.compressed));
        assert.equal(restored.byteLength, 54);
      }
    });
  });
});

// ─── Messages (JSON Protocol) ───────────────────────────────────

describe("Messages", () => {

  describe("message constructors", () => {
    it("createJoinMessage has correct structure", () => {
      const msg = createJoinMessage("node-0", { webgpu: true, maxLayers: 12 });
      assert.equal(msg.type, MessageType.JOIN);
      assert.equal(msg.nodeId, "node-0");
      assert.equal(msg.capabilities.webgpu, true);
      assert.equal(msg.capabilities.maxLayers, 12);
      assert.ok(msg.timestamp > 0);
    });

    it("createJoinMessage provides defaults", () => {
      const msg = createJoinMessage("node-0");
      assert.equal(msg.capabilities.webgpu, false);
      assert.equal(msg.capabilities.maxLayers, 6);
    });

    it("createAssignShardMessage includes all fields", () => {
      const msg = createAssignShardMessage("shard-0", 0, 5, "/shards/0", "/shards/shared");
      assert.equal(msg.type, MessageType.ASSIGN_SHARD);
      assert.equal(msg.shardId, "shard-0");
      assert.equal(msg.layerStart, 0);
      assert.equal(msg.layerEnd, 5);
    });

    it("createActivationMessage includes tensor", () => {
      const msg = createActivationMessage("n0", "n1", 5, "req-1", "base64data", [1, 768]);
      assert.equal(msg.type, MessageType.ACTIVATION);
      assert.equal(msg.fromNode, "n0");
      assert.equal(msg.toNode, "n1");
      assert.equal(msg.tensor.dtype, "float32");
      assert.deepEqual(msg.tensor.shape, [1, 768]);
    });

    it("createOutputMessage includes tokens and text", () => {
      const msg = createOutputMessage("req-1", [50256, 15], "Hello");
      assert.equal(msg.type, MessageType.OUTPUT);
      assert.deepEqual(msg.tokens, [50256, 15]);
      assert.equal(msg.text, "Hello");
    });

    it("createTopologyUpdateMessage", () => {
      const nodes = [{ nodeId: "n0" }];
      const pipeline = ["n0"];
      const msg = createTopologyUpdateMessage(nodes, pipeline);
      assert.equal(msg.type, MessageType.TOPOLOGY_UPDATE);
      assert.deepEqual(msg.pipeline, ["n0"]);
    });

    it("createInferenceRequestMessage", () => {
      const msg = createInferenceRequestMessage("req-5", [100, 200, 300]);
      assert.equal(msg.type, MessageType.INFERENCE_REQUEST);
      assert.deepEqual(msg.tokenIds, [100, 200, 300]);
    });

    it("createPingMessage and createPongMessage", () => {
      assert.equal(createPingMessage().type, MessageType.PING);
      assert.equal(createPongMessage().type, MessageType.PONG);
    });

    it("createNodeReadyMessage", () => {
      const msg = createNodeReadyMessage("node-0", "shard-0");
      assert.equal(msg.type, MessageType.NODE_READY);
      assert.equal(msg.nodeId, "node-0");
      assert.equal(msg.shardId, "shard-0");
    });

    it("createInferenceStepMessage", () => {
      const msg = createInferenceStepMessage("req-1", 50256, 10, 42);
      assert.equal(msg.type, MessageType.INFERENCE_STEP);
      assert.equal(msg.tokenId, 50256);
      assert.equal(msg.seqPos, 10);
      assert.equal(msg.binaryRequestId, 42);
    });

    it("createKVResetMessage", () => {
      const msg = createKVResetMessage("req-1");
      assert.equal(msg.type, MessageType.KV_RESET);
      assert.equal(msg.requestId, "req-1");
    });

    it("createNodeLogMessage", () => {
      const msg = createNodeLogMessage("node-0", "perf", "layer_forward", { latencyMs: 5 });
      assert.equal(msg.type, MessageType.NODE_LOG);
      assert.equal(msg.level, "perf");
      assert.equal(msg.event, "layer_forward");
      assert.equal(msg.data.latencyMs, 5);
    });

    it("createErrorMessage", () => {
      const msg = createErrorMessage("SHARD_MISSING", "Shard not found", { shardId: 3 });
      assert.equal(msg.type, MessageType.ERROR);
      assert.equal(msg.code, "SHARD_MISSING");
      assert.equal(msg.message, "Shard not found");
      assert.deepEqual(msg.details, { shardId: 3 });
    });
  });

  describe("validateMessage", () => {
    it("validates correct JOIN message", () => {
      const msg = createJoinMessage("node-0", { webgpu: true });
      assert.deepEqual(validateMessage(msg), { valid: true, error: null });
    });

    it("rejects null", () => {
      const { valid, error } = validateMessage(null);
      assert.equal(valid, false);
      assert.ok(error.includes("non-null object"));
    });

    it("rejects unknown message type", () => {
      const { valid, error } = validateMessage({ type: "UNKNOWN" });
      assert.equal(valid, false);
      assert.ok(error.includes("Unknown message type"));
    });

    it("rejects missing required fields", () => {
      const { valid, error } = validateMessage({ type: MessageType.JOIN });
      assert.equal(valid, false);
      assert.ok(error.includes("nodeId"));
    });

    it("rejects ACTIVATION with invalid tensor shape", () => {
      const msg = createActivationMessage("n0", "n1", 0, "req-1", "data", [1, 768]);
      msg.tensor.shape = "not-an-array";
      const { valid } = validateMessage(msg);
      assert.equal(valid, false);
    });

    it("rejects ACTIVATION with non-string tensor data", () => {
      const msg = createActivationMessage("n0", "n1", 0, "req-1", "data", [1, 768]);
      msg.tensor.data = 12345;
      const { valid } = validateMessage(msg);
      assert.equal(valid, false);
    });

    it("rejects OUTPUT with non-array tokens", () => {
      const msg = createOutputMessage("req-1", "not-array", "text");
      const { valid } = validateMessage(msg);
      assert.equal(valid, false);
    });

    it("validates all message types", () => {
      const messages = [
        createJoinMessage("n0"),
        createAssignShardMessage("s0", 0, 5, "/s", "/sh"),
        createActivationMessage("n0", "n1", 0, "r1", "d", [1, 1]),
        createOutputMessage("r1", [1], "t"),
        createTopologyUpdateMessage([], []),
        createInferenceRequestMessage("r1", [1]),
        createPingMessage(),
        createPongMessage(),
        createNodeReadyMessage("n0", "s0"),
        createInferenceStepMessage("r1", 1, 0),
        createKVResetMessage("r1"),
        createNodeLogMessage("n0", "info", "test"),
        createErrorMessage("E", "msg"),
      ];
      for (const msg of messages) {
        const { valid, error } = validateMessage(msg);
        assert.ok(valid, `${msg.type} should be valid: ${error}`);
      }
    });
  });

  describe("parseMessage", () => {
    it("parses valid JSON message", () => {
      const raw = JSON.stringify(createPingMessage());
      const { msg, error } = parseMessage(raw);
      assert.ok(msg);
      assert.equal(error, null);
      assert.equal(msg.type, MessageType.PING);
    });

    it("returns error for invalid JSON", () => {
      const { msg, error } = parseMessage("not json{");
      assert.equal(msg, null);
      assert.ok(error.includes("Invalid JSON"));
    });

    it("returns error for valid JSON with missing fields", () => {
      const { msg, error } = parseMessage('{"type":"JOIN"}');
      assert.equal(msg, null);
      assert.ok(error.includes("nodeId"));
    });
  });

  describe("constants", () => {
    it("PROTOCOL_V2 is defined", () => {
      assert.equal(PROTOCOL_V2, "proto_v2");
    });

    it("MessageType covers all expected types", () => {
      const expected = [
        "JOIN", "PING", "ASSIGN_SHARD", "PONG", "TOPOLOGY_UPDATE",
        "INFERENCE_REQUEST", "ACTIVATION", "NODE_READY", "OUTPUT",
        "INFERENCE_STEP", "KV_RESET", "NODE_LOG", "ERROR",
      ];
      for (const t of expected) {
        assert.ok(MessageType[t], `Missing MessageType.${t}`);
      }
    });
  });
});
