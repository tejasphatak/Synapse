/**
 * SYN1 Binary Wire Protocol Tests
 *
 * Covers: encode/decode round-trips, flag manipulation, request ID mapping,
 * output encoding, detection, peek helpers, edge cases.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";

import {
  MAGIC, HEADER_SIZE,
  BinaryMsgType, Flags, QuantMode,
  getQuantMode, setQuantFlags,
  requestIdToUint32, uint32ToRequestId, registerRequestId,
  encodeBinaryMessage, decodeBinaryMessage,
  encodeBinaryOutput, decodeOutputTokens,
  isBinaryMessage,
  peekRequestId, peekMessageType,
} from "../protocol/binary.js";

// ─── Helpers ─────────────────────────────────────────────────

function f32buf(...vals) {
  return new Float32Array(vals).buffer;
}

function randomFloat32(len) {
  const arr = new Float32Array(len);
  for (let i = 0; i < len; i++) arr[i] = Math.random() * 2 - 1;
  return arr;
}

// ─── Constants ───────────────────────────────────────────────

describe("Binary Protocol Constants", () => {
  it("MAGIC equals ASCII SYN1", () => {
    // S=0x53, Y=0x59, N=0x4E, 1=0x31
    assert.equal(MAGIC, 0x53594E31);
  });

  it("HEADER_SIZE is 24 bytes", () => {
    assert.equal(HEADER_SIZE, 24);
  });

  it("BinaryMsgType values are distinct", () => {
    const vals = Object.values(BinaryMsgType);
    assert.equal(new Set(vals).size, vals.length);
  });

  it("Flag bits don't overlap", () => {
    // Each flag should be a distinct bit (except QUANT_MASK which overlaps first 2 bits)
    const flags = [Flags.COMPRESSED, Flags.PREDICTED, Flags.EARLY_EXIT, Flags.DELTA];
    for (let i = 0; i < flags.length; i++) {
      for (let j = i + 1; j < flags.length; j++) {
        assert.equal(flags[i] & flags[j], 0, `Flags ${flags[i]} and ${flags[j]} overlap`);
      }
    }
    // QUANT_MASK should not overlap with other flags
    for (const f of flags) {
      assert.equal(Flags.QUANT_MASK & f, 0, `QUANT_MASK overlaps with ${f}`);
    }
  });
});

// ─── Quantization Flag Helpers ───────────────────────────────

describe("Quantization Flags", () => {
  it("getQuantMode extracts quant bits from flags", () => {
    assert.equal(getQuantMode(0b00000000), QuantMode.NONE);
    assert.equal(getQuantMode(0b00000001), QuantMode.INT8);
    assert.equal(getQuantMode(0b00000010), QuantMode.INT4);
  });

  it("getQuantMode ignores other flags", () => {
    assert.equal(getQuantMode(0b11111100), QuantMode.NONE);
    assert.equal(getQuantMode(0b11111101), QuantMode.INT8);
    assert.equal(getQuantMode(0b00101110), QuantMode.INT4);
  });

  it("setQuantFlags sets quant bits without affecting others", () => {
    const base = Flags.COMPRESSED | Flags.DELTA; // 0b00100100
    const result = setQuantFlags(base, QuantMode.INT8);
    assert.equal(getQuantMode(result), QuantMode.INT8);
    assert.ok(result & Flags.COMPRESSED);
    assert.ok(result & Flags.DELTA);
  });

  it("setQuantFlags clears previous quant mode", () => {
    let flags = setQuantFlags(0, QuantMode.INT8);
    assert.equal(getQuantMode(flags), QuantMode.INT8);
    flags = setQuantFlags(flags, QuantMode.INT4);
    assert.equal(getQuantMode(flags), QuantMode.INT4);
    flags = setQuantFlags(flags, QuantMode.NONE);
    assert.equal(getQuantMode(flags), QuantMode.NONE);
  });
});

// ─── Request ID Mapping ──────────────────────────────────────

describe("Request ID Mapping", () => {
  it("maps string to uint32 deterministically", () => {
    const id = requestIdToUint32("test-req-binary-1");
    assert.equal(typeof id, "number");
    assert.equal(requestIdToUint32("test-req-binary-1"), id); // same input → same output
  });

  it("reverse lookup returns original string", () => {
    const numId = requestIdToUint32("test-req-binary-2");
    assert.equal(uint32ToRequestId(numId), "test-req-binary-2");
  });

  it("unknown uint32 returns fallback", () => {
    assert.equal(uint32ToRequestId(99999999), "req-99999999");
  });

  it("registerRequestId creates bidirectional mapping", () => {
    registerRequestId("manual-req", 42424242);
    assert.equal(requestIdToUint32("manual-req"), 42424242);
    assert.equal(uint32ToRequestId(42424242), "manual-req");
  });

  it("different strings get different IDs", () => {
    const a = requestIdToUint32("req-aaa-bin");
    const b = requestIdToUint32("req-bbb-bin");
    assert.notEqual(a, b);
  });
});

// ─── Encode / Decode Round-Trip ──────────────────────────────

describe("Encode / Decode", () => {
  it("round-trips ACTIVATION message with float32 data", () => {
    const tensor = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const encoded = encodeBinaryMessage(
      BinaryMsgType.ACTIVATION, 0, 42, 100, [1, 4], tensor
    );

    assert.ok(encoded instanceof ArrayBuffer);
    assert.equal(encoded.byteLength, HEADER_SIZE + 16); // 4 floats × 4 bytes

    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.type, BinaryMsgType.ACTIVATION);
    assert.equal(decoded.flags, 0);
    assert.equal(decoded.seqPos, 42);
    assert.equal(decoded.requestId, 100);
    assert.equal(decoded.payloadSize, 16);
    assert.deepEqual(decoded.shape, [1, 4]);

    const floats = new Float32Array(decoded.payload.buffer, decoded.payload.byteOffset, 4);
    assert.deepEqual(Array.from(floats), [1, 2, 3, 4]);
  });

  it("round-trips all flag combinations", () => {
    const tensor = new Float32Array([0.5]);
    const flagCombos = [
      0,
      Flags.COMPRESSED,
      Flags.PREDICTED,
      Flags.EARLY_EXIT,
      Flags.DELTA,
      Flags.COMPRESSED | Flags.PREDICTED,
      Flags.DELTA | Flags.EARLY_EXIT | Flags.COMPRESSED,
      setQuantFlags(Flags.PREDICTED, QuantMode.INT8),
      setQuantFlags(Flags.DELTA, QuantMode.INT4),
      0xFF, // all bits set
    ];

    for (const flags of flagCombos) {
      const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, flags, 0, 1, [1, 1], tensor);
      const decoded = decodeBinaryMessage(encoded);
      assert.equal(decoded.flags, flags, `Flag mismatch for 0b${flags.toString(2)}`);
    }
  });

  it("handles max seqPos (uint16 max)", () => {
    const tensor = new Float32Array([1.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 65535, 1, [1, 1], tensor);
    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.seqPos, 65535);
  });

  it("handles seqPos 0", () => {
    const tensor = new Float32Array([1.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 1, [1, 1], tensor);
    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.seqPos, 0);
  });

  it("handles large request IDs", () => {
    const tensor = new Float32Array([1.0]);
    const largeId = 0xFFFFFFFE;
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, largeId, [1, 1], tensor);
    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.requestId, largeId);
  });

  it("round-trips Uint8Array tensor data", () => {
    const data = new Uint8Array([10, 20, 30, 40, 50]);
    const encoded = encodeBinaryMessage(
      BinaryMsgType.ACTIVATION, setQuantFlags(0, QuantMode.INT8), 0, 1, [1, 5], data
    );
    const decoded = decodeBinaryMessage(encoded);
    assert.deepEqual(Array.from(decoded.payload), [10, 20, 30, 40, 50]);
  });

  it("round-trips Int8Array tensor data", () => {
    const data = new Int8Array([-128, -1, 0, 1, 127]);
    const encoded = encodeBinaryMessage(
      BinaryMsgType.ACTIVATION, setQuantFlags(0, QuantMode.INT8), 0, 1, [1, 5], data
    );
    const decoded = decodeBinaryMessage(encoded);
    const int8view = new Int8Array(decoded.payload.buffer, decoded.payload.byteOffset, 5);
    assert.deepEqual(Array.from(int8view), [-128, -1, 0, 1, 127]);
  });

  it("round-trips ArrayBuffer directly", () => {
    const buf = new Float32Array([3.14, 2.71]).buffer;
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 1, [1, 2], buf);
    const decoded = decodeBinaryMessage(encoded);
    const floats = new Float32Array(decoded.payload.buffer, decoded.payload.byteOffset, 2);
    assert.ok(Math.abs(floats[0] - 3.14) < 0.001);
    assert.ok(Math.abs(floats[1] - 2.71) < 0.001);
  });

  it("round-trips all message types", () => {
    const tensor = new Float32Array([1.0]);
    for (const [name, type] of Object.entries(BinaryMsgType)) {
      const encoded = encodeBinaryMessage(type, 0, 0, 1, [1, 1], tensor);
      const decoded = decodeBinaryMessage(encoded);
      assert.equal(decoded.type, type, `Type mismatch for ${name}`);
    }
  });

  it("preserves shape dimensions", () => {
    const tensor = new Float32Array(12).fill(0);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 1, [3, 4], tensor);
    const decoded = decodeBinaryMessage(encoded);
    assert.deepEqual(decoded.shape, [3, 4]);
  });

  it("handles zero-dimension shape", () => {
    const tensor = new Uint8Array(0);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 1, [0, 0], tensor);
    const decoded = decodeBinaryMessage(encoded);
    assert.deepEqual(decoded.shape, [0, 0]);
    assert.equal(decoded.payloadSize, 0);
  });

  it("handles large tensors (4K elements)", () => {
    const tensor = randomFloat32(4096);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 5, 99, [1, 4096], tensor);
    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.payloadSize, 4096 * 4);
    assert.deepEqual(decoded.shape, [1, 4096]);

    const floats = new Float32Array(decoded.payload.buffer, decoded.payload.byteOffset, 4096);
    for (let i = 0; i < 4096; i++) {
      assert.equal(floats[i], tensor[i]);
    }
  });
});

// ─── Output Encoding ────────────────────────────────────────

describe("Output Encoding", () => {
  it("encodes and decodes token IDs", () => {
    const tokens = [1234, 5678, 42, 0, 50256];
    const encoded = encodeBinaryOutput(200, tokens, 10);

    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.type, BinaryMsgType.OUTPUT);
    assert.equal(decoded.flags, 0);
    assert.equal(decoded.seqPos, 10);
    assert.equal(decoded.requestId, 200);
    assert.deepEqual(decoded.shape, [5, 1]);

    const decodedTokens = decodeOutputTokens(decoded.payload);
    assert.deepEqual(decodedTokens, tokens);
  });

  it("handles single token", () => {
    const encoded = encodeBinaryOutput(1, [42]);
    const decoded = decodeBinaryMessage(encoded);
    const tokens = decodeOutputTokens(decoded.payload);
    assert.deepEqual(tokens, [42]);
  });

  it("handles empty token array", () => {
    const encoded = encodeBinaryOutput(1, []);
    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.payloadSize, 0);
    assert.deepEqual(decoded.shape, [0, 1]);
  });

  it("handles large token IDs (uint32 range)", () => {
    const tokens = [0, 1, 0xFFFFFFFF, 50257];
    const encoded = encodeBinaryOutput(1, tokens);
    const decoded = decodeBinaryMessage(encoded);
    const result = decodeOutputTokens(decoded.payload);
    assert.deepEqual(result, tokens);
  });

  it("defaults seqPos to 0", () => {
    const encoded = encodeBinaryOutput(1, [42]);
    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.seqPos, 0);
  });
});

// ─── Detection ──────────────────────────────────────────────

describe("isBinaryMessage", () => {
  it("detects valid ArrayBuffer binary message", () => {
    const tensor = new Float32Array([1.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 1, [1, 1], tensor);
    assert.ok(isBinaryMessage(encoded));
  });

  it("detects valid Buffer binary message", () => {
    const tensor = new Float32Array([1.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 1, [1, 1], tensor);
    const buf = Buffer.from(encoded);
    assert.ok(isBinaryMessage(buf));
  });

  it("rejects JSON string", () => {
    assert.ok(!isBinaryMessage('{"type":"PING"}'));
  });

  it("rejects empty ArrayBuffer", () => {
    assert.ok(!isBinaryMessage(new ArrayBuffer(0)));
  });

  it("rejects short ArrayBuffer", () => {
    assert.ok(!isBinaryMessage(new ArrayBuffer(3)));
  });

  it("rejects wrong magic", () => {
    const buf = new ArrayBuffer(24);
    new DataView(buf).setUint32(0, 0xDEADBEEF, false);
    assert.ok(!isBinaryMessage(buf));
  });

  it("rejects empty Buffer", () => {
    assert.ok(!isBinaryMessage(Buffer.alloc(0)));
  });

  it("rejects short Buffer", () => {
    assert.ok(!isBinaryMessage(Buffer.alloc(2)));
  });

  it("rejects Buffer with wrong magic", () => {
    const buf = Buffer.alloc(24);
    buf.writeUInt32BE(0x12345678, 0);
    assert.ok(!isBinaryMessage(buf));
  });

  it("rejects non-buffer types", () => {
    assert.ok(!isBinaryMessage(null));
    assert.ok(!isBinaryMessage(undefined));
    assert.ok(!isBinaryMessage(42));
    assert.ok(!isBinaryMessage({}));
    assert.ok(!isBinaryMessage([]));
  });

  it("detects minimal 4-byte ArrayBuffer with magic", () => {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, MAGIC, false);
    assert.ok(isBinaryMessage(buf));
  });
});

// ─── Peek Helpers ───────────────────────────────────────────

describe("Peek Helpers", () => {
  it("peekRequestId returns correct ID without full decode", () => {
    const tensor = new Float32Array([1.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 12345, [1, 1], tensor);
    assert.equal(peekRequestId(encoded), 12345);
  });

  it("peekRequestId works with Buffer", () => {
    const tensor = new Float32Array([1.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 777, [1, 1], tensor);
    const buf = Buffer.from(encoded);
    assert.equal(peekRequestId(buf), 777);
  });

  it("peekMessageType returns correct type", () => {
    const tensor = new Float32Array([1.0]);
    for (const [name, type] of Object.entries(BinaryMsgType)) {
      const encoded = encodeBinaryMessage(type, 0, 0, 1, [1, 1], tensor);
      assert.equal(peekMessageType(encoded), type, `Type mismatch for ${name}`);
    }
  });

  it("peekMessageType works with Buffer", () => {
    const tensor = new Float32Array([1.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.OUTPUT, 0, 0, 1, [1, 1], tensor);
    const buf = Buffer.from(encoded);
    assert.equal(peekMessageType(buf), BinaryMsgType.OUTPUT);
  });
});

// ─── Error Cases ────────────────────────────────────────────

describe("Decode Error Handling", () => {
  it("throws on buffer shorter than header", () => {
    assert.throws(
      () => decodeBinaryMessage(new ArrayBuffer(10)),
      /too short/
    );
  });

  it("throws on wrong magic", () => {
    const buf = new ArrayBuffer(24);
    new DataView(buf).setUint32(0, 0xBADCAFE, false);
    assert.throws(
      () => decodeBinaryMessage(buf),
      /Invalid magic/
    );
  });

  it("throws on empty buffer", () => {
    assert.throws(
      () => decodeBinaryMessage(new ArrayBuffer(0)),
      /too short/
    );
  });

  it("decodes header-only message (no payload)", () => {
    const buf = new ArrayBuffer(HEADER_SIZE);
    const view = new DataView(buf);
    view.setUint32(0, MAGIC, false);
    view.setUint8(4, BinaryMsgType.ACTIVATION);
    view.setUint8(5, 0);
    view.setUint16(6, 0, true);
    view.setUint32(8, 1, true);
    view.setUint32(12, 0, true); // payload size = 0
    view.setUint32(16, 0, true);
    view.setUint32(20, 0, true);

    const decoded = decodeBinaryMessage(buf);
    assert.equal(decoded.payloadSize, 0);
    assert.equal(decoded.payload.length, 0);
  });
});

// ─── Node.js Buffer Interop ─────────────────────────────────

describe("Node.js Buffer Interop", () => {
  it("decodes from Node.js Buffer", () => {
    const tensor = new Float32Array([1.5, 2.5]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 7, 99, [1, 2], tensor);
    const nodeBuf = Buffer.from(encoded);

    const decoded = decodeBinaryMessage(nodeBuf);
    assert.equal(decoded.type, BinaryMsgType.ACTIVATION);
    assert.equal(decoded.seqPos, 7);
    assert.equal(decoded.requestId, 99);

    const floats = new Float32Array(decoded.payload.buffer, decoded.payload.byteOffset, 2);
    assert.ok(Math.abs(floats[0] - 1.5) < 0.001);
    assert.ok(Math.abs(floats[1] - 2.5) < 0.001);
  });

  it("decodes from Buffer slice (byteOffset != 0)", () => {
    // Node.js WS can deliver Buffers that are views into a larger pool
    const tensor = new Float32Array([9.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 1, [1, 1], tensor);

    // Simulate a pool buffer with padding before our data
    const pool = Buffer.alloc(100);
    const msgBuf = Buffer.from(encoded);
    msgBuf.copy(pool, 20);
    const slice = pool.subarray(20, 20 + msgBuf.length);

    const decoded = decodeBinaryMessage(slice);
    assert.equal(decoded.type, BinaryMsgType.ACTIVATION);
    const floats = new Float32Array(decoded.payload.buffer, decoded.payload.byteOffset, 1);
    assert.ok(Math.abs(floats[0] - 9.0) < 0.001);
  });
});

// ─── Endianness ──────────────────────────────────────────────

describe("Endianness", () => {
  it("magic is big-endian, fields are little-endian", () => {
    const tensor = new Float32Array([1.0]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 256, 1, [1, 1], tensor);
    const view = new DataView(encoded);

    // Magic: big-endian
    assert.equal(view.getUint32(0, false), MAGIC);
    // SeqPos: little-endian — 256 as uint16 LE = 0x00 0x01
    assert.equal(view.getUint16(6, true), 256);
  });
});
