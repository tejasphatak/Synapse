/**
 * Synapse Phase 1 Validation Framework
 *
 * Tests old vs new paths for:
 *   1. Binary protocol — round-trip bit-exact comparison
 *   2. Int8 quantization — error bounds and quality
 *   3. KV cache data structures — append/reset correctness
 *   4. End-to-end wire format — old (JSON+base64) vs new (binary+int8)
 *
 * Run: node --experimental-vm-modules test/validate-phase1.js
 *   or: node test/validate-phase1.js
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

// ─── Protocol imports ────────────────────────────────────────────

import {
  encodeBinaryMessage,
  decodeBinaryMessage,
  encodeBinaryOutput,
  decodeOutputTokens,
  isBinaryMessage,
  requestIdToUint32,
  uint32ToRequestId,
  registerRequestId,
  BinaryMsgType,
  QuantMode,
  Flags,
  getQuantMode,
  setQuantFlags,
  MAGIC,
  HEADER_SIZE,
} from "../protocol/binary.js";

import {
  quantizeInt8,
  dequantizeInt8,
  packQuantized,
  unpackQuantized,
} from "../protocol/quantize.js";

import {
  MessageType,
  PROTOCOL_V2,
  createActivationMessage,
  createInferenceStepMessage,
  createKVResetMessage,
  validateMessage,
  parseMessage,
} from "../protocol/messages.js";

// ─── Test Utilities ──────────────────────────────────────────────

/**
 * Generate a random Float32Array simulating a transformer activation.
 * Values centered around 0 with some outliers, mimicking real hidden states.
 */
function randomActivation(size, scale = 1.0) {
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    // Normal-ish distribution via Box-Muller
    const u1 = Math.random();
    const u2 = Math.random();
    data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * scale;
  }
  return data;
}

/**
 * Compute max absolute error between two Float32Arrays.
 */
function maxAbsError(a, b) {
  assert.equal(a.length, b.length, "Arrays must be same length");
  let maxErr = 0;
  for (let i = 0; i < a.length; i++) {
    maxErr = Math.max(maxErr, Math.abs(a[i] - b[i]));
  }
  return maxErr;
}

/**
 * Compute cosine similarity between two Float32Arrays.
 */
function cosineSimilarity(a, b) {
  assert.equal(a.length, b.length, "Arrays must be same length");
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Simulate the OLD serialization path (JSON + base64).
 * This is what the system did before Phase 1.
 */
function serializeOld(float32Array, shape) {
  const bytes = new Uint8Array(float32Array.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return {
    type: "ACTIVATION",
    fromNode: "node-test",
    toNode: null,
    layer: 5,
    requestId: "req-test-123",
    tensor: {
      shape,
      dtype: "float32",
      data: Buffer.from(bytes).toString("base64"),
    },
    timestamp: Date.now(),
  };
}

/**
 * Simulate the OLD deserialization path.
 */
function deserializeOld(msg) {
  const base64 = msg.tensor.data;
  const bytes = Buffer.from(base64, "base64");
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/**
 * Simulate the NEW binary serialization path.
 */
function serializeNew(float32Array, shape, requestId = 1) {
  return encodeBinaryMessage(
    BinaryMsgType.ACTIVATION,
    0, // no quant
    0, // seqPos
    requestId,
    shape,
    float32Array.buffer
  );
}

/**
 * Simulate the NEW binary + int8 serialization path.
 */
function serializeNewQuantized(float32Array, shape, requestId = 1) {
  const { data: int8Data, scale } = quantizeInt8(float32Array);
  const packed = packQuantized(int8Data, scale);
  const flags = setQuantFlags(0, QuantMode.INT8);
  return encodeBinaryMessage(
    BinaryMsgType.ACTIVATION,
    flags,
    0,
    requestId,
    shape,
    packed
  );
}

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE 1: Binary Protocol Round-Trip
// ═══════════════════════════════════════════════════════════════════

describe("Binary Protocol", () => {
  it("encodes and decodes magic bytes correctly", () => {
    const data = new Float32Array([1, 2, 3]);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 42, [1, 3], data);
    const decoded = decodeBinaryMessage(encoded);

    assert.equal(decoded.type, BinaryMsgType.ACTIVATION);
    assert.equal(decoded.requestId, 42);
    assert.deepEqual(decoded.shape, [1, 3]);
  });

  it("round-trips float32 tensor data bit-exactly", () => {
    const original = randomActivation(768); // GPT-2 hidden size
    const encoded = serializeNew(original, [1, 768]);
    const decoded = decodeBinaryMessage(encoded);

    const recovered = new Float32Array(
      decoded.payload.buffer,
      decoded.payload.byteOffset,
      decoded.payload.byteLength / 4
    );

    assert.equal(recovered.length, original.length);
    for (let i = 0; i < original.length; i++) {
      assert.equal(recovered[i], original[i], `Mismatch at index ${i}`);
    }
  });

  it("round-trips large tensors (full sequence)", () => {
    // Simulate a prefill activation: [128, 768]
    const seqLen = 128;
    const hiddenSize = 768;
    const original = randomActivation(seqLen * hiddenSize);
    const encoded = serializeNew(original, [seqLen, hiddenSize]);
    const decoded = decodeBinaryMessage(encoded);

    const recovered = new Float32Array(
      decoded.payload.buffer,
      decoded.payload.byteOffset,
      decoded.payload.byteLength / 4
    );

    assert.equal(recovered.length, original.length);
    assert.equal(maxAbsError(original, recovered), 0, "Binary round-trip must be bit-exact");
  });

  it("preserves seqPos and flags", () => {
    const data = new Float32Array([1]);
    const flags = setQuantFlags(0, QuantMode.INT8) | Flags.DELTA;
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, flags, 512, 99, [1, 1], data);
    const decoded = decodeBinaryMessage(encoded);

    assert.equal(decoded.seqPos, 512);
    assert.equal(getQuantMode(decoded.flags), QuantMode.INT8);
    assert.ok(decoded.flags & Flags.DELTA);
  });

  it("handles OUTPUT messages with token IDs", () => {
    const tokens = [15496, 345, 50256, 0, 42];
    const encoded = encodeBinaryOutput(7, tokens);
    const decoded = decodeBinaryMessage(encoded);

    assert.equal(decoded.type, BinaryMsgType.OUTPUT);
    assert.equal(decoded.requestId, 7);

    const recoveredTokens = decodeOutputTokens(decoded.payload);
    assert.deepEqual(recoveredTokens, tokens);
  });

  it("detects binary vs JSON messages", () => {
    const binaryMsg = encodeBinaryMessage(BinaryMsgType.ACTIVATION, 0, 0, 1, [1, 1], new Float32Array([0]));
    assert.ok(isBinaryMessage(binaryMsg));
    assert.ok(isBinaryMessage(Buffer.from(binaryMsg)));
    assert.ok(!isBinaryMessage('{"type":"ACTIVATION"}'));
    assert.ok(!isBinaryMessage("hello"));
  });

  it("rejects messages with wrong magic", () => {
    const buf = new ArrayBuffer(HEADER_SIZE + 4);
    new DataView(buf).setUint32(0, 0xDEADBEEF, false);
    assert.throws(() => decodeBinaryMessage(buf), /Invalid magic/);
  });

  it("rejects messages shorter than header", () => {
    assert.throws(() => decodeBinaryMessage(new ArrayBuffer(10)), /too short/);
  });

  it("request ID mapping is bidirectional", () => {
    registerRequestId("gen-test-abc", 999);
    assert.equal(requestIdToUint32("gen-test-abc"), 999);
    assert.equal(uint32ToRequestId(999), "gen-test-abc");
  });
});

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE 2: Old vs New Serialization Comparison
// ═══════════════════════════════════════════════════════════════════

describe("Old vs New Serialization", () => {
  it("produces identical float32 values via JSON+base64 and binary paths", () => {
    const original = randomActivation(768);
    const shape = [1, 768];

    // Old path: Float32 → base64 → JSON
    const oldMsg = serializeOld(original, shape);
    const oldRecovered = deserializeOld(oldMsg);

    // New path: Float32 → binary frame
    const newEncoded = serializeNew(original, shape);
    const newDecoded = decodeBinaryMessage(newEncoded);
    const newRecovered = new Float32Array(
      newDecoded.payload.buffer,
      newDecoded.payload.byteOffset,
      newDecoded.payload.byteLength / 4
    );

    // Both must be bit-exact to original
    assert.equal(maxAbsError(original, oldRecovered), 0, "Old path must be bit-exact");
    assert.equal(maxAbsError(original, newRecovered), 0, "New path must be bit-exact");
    assert.equal(maxAbsError(oldRecovered, newRecovered), 0, "Old and new must match");
  });

  it("binary format is smaller than JSON+base64", () => {
    const original = randomActivation(768);
    const shape = [1, 768];

    const oldMsg = serializeOld(original, shape);
    const oldSize = JSON.stringify(oldMsg).length;

    const newEncoded = serializeNew(original, shape);
    const newSize = newEncoded.byteLength;

    const ratio = oldSize / newSize;
    console.log(`    Size comparison: old=${oldSize} bytes, new=${newSize} bytes, ratio=${ratio.toFixed(2)}x`);

    assert.ok(newSize < oldSize, `Binary (${newSize}) should be smaller than JSON+base64 (${oldSize})`);
    assert.ok(ratio > 1.2, `Expected >1.2x reduction, got ${ratio.toFixed(2)}x`);
  });

  it("quantized binary is ~5x smaller than JSON+base64", () => {
    const original = randomActivation(768);
    const shape = [1, 768];

    const oldMsg = serializeOld(original, shape);
    const oldSize = JSON.stringify(oldMsg).length;

    const newEncoded = serializeNewQuantized(original, shape);
    const newSize = newEncoded.byteLength;

    const ratio = oldSize / newSize;
    console.log(`    Quantized comparison: old=${oldSize} bytes, new=${newSize} bytes, ratio=${ratio.toFixed(2)}x`);

    assert.ok(ratio > 3.0, `Expected >3x reduction with int8, got ${ratio.toFixed(2)}x`);
  });
});

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE 3: Int8 Quantization Quality
// ═══════════════════════════════════════════════════════════════════

describe("Int8 Quantization", () => {
  it("round-trips with bounded error", () => {
    const original = randomActivation(768, 2.0);
    const { data: quantized, scale } = quantizeInt8(original);
    const recovered = dequantizeInt8(quantized, scale);

    const maxErr = maxAbsError(original, recovered);
    const expectedMaxErr = Math.max(...original.map(Math.abs)) / 127;

    console.log(`    Max error: ${maxErr.toFixed(6)}, expected bound: ${expectedMaxErr.toFixed(6)}`);
    assert.ok(maxErr <= expectedMaxErr + 1e-6, `Max error ${maxErr} exceeds bound ${expectedMaxErr}`);
  });

  it("maintains high cosine similarity (>0.999)", () => {
    const original = randomActivation(768);
    const { data: quantized, scale } = quantizeInt8(original);
    const recovered = dequantizeInt8(quantized, scale);

    const similarity = cosineSimilarity(original, recovered);
    console.log(`    Cosine similarity: ${similarity.toFixed(6)}`);
    assert.ok(similarity > 0.999, `Cosine similarity ${similarity} is too low`);
  });

  it("handles zero tensor", () => {
    const zeros = new Float32Array(768);
    const { data: quantized, scale } = quantizeInt8(zeros);
    const recovered = dequantizeInt8(quantized, scale);

    assert.equal(maxAbsError(zeros, recovered), 0);
  });

  it("handles single large outlier", () => {
    const data = new Float32Array(768);
    for (let i = 0; i < 768; i++) data[i] = 0.01 * (Math.random() - 0.5);
    data[0] = 100.0; // large outlier

    const { data: quantized, scale } = quantizeInt8(data);
    const recovered = dequantizeInt8(quantized, scale);

    // Outlier should be preserved well
    const outlierErr = Math.abs(data[0] - recovered[0]);
    console.log(`    Outlier error: ${outlierErr.toFixed(6)} (value: ${data[0]})`);
    assert.ok(outlierErr < 1.0, "Outlier should be within 1.0 of original");

    // But small values lose precision (this is expected behavior)
    const similarity = cosineSimilarity(data, recovered);
    console.log(`    Cosine similarity with outlier: ${similarity.toFixed(6)}`);
  });

  it("pack/unpack preserves quantized data", () => {
    const original = randomActivation(768);
    const { data: int8Data, scale } = quantizeInt8(original);

    const packed = packQuantized(int8Data, scale);
    const { int8Data: unpacked, scale: unpackedScale } = unpackQuantized(packed);

    // Scale is stored as float32, so compare with float32 precision
    assert.ok(Math.abs(unpackedScale - scale) < 1e-6, `Scale mismatch: ${unpackedScale} vs ${scale}`);
    assert.equal(unpacked.length, int8Data.length);
    for (let i = 0; i < int8Data.length; i++) {
      assert.equal(unpacked[i], int8Data[i]);
    }
  });

  it("quantized binary round-trip through wire format", () => {
    const original = randomActivation(768);
    const shape = [1, 768];

    // Quantize + encode
    const { data: int8Data, scale } = quantizeInt8(original);
    const packed = packQuantized(int8Data, scale);
    const flags = setQuantFlags(0, QuantMode.INT8);
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, flags, 0, 1, shape, packed);

    // Decode + dequantize
    const decoded = decodeBinaryMessage(encoded);
    assert.equal(getQuantMode(decoded.flags), QuantMode.INT8);

    const { int8Data: recoveredInt8, scale: recoveredScale } = unpackQuantized(decoded.payload);
    const recovered = dequantizeInt8(recoveredInt8, recoveredScale);

    const similarity = cosineSimilarity(original, recovered);
    console.log(`    Full wire round-trip cosine similarity: ${similarity.toFixed(6)}`);
    assert.ok(similarity > 0.999);
  });

  it("statistical quality over 100 random tensors", () => {
    let totalSimilarity = 0;
    let worstSimilarity = 1.0;
    const N = 100;

    for (let trial = 0; trial < N; trial++) {
      const original = randomActivation(768, 1.0 + Math.random() * 5.0);
      const { data: quantized, scale } = quantizeInt8(original);
      const recovered = dequantizeInt8(quantized, scale);
      const sim = cosineSimilarity(original, recovered);
      totalSimilarity += sim;
      if (sim < worstSimilarity) worstSimilarity = sim;
    }

    const avgSimilarity = totalSimilarity / N;
    console.log(`    Average cosine similarity (${N} trials): ${avgSimilarity.toFixed(6)}`);
    console.log(`    Worst cosine similarity: ${worstSimilarity.toFixed(6)}`);

    assert.ok(avgSimilarity > 0.9995, `Average similarity ${avgSimilarity} too low`);
    assert.ok(worstSimilarity > 0.999, `Worst similarity ${worstSimilarity} too low`);
  });
});

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE 4: Message Schema (new types)
// ═══════════════════════════════════════════════════════════════════

describe("Protocol Messages", () => {
  it("validates INFERENCE_STEP messages", () => {
    const msg = createInferenceStepMessage("req-123", 42, 10, 1);
    const result = validateMessage(msg);
    assert.ok(result.valid, result.error);
    assert.equal(msg.type, MessageType.INFERENCE_STEP);
    assert.equal(msg.tokenId, 42);
    assert.equal(msg.seqPos, 10);
  });

  it("validates KV_RESET messages", () => {
    const msg = createKVResetMessage("req-123");
    const result = validateMessage(msg);
    assert.ok(result.valid, result.error);
    assert.equal(msg.type, MessageType.KV_RESET);
  });

  it("rejects INFERENCE_STEP without required fields", () => {
    const result = validateMessage({ type: MessageType.INFERENCE_STEP, requestId: "x" });
    assert.ok(!result.valid);
    assert.match(result.error, /tokenId|seqPos/);
  });

  it("PROTOCOL_V2 constant is defined", () => {
    assert.equal(PROTOCOL_V2, "proto_v2");
  });

  it("parseMessage handles valid JSON", () => {
    const msg = createInferenceStepMessage("req-1", 5, 3);
    const raw = JSON.stringify(msg);
    const { msg: parsed, error } = parseMessage(raw);
    assert.ok(!error);
    assert.equal(parsed.type, MessageType.INFERENCE_STEP);
  });
});

// ═══════════════════════════════════════════════════════════════════
//  TEST SUITE 5: Size Comparison Report
// ═══════════════════════════════════════════════════════════════════

describe("Wire Size Report", () => {
  it("generates size comparison for all tensor shapes", () => {
    const shapes = [
      { name: "Single token (cached)", shape: [1, 768] },
      { name: "Short prompt (10 tokens)", shape: [10, 768] },
      { name: "Medium prompt (128 tokens)", shape: [128, 768] },
      { name: "Long prompt (512 tokens)", shape: [512, 768] },
      { name: "Max prompt (1024 tokens)", shape: [1024, 768] },
    ];

    console.log("\n    ┌─────────────────────────────┬──────────┬──────────┬──────────┬───────┬───────┐");
    console.log("    │ Tensor Shape                │ Old (B)  │ New (B)  │ Quant(B) │ Old/N │ Old/Q │");
    console.log("    ├─────────────────────────────┼──────────┼──────────┼──────────┼───────┼───────┤");

    for (const { name, shape } of shapes) {
      const size = shape[0] * shape[1];
      const data = randomActivation(size);

      const oldSize = JSON.stringify(serializeOld(data, shape)).length;
      const newSize = serializeNew(data, shape).byteLength;
      const quantSize = serializeNewQuantized(data, shape).byteLength;

      const ratioNew = (oldSize / newSize).toFixed(1);
      const ratioQuant = (oldSize / quantSize).toFixed(1);

      const pad = (s, w) => String(s).padStart(w);
      console.log(
        `    │ ${name.padEnd(27)} │ ${pad(oldSize, 8)} │ ${pad(newSize, 8)} │ ${pad(quantSize, 8)} │ ${pad(ratioNew, 5)}x│ ${pad(ratioQuant, 5)}x│`
      );
    }

    console.log("    └─────────────────────────────┴──────────┴──────────┴──────────┴───────┴───────┘\n");
  });
});
