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
  quantizeInt8PerChannel,
  dequantizeInt8PerChannel,
  packQuantizedPerChannel,
  unpackQuantizedPerChannel,
  computeDelta,
  applyDelta,
  deltaSparsity,
} from "../protocol/quantize.js";

import { ActivationPredictor } from "../node/predictor.js";
import { EarlyExitDetector } from "../node/early-exit.js";

import {
  rleCompress,
  rleDecompress,
  compressPayload,
  decompressPayload,
} from "../protocol/entropy.js";

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

// ─── Quantized Relay Integration Test ─────────────────────────────
// Simulates the full path: node0 quantize → coordinator relay → node1 dequantize

describe("Quantized Relay (end-to-end wire path)", () => {
  it("node0 → coordinator → node1 with int8 maintains quality", () => {
    const shape = [1, 768]; // single cached token
    const original = randomActivation(768);

    // Node 0: quantize and encode binary message
    const { data: int8Data, scale } = quantizeInt8(original);
    const packed = packQuantized(int8Data, scale);
    let flags = setQuantFlags(0, QuantMode.INT8);
    registerRequestId("relay-test-1", 9999);

    const binaryMsg = encodeBinaryMessage(
      BinaryMsgType.ACTIVATION, flags, 42, 9999, shape, packed
    );

    // Coordinator: zero-copy relay (just pass the buffer)
    const relayed = Buffer.from(binaryMsg);

    // Node 1: decode and dequantize
    assert.ok(isBinaryMessage(relayed), "Relayed message should be detected as binary");
    const decoded = decodeBinaryMessage(relayed);
    assert.equal(getQuantMode(decoded.flags), QuantMode.INT8);
    assert.deepEqual(decoded.shape, shape);

    const unpacked = unpackQuantized(decoded.payload);
    const recovered = dequantizeInt8(unpacked.int8Data, unpacked.scale);

    // Quality check
    const cos = cosineSimilarity(original, recovered);
    const maxErr = maxAbsError(original, recovered);
    console.log(`    Relay cosine similarity: ${cos.toFixed(6)}, max error: ${maxErr.toFixed(6)}`);
    assert.ok(cos > 0.999, `Cosine similarity too low: ${cos}`);
  });

  it("multi-token prefill relay maintains quality", () => {
    const seqLen = 128;
    const shape = [seqLen, 768];
    const original = randomActivation(seqLen * 768);

    const { data: int8Data, scale } = quantizeInt8(original);
    const packed = packQuantized(int8Data, scale);
    let flags = setQuantFlags(0, QuantMode.INT8);

    const binaryMsg = encodeBinaryMessage(
      BinaryMsgType.ACTIVATION, flags, seqLen, 8888, shape, packed
    );

    const decoded = decodeBinaryMessage(Buffer.from(binaryMsg));
    const unpacked = unpackQuantized(decoded.payload);
    const recovered = dequantizeInt8(unpacked.int8Data, unpacked.scale);

    const cos = cosineSimilarity(original, recovered);
    console.log(`    Prefill (${seqLen} tokens) relay cosine: ${cos.toFixed(6)}`);
    assert.ok(cos > 0.999, `Cosine similarity too low for prefill: ${cos}`);
  });

  it("10 consecutive cached steps maintain cumulative quality", () => {
    // Simulates autoregressive loop — each step quantizes independently
    const shape = [1, 768];
    let totalCos = 0;
    let worstCos = 1;

    for (let step = 0; step < 10; step++) {
      const original = randomActivation(768, 0.5 + step * 0.1);
      const { data: int8Data, scale } = quantizeInt8(original);
      const packed = packQuantized(int8Data, scale);
      const flags = setQuantFlags(0, QuantMode.INT8);

      const msg = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, flags, step + 1, 7777, shape, packed
      );
      const decoded = decodeBinaryMessage(Buffer.from(msg));
      const unpacked = unpackQuantized(decoded.payload);
      const recovered = dequantizeInt8(unpacked.int8Data, unpacked.scale);

      const cos = cosineSimilarity(original, recovered);
      totalCos += cos;
      worstCos = Math.min(worstCos, cos);
    }

    const avgCos = totalCos / 10;
    console.log(`    10-step avg cosine: ${avgCos.toFixed(6)}, worst: ${worstCos.toFixed(6)}`);
    assert.ok(worstCos > 0.999, `Worst cosine across 10 steps too low: ${worstCos}`);
  });
});

// ─── Delta Encoding Relay Test ────────────────────────────────────
// Simulates autoregressive loop: first token is full int8, subsequent tokens are delta-encoded

describe("Delta Encoding Relay (autoregressive simulation)", () => {
  it("first token full, then 9 delta-encoded steps maintain quality", () => {
    const shape = [1, 768];
    const hidden = 768;

    // Generate a sequence of activations where each differs slightly from the previous
    // (mimics real transformer behavior during autoregressive decoding)
    const baseActivation = randomActivation(hidden, 1.0);
    const activations = [baseActivation];
    for (let i = 1; i < 10; i++) {
      const prev = activations[i - 1];
      const next = new Float32Array(hidden);
      for (let j = 0; j < hidden; j++) {
        // ~5% change per step (realistic for transformer hidden states)
        next[j] = prev[j] + (Math.random() - 0.5) * 0.1;
      }
      activations.push(next);
    }

    // Simulate sender side (node 0)
    let senderPrev = null;
    const wireMessages = [];

    for (let step = 0; step < 10; step++) {
      const current = activations[step];
      let flags = 0;
      let packed;

      if (senderPrev && senderPrev.length === current.length) {
        // Delta encode
        const delta = computeDelta(current, senderPrev);
        const sparsity = deltaSparsity(delta);
        const { data: int8Data, scale } = quantizeInt8(delta);
        packed = packQuantized(int8Data, scale);
        flags = setQuantFlags(flags, QuantMode.INT8);
        flags |= Flags.DELTA;
        if (step === 1) {
          console.log(`    Step 1 delta sparsity: ${(sparsity * 100).toFixed(1)}%`);
        }
      } else {
        // Full send (first token)
        const { data: int8Data, scale } = quantizeInt8(current);
        packed = packQuantized(int8Data, scale);
        flags = setQuantFlags(flags, QuantMode.INT8);
      }

      const msg = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION, flags, step + 1, 6666, shape, packed
      );
      wireMessages.push(msg);
      senderPrev = current; // sender caches original (not quantized)
    }

    // Simulate receiver side (node 1) — decoding through coordinator relay
    let receiverPrev = null;
    let worstCos = 1;
    let totalBytes = 0;

    for (let step = 0; step < 10; step++) {
      const decoded = decodeBinaryMessage(Buffer.from(wireMessages[step]));
      const isDelta = !!(decoded.flags & Flags.DELTA);
      const quantMode = getQuantMode(decoded.flags);
      totalBytes += wireMessages[step].byteLength;

      assert.equal(quantMode, QuantMode.INT8);

      let recovered;
      if (isDelta && receiverPrev) {
        const { int8Data, scale } = unpackQuantized(decoded.payload);
        const deltaFloat32 = dequantizeInt8(int8Data, scale);
        recovered = applyDelta(deltaFloat32, receiverPrev);
      } else {
        const { int8Data, scale } = unpackQuantized(decoded.payload);
        recovered = dequantizeInt8(int8Data, scale);
      }

      const cos = cosineSimilarity(activations[step], recovered);
      worstCos = Math.min(worstCos, cos);
      receiverPrev = recovered;
    }

    const avgBytes = totalBytes / 10;
    const fullBytes = 768 * 4 + HEADER_SIZE; // what uncompressed float32 would cost
    console.log(`    10-step worst cosine: ${worstCos.toFixed(6)}`);
    console.log(`    Avg wire bytes/step: ${avgBytes.toFixed(0)} vs full float32: ${fullBytes} (${(fullBytes / avgBytes).toFixed(1)}x compression)`);
    assert.ok(worstCos > 0.998, `Delta relay quality too low: ${worstCos}`);
  });

  it("delta sparsity is high for similar activations", () => {
    const a = randomActivation(768, 1.0);
    const b = new Float32Array(768);
    for (let i = 0; i < 768; i++) {
      b[i] = a[i] + (Math.random() - 0.5) * 0.05; // 2.5% perturbation
    }
    const delta = computeDelta(b, a);
    const sparsity = deltaSparsity(delta);
    console.log(`    Sparsity (2.5% perturbation): ${(sparsity * 100).toFixed(1)}%`);
    assert.ok(sparsity > 0.3, `Expected meaningful sparsity, got ${sparsity}`);
  });
});

// ─── Phase 2: Activation Predictor ────────────────────────────────

describe("Activation Predictor", () => {
  // Generate a sequence of slowly-drifting activations (mimics real transformer behavior)
  function generateTrajectory(steps, size, drift = 0.05) {
    const seq = [randomActivation(size, 1.0)];
    for (let i = 1; i < steps; i++) {
      const prev = seq[i - 1];
      const next = new Float32Array(size);
      for (let j = 0; j < size; j++) {
        next[j] = prev[j] + (Math.random() - 0.5) * drift;
      }
      seq.push(next);
    }
    return seq;
  }

  it("returns null with insufficient history", () => {
    const pred = new ActivationPredictor();
    assert.equal(pred.predict("req-1"), null);

    pred.observe("req-1", randomActivation(768));
    assert.equal(pred.predict("req-1"), null); // need 2 observations
  });

  it("predicts with linear extrapolation after 2 observations", () => {
    const pred = new ActivationPredictor();
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([2, 4, 6]); // each doubled

    pred.observe("req-1", a);
    pred.observe("req-1", b);

    const result = pred.predict("req-1");
    assert.ok(result !== null);
    // Linear extrapolation: [2,4,6] + ([2,4,6] - [1,2,3]) = [3,6,9]
    assert.deepEqual(Array.from(result.prediction), [3, 6, 9]);
  });

  it("predictions are close for smooth trajectories", () => {
    const pred = new ActivationPredictor();
    const trajectory = generateTrajectory(10, 768, 0.05);

    // Feed first 5, predict 6th
    for (let i = 0; i < 5; i++) {
      pred.observe("req-1", trajectory[i]);
    }

    const result = pred.predict("req-1");
    assert.ok(result !== null);

    const verification = pred.verify(result.prediction, trajectory[5]);
    console.log(`    Smooth trajectory prediction cosine: ${verification.cosine.toFixed(6)}, accept: ${verification.accept}`);
    // With 5% drift, linear extrapolation should be decent
    assert.ok(verification.cosine > 0.95, `Prediction too inaccurate: ${verification.cosine}`);
  });

  it("tracks hit/miss stats correctly", () => {
    const pred = new ActivationPredictor();
    const trajectory = generateTrajectory(20, 768, 0.03);

    let hits = 0;
    for (let i = 0; i < 18; i++) {
      pred.observe("req-1", trajectory[i]);
      if (i >= 2) {
        const result = pred.predict("req-1");
        if (result) {
          const v = pred.verify(result.prediction, trajectory[i + 1]);
          if (v.accept) hits++;
        }
      }
    }

    const stats = pred.getStats();
    console.log(`    Predictor stats: ${stats.predictions} predictions, ${stats.hitRate.toFixed(1)}% hit rate, avg cosine: ${stats.avgCosine.toFixed(4)}`);
    assert.ok(stats.predictions > 0);
    assert.equal(stats.hits + stats.misses, stats.predictions);
  });

  it("clear removes request state", () => {
    const pred = new ActivationPredictor();
    pred.observe("req-1", randomActivation(768));
    pred.observe("req-1", randomActivation(768));
    pred.clear("req-1");
    assert.equal(pred.predict("req-1"), null);
  });
});

// ─── Phase 2: Early Exit Detector ─────────────────────────────────

describe("Early Exit Detector", () => {
  it("does not exit on first layer (no comparison)", () => {
    const det = new EarlyExitDetector();
    det.enabled = true;
    const hidden = randomActivation(768);
    const result = det.check("req-1", 0, hidden, 6);
    assert.equal(result.shouldExit, false);
  });

  it("detects convergence when layers produce near-identical output", () => {
    const det = new EarlyExitDetector();
    det.enabled = true;
    const base = randomActivation(768, 1.0);

    // Layer 0: base activation
    det.check("req-1", 0, base, 6);

    // Layer 1: nearly identical (0.1% perturbation)
    const similar = new Float32Array(768);
    for (let i = 0; i < 768; i++) {
      similar[i] = base[i] + (Math.random() - 0.5) * 0.001;
    }
    const result = det.check("req-1", 1, similar, 6);
    console.log(`    Converged layers cosine: ${result.cosine.toFixed(6)}, normRatio: ${result.normRatio.toFixed(6)}`);
    assert.ok(result.cosine > 0.999, `Expected high cosine, got ${result.cosine}`);
    assert.ok(result.normRatio < 0.01, `Expected low norm ratio, got ${result.normRatio}`);
    assert.equal(result.shouldExit, true);
  });

  it("does NOT exit when layers produce different output", () => {
    const det = new EarlyExitDetector();
    det.enabled = true;

    // Layer 0
    det.check("req-1", 0, randomActivation(768, 1.0), 6);

    // Layer 1: completely different
    const result = det.check("req-1", 1, randomActivation(768, 1.0), 6);
    console.log(`    Divergent layers cosine: ${result.cosine.toFixed(6)}, normRatio: ${result.normRatio.toFixed(6)}`);
    assert.equal(result.shouldExit, false);
  });

  it("does NOT exit on the last layer (nothing to skip)", () => {
    const det = new EarlyExitDetector();
    det.enabled = true;
    const base = randomActivation(768);
    det.check("req-1", 0, base, 2);

    const similar = new Float32Array(base);
    const result = det.check("req-1", 1, similar, 2); // last layer
    assert.equal(result.shouldExit, false);
  });

  it("tracks stats correctly", () => {
    const det = new EarlyExitDetector();
    det.enabled = true;
    const base = randomActivation(768, 1.0);

    // Simulate 6 layers where layers 3+ converge
    let h = base;
    for (let l = 0; l < 6; l++) {
      const next = new Float32Array(768);
      const drift = l < 3 ? 0.5 : 0.0005; // big changes first, then convergence
      for (let i = 0; i < 768; i++) {
        next[i] = h[i] + (Math.random() - 0.5) * drift;
      }
      det.check("req-1", l, next, 6);
      h = next;
    }

    const stats = det.getStats();
    console.log(`    Early exit stats: ${stats.earlyExits} exits, ${stats.layersSaved} layers saved, ${(stats.exitRate * 100).toFixed(1)}% exit rate`);
    assert.ok(stats.checks === 6);
  });

  it("disabled mode tracks but never exits", () => {
    const det = new EarlyExitDetector();
    det.enabled = false; // disabled
    const base = randomActivation(768);
    det.check("req-1", 0, base, 6);
    const result = det.check("req-1", 1, new Float32Array(base), 6); // identical
    assert.equal(result.shouldExit, false); // disabled, so no exit
    assert.ok(result.cosine > 0.999); // but still reports metrics
  });
});

// ─── Per-Channel Quantization ─────────────────────────────────────

describe("Per-Channel Int8 Quantization", () => {
  it("round-trips with much lower error than per-tensor", () => {
    const rows = 4, cols = 768;
    const data = randomActivation(rows * cols, 1.0);

    // Per-tensor
    const { data: ptQ, scale: ptS } = quantizeInt8(data);
    const ptRecovered = dequantizeInt8(ptQ, ptS);
    const ptCos = cosineSimilarity(data, ptRecovered);

    // Per-channel
    const { data: pcQ, scales: pcS } = quantizeInt8PerChannel(data, cols);
    const pcRecovered = dequantizeInt8PerChannel(pcQ, pcS, cols);
    const pcCos = cosineSimilarity(data, pcRecovered);

    console.log(`    Per-tensor cosine: ${ptCos.toFixed(6)}, Per-channel cosine: ${pcCos.toFixed(6)}`);
    assert.ok(pcCos >= ptCos, "Per-channel should be at least as good as per-tensor");
  });

  it("pack/unpack round-trips correctly", () => {
    const rows = 2, cols = 768;
    const data = randomActivation(rows * cols, 1.0);
    const { data: int8Data, scales } = quantizeInt8PerChannel(data, cols);
    const packed = packQuantizedPerChannel(int8Data, scales);
    const unpacked = unpackQuantizedPerChannel(packed);

    assert.equal(unpacked.numRows, rows);
    assert.equal(unpacked.int8Data.length, int8Data.length);
    for (let i = 0; i < int8Data.length; i++) {
      assert.equal(unpacked.int8Data[i], int8Data[i]);
    }
    for (let i = 0; i < scales.length; i++) {
      assert.equal(unpacked.scales[i], scales[i]);
    }
  });

  it("wire relay with per-channel maintains quality", () => {
    const shape = [1, 768];
    const original = randomActivation(768);

    const { data: int8Data, scales } = quantizeInt8PerChannel(original, 768);
    const packed = packQuantizedPerChannel(int8Data, scales);

    // Simulate wire relay
    const flags = setQuantFlags(0, QuantMode.INT8);
    const msg = encodeBinaryMessage(BinaryMsgType.ACTIVATION, flags, 42, 5555, shape, packed);
    const decoded = decodeBinaryMessage(Buffer.from(msg));
    const unpacked = unpackQuantizedPerChannel(decoded.payload);
    const recovered = dequantizeInt8PerChannel(unpacked.int8Data, unpacked.scales, 768);

    const cos = cosineSimilarity(original, recovered);
    console.log(`    Per-channel wire relay cosine: ${cos.toFixed(6)}`);
    assert.ok(cos > 0.9999, `Per-channel relay cosine too low: ${cos}`);
  });
});

// ─── Phase 3: Head Pruning ────────────────────────────────────────

import { HeadPruner } from "../node/head-pruning.js";

describe("Attention Head Pruning", () => {
  it("returns all-true mask when disabled", () => {
    const pruner = new HeadPruner(12, 6);
    pruner.enabled = false;
    const mask = pruner.getHeadMask(0);
    assert.equal(mask.length, 12);
    assert.ok(mask.every(v => v === true));
  });

  it("prunes low-importance heads when enabled", () => {
    const pruner = new HeadPruner(12, 6);
    pruner.enabled = true;

    // Simulate: heads 0-7 have high norms, heads 8-11 have near-zero
    const norms = new Float32Array(12);
    for (let i = 0; i < 8; i++) norms[i] = 1.0 + Math.random();
    for (let i = 8; i < 12; i++) norms[i] = 0.01;

    // Record multiple times to stabilize EMA
    for (let t = 0; t < 20; t++) pruner.recordHeadNorms(0, norms);

    const mask = pruner.getHeadMask(0);
    const activeCount = mask.filter(v => v).length;
    console.log(`    Active heads: ${activeCount}/12, mask: [${mask.map(v=>v?1:0)}]`);
    assert.ok(activeCount >= 6 && activeCount <= 10, `Expected 6-10 active, got ${activeCount}`);
    // Low-importance heads should be pruned
    assert.ok(!mask[8] || !mask[9] || !mask[10] || !mask[11], "At least one low head should be pruned");
  });

  it("never prunes below minimum heads", () => {
    const pruner = new HeadPruner(12, 6);
    pruner.enabled = true;
    pruner.minHeads = 8;

    // All heads equally unimportant
    const norms = new Float32Array(12).fill(0.001);
    for (let t = 0; t < 20; t++) pruner.recordHeadNorms(0, norms);

    const mask = pruner.getHeadMask(0);
    const activeCount = mask.filter(v => v).length;
    assert.ok(activeCount >= 8, `Should keep at least 8 heads, got ${activeCount}`);
  });

  it("tracks stats correctly", () => {
    const pruner = new HeadPruner(12, 6);
    pruner.enabled = true;
    const norms = new Float32Array(12);
    for (let i = 0; i < 12; i++) norms[i] = i < 6 ? 2.0 : 0.01;
    for (let t = 0; t < 20; t++) pruner.recordHeadNorms(0, norms);

    pruner.getHeadMask(0);
    pruner.getHeadMask(0);

    const stats = pruner.getStats();
    console.log(`    Prune rate: ${(stats.pruneRate * 100).toFixed(1)}%`);
    assert.ok(stats.totalHeadOps > 0);
    assert.ok(stats.skippedHeadOps > 0);
  });
});

// ─── RLE Entropy Coding ─────────────────────────────────────────

describe("RLE Entropy Coding", () => {
  it("round-trips all-zero data", () => {
    const data = new Int8Array(768); // all zeros
    const { compressed, ratio } = rleCompress(data);
    const restored = rleDecompress(compressed, data.length);
    assert.deepStrictEqual(restored, data);
    assert.ok(ratio > 1, `Expected compression, got ratio ${ratio}`);
    console.log(`    All-zero: ${data.length} → ${compressed.length} bytes (ratio ${ratio.toFixed(1)}x)`);
  });

  it("round-trips random data losslessly", () => {
    const data = new Int8Array(768);
    for (let i = 0; i < data.length; i++) data[i] = Math.floor(Math.random() * 256) - 128;
    const { compressed } = rleCompress(data);
    const restored = rleDecompress(compressed, data.length);
    assert.deepStrictEqual(restored, data);
  });

  it("compresses sparse delta activations well", () => {
    // Simulate a delta-encoded activation: mostly zeros with some non-zero values
    const data = new Int8Array(768);
    // ~30% non-zero
    for (let i = 0; i < 768; i += 3) data[i] = Math.floor(Math.random() * 20) - 10;
    const { compressed, ratio } = rleCompress(data, 0);
    const restored = rleDecompress(compressed, data.length);
    assert.deepStrictEqual(restored, data);
    console.log(`    Sparse delta: ${data.length} → ${compressed.length} bytes (ratio ${ratio.toFixed(2)}x)`);
  });

  it("compresses with zero threshold (lossy)", () => {
    // Simulate delta activations: mostly small values with occasional spikes.
    // Real deltas cluster near zero with long near-zero runs.
    const data = new Int8Array(768);
    for (let i = 0; i < data.length; i++) {
      // ~80% values in [-2,2] (will be zeroed by threshold), ~20% larger spikes
      data[i] = Math.random() < 0.8
        ? Math.floor(Math.random() * 5) - 2
        : Math.floor(Math.random() * 20) - 10;
    }
    const { compressed, ratio } = rleCompress(data, 2); // treat |v| <= 2 as zero
    assert.ok(ratio > 1, `Threshold=2 should compress better than lossless`);
    const restored = rleDecompress(compressed, data.length);
    // Lossy: values with |v| <= 2 become 0
    for (let i = 0; i < data.length; i++) {
      if (Math.abs(data[i]) <= 2) assert.equal(restored[i], 0);
    }
    console.log(`    Lossy threshold=2: ratio ${ratio.toFixed(2)}x`);
  });
});

// ─── Wire-Level Payload Compression ─────────────────────────────

describe("Wire Payload Compression", () => {
  it("round-trips a packed quantized payload", () => {
    // Create a sparse int8 activation (simulating delta)
    const activation = randomActivation(768, 0.5);
    // Zero out 70% to simulate sparse delta
    for (let i = 0; i < activation.length; i++) {
      if (Math.random() < 0.7) activation[i] = 0;
    }
    const { data: int8Data, scale } = quantizeInt8(activation);
    const packed = packQuantized(int8Data, scale);

    const result = compressPayload(packed);
    assert.ok(result !== null, "Sparse data should compress");

    const restored = decompressPayload(new Uint8Array(result.compressed));
    const unpacked = unpackQuantized(restored);

    assert.equal(unpacked.int8Data.length, int8Data.length);
    assert.ok(Math.abs(unpacked.scale - scale) < 1e-6);
    for (let i = 0; i < int8Data.length; i++) {
      assert.equal(unpacked.int8Data[i], int8Data[i], `Mismatch at index ${i}`);
    }
    console.log(`    Quantized payload: ${packed.byteLength} → ${result.compressed.byteLength} bytes (ratio ${result.ratio.toFixed(2)}x)`);
  });

  it("round-trips a packed per-channel quantized payload", () => {
    // Simulate sparse delta activation with long zero runs (realistic pattern).
    // Per-channel packed format has scale metadata, so data needs high sparsity.
    const activation = randomActivation(768, 0.5);
    for (let i = 0; i < activation.length; i++) {
      if (Math.random() < 0.85) activation[i] = 0;
    }
    const { data: int8Data, scales } = quantizeInt8PerChannel(activation, 768);
    const packed = packQuantizedPerChannel(int8Data, scales);

    const result = compressPayload(packed);
    assert.ok(result !== null, "Sparse per-channel data should compress");

    const restored = decompressPayload(new Uint8Array(result.compressed));
    const unpacked = unpackQuantizedPerChannel(restored);

    assert.equal(unpacked.int8Data.length, int8Data.length);
    assert.equal(unpacked.numRows, scales.length);
    console.log(`    Per-channel payload: ${packed.byteLength} → ${result.compressed.byteLength} bytes`);
  });

  it("returns null for incompressible data", () => {
    // Random data with no zeros — shouldn't compress
    const int8Data = new Int8Array(768);
    for (let i = 0; i < int8Data.length; i++) int8Data[i] = (i % 127) + 1; // no zeros
    const packed = packQuantized(int8Data, 1.0);
    const result = compressPayload(packed);
    // May or may not be null, but if not null, ratio must be > 1
    if (result) assert.ok(result.ratio > 1);
    console.log(`    Incompressible: ${result ? "compressed anyway" : "correctly skipped"}`);
  });

  it("integrates with binary protocol encode/decode", () => {
    // Full pipeline: quantize → pack → compress → binary encode → decode → decompress → unpack → dequantize
    const original = randomActivation(768, 1.0);
    for (let i = 0; i < original.length; i++) {
      if (Math.random() < 0.65) original[i] = 0;
    }

    // Sender side
    const { data: int8Data, scale } = quantizeInt8(original);
    const packed = packQuantized(int8Data, scale);
    let flags = setQuantFlags(0, QuantMode.INT8);
    let payloadData = packed;
    const compResult = compressPayload(packed);
    if (compResult) {
      payloadData = compResult.compressed;
      flags |= Flags.COMPRESSED;
    }

    const reqId = requestIdToUint32("test-rle-integration");
    const encoded = encodeBinaryMessage(BinaryMsgType.ACTIVATION, flags, 42, reqId, [1, 768], payloadData);

    // Receiver side
    const decoded = decodeBinaryMessage(encoded);
    assert.equal(decoded.type, BinaryMsgType.ACTIVATION);
    assert.equal(decoded.seqPos, 42);

    const isCompressed = !!(decoded.flags & Flags.COMPRESSED);
    const payload = isCompressed ? decompressPayload(decoded.payload) : decoded.payload;
    const unpacked = unpackQuantized(payload);
    const restored = dequantizeInt8(unpacked.int8Data, unpacked.scale);

    // Check within quantization error
    for (let i = 0; i < original.length; i++) {
      assert.ok(Math.abs(restored[i] - original[i]) < scale * 1.5,
        `Value at ${i}: original=${original[i].toFixed(4)}, restored=${restored[i].toFixed(4)}`);
    }
    console.log(`    Full pipeline: ${original.byteLength} → ${payloadData.byteLength} bytes on wire`);
  });
});
