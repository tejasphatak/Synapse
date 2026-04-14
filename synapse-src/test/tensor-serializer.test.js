/**
 * TensorSerializer Tests — JSON, binary, int8, int4, and delta
 * serialization round-trips with mocked GPU device.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";

// Mock WebGPU globals
globalThis.GPUBufferUsage = { STORAGE: 0x80, COPY_SRC: 0x04, COPY_DST: 0x08 };

if (typeof globalThis.btoa === "undefined") {
  globalThis.btoa = (str) => Buffer.from(str, "binary").toString("base64");
  globalThis.atob = (b64) => Buffer.from(b64, "base64").toString("binary");
}

import { TensorSerializer } from "../node/tensor-serializer.js";

// ─── Mock GPU ──────────────────────────────────────────────

function makeMockDevice() {
  const writtenBuffers = [];
  return {
    writtenBuffers,
    queue: {
      writeBuffer(buf, offset, data) {
        buf._data = new Uint8Array(data.buffer || data, data.byteOffset || 0, data.byteLength);
        writtenBuffers.push({ buf, offset, data });
      },
    },
  };
}

function makeSerializer(device) {
  const createBuffer = (label, size, usage) => {
    const buf = { label, size, usage, _data: null };
    return buf;
  };
  const readBuffer = async (buffer, offset, size) => {
    // Return the data that was written to this buffer, or zeros
    if (buffer._data) {
      return buffer._data.buffer.slice(
        buffer._data.byteOffset,
        buffer._data.byteOffset + buffer._data.byteLength
      );
    }
    return new ArrayBuffer(size);
  };
  return new TensorSerializer(device, createBuffer, readBuffer);
}

function makeTensor(float32Array, shape) {
  const buf = { _data: new Uint8Array(float32Array.buffer, float32Array.byteOffset, float32Array.byteLength) };
  return { buffer: buf, shape };
}

function meanAbsError(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

// ─── JSON (base64) ─────────────────────────────────────────

describe("TensorSerializer — JSON base64", () => {
  let device, serializer;

  beforeEach(() => {
    device = makeMockDevice();
    serializer = makeSerializer(device);
  });

  it("round-trips float32 data through base64 JSON", async () => {
    const original = new Float32Array([1.0, -2.5, 3.14, 0.0]);
    const tensor = makeTensor(original, [1, 4]);

    const msg = await serializer.serialize(tensor);
    assert.deepStrictEqual(msg.shape, [1, 4]);
    assert.equal(msg.dtype, "float32");
    assert.equal(typeof msg.data, "string"); // base64

    const result = serializer.deserialize(msg);
    assert.deepStrictEqual(result.shape, [1, 4]);
    // Verify data was uploaded to GPU
    const uploaded = new Float32Array(
      result.buffer._data.buffer,
      result.buffer._data.byteOffset,
      result.buffer._data.byteLength / 4
    );
    assert.deepStrictEqual(Array.from(uploaded), [1.0, -2.5, Math.fround(3.14), 0.0]);
  });

  it("handles empty-ish single element", async () => {
    const original = new Float32Array([42.0]);
    const tensor = makeTensor(original, [1, 1]);
    const msg = await serializer.serialize(tensor);
    const result = serializer.deserialize(msg);
    const uploaded = new Float32Array(
      result.buffer._data.buffer,
      result.buffer._data.byteOffset,
      result.buffer._data.byteLength / 4
    );
    assert.equal(uploaded[0], 42.0);
  });
});

// ─── Binary ────────────────────────────────────────────────

describe("TensorSerializer — Binary", () => {
  let device, serializer;

  beforeEach(() => {
    device = makeMockDevice();
    serializer = makeSerializer(device);
  });

  it("round-trips float32 data through raw binary", async () => {
    const original = new Float32Array([1.0, -2.5, 3.14, 0.0, 100.0]);
    const tensor = makeTensor(original, [1, 5]);

    const msg = await serializer.serializeBinary(tensor);
    assert.deepStrictEqual(msg.shape, [1, 5]);
    assert.ok(msg.data instanceof ArrayBuffer);

    const payload = new Uint8Array(msg.data);
    const result = serializer.deserializeBinary(payload, msg.shape);
    const uploaded = new Float32Array(
      result.buffer._data.buffer,
      result.buffer._data.byteOffset,
      result.buffer._data.byteLength / 4
    );
    assert.deepStrictEqual(Array.from(uploaded), [1.0, -2.5, Math.fround(3.14), 0.0, 100.0]);
  });

  it("binary is smaller than base64 JSON", async () => {
    const original = new Float32Array(256);
    for (let i = 0; i < 256; i++) original[i] = Math.random() * 10 - 5;
    const tensor = makeTensor(original, [1, 256]);

    const jsonMsg = await serializer.serialize(tensor);
    const binMsg = await serializer.serializeBinary(tensor);

    const jsonSize = JSON.stringify(jsonMsg).length;
    const binSize = binMsg.data.byteLength;
    assert.ok(binSize < jsonSize, `binary (${binSize}) should be smaller than JSON (${jsonSize})`);
  });
});

// ─── Int8 Quantized ────────────────────────────────────────

describe("TensorSerializer — Int8 Quantized", () => {
  let device, serializer;

  beforeEach(() => {
    device = makeMockDevice();
    serializer = makeSerializer(device);
  });

  it("round-trips single-token (per-tensor) with acceptable error", async () => {
    const original = new Float32Array(768);
    for (let i = 0; i < 768; i++) original[i] = Math.random() * 4 - 2;
    const tensor = makeTensor(original, [1, 768]);

    const msg = await serializer.serializeQuantized(tensor);
    assert.deepStrictEqual(msg.shape, [1, 768]);
    assert.ok(!msg.fallbackUnquantized);

    const payload = msg.data instanceof Uint8Array ? msg.data : new Uint8Array(msg.data);
    const result = serializer.deserializeQuantized(payload, msg.shape);
    const uploaded = new Float32Array(
      result.buffer._data.buffer,
      result.buffer._data.byteOffset,
      result.buffer._data.byteLength / 4
    );
    assert.equal(uploaded.length, 768);
    const mae = meanAbsError(original, uploaded);
    assert.ok(mae < 0.1, `Mean abs error ${mae} too high for int8`);
  });

  it("round-trips multi-token (per-channel) with acceptable error", async () => {
    const rows = 4, cols = 768;
    const original = new Float32Array(rows * cols);
    for (let i = 0; i < original.length; i++) original[i] = Math.random() * 6 - 3;
    const tensor = makeTensor(original, [rows, cols]);

    const msg = await serializer.serializeQuantized(tensor);
    assert.deepStrictEqual(msg.shape, [rows, cols]);

    const payload = msg.data instanceof Uint8Array ? msg.data : new Uint8Array(msg.data);
    const result = serializer.deserializeQuantized(payload, msg.shape);
    const uploaded = new Float32Array(
      result.buffer._data.buffer,
      result.buffer._data.byteOffset,
      result.buffer._data.byteLength / 4
    );
    assert.equal(uploaded.length, rows * cols);
    const mae = meanAbsError(original, uploaded);
    assert.ok(mae < 0.1, `Mean abs error ${mae} too high for per-channel int8`);
  });

  it("falls back to unquantized on NaN input", async () => {
    const original = new Float32Array([1.0, NaN, 3.0, 4.0]);
    const tensor = makeTensor(original, [1, 4]);

    const msg = await serializer.serializeQuantized(tensor);
    assert.ok(msg.fallbackUnquantized, "Should flag fallback on NaN");
  });

  it("falls back to unquantized on Infinity input", async () => {
    const original = new Float32Array([1.0, Infinity, 3.0, 4.0]);
    const tensor = makeTensor(original, [1, 4]);

    const msg = await serializer.serializeQuantized(tensor);
    assert.ok(msg.fallbackUnquantized, "Should flag fallback on Infinity");
  });

  it("int8 data is ~4x smaller than float32", async () => {
    const original = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) original[i] = Math.random() * 2 - 1;
    const tensor = makeTensor(original, [1, 1024]);

    const msg = await serializer.serializeQuantized(tensor);
    const quantizedSize = msg.data.byteLength;
    const float32Size = 1024 * 4;
    const ratio = float32Size / quantizedSize;
    assert.ok(ratio > 3.5, `Compression ratio ${ratio.toFixed(2)} should be ~4x`);
  });
});

// ─── Int4 Quantized ────────────────────────────────────────

describe("TensorSerializer — Int4 Quantized", () => {
  let device, serializer;

  beforeEach(() => {
    device = makeMockDevice();
    serializer = makeSerializer(device);
  });

  it("round-trips through int4 with bounded error", async () => {
    const original = new Float32Array(768);
    for (let i = 0; i < 768; i++) original[i] = Math.random() * 2 - 1;
    const tensor = makeTensor(original, [1, 768]);

    const msg = await serializer.serializeInt4(tensor);
    assert.deepStrictEqual(msg.shape, [1, 768]);
    assert.ok(!msg.fallbackUnquantized);

    const payload = msg.data instanceof Uint8Array ? msg.data : new Uint8Array(msg.data);
    const result = serializer.deserializeInt4(payload, msg.shape);
    const uploaded = new Float32Array(
      result.buffer._data.buffer,
      result.buffer._data.byteOffset,
      result.buffer._data.byteLength / 4
    );
    assert.equal(uploaded.length, 768);
    // Int4 has higher error than int8 — 16 levels vs 256
    const mae = meanAbsError(original, uploaded);
    assert.ok(mae < 0.3, `Mean abs error ${mae} too high for int4`);
  });

  it("falls back to unquantized on NaN input", async () => {
    const original = new Float32Array([NaN, 1.0, 2.0, 3.0]);
    const tensor = makeTensor(original, [1, 4]);

    const msg = await serializer.serializeInt4(tensor);
    assert.ok(msg.fallbackUnquantized, "Should flag fallback on NaN");
  });

  it("int4 is ~8x smaller than float32", async () => {
    const original = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) original[i] = Math.random() * 2 - 1;
    const tensor = makeTensor(original, [1, 1024]);

    const msg = await serializer.serializeInt4(tensor);
    const packedSize = msg.data.byteLength;
    const float32Size = 1024 * 4;
    const ratio = float32Size / packedSize;
    assert.ok(ratio > 6, `Compression ratio ${ratio.toFixed(2)} should be ~8x`);
  });
});

// ─── Delta-Encoded ─────────────────────────────────────────

describe("TensorSerializer — Delta Encoding", () => {
  let device, serializer;

  beforeEach(() => {
    device = makeMockDevice();
    serializer = makeSerializer(device);
  });

  it("sends full tensor when no previous activation", async () => {
    const original = new Float32Array(768);
    for (let i = 0; i < 768; i++) original[i] = Math.random() * 2 - 1;
    const tensor = makeTensor(original, [1, 768]);

    const msg = await serializer.serializeDelta(tensor, null);
    assert.equal(msg.isDelta, false);
    assert.equal(msg.sparsity, 0);
    assert.ok(msg.currentFloat32 instanceof Float32Array);
    assert.equal(msg.currentFloat32.length, 768);
  });

  it("sends full tensor when shape mismatch", async () => {
    const original = new Float32Array(768);
    for (let i = 0; i < 768; i++) original[i] = Math.random() * 2 - 1;
    const tensor = makeTensor(original, [1, 768]);
    const previousWrongSize = new Float32Array(512);

    const msg = await serializer.serializeDelta(tensor, previousWrongSize);
    assert.equal(msg.isDelta, false);
  });

  it("sends delta when previous activation matches", async () => {
    const previous = new Float32Array(768);
    const current = new Float32Array(768);
    for (let i = 0; i < 768; i++) {
      previous[i] = Math.random() * 2 - 1;
      current[i] = previous[i] + (Math.random() * 0.1 - 0.05); // small perturbation
    }
    const tensor = makeTensor(current, [1, 768]);

    const msg = await serializer.serializeDelta(tensor, previous);
    assert.equal(msg.isDelta, true);
    assert.ok(msg.sparsity >= 0 && msg.sparsity <= 1);
    assert.ok(msg.currentFloat32 instanceof Float32Array);
  });

  it("delta round-trip reconstructs with bounded error", async () => {
    const previous = new Float32Array(768);
    const current = new Float32Array(768);
    for (let i = 0; i < 768; i++) {
      previous[i] = Math.random() * 4 - 2;
      current[i] = previous[i] + (Math.random() * 0.2 - 0.1);
    }
    const tensor = makeTensor(current, [1, 768]);

    const msg = await serializer.serializeDelta(tensor, previous);
    assert.equal(msg.isDelta, true);

    // Deserialize delta and apply to previous
    const payload = msg.data instanceof Uint8Array ? msg.data : new Uint8Array(msg.data);
    const result = serializer.deserializeDeltaApply(payload, msg.shape, previous);
    assert.ok(result.currentFloat32 instanceof Float32Array);
    assert.equal(result.currentFloat32.length, 768);

    const mae = meanAbsError(current, result.currentFloat32);
    assert.ok(mae < 0.1, `Delta round-trip MAE ${mae} too high`);
  });

  it("high sparsity when activations are identical", async () => {
    const data = new Float32Array(768);
    for (let i = 0; i < 768; i++) data[i] = Math.random() * 2 - 1;
    const tensor = makeTensor(new Float32Array(data), [1, 768]);

    const msg = await serializer.serializeDelta(tensor, data);
    assert.equal(msg.isDelta, true);
    // Identical activations should yield very high sparsity (all zeros delta)
    assert.ok(msg.sparsity > 0.9, `Sparsity ${msg.sparsity} should be >0.9 for identical data`);
  });
});

// ─── Constructor ───────────────────────────────────────────

describe("TensorSerializer — Construction", () => {
  it("stores device, createBuffer, and readBuffer", () => {
    const device = makeMockDevice();
    const createBuffer = () => {};
    const readBuffer = async () => new ArrayBuffer(0);
    const s = new TensorSerializer(device, createBuffer, readBuffer);
    assert.equal(s.device, device);
    assert.equal(s._createBuffer, createBuffer);
    assert.equal(s._readBuffer, readBuffer);
  });
});
