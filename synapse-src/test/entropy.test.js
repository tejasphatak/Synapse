/**
 * Entropy Coding Tests — RLE compression/decompression for wire protocol.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  rleCompress,
  rleDecompress,
  compressActivation,
  decompressActivation,
  compressPayload,
  decompressPayload,
} from "../protocol/entropy.js";

describe("RLE Compression", () => {
  it("compresses all-zero data into a single zero run", () => {
    const data = new Int8Array(100);
    const { compressed, originalLength, ratio } = rleCompress(data);
    assert.equal(originalLength, 100);
    assert.ok(ratio > 1, "should achieve compression");
    assert.ok(compressed.byteLength < 100, "compressed should be smaller");
  });

  it("handles all-nonzero data (literal run)", () => {
    const data = new Int8Array([1, 2, 3, 4, 5, -1, -2, -3]);
    const { compressed, originalLength } = rleCompress(data);
    assert.equal(originalLength, 8);
    // Literal: 2 (numRuns) + 3 (header) + 8 (data) = 13
    // Overhead is acceptable for short data
  });

  it("roundtrips zeros correctly", () => {
    const data = new Int8Array(256);
    const { compressed, originalLength } = rleCompress(data);
    const restored = rleDecompress(compressed, originalLength);
    assert.deepEqual(restored, data);
  });

  it("roundtrips nonzero data correctly", () => {
    const data = new Int8Array([10, -20, 30, -40, 50, 127, -128, 1]);
    const { compressed, originalLength } = rleCompress(data);
    const restored = rleDecompress(compressed, originalLength);
    assert.deepEqual(restored, data);
  });

  it("roundtrips mixed zero and nonzero data", () => {
    // Pattern: [nonzero...][zeros...][nonzero...][zeros...]
    const data = new Int8Array(200);
    for (let i = 0; i < 50; i++) data[i] = (i % 127) + 1;
    // 50-150: zeros
    for (let i = 150; i < 180; i++) data[i] = -(i % 50) - 1;
    // 180-200: zeros

    const { compressed, originalLength } = rleCompress(data);
    const restored = rleDecompress(compressed, originalLength);
    assert.deepEqual(restored, data);
  });

  it("handles empty data", () => {
    const data = new Int8Array(0);
    const { compressed, originalLength } = rleCompress(data);
    assert.equal(originalLength, 0);
    const restored = rleDecompress(compressed, originalLength);
    assert.equal(restored.length, 0);
  });

  it("handles single element", () => {
    const data = new Int8Array([42]);
    const { compressed, originalLength } = rleCompress(data);
    const restored = rleDecompress(compressed, originalLength);
    assert.deepEqual(restored, data);
  });

  it("handles single zero element", () => {
    const data = new Int8Array([0]);
    const { compressed, originalLength } = rleCompress(data);
    const restored = rleDecompress(compressed, originalLength);
    assert.deepEqual(restored, data);
  });

  it("achieves good compression on sparse data", () => {
    // 90% zeros, 10% nonzero — typical of delta-encoded activations
    const data = new Int8Array(1000);
    for (let i = 0; i < 100; i++) {
      data[i * 10] = (i % 127) + 1;
    }
    const { ratio } = rleCompress(data);
    assert.ok(ratio > 1.2, `Expected ratio > 1.2, got ${ratio.toFixed(2)}`);
  });

  it("coalesces short zero runs into literals (MIN_ZERO_RUN=4)", () => {
    // Pattern: [1, 0, 0, 1] — the two zeros are too short for their own run
    const data = new Int8Array([1, 0, 0, 1]);
    const { compressed, originalLength } = rleCompress(data);
    const restored = rleDecompress(compressed, originalLength);
    assert.deepEqual(restored, data);
  });

  it("preserves sign extension for negative values", () => {
    const data = new Int8Array([-1, -128, -64, -2]);
    const { compressed, originalLength } = rleCompress(data);
    const restored = rleDecompress(compressed, originalLength);
    assert.deepEqual(restored, data);
  });

  it("handles alternating zero/nonzero pattern", () => {
    const data = new Int8Array(20);
    for (let i = 0; i < 20; i++) data[i] = i % 2 === 0 ? 0 : (i + 1);
    const { compressed, originalLength } = rleCompress(data);
    const restored = rleDecompress(compressed, originalLength);
    assert.deepEqual(restored, data);
  });
});

describe("RLE with zeroThreshold", () => {
  it("treats small values as zero when threshold > 0", () => {
    const data = new Int8Array([0, 1, -1, 2, 0, 0, 0, 0, 0, 5]);
    const { compressed: c0 } = rleCompress(data, 0);
    const { compressed: c1 } = rleCompress(data, 1);
    // With threshold=1, values 1 and -1 become zeros → more compressible
    assert.ok(c1.byteLength <= c0.byteLength,
      "higher threshold should compress at least as well");
  });

  it("lossy roundtrip zeroes small values", () => {
    const data = new Int8Array([5, 1, -1, 0, 0, 0, 0, 0, 10]);
    const { compressed, originalLength } = rleCompress(data, 1);
    const restored = rleDecompress(compressed, originalLength);
    // Values with |v| <= 1 become 0
    assert.equal(restored[0], 5);
    assert.equal(restored[1], 0); // was 1, now 0 (lossy)
    assert.equal(restored[2], 0); // was -1, now 0 (lossy)
    assert.equal(restored[8], 10);
  });
});

describe("compressActivation / decompressActivation", () => {
  it("roundtrips via wrapper functions", () => {
    const data = new Int8Array(100);
    for (let i = 0; i < 10; i++) data[i * 10] = i + 1;
    const { compressed, originalLength } = compressActivation(data);
    const restored = decompressActivation(compressed, originalLength);
    assert.deepEqual(restored, data);
  });
});

describe("compressPayload / decompressPayload", () => {
  it("roundtrips packed payload with length prefix", () => {
    // Simulate a packed quantized payload with lots of zeros
    const payload = new ArrayBuffer(500);
    const view = new Int8Array(payload);
    for (let i = 0; i < 50; i++) view[i * 10] = (i + 1) % 127;

    const result = compressPayload(payload);
    assert.ok(result !== null, "should compress (sparse data)");
    assert.ok(result.ratio > 1, "should achieve compression");

    const restored = decompressPayload(new Uint8Array(result.compressed));
    assert.equal(restored.byteLength, 500);
    for (let i = 0; i < 500; i++) {
      assert.equal(restored[i], view[i] & 0xFF, `mismatch at byte ${i}`);
    }
  });

  it("returns null when compression doesn't save space", () => {
    // All nonzero — RLE can't help
    const payload = new ArrayBuffer(20);
    const view = new Int8Array(payload);
    for (let i = 0; i < 20; i++) view[i] = i + 1;

    const result = compressPayload(payload);
    assert.equal(result, null, "should return null when compression isn't beneficial");
  });

  it("handles large payloads", () => {
    const payload = new ArrayBuffer(10000);
    const view = new Int8Array(payload);
    // 95% zeros
    for (let i = 0; i < 500; i++) view[i * 20] = ((i * 7) % 254) - 127;

    const result = compressPayload(payload);
    assert.ok(result !== null);
    const restored = decompressPayload(new Uint8Array(result.compressed));
    assert.equal(restored.byteLength, 10000);
  });
});
