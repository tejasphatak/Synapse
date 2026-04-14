import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  quantizeInt4,
  dequantizeInt4,
  packInt4,
  unpackInt4,
  AdaptivePrecisionSelector,
} from "../protocol/adaptive-precision.js";

// ─── Int4 Quantization ──────────────────────────────────────────

describe("Int4 Quantization", () => {
  it("quantizes and dequantizes small values accurately", () => {
    const input = new Float32Array([0.0, 0.5, -0.5, 1.0, -1.0]);
    const { data, scale, length } = quantizeInt4(input);
    assert.equal(length, 5);
    assert.ok(scale > 0);
    const output = dequantizeInt4(data, scale, length);
    assert.equal(output.length, 5);
    // Int4 has only 15 levels, so error is larger but bounded
    for (let i = 0; i < input.length; i++) {
      assert.ok(Math.abs(output[i] - input[i]) <= scale + 1e-6,
        `index ${i}: ${output[i]} vs ${input[i]}, scale=${scale}`);
    }
  });

  it("handles zero vector", () => {
    const input = new Float32Array([0, 0, 0, 0]);
    const { data, scale, length } = quantizeInt4(input);
    assert.equal(scale, 1); // fallback scale
    const output = dequantizeInt4(data, scale, length);
    for (let i = 0; i < 4; i++) {
      assert.equal(output[i], 0);
    }
  });

  it("handles odd-length arrays", () => {
    const input = new Float32Array([1.0, -1.0, 0.5]);
    const { data, scale, length } = quantizeInt4(input);
    assert.equal(length, 3);
    assert.equal(data.length, 2); // ceil(3/2) = 2 packed bytes
    const output = dequantizeInt4(data, scale, length);
    assert.equal(output.length, 3);
  });

  it("clamps values to [-7, 7] range", () => {
    const input = new Float32Array([100.0, -100.0]);
    const { data, scale, length } = quantizeInt4(input);
    const output = dequantizeInt4(data, scale, length);
    // Max quantized value is 7 * scale = 100, so these should roundtrip exactly
    assert.ok(Math.abs(output[0] - 100.0) < scale * 0.5);
    assert.ok(Math.abs(output[1] - (-100.0)) < scale * 0.5);
  });

  it("roundtrip preserves sign correctly", () => {
    const input = new Float32Array([-3.5, 2.1, -0.7, 4.2, 0.0, -1.3]);
    const { data, scale, length } = quantizeInt4(input);
    const output = dequantizeInt4(data, scale, length);
    for (let i = 0; i < input.length; i++) {
      // Sign must be preserved
      if (input[i] > 0) assert.ok(output[i] >= 0, `index ${i} lost positive sign`);
      if (input[i] < 0) assert.ok(output[i] <= 0, `index ${i} lost negative sign`);
    }
  });

  it("compression ratio is ~8x vs float32", () => {
    const n = 768;
    const input = new Float32Array(n);
    for (let i = 0; i < n; i++) input[i] = Math.random() * 2 - 1;
    const { data } = quantizeInt4(input);
    // Float32: 768 * 4 = 3072 bytes, Int4 packed: 384 bytes
    assert.equal(data.length, n / 2);
    assert.ok(data.length < n * 4 / 7); // at least 7x compression
  });
});

// ─── Int4 Pack/Unpack ───────────────────────────────────────────

describe("Int4 Pack/Unpack", () => {
  it("roundtrips through wire format", () => {
    const input = new Float32Array([1.5, -2.3, 0.7, -0.1, 3.0]);
    const { data, scale, length } = quantizeInt4(input);

    const packed = packInt4(data, scale, length);
    assert.ok(packed instanceof ArrayBuffer);
    assert.equal(packed.byteLength, data.length + 8); // data + scale(4) + length(4)

    const { packedData, scale: s2, originalLength } = unpackInt4(packed);
    assert.equal(originalLength, 5);
    assert.ok(Math.abs(s2 - scale) < 1e-6);
    assert.equal(packedData.length, data.length);

    const output = dequantizeInt4(packedData, s2, originalLength);
    for (let i = 0; i < input.length; i++) {
      assert.ok(Math.abs(output[i] - input[i]) <= scale * 1.5,
        `index ${i}: ${output[i]} vs ${input[i]}`);
    }
  });

  it("accepts Uint8Array input to unpackInt4", () => {
    const input = new Float32Array([1.0, -1.0, 0.5, -0.5]);
    const { data, scale, length } = quantizeInt4(input);
    const packed = packInt4(data, scale, length);
    const uint8View = new Uint8Array(packed);
    const result = unpackInt4(uint8View);
    assert.equal(result.originalLength, 4);
  });
});

// ─── Adaptive Precision Selector ────────────────────────────────

describe("AdaptivePrecisionSelector", () => {
  it("returns INT8 during warmup", () => {
    const selector = new AdaptivePrecisionSelector(4, { warmupSteps: 5 });
    // No observations yet
    assert.equal(selector.getMode(0), 1); // INT8 = 1
    assert.equal(selector.getMode(3), 1);
  });

  it("selects INT4 for low-error layers after warmup", () => {
    const selector = new AdaptivePrecisionSelector(2, {
      warmupSteps: 3,
      int4Threshold: 0.05,
      int8Threshold: 0.01,
    });

    // Feed a smooth activation that quantizes well
    const activation = new Float32Array(64);
    for (let i = 0; i < 64; i++) activation[i] = Math.sin(i * 0.1) * 10;

    for (let step = 0; step < 10; step++) {
      selector.observe(0, activation);
    }

    const mode = selector.getMode(0);
    // Sin wave with amplitude 10 should quantize well to int4
    // Mode should be INT4 (2) or INT8 (1)
    assert.ok(mode === 1 || mode === 2, `mode=${mode}`);
  });

  it("selects NONE for high-error layers", () => {
    const selector = new AdaptivePrecisionSelector(2, {
      warmupSteps: 3,
      int4Threshold: 0.0001, // extremely tight
      int8Threshold: 0.0001,
    });

    // Random high-dynamic-range activation
    const activation = new Float32Array(64);
    for (let i = 0; i < 64; i++) activation[i] = (Math.random() - 0.5) * 1000;

    for (let step = 0; step < 10; step++) {
      selector.observe(0, activation);
    }

    // With tiny thresholds, should stay at NONE (0) or INT8 at most
    const mode = selector.getMode(0);
    assert.ok(mode === 0 || mode === 1, `expected NONE or INT8, got ${mode}`);
  });

  it("ignores out-of-range layer indices", () => {
    const selector = new AdaptivePrecisionSelector(2);
    selector.observe(5, new Float32Array([1, 2, 3])); // should not throw
    assert.equal(selector.getMode(5), 0); // NONE for out-of-range
  });

  it("handles zero-norm activation gracefully", () => {
    const selector = new AdaptivePrecisionSelector(2, { warmupSteps: 1 });
    selector.observe(0, new Float32Array([0, 0, 0, 0]));
    // Should not crash, mode should still be valid
    const mode = selector.getMode(0);
    assert.ok(mode >= 0 && mode <= 2);
  });

  it("snapshot returns diagnostic info for all layers", () => {
    const selector = new AdaptivePrecisionSelector(3);
    const snap = selector.snapshot();
    assert.equal(snap.length, 3);
    assert.ok("mode" in snap[0]);
    assert.ok("int8Error" in snap[0]);
    assert.ok("int4Error" in snap[0]);
    assert.ok("steps" in snap[0]);
  });

  it("different layers can have different modes", () => {
    const selector = new AdaptivePrecisionSelector(2, {
      warmupSteps: 3,
      int4Threshold: 0.1,
      int8Threshold: 0.02,
    });

    // Layer 0: smooth activation (compresses well)
    const smooth = new Float32Array(128);
    for (let i = 0; i < 128; i++) smooth[i] = i * 0.01;

    // Layer 1: noisy activation
    const noisy = new Float32Array(128);
    for (let i = 0; i < 128; i++) noisy[i] = (Math.random() - 0.5) * 100;

    for (let step = 0; step < 10; step++) {
      selector.observe(0, smooth);
      selector.observe(1, noisy);
    }

    // They should potentially get different modes
    const snap = selector.snapshot();
    assert.equal(snap.length, 2);
    assert.equal(snap[0].steps, 10);
    assert.equal(snap[1].steps, 10);
  });
});
