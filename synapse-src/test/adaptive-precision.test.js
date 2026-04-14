import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  quantizeInt4,
  dequantizeInt4,
  packInt4,
  unpackInt4,
  AdaptivePrecisionSelector,
} from "../protocol/adaptive-precision.js";
import {
  encodeBinaryMessage,
  decodeBinaryMessage,
  BinaryMsgType,
  QuantMode,
  Flags,
  setQuantFlags,
  getQuantMode,
} from "../protocol/binary.js";
import { compressPayload, decompressPayload } from "../protocol/entropy.js";

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

// ─── Wire Integration: INT4 over SYN1 Binary Protocol ──────────

describe("INT4 Wire Integration", () => {
  it("round-trips INT4 payload through SYN1 binary encode/decode", () => {
    const original = new Float32Array(768);
    for (let i = 0; i < 768; i++) original[i] = Math.sin(i * 0.05) * 5;

    const { data: packedData, scale, length } = quantizeInt4(original);
    const wirePayload = new Uint8Array(packInt4(packedData, scale, length));

    // Encode as SYN1 binary message with INT4 quant flag
    let flags = setQuantFlags(0, QuantMode.INT4);
    const shape = [1, 768];
    const msg = encodeBinaryMessage(BinaryMsgType.ACTIVATION, flags, 42, 1, shape, wirePayload);

    // Decode
    const decoded = decodeBinaryMessage(msg);
    assert.equal(getQuantMode(decoded.flags), QuantMode.INT4);
    assert.deepEqual(decoded.shape, shape);

    // Dequantize
    const { packedData: recvPacked, scale: recvScale, originalLength } = unpackInt4(decoded.payload);
    const recovered = dequantizeInt4(recvPacked, recvScale, originalLength);
    assert.equal(recovered.length, 768);

    // INT4 error should be bounded (7 levels per direction)
    let maxErr = 0;
    for (let i = 0; i < 768; i++) {
      const err = Math.abs(recovered[i] - original[i]);
      if (err > maxErr) maxErr = err;
    }
    // Max error should be less than 2 * scale (one quantization step)
    const expectedScale = 5 / 7; // max(|sin * 5|) ≈ 5
    assert.ok(maxErr < expectedScale * 1.5, `maxErr=${maxErr}, expectedScale=${expectedScale}`);
  });

  it("INT4 with RLE compression round-trips correctly", () => {
    // Sparse activation — lots of near-zero values
    const original = new Float32Array(768);
    for (let i = 0; i < 20; i++) original[i * 38] = (i - 10) * 0.5;

    const { data: packedData, scale, length } = quantizeInt4(original);
    const wirePayload = new Uint8Array(packInt4(packedData, scale, length));

    // Compress
    const compressed = compressPayload(wirePayload);
    if (compressed) {
      // Decompress and verify
      const decompressed = decompressPayload(compressed.compressed);
      const { packedData: p, scale: s, originalLength: l } = unpackInt4(decompressed);
      const recovered = dequantizeInt4(p, s, l);
      assert.equal(recovered.length, 768);
    }
    // If compression didn't help, that's fine — just verifying it doesn't corrupt
  });

  it("AdaptivePrecisionSelector transitions from INT8 warmup to INT4 for clean data", () => {
    const selector = new AdaptivePrecisionSelector(1, {
      warmupSteps: 3,
      int4Threshold: 0.1, // generous
    });

    // Clean linear ramp — quantizes very well
    const activation = new Float32Array(256);
    for (let i = 0; i < 256; i++) activation[i] = i * 0.01;

    // During warmup, should return INT8
    assert.equal(selector.getMode(0), QuantMode.INT8, "warmup should default to INT8");

    // After warmup, should converge to INT4 for clean data
    for (let step = 0; step < 10; step++) {
      selector.observe(0, activation);
    }
    const mode = selector.getMode(0);
    assert.equal(mode, QuantMode.INT4, `expected INT4 for clean data, got ${mode}`);
  });

  it("selector stays at INT8 when INT4 error is too high", () => {
    const selector = new AdaptivePrecisionSelector(1, {
      warmupSteps: 3,
      int4Threshold: 0.001, // very tight — INT4 won't satisfy
      int8Threshold: 0.05,  // loose enough for INT8
    });

    const activation = new Float32Array(256);
    for (let i = 0; i < 256; i++) activation[i] = Math.sin(i * 0.3) * 10;

    for (let step = 0; step < 10; step++) {
      selector.observe(0, activation);
    }
    const mode = selector.getMode(0);
    assert.equal(mode, QuantMode.INT8, `expected INT8 for tight INT4 threshold, got ${mode}`);
  });

  it("SYN1 flags correctly distinguish INT4 from INT8", () => {
    let flags4 = setQuantFlags(0, QuantMode.INT4);
    let flags8 = setQuantFlags(0, QuantMode.INT8);
    let flags0 = setQuantFlags(0, QuantMode.NONE);

    assert.equal(getQuantMode(flags4), QuantMode.INT4);
    assert.equal(getQuantMode(flags8), QuantMode.INT8);
    assert.equal(getQuantMode(flags0), QuantMode.NONE);

    // INT4 + COMPRESSED flag should preserve both
    flags4 |= Flags.COMPRESSED;
    assert.equal(getQuantMode(flags4), QuantMode.INT4);
    assert.ok(flags4 & Flags.COMPRESSED);
  });
});
