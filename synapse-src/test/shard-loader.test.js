/**
 * Shard Loader Tests — Dequantization, manifest parsing, tensor lookup
 *
 * Tests the pure-logic portions of ShardLoader without WebGPU or IndexedDB.
 * Focus: float16→float32 conversion, int8/int4 dequantization, model config,
 * and tensor metadata operations.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { ShardLoader } from "../node/shard-loader.js";

// ─── Mock WebGPU Device ────────────────────────────────────────

function mockDevice() {
  return {
    createBuffer({ label, size, usage, mappedAtCreation }) {
      const storage = new ArrayBuffer(size);
      return {
        label,
        size,
        usage,
        _data: storage,
        _mapped: mappedAtCreation,
        getMappedRange() { return storage; },
        unmap() { this._mapped = false; },
        destroy() { this._destroyed = true; },
        _destroyed: false,
      };
    },
  };
}

// GPUBufferUsage constants (not available in Node)
globalThis.GPUBufferUsage = globalThis.GPUBufferUsage || {
  STORAGE: 0x80,
  COPY_SRC: 0x04,
  COPY_DST: 0x08,
};

// ─── Helper: build an ArrayBuffer with encoded tensor data ─────

function buildFloat32Buffer(values) {
  return new Float32Array(values).buffer;
}

function buildFloat16Buffer(f32Values) {
  const buf = new ArrayBuffer(f32Values.length * 2);
  const u16 = new Uint16Array(buf);
  for (let i = 0; i < f32Values.length; i++) {
    u16[i] = float32ToFloat16(f32Values[i]);
  }
  return buf;
}

/** Encode a float32 as float16 (uint16). Inverse of ShardLoader._float16ToFloat32. */
function float32ToFloat16(val) {
  const f32 = new Float32Array([val]);
  const u32 = new Uint32Array(f32.buffer);
  const bits = u32[0];

  const sign = (bits >> 31) & 1;
  const exp = (bits >> 23) & 0xFF;
  const mant = bits & 0x7FFFFF;

  if (exp === 0xFF) {
    // Inf or NaN
    return (sign << 15) | (0x1F << 10) | (mant ? 0x200 : 0);
  }
  if (exp === 0) {
    // Zero or subnormal — becomes zero in float16
    return (sign << 15);
  }

  let newExp = exp - 127 + 15;
  if (newExp >= 0x1F) return (sign << 15) | (0x1F << 10); // overflow → Inf
  if (newExp <= 0) return (sign << 15); // underflow → zero

  return (sign << 15) | (newExp << 10) | (mant >> 13);
}

function buildInt8Buffer(int8Values, scale) {
  const scaleOffset = int8Values.length;
  const totalSize = scaleOffset + 4;
  const buf = new ArrayBuffer(totalSize);
  const i8view = new Int8Array(buf, 0, scaleOffset);
  i8view.set(int8Values);
  new DataView(buf).setFloat32(scaleOffset, scale, true);
  return { buf, scaleOffset, totalSize };
}

function buildInt4Buffer(values, groupSize) {
  // Pack pairs of int4 values into bytes
  const numElements = values.length;
  const packedSize = Math.ceil(numElements / 2);
  const numGroups = Math.ceil(numElements / groupSize);

  // Compute per-group scales
  const scales = new Float32Array(numGroups);
  for (let g = 0; g < numGroups; g++) {
    let maxAbs = 0;
    for (let i = g * groupSize; i < Math.min((g + 1) * groupSize, numElements); i++) {
      maxAbs = Math.max(maxAbs, Math.abs(values[i]));
    }
    scales[g] = maxAbs > 0 ? maxAbs / 7 : 1.0;
  }

  // Quantize to int4 and pack
  const packed = new Uint8Array(packedSize);
  for (let i = 0; i < packedSize; i++) {
    const idx0 = i * 2;
    const idx1 = i * 2 + 1;
    const g0 = Math.floor(idx0 / groupSize);
    const g1 = idx1 < numElements ? Math.floor(idx1 / groupSize) : 0;

    let q0 = Math.round(values[idx0] / scales[g0]);
    q0 = Math.max(-8, Math.min(7, q0));
    if (q0 < 0) q0 += 16; // 4-bit unsigned representation

    let q1 = 0;
    if (idx1 < numElements) {
      q1 = Math.round(values[idx1] / scales[g1]);
      q1 = Math.max(-8, Math.min(7, q1));
      if (q1 < 0) q1 += 16;
    }

    packed[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
  }

  const scaleSizeBytes = numGroups * 4;
  // Align scales offset to 4 bytes for Float32Array
  const scalesStart = Math.ceil(packedSize / 4) * 4;
  const totalSize = scalesStart + scaleSizeBytes;
  const buf = new ArrayBuffer(totalSize);
  new Uint8Array(buf, 0, packedSize).set(packed);
  new Float32Array(buf, scalesStart, numGroups).set(scales);

  return {
    buf,
    packedSize,
    scalesOffset: scalesStart,
    scalesSize: scaleSizeBytes,
    totalSize,
    numElements,
    groupSize,
    scales,
  };
}

// ─── float16 → float32 Conversion ──────────────────────────────

describe("ShardLoader._float16ToFloat32", () => {
  let loader;

  beforeEach(() => {
    loader = new ShardLoader(mockDevice());
  });

  it("converts positive zero", () => {
    assert.equal(loader._float16ToFloat32(0x0000), 0);
  });

  it("converts negative zero", () => {
    assert.equal(loader._float16ToFloat32(0x8000), -0);
    assert.ok(Object.is(loader._float16ToFloat32(0x8000), -0));
  });

  it("converts 1.0", () => {
    // float16: sign=0, exp=15, mantissa=0 → 0 01111 0000000000 = 0x3C00
    assert.equal(loader._float16ToFloat32(0x3C00), 1.0);
  });

  it("converts -1.0", () => {
    // float16: sign=1, exp=15, mantissa=0 → 1 01111 0000000000 = 0xBC00
    assert.equal(loader._float16ToFloat32(0xBC00), -1.0);
  });

  it("converts 0.5", () => {
    // float16: sign=0, exp=14, mantissa=0 → 0 01110 0000000000 = 0x3800
    assert.equal(loader._float16ToFloat32(0x3800), 0.5);
  });

  it("converts 2.0", () => {
    // float16: sign=0, exp=16, mantissa=0 → 0 10000 0000000000 = 0x4000
    assert.equal(loader._float16ToFloat32(0x4000), 2.0);
  });

  it("converts positive infinity", () => {
    // float16: sign=0, exp=31, mantissa=0 → 0 11111 0000000000 = 0x7C00
    assert.equal(loader._float16ToFloat32(0x7C00), Infinity);
  });

  it("converts negative infinity", () => {
    // float16: sign=1, exp=31, mantissa=0 → 1 11111 0000000000 = 0xFC00
    assert.equal(loader._float16ToFloat32(0xFC00), -Infinity);
  });

  it("converts NaN", () => {
    // float16: exp=31, mantissa!=0 → NaN
    assert.ok(Number.isNaN(loader._float16ToFloat32(0x7E00)));
  });

  it("converts subnormal float16", () => {
    // float16: sign=0, exp=0, mantissa=1 → smallest subnormal
    // value = (1/1024) * 2^(-14) ≈ 5.96e-8
    const val = loader._float16ToFloat32(0x0001);
    assert.ok(val > 0);
    assert.ok(val < 1e-6);
    assert.ok(Math.abs(val - 5.960464477539063e-8) < 1e-15);
  });

  it("converts negative subnormal", () => {
    const val = loader._float16ToFloat32(0x8001);
    assert.ok(val < 0);
    assert.ok(val > -1e-6);
  });

  it("converts 65504 (max finite float16)", () => {
    // 0 11110 1111111111 = 0x7BFF
    const val = loader._float16ToFloat32(0x7BFF);
    assert.equal(val, 65504);
  });

  it("roundtrips common values through encode/decode", () => {
    const testValues = [0, 1, -1, 0.5, -0.5, 2, 10, 100, -100, 0.125];
    for (const v of testValues) {
      const encoded = float32ToFloat16(v);
      const decoded = loader._float16ToFloat32(encoded);
      assert.ok(
        Math.abs(decoded - v) < Math.abs(v) * 0.002 + 1e-6,
        `Roundtrip failed for ${v}: got ${decoded}`
      );
    }
  });
});

// ─── _dequantizeTensor ──────────────────────────────────────────

describe("ShardLoader._dequantizeTensor", () => {
  let loader;

  beforeEach(() => {
    loader = new ShardLoader(mockDevice());
  });

  describe("float32", () => {
    it("returns Float32Array view of source data", () => {
      const values = [1.0, 2.0, 3.0, 4.0];
      const buf = buildFloat32Buffer(values);
      const entry = { dtype: "float32", offset: 0, size: 16 };

      const result = loader._dequantizeTensor(buf, entry);
      assert.ok(result instanceof Float32Array);
      assert.equal(result.length, 4);
      assert.deepEqual(Array.from(result), values);
    });

    it("handles non-zero offset", () => {
      // 8 bytes of padding + 4 floats
      const buf = new ArrayBuffer(8 + 16);
      new Float32Array(buf, 8, 4).set([5, 6, 7, 8]);
      const entry = { dtype: "float32", offset: 8, size: 16 };

      const result = loader._dequantizeTensor(buf, entry);
      assert.deepEqual(Array.from(result), [5, 6, 7, 8]);
    });
  });

  describe("float16", () => {
    it("dequantizes float16 to float32", () => {
      const originalValues = [1.0, -1.0, 0.5, 2.0];
      const buf = buildFloat16Buffer(originalValues);
      const entry = { dtype: "float16", offset: 0, size: 8 };

      const result = loader._dequantizeTensor(buf, entry);
      assert.ok(result instanceof Float32Array);
      assert.equal(result.length, 4);

      for (let i = 0; i < originalValues.length; i++) {
        assert.ok(
          Math.abs(result[i] - originalValues[i]) < 0.01,
          `float16 dequant [${i}]: expected ${originalValues[i]}, got ${result[i]}`
        );
      }
    });

    it("dequantizes float16 zero", () => {
      const buf = buildFloat16Buffer([0]);
      const entry = { dtype: "float16", offset: 0, size: 2 };
      const result = loader._dequantizeTensor(buf, entry);
      assert.equal(result[0], 0);
    });
  });

  describe("int8", () => {
    it("dequantizes int8 with scale", () => {
      const int8Vals = new Int8Array([10, -10, 127, -127, 0]);
      const scale = 0.01;
      const { buf, scaleOffset, totalSize } = buildInt8Buffer(int8Vals, scale);
      const entry = {
        dtype: "int8",
        offset: 0,
        size: totalSize,
        quant: { scale_offset: scaleOffset },
      };

      const result = loader._dequantizeTensor(buf, entry);
      assert.ok(result instanceof Float32Array);
      assert.equal(result.length, 5);
      assert.ok(Math.abs(result[0] - 0.10) < 1e-6);
      assert.ok(Math.abs(result[1] - (-0.10)) < 1e-6);
      assert.ok(Math.abs(result[2] - 1.27) < 1e-6);
      assert.ok(Math.abs(result[3] - (-1.27)) < 1e-6);
      assert.equal(result[4], 0);
    });

    it("dequantizes with scale=1.0", () => {
      const int8Vals = new Int8Array([1, 2, 3]);
      const { buf, scaleOffset, totalSize } = buildInt8Buffer(int8Vals, 1.0);
      const entry = {
        dtype: "int8",
        offset: 0,
        size: totalSize,
        quant: { scale_offset: scaleOffset },
      };

      const result = loader._dequantizeTensor(buf, entry);
      assert.deepEqual(Array.from(result), [1, 2, 3]);
    });

    it("handles non-zero buffer offset", () => {
      const int8Vals = new Int8Array([50, -50]);
      const scale = 0.1;
      const padding = 16;

      const scaleOffset = int8Vals.length;
      const dataSize = scaleOffset + 4;
      const buf = new ArrayBuffer(padding + dataSize);
      new Int8Array(buf, padding, scaleOffset).set(int8Vals);
      new DataView(buf).setFloat32(padding + scaleOffset, scale, true);

      const entry = {
        dtype: "int8",
        offset: padding,
        size: dataSize,
        quant: { scale_offset: scaleOffset },
      };

      const result = loader._dequantizeTensor(buf, entry);
      assert.ok(Math.abs(result[0] - 5.0) < 1e-5);
      assert.ok(Math.abs(result[1] - (-5.0)) < 1e-5);
    });
  });

  describe("int4", () => {
    it("dequantizes simple int4 values", () => {
      // Pack two values: 3 and -2, with groupSize=4, scale should handle them
      const values = [3, -2, 7, -7];
      const groupSize = 4;
      const info = buildInt4Buffer(values, groupSize);

      const entry = {
        dtype: "int4",
        offset: 0,
        size: info.totalSize,
        quant: {
          packed_size: info.packedSize,
          scales_offset: info.scalesOffset,
          scales_size: info.scalesSize,
          original_numel: info.numElements,
          group_size: groupSize,
        },
      };

      const result = loader._dequantizeTensor(info.buf, entry);
      assert.ok(result instanceof Float32Array);
      assert.equal(result.length, 4);

      // Verify values are approximately correct (int4 is lossy)
      for (let i = 0; i < values.length; i++) {
        assert.ok(
          Math.abs(result[i] - values[i]) < Math.abs(values[i]) * 0.5 + 1,
          `int4 dequant [${i}]: expected ~${values[i]}, got ${result[i]}`
        );
      }
    });

    it("handles odd number of elements", () => {
      const values = [1, -1, 3];
      const groupSize = 4;
      const info = buildInt4Buffer(values, groupSize);

      const entry = {
        dtype: "int4",
        offset: 0,
        size: info.totalSize,
        quant: {
          packed_size: info.packedSize,
          scales_offset: info.scalesOffset,
          scales_size: info.scalesSize,
          original_numel: 3,
          group_size: groupSize,
        },
      };

      const result = loader._dequantizeTensor(info.buf, entry);
      assert.equal(result.length, 3);
    });

    it("handles multiple groups", () => {
      const values = [1, 2, 3, 4, 5, 6, 7, -7];
      const groupSize = 4;
      const info = buildInt4Buffer(values, groupSize);

      const entry = {
        dtype: "int4",
        offset: 0,
        size: info.totalSize,
        quant: {
          packed_size: info.packedSize,
          scales_offset: info.scalesOffset,
          scales_size: info.scalesSize,
          original_numel: 8,
          group_size: groupSize,
        },
      };

      const result = loader._dequantizeTensor(info.buf, entry);
      assert.equal(result.length, 8);
    });
  });

  describe("unknown dtype", () => {
    it("falls back to float32", () => {
      const values = [1.5, 2.5];
      const buf = buildFloat32Buffer(values);
      const entry = { dtype: "bfloat16", offset: 0, size: 8 };

      const result = loader._dequantizeTensor(buf, entry);
      assert.deepEqual(Array.from(result), values);
    });
  });
});

// ─── getModelConfig ─────────────────────────────────────────────

describe("ShardLoader.getModelConfig", () => {
  let loader;

  beforeEach(() => {
    loader = new ShardLoader(mockDevice());
  });

  it("returns null when no manifest loaded", () => {
    assert.equal(loader.getModelConfig(), null);
  });

  it("returns correct config from manifest", () => {
    loader.manifest = {
      model: "gpt2",
      arch: "gpt2",
      num_layers: 12,
      hidden_size: 768,
      num_heads: 12,
      head_dim: 64,
      vocab_size: 50257,
      max_seq_len: 1024,
      dtype: "float16",
      num_shards: 2,
    };

    const config = loader.getModelConfig();
    assert.equal(config.model, "gpt2");
    assert.equal(config.arch, "gpt2");
    assert.equal(config.numLayers, 12);
    assert.equal(config.hiddenSize, 768);
    assert.equal(config.numHeads, 12);
    assert.equal(config.headDim, 64);
    assert.equal(config.vocabSize, 50257);
    assert.equal(config.maxSeqLen, 1024);
    assert.equal(config.dtype, "float16");
    assert.equal(config.numShards, 2);
  });

  it("provides defaults for optional fields", () => {
    loader.manifest = {
      model: "gpt2",
      num_layers: 12,
      hidden_size: 768,
      num_heads: 12,
      head_dim: 64,
      vocab_size: 50257,
      max_seq_len: 1024,
    };

    const config = loader.getModelConfig();
    assert.equal(config.arch, "gpt2");       // default
    assert.equal(config.dtype, "float32");   // default
    assert.equal(config.numShards, 2);       // default
  });
});

// ─── Tensor Lookup Methods ──────────────────────────────────────

describe("ShardLoader tensor lookups", () => {
  let loader;

  beforeEach(() => {
    loader = new ShardLoader(mockDevice());
    // Simulate loaded tensors
    loader.buffers.set("layer.0.attn.weight", { id: "buf0" });
    loader.buffers.set("layer.0.attn.bias", { id: "buf1" });
    loader.buffers.set("layer.0.mlp.weight", { id: "buf2" });
    loader.buffers.set("layer.1.attn.weight", { id: "buf3" });
    loader.buffers.set("embed.weight", { id: "buf4" });

    loader.metadata.set("layer.0.attn.weight", { shape: [768, 768], dtype: "float16" });
    loader.metadata.set("layer.0.attn.bias", { shape: [768], dtype: "float16" });
    loader.metadata.set("layer.0.mlp.weight", { shape: [3072, 768], dtype: "float16" });
    loader.metadata.set("layer.1.attn.weight", { shape: [768, 768], dtype: "float16" });
    loader.metadata.set("embed.weight", { shape: [50257, 768], dtype: "float16" });
  });

  describe("getBuffer", () => {
    it("returns buffer for known tensor", () => {
      const buf = loader.getBuffer("layer.0.attn.weight");
      assert.deepEqual(buf, { id: "buf0" });
    });

    it("returns null for unknown tensor", () => {
      assert.equal(loader.getBuffer("nonexistent"), null);
    });
  });

  describe("getTensorMeta", () => {
    it("returns metadata for known tensor", () => {
      const meta = loader.getTensorMeta("layer.0.attn.weight");
      assert.deepEqual(meta.shape, [768, 768]);
      assert.equal(meta.dtype, "float16");
    });

    it("returns null for unknown tensor", () => {
      assert.equal(loader.getTensorMeta("nonexistent"), null);
    });
  });

  describe("getTensorsByPrefix", () => {
    it("finds all tensors matching prefix", () => {
      const results = loader.getTensorsByPrefix("layer.0.");
      assert.equal(results.length, 3);
      const names = results.map((r) => r.name).sort();
      assert.deepEqual(names, [
        "layer.0.attn.bias",
        "layer.0.attn.weight",
        "layer.0.mlp.weight",
      ]);
    });

    it("returns empty for no matches", () => {
      const results = loader.getTensorsByPrefix("layer.5.");
      assert.deepEqual(results, []);
    });

    it("includes buffer and meta in results", () => {
      const results = loader.getTensorsByPrefix("embed.");
      assert.equal(results.length, 1);
      assert.deepEqual(results[0].buffer, { id: "buf4" });
      assert.deepEqual(results[0].meta.shape, [50257, 768]);
    });

    it("matches all tensors with empty prefix", () => {
      const results = loader.getTensorsByPrefix("");
      assert.equal(results.length, 5);
    });
  });
});

// ─── GPU Buffer Creation ────────────────────────────────────────

describe("ShardLoader._createGPUBuffer", () => {
  let loader;

  beforeEach(() => {
    loader = new ShardLoader(mockDevice());
  });

  it("creates buffer with correct size", () => {
    const data = new Float32Array([1, 2, 3, 4]);
    const buf = loader._createGPUBuffer("test", data, [1, 4]);
    assert.equal(buf.size, 16);
    assert.equal(buf.label, "test");
  });

  it("copies data into mapped buffer", () => {
    const data = new Float32Array([1.5, 2.5, 3.5]);
    const buf = loader._createGPUBuffer("test", data, [1, 3]);
    const stored = new Float32Array(buf._data);
    assert.deepEqual(Array.from(stored), [1.5, 2.5, 3.5]);
  });

  it("unmaps buffer after creation", () => {
    const data = new Float32Array([1]);
    const buf = loader._createGPUBuffer("test", data, [1, 1]);
    assert.equal(buf._mapped, false);
  });
});

// ─── Destroy ────────────────────────────────────────────────────

describe("ShardLoader.destroy", () => {
  it("destroys all GPU buffers and clears maps", () => {
    const loader = new ShardLoader(mockDevice());
    const data = new Float32Array([1]);

    const buf1 = loader._createGPUBuffer("a", data, [1, 1]);
    const buf2 = loader._createGPUBuffer("b", data, [1, 1]);
    loader.buffers.set("a", buf1);
    loader.buffers.set("b", buf2);
    loader.metadata.set("a", { shape: [1, 1] });
    loader.metadata.set("b", { shape: [1, 1] });

    loader.destroy();
    assert.equal(loader.buffers.size, 0);
    assert.equal(loader.metadata.size, 0);
    assert.equal(buf1._destroyed, true);
    assert.equal(buf2._destroyed, true);
  });

  it("handles destroy on empty loader", () => {
    const loader = new ShardLoader(mockDevice());
    loader.destroy(); // should not throw
    assert.equal(loader.buffers.size, 0);
  });
});
