/**
 * Pipeline Tests — Forward pass orchestration, KV cache management,
 * token sampling, serialization, and buffer utilities.
 *
 * GPU compute dispatch is mocked — these tests verify control flow,
 * state management, and CPU-side logic (sampling, quantization round-trips).
 */

import { describe, it, beforeEach, mock } from "node:test";
import assert from "node:assert/strict";

// Mock WebGPU globals before importing Pipeline
globalThis.GPUBufferUsage = { STORAGE: 0x80, COPY_SRC: 0x04, COPY_DST: 0x08, UNIFORM: 0x40, MAP_READ: 0x01, COPY_DST: 0x08 };
globalThis.GPUShaderStage = { COMPUTE: 0x04 };
globalThis.GPUMapMode = { READ: 0x01 };

// We need to mock fetch for shader loading in init()
globalThis.fetch = async (url) => ({
  text: async () => "// mock WGSL shader code",
});

// Mock btoa/atob for JSON serialization tests
if (typeof globalThis.btoa === "undefined") {
  globalThis.btoa = (str) => Buffer.from(str, "binary").toString("base64");
  globalThis.atob = (b64) => Buffer.from(b64, "base64").toString("binary");
}

import { Pipeline } from "../node/pipeline.js";

// ─── Mock GPU Device ─────────────────────────────────────────

function makeMockDevice() {
  const buffers = [];
  const copies = [];
  const submits = [];

  return {
    buffers,
    copies,
    submits,
    createBuffer(desc) {
      const buf = {
        label: desc.label || "unlabeled",
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() { this.destroyed = true; },
        _data: null,
        // For mapAsync mock
        async mapAsync() {},
        getMappedRange() {
          return buf._data || new ArrayBuffer(buf.size);
        },
        unmap() {},
      };
      buffers.push(buf);
      return buf;
    },
    createCommandEncoder() {
      const cmds = [];
      return {
        copyBufferToBuffer(src, srcOff, dst, dstOff, size) {
          copies.push({ src, srcOff, dst, dstOff, size });
        },
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups() {},
            end() {},
          };
        },
        finish() { return cmds; },
      };
    },
    createShaderModule(desc) {
      return { label: desc.label };
    },
    createBindGroupLayout(desc) {
      return { entries: desc.entries };
    },
    createPipelineLayout(desc) {
      return { bindGroupLayouts: desc.bindGroupLayouts };
    },
    createComputePipeline(desc) {
      return {
        label: desc.compute?.entryPoint,
        getBindGroupLayout: () => ({
          entries: [],
        }),
      };
    },
    createBindGroup(desc) {
      return { layout: desc.layout, entries: desc.entries };
    },
    queue: {
      submit(cmdBuffers) { submits.push(cmdBuffers); },
      writeBuffer() {},
    },
  };
}

function makeMockShardLoader(config) {
  const buffers = new Map();
  return {
    buffers,
    getModelConfig() {
      return {
        hiddenSize: 768,
        vocabSize: 50257,
        numHeads: 12,
        headDim: 64,
        maxSeqLen: 1024,
        numLayers: 12,
        ...config,
      };
    },
    getBuffer(name) {
      if (!buffers.has(name)) {
        buffers.set(name, { label: name, destroyed: false, destroy() { this.destroyed = true; } });
      }
      return buffers.get(name);
    },
  };
}

// ─── Constructor ──────────────────────────────────────────────

describe("Pipeline", () => {
  let device;
  let loader;
  let pipeline;

  beforeEach(() => {
    device = makeMockDevice();
    loader = makeMockShardLoader();
    pipeline = new Pipeline(device, loader);
  });

  describe("constructor", () => {
    it("sets device and loader references", () => {
      assert.equal(pipeline.device, device);
      assert.equal(pipeline.loader, loader);
    });

    it("starts uninitialized", () => {
      assert.equal(pipeline._initialized, false);
    });

    it("starts with empty pipeline cache", () => {
      assert.deepEqual(pipeline.pipelines, {});
    });

    it("starts with empty shader module cache", () => {
      assert.deepEqual(pipeline.shaderModules, {});
    });

    it("starts with empty temp buffer list", () => {
      assert.deepEqual(pipeline._tempBuffers, []);
    });

    it("starts with empty KV cache map", () => {
      assert.equal(pipeline.kvCaches.size, 0);
    });

    it("initializes EarlyExitDetector", () => {
      assert.ok(pipeline.earlyExit);
    });

    it("starts with null MoD router (lazy init)", () => {
      assert.equal(pipeline.modRouter, null);
    });

    it("starts with null HeadPruner (lazy init)", () => {
      assert.equal(pipeline.headPruner, null);
    });

    it("reads model config from shard loader", () => {
      assert.equal(pipeline.config.hiddenSize, 768);
      assert.equal(pipeline.config.vocabSize, 50257);
      assert.equal(pipeline.config.numHeads, 12);
      assert.equal(pipeline.config.headDim, 64);
    });
  });

  // ─── init() ───────────────────────────────────────────────────

  describe("init", () => {
    it("loads all shader modules", async () => {
      await pipeline.init();
      const expected = [
        "matmul", "matmul_transB", "attention", "attention_cached",
        "layernorm", "gelu", "residual_add", "embed", "bias_add",
        "head_slice", "head_concat",
      ];
      for (const name of expected) {
        assert.ok(pipeline.shaderModules[name], `shader ${name} should be loaded`);
        assert.equal(pipeline.shaderModules[name].label, name);
      }
    });

    it("sets _initialized to true after loading", async () => {
      await pipeline.init();
      assert.equal(pipeline._initialized, true);
    });

    it("fetches each shader with cache-busting param", async () => {
      const fetched = [];
      globalThis.fetch = async (url) => {
        fetched.push(url);
        return { text: async () => "// mock WGSL" };
      };
      await pipeline.init();
      assert.equal(fetched.length, 11);
      for (const url of fetched) {
        assert.match(url, /\/node\/kernels\/\w+\.wgsl\?v=\d+/);
      }
      // Restore
      globalThis.fetch = async () => ({ text: async () => "// mock" });
    });
  });

  // ─── KV Cache Management ────────────────────────────────────

  describe("getOrCreateKVCache", () => {
    it("creates a new KV cache for unknown request", () => {
      const cache = pipeline.getOrCreateKVCache("req-1", 0, 6);
      assert.ok(cache);
      assert.equal(pipeline.kvCaches.size, 1);
    });

    it("returns existing cache for same request", () => {
      const cache1 = pipeline.getOrCreateKVCache("req-1", 0, 6);
      const cache2 = pipeline.getOrCreateKVCache("req-1", 0, 6);
      assert.equal(cache1, cache2);
      assert.equal(pipeline.kvCaches.size, 1);
    });

    it("creates separate caches for different requests", () => {
      const cache1 = pipeline.getOrCreateKVCache("req-1", 0, 6);
      const cache2 = pipeline.getOrCreateKVCache("req-2", 6, 6);
      assert.notEqual(cache1, cache2);
      assert.equal(pipeline.kvCaches.size, 2);
    });
  });

  describe("clearCache", () => {
    it("removes KV cache for a request", () => {
      pipeline.getOrCreateKVCache("req-1", 0, 6);
      assert.equal(pipeline.kvCaches.size, 1);
      pipeline.clearCache("req-1");
      assert.equal(pipeline.kvCaches.size, 0);
    });

    it("is safe to call with unknown request ID", () => {
      assert.doesNotThrow(() => pipeline.clearCache("nonexistent"));
    });

    it("does not affect other caches", () => {
      pipeline.getOrCreateKVCache("req-1", 0, 6);
      pipeline.getOrCreateKVCache("req-2", 6, 6);
      pipeline.clearCache("req-1");
      assert.equal(pipeline.kvCaches.size, 1);
      assert.ok(pipeline.kvCaches.has("req-2"));
    });
  });

  // ─── sampleToken ────────────────────────────────────────────

  describe("sampleToken", () => {
    it("returns an integer token ID", async () => {
      // Create a logits tensor with known values
      const vocabSize = 10;
      const seqLen = 1;
      const customLoader = makeMockShardLoader({ vocabSize });
      const p = new Pipeline(device, customLoader);

      // Mock _readBuffer to return logits with a clear winner
      const logits = new Float32Array(vocabSize);
      logits[7] = 100.0; // token 7 should win by a huge margin
      p._readBuffer = async () => logits.buffer;

      const tensor = { buffer: {}, shape: [seqLen, vocabSize] };
      const token = await p.sampleToken(tensor, 1.0);
      assert.equal(typeof token, "number");
      assert.ok(Number.isInteger(token));
      assert.equal(token, 7);
    });

    it("respects temperature scaling", async () => {
      const vocabSize = 3;
      const customLoader = makeMockShardLoader({ vocabSize });
      const p = new Pipeline(device, customLoader);

      // With very low temperature, should always pick the max
      const logits = new Float32Array([1.0, 2.0, 0.5]);
      p._readBuffer = async () => logits.buffer;

      const tensor = { buffer: {}, shape: [1, vocabSize] };
      const token = await p.sampleToken(tensor, 0.01);
      assert.equal(token, 1); // index of max logit
    });

    it("reads from last position for multi-token input", async () => {
      const vocabSize = 5;
      const seqLen = 3;
      const customLoader = makeMockShardLoader({ vocabSize });
      const p = new Pipeline(device, customLoader);

      let readOffset = null;
      p._readBuffer = async (buf, offset, size) => {
        readOffset = offset;
        const logits = new Float32Array(vocabSize);
        logits[2] = 50.0;
        return logits.buffer;
      };

      const tensor = { buffer: {}, shape: [seqLen, vocabSize] };
      await p.sampleToken(tensor, 1.0);

      // Should read from offset = (seqLen - 1) * vocabSize * 4
      assert.equal(readOffset, (seqLen - 1) * vocabSize * 4);
    });

    it("handles uniform logits (returns valid token)", async () => {
      const vocabSize = 4;
      const customLoader = makeMockShardLoader({ vocabSize });
      const p = new Pipeline(device, customLoader);

      const logits = new Float32Array([0.0, 0.0, 0.0, 0.0]);
      p._readBuffer = async () => logits.buffer;

      const tensor = { buffer: {}, shape: [1, vocabSize] };
      const token = await p.sampleToken(tensor, 1.0);
      assert.ok(token >= 0 && token < vocabSize);
    });
  });

  // ─── Buffer Utilities ──────────────────────────────────────

  describe("_createBuffer", () => {
    it("creates a GPU buffer via device", () => {
      const buf = pipeline._createBuffer("test", 1024, 0x80);
      assert.equal(buf.label, "test");
      assert.equal(buf.size, 1024);
    });

    it("tracks buffer in _tempBuffers", () => {
      const buf = pipeline._createBuffer("test", 512, 0x80);
      assert.ok(pipeline._tempBuffers.includes(buf));
    });

    it("accumulates multiple buffers", () => {
      pipeline._createBuffer("a", 100, 0x80);
      pipeline._createBuffer("b", 200, 0x80);
      pipeline._createBuffer("c", 300, 0x80);
      assert.equal(pipeline._tempBuffers.length, 3);
    });
  });

  describe("_cleanupTempBuffers", () => {
    it("destroys all temp buffers", () => {
      const a = pipeline._createBuffer("a", 100, 0x80);
      const b = pipeline._createBuffer("b", 200, 0x80);
      pipeline._cleanupTempBuffers();
      assert.equal(a.destroyed, true);
      assert.equal(b.destroyed, true);
    });

    it("clears the temp buffer list", () => {
      pipeline._createBuffer("a", 100, 0x80);
      pipeline._cleanupTempBuffers();
      assert.equal(pipeline._tempBuffers.length, 0);
    });

    it("is safe to call when empty", () => {
      assert.doesNotThrow(() => pipeline._cleanupTempBuffers());
      assert.equal(pipeline._tempBuffers.length, 0);
    });
  });

  // ─── Pipeline Caching ──────────────────────────────────────

  describe("_getOrCreatePipeline", () => {
    beforeEach(async () => {
      await pipeline.init();
    });

    it("creates compute pipeline on first call", () => {
      const p = pipeline._getOrCreatePipeline("matmul", "main", [
        { binding: 0, visibility: 0x04, buffer: { type: "uniform" } },
      ]);
      assert.ok(p);
    });

    it("returns cached pipeline on second call", () => {
      const layout = [{ binding: 0, visibility: 0x04, buffer: { type: "uniform" } }];
      const p1 = pipeline._getOrCreatePipeline("matmul", "main", layout);
      const p2 = pipeline._getOrCreatePipeline("matmul", "main", layout);
      assert.equal(p1, p2);
    });

    it("creates separate pipelines for different entry points", () => {
      const layout = [{ binding: 0, visibility: 0x04, buffer: { type: "uniform" } }];
      const p1 = pipeline._getOrCreatePipeline("attention", "compute_scores", layout);
      const p2 = pipeline._getOrCreatePipeline("attention_softmax", "softmax_rows", layout);
      assert.notEqual(p1, p2);
    });

    it("maps attention_softmax to attention shader module", () => {
      const layout = [{ binding: 0, visibility: 0x04, buffer: { type: "uniform" } }];
      // Should not throw — attention_softmax uses "attention" module
      const p = pipeline._getOrCreatePipeline("attention_softmax", "softmax_rows", layout);
      assert.ok(p);
    });

    it("maps attention_cached_softmax to attention_cached module", () => {
      const layout = [{ binding: 0, visibility: 0x04, buffer: { type: "uniform" } }];
      const p = pipeline._getOrCreatePipeline("attention_cached_softmax", "softmax_cached", layout);
      assert.ok(p);
    });

    it("maps attention_cached_ws to attention_cached module", () => {
      const layout = [{ binding: 0, visibility: 0x04, buffer: { type: "uniform" } }];
      const p = pipeline._getOrCreatePipeline("attention_cached_ws", "weighted_sum_cached", layout);
      assert.ok(p);
    });
  });

  // ─── _readBuffer ───────────────────────────────────────────

  describe("_readBuffer", () => {
    it("creates staging buffer with MAP_READ | COPY_DST", async () => {
      const srcBuf = pipeline._createBuffer("src", 256, 0x80 | 0x04);
      await pipeline._readBuffer(srcBuf, 0, 256);

      // Find the staging buffer (last created, MAP_READ usage)
      const staging = device.buffers[device.buffers.length - 1];
      assert.equal(staging.size, 256);
      assert.ok(staging.destroyed); // should be destroyed after read
    });

    it("returns ArrayBuffer of correct size", async () => {
      const srcBuf = pipeline._createBuffer("src", 128, 0x80 | 0x04);
      const data = await pipeline._readBuffer(srcBuf, 0, 128);
      assert.ok(data instanceof ArrayBuffer);
      assert.equal(data.byteLength, 128);
    });
  });

  // ─── JSON Tensor Serialization ─────────────────────────────

  describe("serializeTensor / deserializeTensor", () => {
    it("round-trips a tensor through base64", async () => {
      const floats = new Float32Array([1.0, 2.0, 3.0, 4.0]);
      pipeline._readBuffer = async () => floats.buffer;

      const tensor = { buffer: {}, shape: [2, 2] };
      const serialized = await pipeline.serializeTensor(tensor);

      assert.deepEqual(serialized.shape, [2, 2]);
      assert.equal(serialized.dtype, "float32");
      assert.equal(typeof serialized.data, "string"); // base64

      // Deserialize
      const result = pipeline.deserializeTensor(serialized);
      assert.deepEqual(result.shape, [2, 2]);
      assert.ok(result.buffer);
    });
  });

  // ─── Binary Tensor Serialization ───────────────────────────

  describe("serializeTensorBinary / deserializeTensorBinary", () => {
    it("returns raw ArrayBuffer (no base64)", async () => {
      const floats = new Float32Array([1.5, 2.5, 3.5]);
      pipeline._readBuffer = async () => floats.buffer;

      const tensor = { buffer: {}, shape: [1, 3] };
      const result = await pipeline.serializeTensorBinary(tensor);

      assert.deepEqual(result.shape, [1, 3]);
      assert.ok(result.data instanceof ArrayBuffer);
      assert.equal(result.data.byteLength, 12); // 3 floats * 4 bytes
    });

    it("deserializes raw ArrayBuffer back to GPU buffer", () => {
      const floats = new Float32Array([1.0, 2.0, 3.0]);
      const payload = new Uint8Array(floats.buffer);

      const result = pipeline.deserializeTensorBinary(payload, [1, 3]);
      assert.deepEqual(result.shape, [1, 3]);
      assert.ok(result.buffer);
    });
  });

  // ─── Quantized Serialization (int8) ────────────────────────

  describe("serializeTensorQuantized / deserializeTensorQuantized", () => {
    it("quantizes single-token tensor (per-tensor)", async () => {
      const floats = new Float32Array([0.1, -0.5, 0.3, 0.8]);
      pipeline._readBuffer = async () => floats.buffer;

      const tensor = { buffer: {}, shape: [1, 4] };
      const result = await pipeline.serializeTensorQuantized(tensor);

      assert.deepEqual(result.shape, [1, 4]);
      assert.ok(result.data); // packed int8 + scale
    });

    it("quantizes multi-token tensor (per-channel)", async () => {
      const floats = new Float32Array([
        0.1, -0.5, 0.3, 0.8,
        -0.2, 0.4, -0.6, 0.1,
      ]);
      pipeline._readBuffer = async () => floats.buffer;

      const tensor = { buffer: {}, shape: [2, 4] };
      const result = await pipeline.serializeTensorQuantized(tensor);

      assert.deepEqual(result.shape, [2, 4]);
      assert.ok(result.data);
    });

    it("falls back to unquantized on NaN", async () => {
      const floats = new Float32Array([NaN, 0.5, 0.3]);
      pipeline._readBuffer = async () => floats.buffer;
      // Also mock serializeTensorBinary for the fallback path
      pipeline.serializeTensorBinary = async () => ({
        shape: [1, 3],
        data: new ArrayBuffer(12),
      });

      const tensor = { buffer: {}, shape: [1, 3] };
      const result = await pipeline.serializeTensorQuantized(tensor);
      assert.equal(result.fallbackUnquantized, true);
    });

    it("round-trips single-token through quantize/dequantize", async () => {
      const original = new Float32Array([0.5, -0.3, 0.8, -0.1]);
      pipeline._readBuffer = async () => original.buffer;

      const tensor = { buffer: {}, shape: [1, 4] };
      const quantized = await pipeline.serializeTensorQuantized(tensor);

      // Dequantize
      const payload = quantized.data instanceof Uint8Array
        ? quantized.data
        : new Uint8Array(quantized.data);
      const result = pipeline.deserializeTensorQuantized(payload, [1, 4]);
      assert.deepEqual(result.shape, [1, 4]);
      assert.ok(result.buffer);
    });

    it("round-trips multi-token through quantize/dequantize", async () => {
      const original = new Float32Array([
        0.1, 0.2, 0.3, 0.4,
        -0.1, -0.2, -0.3, -0.4,
      ]);
      pipeline._readBuffer = async () => original.buffer;

      const tensor = { buffer: {}, shape: [2, 4] };
      const quantized = await pipeline.serializeTensorQuantized(tensor);

      const payload = quantized.data instanceof Uint8Array
        ? quantized.data
        : new Uint8Array(quantized.data);
      const result = pipeline.deserializeTensorQuantized(payload, [2, 4]);
      assert.deepEqual(result.shape, [2, 4]);
    });
  });

  // ─── INT4 Serialization ────────────────────────────────────

  describe("serializeTensorInt4 / deserializeTensorInt4", () => {
    it("quantizes to int4 format", async () => {
      const floats = new Float32Array([0.5, -0.3, 0.8, -0.1, 0.2, -0.6]);
      pipeline._readBuffer = async () => floats.buffer;

      const tensor = { buffer: {}, shape: [1, 6] };
      const result = await pipeline.serializeTensorInt4(tensor);

      assert.deepEqual(result.shape, [1, 6]);
      assert.ok(result.data);
    });

    it("falls back to unquantized on Inf", async () => {
      const floats = new Float32Array([Infinity, 0.5, 0.3]);
      pipeline._readBuffer = async () => floats.buffer;
      pipeline.serializeTensorBinary = async () => ({
        shape: [1, 3],
        data: new ArrayBuffer(12),
      });

      const tensor = { buffer: {}, shape: [1, 3] };
      const result = await pipeline.serializeTensorInt4(tensor);
      assert.equal(result.fallbackUnquantized, true);
    });

    it("round-trips through int4 quantize/dequantize", async () => {
      const original = new Float32Array([0.5, -0.3, 0.8, -0.1, 0.2, -0.6]);
      pipeline._readBuffer = async () => original.buffer;

      const tensor = { buffer: {}, shape: [1, 6] };
      const serialized = await pipeline.serializeTensorInt4(tensor);

      const payload = serialized.data instanceof Uint8Array
        ? serialized.data
        : new Uint8Array(serialized.data);
      const result = pipeline.deserializeTensorInt4(payload, [1, 6]);
      assert.deepEqual(result.shape, [1, 6]);
      assert.ok(result.buffer);
    });
  });

  // ─── Delta-Encoded Serialization ───────────────────────────

  describe("serializeTensorDelta / deserializeTensorDeltaApply", () => {
    it("sends full tensor when no previous activation", async () => {
      const floats = new Float32Array([1.0, 2.0, 3.0]);
      pipeline._readBuffer = async () => floats.buffer;

      const tensor = { buffer: {}, shape: [1, 3] };
      const result = await pipeline.serializeTensorDelta(tensor, null);

      assert.equal(result.isDelta, false);
      assert.equal(result.sparsity, 0);
      assert.ok(result.currentFloat32);
    });

    it("sends full tensor when shape changes", async () => {
      const floats = new Float32Array([1.0, 2.0, 3.0]);
      pipeline._readBuffer = async () => floats.buffer;

      const tensor = { buffer: {}, shape: [1, 3] };
      const prev = new Float32Array([1.0, 2.0]); // different length
      const result = await pipeline.serializeTensorDelta(tensor, prev);

      assert.equal(result.isDelta, false);
    });

    it("computes delta when previous activation exists", async () => {
      const current = new Float32Array([1.0, 2.0, 3.0]);
      pipeline._readBuffer = async () => current.buffer;

      const tensor = { buffer: {}, shape: [1, 3] };
      const prev = new Float32Array([0.9, 1.8, 2.7]);
      const result = await pipeline.serializeTensorDelta(tensor, prev);

      assert.equal(result.isDelta, true);
      assert.ok(typeof result.sparsity === "number");
      assert.ok(result.currentFloat32 instanceof Float32Array);
    });

    it("reports high sparsity for nearly identical activations", async () => {
      const current = new Float32Array([1.0, 2.0, 3.0, 4.0]);
      pipeline._readBuffer = async () => current.buffer;

      const tensor = { buffer: {}, shape: [1, 4] };
      const prev = new Float32Array([1.0, 2.0, 3.0, 4.0]); // identical
      const result = await pipeline.serializeTensorDelta(tensor, prev);

      assert.equal(result.isDelta, true);
      assert.ok(result.sparsity >= 0.9); // mostly zeros
    });

    it("round-trips through delta encode/decode", async () => {
      const current = new Float32Array([1.0, 2.5, -0.3, 0.8]);
      pipeline._readBuffer = async () => current.buffer;

      const prev = new Float32Array([0.8, 2.0, -0.1, 0.5]);
      const tensor = { buffer: {}, shape: [1, 4] };
      const encoded = await pipeline.serializeTensorDelta(tensor, prev);

      assert.equal(encoded.isDelta, true);

      const payload = encoded.data instanceof Uint8Array
        ? encoded.data
        : new Uint8Array(encoded.data);
      const result = pipeline.deserializeTensorDeltaApply(payload, [1, 4], prev);

      assert.deepEqual(result.shape, [1, 4]);
      assert.ok(result.currentFloat32 instanceof Float32Array);
      // Should be approximately equal to current (int8 quantization adds some error)
      for (let i = 0; i < 4; i++) {
        assert.ok(
          Math.abs(result.currentFloat32[i] - current[i]) < 0.1,
          `element ${i}: got ${result.currentFloat32[i]}, expected ~${current[i]}`
        );
      }
    });
  });

  // ─── Multiple Cache Lifecycle ──────────────────────────────

  describe("multi-request cache lifecycle", () => {
    it("handles concurrent generation requests", () => {
      const c1 = pipeline.getOrCreateKVCache("gen-1", 0, 6);
      const c2 = pipeline.getOrCreateKVCache("gen-2", 0, 6);
      const c3 = pipeline.getOrCreateKVCache("gen-3", 0, 6);

      assert.equal(pipeline.kvCaches.size, 3);

      // Complete gen-2
      pipeline.clearCache("gen-2");
      assert.equal(pipeline.kvCaches.size, 2);
      assert.ok(pipeline.kvCaches.has("gen-1"));
      assert.ok(!pipeline.kvCaches.has("gen-2"));
      assert.ok(pipeline.kvCaches.has("gen-3"));
    });

    it("clears early exit state along with cache", () => {
      pipeline.getOrCreateKVCache("req-1", 0, 6);
      // earlyExit.clear should not throw even without prior state
      assert.doesNotThrow(() => pipeline.clearCache("req-1"));
    });
  });
});
