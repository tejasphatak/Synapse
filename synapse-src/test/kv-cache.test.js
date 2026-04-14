/**
 * KVCache Tests
 *
 * Tests GPU buffer management, append/appendBatch, rollback (Phase 2),
 * reset, destroy, and edge cases. GPU operations are mocked.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";

// Mock WebGPU globals
globalThis.GPUBufferUsage = { STORAGE: 0x80, COPY_SRC: 0x04, COPY_DST: 0x08 };

import { KVCache } from "../node/kv-cache.js";

// ─── Mock GPU ───────────────────────────────────────────────

function makeMockDevice() {
  const copies = [];
  const submits = [];
  const buffers = [];

  return {
    copies,
    submits,
    buffers,
    createBuffer(desc) {
      const buf = {
        label: desc.label,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() { this.destroyed = true; },
      };
      buffers.push(buf);
      return buf;
    },
    createCommandEncoder() {
      const cmds = [];
      return {
        copyBufferToBuffer(src, srcOff, dst, dstOff, size) {
          const entry = { src, srcOff, dst, dstOff, size };
          cmds.push(entry);
          copies.push(entry);
        },
        finish() { return cmds; },
      };
    },
    queue: {
      submit(cmdBuffers) { submits.push(cmdBuffers); },
    },
  };
}

// ─── Constructor ────────────────────────────────────────────

describe("KVCache", () => {
  const LAYERS = 3;
  const LAYER_START = 2;
  const HIDDEN = 64;
  const MAX_SEQ = 128;
  let device;
  let cache;

  beforeEach(() => {
    device = makeMockDevice();
    cache = new KVCache(device, LAYERS, LAYER_START, HIDDEN, MAX_SEQ);
  });

  describe("constructor", () => {
    it("starts with seqLen 0", () => {
      assert.equal(cache.seqLen, 0);
    });

    it("creates K and V buffers for each owned layer", () => {
      for (let l = LAYER_START; l < LAYER_START + LAYERS; l++) {
        assert.ok(cache.kBuffers.has(l), `missing K buffer for layer ${l}`);
        assert.ok(cache.vBuffers.has(l), `missing V buffer for layer ${l}`);
      }
      assert.equal(cache.kBuffers.size, LAYERS);
      assert.equal(cache.vBuffers.size, LAYERS);
    });

    it("allocates correct buffer size", () => {
      const expected = MAX_SEQ * HIDDEN * 4;
      for (const buf of cache.kBuffers.values()) {
        assert.equal(buf.size, expected);
      }
    });

    it("sets correct usage flags", () => {
      const expected = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
      for (const buf of cache.kBuffers.values()) {
        assert.equal(buf.usage, expected);
      }
    });

    it("labels buffers with layer index", () => {
      assert.equal(cache.kBuffers.get(2).label, "kv_cache_k_layer2");
      assert.equal(cache.vBuffers.get(3).label, "kv_cache_v_layer3");
    });
  });

  // ─── append ─────────────────────────────────────────────

  describe("append", () => {
    it("copies K and V to correct offset for position 0", () => {
      const mockK = { label: "newK" };
      const mockV = { label: "newV" };
      cache.append(2, mockK, mockV, 0);

      assert.equal(device.copies.length, 2);
      assert.equal(device.copies[0].src, mockK);
      assert.equal(device.copies[0].dstOff, 0);
      assert.equal(device.copies[0].size, HIDDEN * 4);
      assert.equal(device.copies[1].src, mockV);
    });

    it("computes correct byte offset for later positions", () => {
      const pos = 5;
      cache.append(2, {}, {}, pos);

      assert.equal(device.copies[0].dstOff, pos * HIDDEN * 4);
    });

    it("advances seqLen", () => {
      cache.append(2, {}, {}, 0);
      assert.equal(cache.seqLen, 1);
      cache.append(2, {}, {}, 1);
      assert.equal(cache.seqLen, 2);
    });

    it("seqLen tracks max position, not call count", () => {
      cache.append(2, {}, {}, 5);
      assert.equal(cache.seqLen, 6);
      // Writing to an earlier position shouldn't decrease seqLen
      cache.append(3, {}, {}, 2);
      assert.equal(cache.seqLen, 6);
    });

    it("submits GPU commands", () => {
      cache.append(2, {}, {}, 0);
      assert.equal(device.submits.length, 1);
    });

    it("copies to the correct layer's buffer", () => {
      cache.append(3, {}, {}, 0);
      assert.equal(device.copies[0].dst, cache.kBuffers.get(3));
      assert.equal(device.copies[1].dst, cache.vBuffers.get(3));
    });
  });

  // ─── appendBatch ────────────────────────────────────────

  describe("appendBatch", () => {
    it("copies correct size for multiple tokens", () => {
      const numTokens = 10;
      cache.appendBatch(2, {}, {}, 0, numTokens);

      assert.equal(device.copies[0].size, numTokens * HIDDEN * 4);
      assert.equal(device.copies[0].dstOff, 0);
    });

    it("computes correct offset from startPos", () => {
      cache.appendBatch(2, {}, {}, 3, 5);
      assert.equal(device.copies[0].dstOff, 3 * HIDDEN * 4);
    });

    it("updates seqLen to startPos + numTokens", () => {
      cache.appendBatch(2, {}, {}, 0, 10);
      assert.equal(cache.seqLen, 10);
    });

    it("does not reduce seqLen for earlier batch", () => {
      cache.appendBatch(2, {}, {}, 0, 20);
      cache.appendBatch(3, {}, {}, 0, 5);
      assert.equal(cache.seqLen, 20);
    });
  });

  // ─── getKV ──────────────────────────────────────────────

  describe("getKV", () => {
    it("returns K and V buffers for requested layer", () => {
      const result = cache.getKV(2);
      assert.equal(result.kBuffer, cache.kBuffers.get(2));
      assert.equal(result.vBuffer, cache.vBuffers.get(2));
    });

    it("returns current seqLen", () => {
      cache.append(2, {}, {}, 0);
      cache.append(2, {}, {}, 1);
      const result = cache.getKV(2);
      assert.equal(result.seqLen, 2);
    });

    it("seqLen is shared across layers", () => {
      cache.append(2, {}, {}, 0);
      cache.append(3, {}, {}, 1);
      // seqLen is global, not per-layer
      assert.equal(cache.getKV(2).seqLen, 2);
      assert.equal(cache.getKV(4).seqLen, 2);
    });
  });

  // ─── rollback (Phase 2 speculation) ─────────────────────

  describe("rollback", () => {
    it("reduces seqLen to target position", () => {
      cache.append(2, {}, {}, 0);
      cache.append(2, {}, {}, 1);
      cache.append(2, {}, {}, 2);
      assert.equal(cache.seqLen, 3);

      cache.rollback(1);
      assert.equal(cache.seqLen, 1);
    });

    it("does nothing if target >= current seqLen", () => {
      cache.append(2, {}, {}, 0);
      cache.append(2, {}, {}, 1);
      assert.equal(cache.seqLen, 2);

      cache.rollback(2);
      assert.equal(cache.seqLen, 2);

      cache.rollback(5);
      assert.equal(cache.seqLen, 2);
    });

    it("can rollback to 0", () => {
      cache.append(2, {}, {}, 0);
      cache.append(2, {}, {}, 1);
      cache.rollback(0);
      assert.equal(cache.seqLen, 0);
    });

    it("allows re-appending after rollback", () => {
      cache.append(2, {}, {}, 0);
      cache.append(2, {}, {}, 1);
      cache.append(2, {}, {}, 2);
      cache.rollback(1);
      assert.equal(cache.seqLen, 1);

      // Re-append at position 1 with new data
      cache.append(2, {}, {}, 1);
      assert.equal(cache.seqLen, 2);
    });
  });

  // ─── reset ──────────────────────────────────────────────

  describe("reset", () => {
    it("sets seqLen to 0", () => {
      cache.append(2, {}, {}, 0);
      cache.append(2, {}, {}, 1);
      cache.reset();
      assert.equal(cache.seqLen, 0);
    });

    it("keeps buffers intact", () => {
      cache.reset();
      assert.equal(cache.kBuffers.size, LAYERS);
      assert.equal(cache.vBuffers.size, LAYERS);
      for (const buf of cache.kBuffers.values()) {
        assert.equal(buf.destroyed, false);
      }
    });
  });

  // ─── destroy ────────────────────────────────────────────

  describe("destroy", () => {
    it("destroys all GPU buffers", () => {
      const allBuffers = [
        ...cache.kBuffers.values(),
        ...cache.vBuffers.values(),
      ];
      cache.destroy();
      for (const buf of allBuffers) {
        assert.equal(buf.destroyed, true);
      }
    });

    it("clears buffer maps", () => {
      cache.destroy();
      assert.equal(cache.kBuffers.size, 0);
      assert.equal(cache.vBuffers.size, 0);
    });

    it("resets seqLen", () => {
      cache.append(2, {}, {}, 0);
      cache.destroy();
      assert.equal(cache.seqLen, 0);
    });
  });

  // ─── Edge cases ─────────────────────────────────────────

  describe("edge cases", () => {
    it("handles single-layer cache", () => {
      const single = new KVCache(device, 1, 0, HIDDEN, MAX_SEQ);
      assert.equal(single.kBuffers.size, 1);
      assert.ok(single.kBuffers.has(0));
      single.append(0, {}, {}, 0);
      assert.equal(single.seqLen, 1);
    });

    it("handles high layer offset", () => {
      const high = new KVCache(device, 2, 100, HIDDEN, MAX_SEQ);
      assert.ok(high.kBuffers.has(100));
      assert.ok(high.kBuffers.has(101));
      assert.ok(!high.kBuffers.has(99));
    });

    it("rollback then appendBatch works correctly", () => {
      cache.appendBatch(2, {}, {}, 0, 10);
      assert.equal(cache.seqLen, 10);
      cache.rollback(3);
      assert.equal(cache.seqLen, 3);
      cache.appendBatch(2, {}, {}, 3, 4);
      assert.equal(cache.seqLen, 7);
    });
  });
});
