/**
 * EarlyExitDetector Tests
 *
 * Tests convergence detection, stats tracking, and early exit logic.
 * Pure math — no GPU or browser dependencies.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";

import { EarlyExitDetector } from "../node/early-exit.js";

// ─── Helpers ─────────────────────────────────────────────────

function f32(...vals) {
  return new Float32Array(vals);
}

/** Create a vector that's a tiny perturbation of `base` */
function nudge(base, epsilon = 0.001) {
  const out = new Float32Array(base.length);
  for (let i = 0; i < base.length; i++) {
    out[i] = base[i] + epsilon * (i % 2 === 0 ? 1 : -1);
  }
  return out;
}

// ─── Tests ───────────────────────────────────────────────────

describe("EarlyExitDetector", () => {
  let detector;

  beforeEach(() => {
    detector = new EarlyExitDetector();
  });

  describe("initialization", () => {
    it("starts disabled by default", () => {
      assert.equal(detector.enabled, false);
    });

    it("has conservative thresholds", () => {
      assert.equal(detector.cosineThreshold, 0.9995);
      assert.equal(detector.normRatioThreshold, 0.01);
    });

    it("starts with zero stats", () => {
      const s = detector.stats;
      assert.equal(s.checks, 0);
      assert.equal(s.earlyExits, 0);
      assert.equal(s.layersSaved, 0);
    });
  });

  describe("check — first layer (no prior data)", () => {
    it("returns shouldExit=false on first layer", () => {
      const result = detector.check("req1", 0, f32(1, 2, 3), 6);
      assert.equal(result.shouldExit, false);
    });

    it("returns cosine=0, normRatio=1 on first layer", () => {
      const result = detector.check("req1", 0, f32(1, 2, 3), 6);
      assert.equal(result.cosine, 0);
      assert.equal(result.normRatio, 1);
    });

    it("records the hidden state for next comparison", () => {
      detector.check("req1", 0, f32(1, 2, 3), 6);
      assert.ok(detector.layerOutputs.has("req1"));
    });

    it("increments check count", () => {
      detector.check("req1", 0, f32(1, 2, 3), 6);
      assert.equal(detector.stats.checks, 1);
    });
  });

  describe("check — convergence detection (enabled)", () => {
    beforeEach(() => {
      detector.enabled = true;
    });

    it("detects convergence when layers produce nearly identical output", () => {
      const base = f32(1, 2, 3, 4, 5);
      detector.check("req1", 0, base, 6);

      // Tiny perturbation — cosine ≈ 1.0, normRatio ≈ 0
      const similar = nudge(base, 0.0001);
      const result = detector.check("req1", 1, similar, 6);

      assert.equal(result.shouldExit, true);
      assert.ok(result.cosine > 0.999);
      assert.ok(result.normRatio < 0.01);
    });

    it("does NOT exit when layers differ significantly", () => {
      detector.check("req1", 0, f32(1, 0, 0), 6);
      const result = detector.check("req1", 1, f32(0, 1, 0), 6);

      assert.equal(result.shouldExit, false);
    });

    it("does NOT exit on the last layer", () => {
      const base = f32(1, 2, 3);
      detector.check("req1", 4, base, 6);
      const result = detector.check("req1", 5, nudge(base, 0.0001), 6);

      assert.equal(result.shouldExit, false);
      assert.ok(result.cosine > 0.999); // metrics still computed
    });

    it("tracks layers saved correctly", () => {
      const base = f32(1, 2, 3, 4, 5);
      detector.check("req1", 0, base, 6);
      detector.check("req1", 1, nudge(base, 0.0001), 6);

      // Exit at layer 1 of 6 → saved layers 2,3,4,5 = 4 layers
      assert.equal(detector.stats.earlyExits, 1);
      assert.equal(detector.stats.layersSaved, 4);
    });
  });

  describe("check — disabled mode", () => {
    it("never exits when disabled, even with converged output", () => {
      assert.equal(detector.enabled, false);

      const base = f32(1, 2, 3, 4, 5);
      detector.check("req1", 0, base, 6);
      const result = detector.check("req1", 1, nudge(base, 0.0001), 6);

      assert.equal(result.shouldExit, false);
      // But metrics are still computed
      assert.ok(result.cosine > 0.999);
    });
  });

  describe("check — threshold boundaries", () => {
    beforeEach(() => {
      detector.enabled = true;
    });

    it("requires BOTH cosine and normRatio to agree", () => {
      // High cosine but large normRatio: scaled vector
      const a = f32(1, 2, 3);
      detector.check("req1", 0, a, 6);

      // Same direction but 2x magnitude — cosine=1.0 but normRatio >> 0.01
      const b = f32(2, 4, 6);
      const result = detector.check("req1", 1, b, 6);

      assert.ok(result.cosine > 0.99);
      assert.ok(result.normRatio > 0.01);
      assert.equal(result.shouldExit, false);
    });

    it("exact duplicate triggers exit", () => {
      const v = f32(3, 1, 4, 1, 5);
      detector.check("req1", 0, v, 6);
      const result = detector.check("req1", 1, new Float32Array(v), 6);

      assert.ok(Math.abs(result.cosine - 1) < 1e-10);
      assert.equal(result.normRatio, 0);
      assert.equal(result.shouldExit, true);
    });
  });

  describe("check — multi-request isolation", () => {
    it("tracks each request independently", () => {
      detector.enabled = true;

      // req1: converging
      const base = f32(1, 2, 3);
      detector.check("req1", 0, base, 6);

      // req2: diverging
      detector.check("req2", 0, f32(1, 0, 0), 6);

      // req1 converges
      const r1 = detector.check("req1", 1, nudge(base, 0.0001), 6);
      // req2 diverges
      const r2 = detector.check("req2", 1, f32(0, 0, 1), 6);

      assert.equal(r1.shouldExit, true);
      assert.equal(r2.shouldExit, false);
    });
  });

  describe("check — successive layers", () => {
    it("compares each layer to its immediate predecessor", () => {
      detector.enabled = true;

      // Layer 0 → Layer 1: big jump (no exit)
      detector.check("req1", 0, f32(1, 0, 0), 6);
      const r1 = detector.check("req1", 1, f32(0, 1, 0), 6);
      assert.equal(r1.shouldExit, false);

      // Layer 1 → Layer 2: converging (exit)
      const r2 = detector.check("req1", 2, nudge(f32(0, 1, 0), 0.0001), 6);
      assert.equal(r2.shouldExit, true);
    });
  });

  describe("_cosineSimilarity", () => {
    it("returns 1 for identical vectors", () => {
      const v = f32(1, 2, 3);
      assert.equal(detector._cosineSimilarity(v, v), 1);
    });

    it("returns 0 for orthogonal vectors", () => {
      const a = f32(1, 0, 0);
      const b = f32(0, 1, 0);
      assert.ok(Math.abs(detector._cosineSimilarity(a, b)) < 1e-10);
    });

    it("returns -1 for opposite vectors", () => {
      const a = f32(1, 2, 3);
      const b = f32(-1, -2, -3);
      assert.ok(Math.abs(detector._cosineSimilarity(a, b) - (-1)) < 1e-10);
    });

    it("returns 0 for zero vector", () => {
      const a = f32(0, 0, 0);
      const b = f32(1, 2, 3);
      assert.equal(detector._cosineSimilarity(a, b), 0);
    });

    it("is scale-invariant", () => {
      const a = f32(1, 2, 3);
      const b = f32(10, 20, 30);
      assert.ok(Math.abs(detector._cosineSimilarity(a, b) - 1) < 1e-10);
    });
  });

  describe("_deltaNormRatio", () => {
    it("returns 0 for identical vectors", () => {
      const v = f32(1, 2, 3);
      assert.equal(detector._deltaNormRatio(v, new Float32Array(v)), 0);
    });

    it("returns 1 for zero-to-nonzero", () => {
      const a = f32(0, 0, 0);
      const b = f32(1, 2, 3);
      assert.equal(detector._deltaNormRatio(a, b), 1);
    });

    it("correctly computes ratio for known values", () => {
      // a = [3, 4], norm = 5
      // b = [3.1, 4], delta = [0.1, 0], deltaNorm = 0.1
      // ratio = 0.1 / 5 = 0.02
      const a = f32(3, 4);
      const b = f32(3.1, 4);
      const ratio = detector._deltaNormRatio(a, b);
      assert.ok(Math.abs(ratio - 0.02) < 1e-6);
    });
  });

  describe("clear", () => {
    it("removes state for specified request", () => {
      detector.check("req1", 0, f32(1, 2, 3), 6);
      assert.ok(detector.layerOutputs.has("req1"));

      detector.clear("req1");
      assert.equal(detector.layerOutputs.has("req1"), false);
    });

    it("does not affect other requests", () => {
      detector.check("req1", 0, f32(1, 2, 3), 6);
      detector.check("req2", 0, f32(4, 5, 6), 6);

      detector.clear("req1");
      assert.ok(detector.layerOutputs.has("req2"));
    });

    it("allows fresh tracking after clear", () => {
      detector.check("req1", 0, f32(1, 2, 3), 6);
      detector.clear("req1");

      // Should behave as first layer again
      const result = detector.check("req1", 0, f32(7, 8, 9), 6);
      assert.equal(result.cosine, 0);
      assert.equal(result.normRatio, 1);
    });
  });

  describe("getStats", () => {
    it("returns zero exitRate with no checks", () => {
      const s = detector.getStats();
      assert.equal(s.exitRate, 0);
      assert.equal(s.avgLayersSaved, 0);
    });

    it("computes exitRate correctly", () => {
      detector.stats.checks = 20;
      detector.stats.earlyExits = 5;

      assert.equal(detector.getStats().exitRate, 0.25);
    });

    it("computes avgLayersSaved correctly", () => {
      detector.stats.earlyExits = 4;
      detector.stats.layersSaved = 12;

      assert.equal(detector.getStats().avgLayersSaved, 3);
    });

    it("includes all raw stats", () => {
      detector.stats.checks = 10;
      detector.stats.earlyExits = 2;
      detector.stats.layersSaved = 6;

      const s = detector.getStats();
      assert.equal(s.checks, 10);
      assert.equal(s.earlyExits, 2);
      assert.equal(s.layersSaved, 6);
    });
  });
});
