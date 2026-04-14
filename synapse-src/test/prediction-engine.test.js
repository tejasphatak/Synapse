/**
 * Phase 2-4 Prediction & Optimization Engine Tests
 *
 * Covers: ActivationPredictor, EarlyExitDetector, HeadPruner, MixtureOfDepthsRouter
 * All pure JS — no WebGPU needed.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { ActivationPredictor } from "../node/predictor.js";
import { EarlyExitDetector } from "../node/early-exit.js";
import { HeadPruner } from "../node/head-pruning.js";
import { MixtureOfDepthsRouter } from "../node/mixture-of-depths.js";

// ─── Helpers ─────────────────────────────────────────────────

function f32(...vals) {
  return new Float32Array(vals);
}

function linearSequence(start, step, len) {
  // Produces a Float32Array with linearly increasing values
  const arr = new Float32Array(len);
  for (let i = 0; i < len; i++) arr[i] = start + step * i;
  return arr;
}

function cosineSim(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// ─── ActivationPredictor ─────────────────────────────────────

describe("ActivationPredictor", () => {
  let predictor;

  beforeEach(() => {
    predictor = new ActivationPredictor();
  });

  describe("observe + predict basics", () => {
    it("returns null with 0 observations", () => {
      assert.equal(predictor.predict("req1"), null);
    });

    it("returns null with 1 observation", () => {
      predictor.observe("req1", f32(1, 2, 3));
      assert.equal(predictor.predict("req1"), null);
    });

    it("predicts after 2 observations via linear extrapolation", () => {
      predictor.observe("req1", f32(1, 2, 3));
      predictor.observe("req1", f32(2, 4, 6));
      const result = predictor.predict("req1");
      assert.ok(result);
      // next = curr + (curr - prev) = [2,4,6] + [1,2,3] = [3,6,9]
      assert.deepEqual(Array.from(result.prediction), [3, 6, 9]);
    });

    it("extrapolation works for constant activations", () => {
      predictor.observe("req1", f32(5, 5, 5));
      predictor.observe("req1", f32(5, 5, 5));
      const result = predictor.predict("req1");
      // constant → delta = 0 → prediction = same
      assert.deepEqual(Array.from(result.prediction), [5, 5, 5]);
    });
  });

  describe("confidence estimation", () => {
    it("confidence is 1.0 with only 2 observations", () => {
      predictor.observe("req1", f32(1, 0));
      predictor.observe("req1", f32(2, 0));
      const result = predictor.predict("req1");
      assert.equal(result.confidence, 1.0);
    });

    it("high confidence for consistent linear trajectory", () => {
      predictor.observe("req1", f32(1, 2, 3));
      predictor.observe("req1", f32(2, 4, 6));
      predictor.observe("req1", f32(3, 6, 9));
      const result = predictor.predict("req1");
      // deltas are [1,2,3] both times → cosine = 1.0
      assert.ok(result.confidence > 0.99);
    });

    it("low confidence for erratic trajectory", () => {
      predictor.observe("req1", f32(1, 0, 0));
      predictor.observe("req1", f32(0, 1, 0));
      predictor.observe("req1", f32(0, 0, 1));
      const result = predictor.predict("req1");
      // deltas are [-1,1,0] then [0,-1,1] → cosine is negative → clamped to 0
      assert.ok(result.confidence < 0.5);
    });
  });

  describe("history ring buffer", () => {
    it("caps history at maxHistory", () => {
      for (let i = 0; i < 10; i++) {
        predictor.observe("req1", f32(i, i * 2));
      }
      const entry = predictor.requests.get("req1");
      assert.equal(entry.history.length, predictor.maxHistory);
    });

    it("prediction uses most recent observations after overflow", () => {
      for (let i = 0; i < 10; i++) {
        predictor.observe("req1", f32(i * 10));
      }
      const result = predictor.predict("req1");
      // last two: [80], [90] → next = 90 + 10 = [100]
      assert.deepEqual(Array.from(result.prediction), [100]);
    });
  });

  describe("verify", () => {
    it("accepts identical predictions", () => {
      const result = predictor.verify(f32(1, 2, 3), f32(1, 2, 3));
      assert.equal(result.accept, true);
      assert.ok(result.cosine > 0.999);
    });

    it("rejects orthogonal vectors", () => {
      const result = predictor.verify(f32(1, 0, 0), f32(0, 1, 0));
      assert.equal(result.accept, false);
      assert.ok(Math.abs(result.cosine) < 0.01);
    });

    it("accepts very similar vectors above threshold", () => {
      const a = f32(1, 2, 3, 4, 5);
      const b = new Float32Array(a);
      b[4] = 5.001; // tiny perturbation
      const result = predictor.verify(a, b);
      assert.equal(result.accept, true);
    });

    it("rejects reversed vectors", () => {
      const result = predictor.verify(f32(1, 2, 3), f32(-1, -2, -3));
      assert.equal(result.accept, false);
    });

    it("handles zero vectors gracefully", () => {
      const result = predictor.verify(f32(0, 0, 0), f32(1, 2, 3));
      assert.equal(result.accept, false);
      assert.equal(result.cosine, 0);
    });
  });

  describe("stats tracking", () => {
    it("counts predictions", () => {
      predictor.observe("r1", f32(1));
      predictor.observe("r1", f32(2));
      predictor.predict("r1");
      predictor.predict("r1");
      assert.equal(predictor.getStats().predictions, 2);
    });

    it("tracks hit rate from verify calls", () => {
      predictor.verify(f32(1, 2), f32(1, 2)); // hit
      predictor.verify(f32(1, 0), f32(0, 1)); // miss
      const stats = predictor.getStats();
      assert.equal(stats.hits, 1);
      assert.equal(stats.misses, 1);
      assert.ok(Math.abs(stats.hitRate - 0.5) < 0.001);
    });
  });

  describe("multi-request isolation", () => {
    it("tracks separate histories per request", () => {
      predictor.observe("r1", f32(1, 0));
      predictor.observe("r1", f32(2, 0));
      predictor.observe("r2", f32(0, 1));
      predictor.observe("r2", f32(0, 2));

      const p1 = predictor.predict("r1");
      const p2 = predictor.predict("r2");
      assert.deepEqual(Array.from(p1.prediction), [3, 0]);
      assert.deepEqual(Array.from(p2.prediction), [0, 3]);
    });

    it("clear removes only the specified request", () => {
      predictor.observe("r1", f32(1));
      predictor.observe("r2", f32(2));
      predictor.clear("r1");
      assert.equal(predictor.requests.has("r1"), false);
      assert.equal(predictor.requests.has("r2"), true);
    });

    it("clearAll resets everything", () => {
      predictor.observe("r1", f32(1));
      predictor.observe("r1", f32(2));
      predictor.predict("r1");
      predictor.verify(f32(1), f32(1));
      predictor.clearAll();
      assert.equal(predictor.requests.size, 0);
      assert.equal(predictor.getStats().predictions, 0);
      assert.equal(predictor.getStats().hits, 0);
    });
  });

  describe("realistic activation patterns", () => {
    it("predicts well on smoothly evolving activations", () => {
      const dim = 768;
      // Simulate activations that shift gradually
      for (let step = 0; step < 5; step++) {
        const act = new Float32Array(dim);
        for (let i = 0; i < dim; i++) {
          act[i] = Math.sin(i * 0.01 + step * 0.1);
        }
        predictor.observe("req1", act);
      }

      const result = predictor.predict("req1");
      assert.ok(result);
      assert.ok(result.confidence > 0.9);

      // Build the expected "actual" next step
      const actual = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        actual[i] = Math.sin(i * 0.01 + 5 * 0.1);
      }

      const verification = predictor.verify(result.prediction, actual);
      // Linear extrapolation on sin isn't perfect but should be decent
      assert.ok(verification.cosine > 0.95);
    });
  });

  describe("predictMulti — batch speculation", () => {
    it("returns null with fewer than 2 observations", () => {
      assert.equal(predictor.predictMulti("req1"), null);
      predictor.observe("req1", f32(1, 2, 3));
      assert.equal(predictor.predictMulti("req1"), null);
    });

    it("returns k predictions with correct stepsAhead", () => {
      predictor.observe("req1", f32(1, 2));
      predictor.observe("req1", f32(3, 4));
      const results = predictor.predictMulti("req1", 4);
      assert.equal(results.length, 4);
      for (let i = 0; i < 4; i++) {
        assert.equal(results[i].stepsAhead, i + 1);
      }
    });

    it("step 1 matches single predict()", () => {
      predictor.observe("req1", f32(10, 20, 30));
      predictor.observe("req1", f32(12, 24, 36));
      // delta = [2, 4, 6]
      const single = predictor.predict("req1");
      // Reset stats to get clean predictMulti
      const p2 = new ActivationPredictor();
      p2.observe("req1", f32(10, 20, 30));
      p2.observe("req1", f32(12, 24, 36));
      const multi = p2.predictMulti("req1", 3);
      assert.deepEqual(Array.from(multi[0].prediction), Array.from(single.prediction));
    });

    it("extrapolates linearly: curr + step * delta", () => {
      predictor.observe("req1", f32(0, 0));
      predictor.observe("req1", f32(10, 5));
      // delta = [10, 5]
      const results = predictor.predictMulti("req1", 3);
      // step 1: [10+10, 5+5] = [20, 10]
      assert.deepEqual(Array.from(results[0].prediction), [20, 10]);
      // step 2: [10+20, 5+10] = [30, 15]
      assert.deepEqual(Array.from(results[1].prediction), [30, 15]);
      // step 3: [10+30, 5+15] = [40, 20]
      assert.deepEqual(Array.from(results[2].prediction), [40, 20]);
    });

    it("confidence decays by 0.85 per step", () => {
      // With only 2 observations, base confidence = 1.0
      predictor.observe("req1", f32(1, 2));
      predictor.observe("req1", f32(2, 4));
      const results = predictor.predictMulti("req1", 4);
      assert.ok(Math.abs(results[0].confidence - 1.0) < 0.001);
      assert.ok(Math.abs(results[1].confidence - 0.85) < 0.001);
      assert.ok(Math.abs(results[2].confidence - 0.85 * 0.85) < 0.001);
      assert.ok(Math.abs(results[3].confidence - Math.pow(0.85, 3)) < 0.001);
    });

    it("base confidence reflects trajectory smoothness", () => {
      // Smooth trajectory: consistent deltas
      predictor.observe("req1", f32(1, 2, 3));
      predictor.observe("req1", f32(2, 4, 6));
      predictor.observe("req1", f32(3, 6, 9));
      const smooth = predictor.predictMulti("req1", 2);
      assert.ok(smooth[0].confidence > 0.99, "smooth trajectory → high base confidence");

      // Erratic trajectory
      const p2 = new ActivationPredictor();
      p2.observe("req2", f32(1, 0, 0));
      p2.observe("req2", f32(0, 1, 0));
      p2.observe("req2", f32(0, 0, 1));
      const erratic = p2.predictMulti("req2", 2);
      assert.ok(erratic[0].confidence < 0.5, "erratic trajectory → low base confidence");
    });

    it("defaults to k=3 when omitted", () => {
      predictor.observe("req1", f32(1));
      predictor.observe("req1", f32(2));
      const results = predictor.predictMulti("req1");
      assert.equal(results.length, 3);
    });

    it("increments stats.predictions for each step", () => {
      predictor.observe("req1", f32(1));
      predictor.observe("req1", f32(2));
      const before = predictor.getStats().predictions;
      predictor.predictMulti("req1", 4);
      assert.equal(predictor.getStats().predictions, before + 4);
    });

    it("handles constant activations (zero delta)", () => {
      predictor.observe("req1", f32(5, 5, 5));
      predictor.observe("req1", f32(5, 5, 5));
      const results = predictor.predictMulti("req1", 3);
      // All predictions should equal the constant
      for (const r of results) {
        assert.deepEqual(Array.from(r.prediction), [5, 5, 5]);
      }
    });

    it("works with high-dimensional activations", () => {
      const dim = 768;
      const a = new Float32Array(dim);
      const b = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        a[i] = Math.sin(i * 0.01);
        b[i] = Math.sin(i * 0.01) + 0.1;
      }
      predictor.observe("req1", a);
      predictor.observe("req1", b);
      const results = predictor.predictMulti("req1", 3);
      assert.equal(results.length, 3);
      assert.equal(results[0].prediction.length, dim);
      // Step 1 should be b + 0.1 ≈ sin + 0.2
      for (let i = 0; i < dim; i++) {
        const expected = b[i] + 0.1;
        assert.ok(Math.abs(results[0].prediction[i] - expected) < 0.001);
      }
    });
  });
});

// ─── EarlyExitDetector ───────────────────────────────────────

describe("EarlyExitDetector", () => {
  let detector;

  beforeEach(() => {
    detector = new EarlyExitDetector();
    detector.enabled = true;
  });

  describe("basic behavior", () => {
    it("does not exit on first layer (no comparison)", () => {
      const result = detector.check("r1", 0, f32(1, 2, 3), 6);
      assert.equal(result.shouldExit, false);
    });

    it("does not exit on the last layer", () => {
      detector.check("r1", 0, f32(1, 2, 3), 6);
      const result = detector.check("r1", 5, f32(1, 2, 3), 6);
      assert.equal(result.shouldExit, false);
      // But cosine should be 1.0 since vectors are identical
      assert.ok(result.cosine > 0.999);
    });

    it("exits when consecutive layers produce identical output", () => {
      const hidden = f32(1, 2, 3, 4, 5);
      detector.check("r1", 0, hidden, 6);
      const result = detector.check("r1", 1, hidden, 6);
      assert.equal(result.shouldExit, true);
      assert.ok(result.cosine > 0.9999);
      assert.ok(result.normRatio < 0.001);
    });

    it("does not exit when layers produce very different output", () => {
      detector.check("r1", 0, f32(1, 0, 0, 0), 6);
      const result = detector.check("r1", 1, f32(0, 0, 0, 1), 6);
      assert.equal(result.shouldExit, false);
    });
  });

  describe("disabled mode", () => {
    it("never exits when disabled", () => {
      detector.enabled = false;
      const hidden = f32(1, 2, 3);
      detector.check("r1", 0, hidden, 6);
      const result = detector.check("r1", 1, hidden, 6);
      assert.equal(result.shouldExit, false);
    });
  });

  describe("convergence detection", () => {
    it("detects convergence with tiny perturbation", () => {
      const base = new Float32Array(100);
      for (let i = 0; i < 100; i++) base[i] = Math.sin(i);

      const perturbed = new Float32Array(base);
      perturbed[0] += 1e-5; // tiny change

      detector.check("r1", 0, base, 6);
      const result = detector.check("r1", 1, perturbed, 6);
      assert.equal(result.shouldExit, true);
    });

    it("does not exit for medium-sized changes", () => {
      const base = new Float32Array(100);
      for (let i = 0; i < 100; i++) base[i] = Math.sin(i);

      const changed = new Float32Array(100);
      for (let i = 0; i < 100; i++) changed[i] = Math.sin(i) + 0.1 * Math.cos(i);

      detector.check("r1", 0, base, 6);
      const result = detector.check("r1", 1, changed, 6);
      assert.equal(result.shouldExit, false);
    });
  });

  describe("stats", () => {
    it("tracks checks and exits", () => {
      const hidden = f32(1, 2, 3);
      detector.check("r1", 0, hidden, 6);
      detector.check("r1", 1, hidden, 6); // early exit
      detector.check("r1", 2, f32(10, 20, 30), 6); // no exit (big change)

      const stats = detector.getStats();
      assert.equal(stats.checks, 3);
      assert.equal(stats.earlyExits, 1);
      assert.equal(stats.layersSaved, 4); // layers 2,3,4,5 skipped from layer 1
    });

    it("computes exit rate", () => {
      const h = f32(1, 2, 3);
      detector.check("r1", 0, h, 6);
      detector.check("r1", 1, h, 6); // exit
      assert.ok(detector.getStats().exitRate > 0);
    });
  });

  describe("multi-request isolation", () => {
    it("tracks each request independently", () => {
      detector.check("r1", 0, f32(1, 0), 6);
      detector.check("r2", 0, f32(0, 1), 6);

      // r1 converges
      const r1 = detector.check("r1", 1, f32(1, 0), 6);
      // r2 diverges
      const r2 = detector.check("r2", 1, f32(1, 1), 6);

      assert.equal(r1.shouldExit, true);
      assert.equal(r2.shouldExit, false);
    });

    it("clear removes only the specified request", () => {
      detector.check("r1", 0, f32(1));
      detector.check("r2", 0, f32(2));
      detector.clear("r1");
      assert.equal(detector.layerOutputs.has("r1"), false);
      assert.equal(detector.layerOutputs.has("r2"), true);
    });
  });

  describe("edge cases", () => {
    it("handles zero vectors", () => {
      detector.check("r1", 0, f32(0, 0, 0), 6);
      const result = detector.check("r1", 1, f32(0, 0, 0), 6);
      // Zero norm → cosine returns 0, normRatio returns 1
      assert.equal(result.shouldExit, false);
    });

    it("handles single-element vectors", () => {
      detector.check("r1", 0, f32(5), 6);
      const result = detector.check("r1", 1, f32(5), 6);
      assert.equal(result.shouldExit, true);
    });
  });
});

// ─── HeadPruner ──────────────────────────────────────────────

describe("HeadPruner", () => {
  let pruner;

  beforeEach(() => {
    // 12 heads, 6 layers (GPT-2 small config)
    pruner = new HeadPruner(12, 6);
    pruner.enabled = true;
  });

  describe("initialization", () => {
    it("starts with equal importance for all heads", () => {
      const ranking = pruner.getImportanceRanking(0);
      assert.equal(ranking.length, 12);
      assert.ok(ranking.every(r => r.importance === 1.0));
    });

    it("sets minHeads correctly", () => {
      assert.equal(pruner.minHeads, 6); // max(4, floor(12*0.5))
    });

    it("handles small head counts", () => {
      const small = new HeadPruner(4, 2);
      assert.equal(small.minHeads, 4); // max(4, floor(4*0.5)) = max(4, 2) = 4
    });
  });

  describe("recordHeadNorms", () => {
    it("updates importance via EMA", () => {
      const norms = new Float32Array(12);
      norms[0] = 10.0; // head 0 is important
      norms[11] = 0.1; // head 11 is not

      pruner.recordHeadNorms(0, norms);

      const ranking = pruner.getImportanceRanking(0);
      // head 0 should be most important after one update
      assert.equal(ranking[0].head, 0);
      assert.ok(ranking[0].importance > ranking[11].importance);
    });

    it("ignores out-of-range layer indices", () => {
      // Should not throw
      pruner.recordHeadNorms(100, new Float32Array(12));
    });

    it("multiple updates converge toward true norms", () => {
      const norms = new Float32Array(12);
      norms[0] = 10.0;
      for (let i = 1; i < 12; i++) norms[i] = 1.0;

      for (let t = 0; t < 20; t++) {
        pruner.recordHeadNorms(0, norms);
      }

      const ranking = pruner.getImportanceRanking(0);
      // After many updates, head 0 should dominate
      assert.equal(ranking[0].head, 0);
      assert.ok(ranking[0].importance > 5.0);
    });
  });

  describe("getHeadMask", () => {
    it("returns all true when disabled", () => {
      pruner.enabled = false;
      const mask = pruner.getHeadMask(0);
      assert.equal(mask.length, 12);
      assert.ok(mask.every(v => v === true));
    });

    it("prunes low-importance heads", () => {
      // Make heads 0-2 very important, rest unimportant
      const norms = new Float32Array(12).fill(0.01);
      norms[0] = 10; norms[1] = 9; norms[2] = 8;

      for (let t = 0; t < 30; t++) {
        pruner.recordHeadNorms(0, norms);
      }

      const mask = pruner.getHeadMask(0);
      // Heads 0, 1, 2 must be kept
      assert.equal(mask[0], true);
      assert.equal(mask[1], true);
      assert.equal(mask[2], true);
      // Some heads should be pruned
      const pruned = mask.filter(v => !v).length;
      assert.ok(pruned > 0);
    });

    it("never prunes below minHeads", () => {
      // Make only 1 head important
      const norms = new Float32Array(12).fill(0);
      norms[0] = 100;

      for (let t = 0; t < 50; t++) {
        pruner.recordHeadNorms(0, norms);
      }

      const mask = pruner.getHeadMask(0);
      const activeCount = mask.filter(v => v).length;
      assert.ok(activeCount >= pruner.minHeads);
    });

    it("returns all true for out-of-range layer", () => {
      const mask = pruner.getHeadMask(100);
      assert.ok(mask.every(v => v === true));
    });
  });

  describe("stats", () => {
    it("tracks total and skipped head ops", () => {
      const norms = new Float32Array(12).fill(0.01);
      norms[0] = 100;
      for (let t = 0; t < 30; t++) pruner.recordHeadNorms(0, norms);

      pruner.getHeadMask(0);

      const stats = pruner.getStats();
      assert.equal(stats.totalHeadOps, 12);
      assert.ok(stats.skippedHeadOps > 0);
      assert.ok(stats.pruneRate > 0);
    });
  });

  describe("per-layer independence", () => {
    it("different layers can have different importance patterns", () => {
      const normsL0 = new Float32Array(12).fill(1);
      normsL0[0] = 100;

      const normsL1 = new Float32Array(12).fill(1);
      normsL1[11] = 100;

      for (let t = 0; t < 30; t++) {
        pruner.recordHeadNorms(0, normsL0);
        pruner.recordHeadNorms(1, normsL1);
      }

      const rankL0 = pruner.getImportanceRanking(0);
      const rankL1 = pruner.getImportanceRanking(1);
      assert.equal(rankL0[0].head, 0);
      assert.equal(rankL1[0].head, 11);
    });
  });

  describe("reset", () => {
    it("resets importance to uniform and clears stats", () => {
      const norms = new Float32Array(12);
      norms[0] = 100;
      pruner.recordHeadNorms(0, norms);
      pruner.getHeadMask(0);

      pruner.reset();

      const ranking = pruner.getImportanceRanking(0);
      assert.ok(ranking.every(r => r.importance === 1.0));
      assert.equal(pruner.getStats().totalHeadOps, 0);
    });
  });
});

// ─── MixtureOfDepthsRouter ─────────────────────────────────────

describe("MixtureOfDepthsRouter", () => {
  let router;

  beforeEach(() => {
    router = new MixtureOfDepthsRouter(6);
  });

  describe("constructor", () => {
    it("initializes with correct number of layers", () => {
      assert.equal(router.numLayers, 6);
      assert.equal(router.layerDifficulty.length, 6);
      assert.equal(router.skipThresholds.length, 6);
    });

    it("starts disabled", () => {
      assert.equal(router.enabled, false);
    });

    it("protects first and last layers", () => {
      assert.ok(router.protectedLayers.has(0));
      assert.ok(router.protectedLayers.has(5));
      assert.ok(!router.protectedLayers.has(2));
    });

    it("initializes difficulty at 0.5", () => {
      for (let i = 0; i < 6; i++) {
        assert.equal(router.layerDifficulty[i], 0.5);
      }
    });
  });

  describe("route()", () => {
    it("never skips protected layers even when enabled", () => {
      router.enabled = true;
      const hidden = f32(1, 2, 3, 4);

      const r0 = router.route(0, "req1", hidden);
      assert.equal(r0.skip, false);
      assert.equal(r0.reason, "protected");

      const r5 = router.route(5, "req1", hidden);
      assert.equal(r5.skip, false);
      assert.equal(r5.reason, "protected");
    });

    it("never skips when disabled", () => {
      router.enabled = false;
      const hidden = f32(1, 2, 3, 4);

      for (let l = 0; l < 6; l++) {
        const r = router.route(l, "req1", hidden);
        assert.equal(r.skip, false);
      }
    });

    it("tracks stats correctly", () => {
      const hidden = f32(1, 2, 3);
      router.route(0, "req1", hidden);
      router.route(1, "req1", hidden);

      const stats = router.getStats();
      assert.equal(stats.totalRouted, 2);
      assert.equal(stats.layersProcessed, 2);
      assert.equal(stats.layersSkipped, 0);
    });

    it("can skip middle layers when enabled with low difficulty", () => {
      router.enabled = true;
      router.capacity = 0.5; // aggressive — skip 50%

      // Set middle layers to very low difficulty
      router.layerDifficulty[1] = 0.01;
      router.layerDifficulty[2] = 0.01;
      router.layerDifficulty[3] = 0.01;
      router.layerDifficulty[4] = 0.01;

      // Token with stable norms → easy token (low difficulty)
      router.tokenDifficulty.set("req1", { normHistory: [10, 10, 10, 10, 10] });

      const r2 = router.route(2, "req1", f32(1, 2, 3));
      // Easy token + low difficulty layer → high skipScore → should skip
      assert.equal(r2.skip, true);
    });

    it("does not skip high-difficulty layers", () => {
      router.enabled = true;
      router.capacity = 0.5;

      // Layer 2 is very important
      router.layerDifficulty[2] = 0.99;
      router.tokenDifficulty.set("req1", { normHistory: [10, 10, 10, 10, 10] });

      const r2 = router.route(2, "req1", f32(1, 2, 3));
      assert.equal(r2.skip, false);
    });
  });

  describe("recordLayerEffect()", () => {
    it("updates layer difficulty based on hidden state delta", () => {
      const input = f32(1, 0, 0, 0);
      const output = f32(1, 0, 0, 0.01); // very small change

      const before = router.layerDifficulty[2];
      router.recordLayerEffect(2, input, output);
      const after = router.layerDifficulty[2];

      // Small delta → low ratio → difficulty should decrease from 0.5
      assert.ok(after < before, `Expected ${after} < ${before}`);
    });

    it("increases difficulty for large changes", () => {
      const input = f32(1, 0, 0, 0);
      const output = f32(0, 1, 1, 1); // large change

      // First bring difficulty down
      router.layerDifficulty[3] = 0.1;

      router.recordLayerEffect(3, input, output);
      assert.ok(router.layerDifficulty[3] > 0.1);
    });

    it("ignores out-of-range layer indices", () => {
      const a = f32(1, 2, 3);
      router.recordLayerEffect(99, a, a); // should not throw
    });
  });

  describe("getLayerMask()", () => {
    it("returns all-true mask when disabled", () => {
      const mask = router.getLayerMask("req1", f32(1, 2, 3));
      assert.equal(mask.length, 6);
      assert.ok(mask.every(v => v === true));
    });

    it("does not pollute stats", () => {
      router.getLayerMask("req1", f32(1, 2, 3));
      const stats = router.getStats();
      assert.equal(stats.totalRouted, 0);
      assert.equal(stats.layersProcessed, 0);
    });

    it("returns correct mask length", () => {
      router.enabled = true;
      const mask = router.getLayerMask("req1", f32(1, 2, 3));
      assert.equal(mask.length, 6);
      // First and last always true (protected)
      assert.equal(mask[0], true);
      assert.equal(mask[5], true);
    });
  });

  describe("observeToken()", () => {
    it("tracks norm history per request", () => {
      router.observeToken("req1", f32(3, 4)); // norm = 5
      router.observeToken("req1", f32(6, 8)); // norm = 10

      const entry = router.tokenDifficulty.get("req1");
      assert.equal(entry.normHistory.length, 2);
      assert.ok(Math.abs(entry.normHistory[0] - 5) < 0.01);
      assert.ok(Math.abs(entry.normHistory[1] - 10) < 0.01);
    });

    it("caps history at 16 entries", () => {
      for (let i = 0; i < 20; i++) {
        router.observeToken("req1", f32(i + 1, 0));
      }
      const entry = router.tokenDifficulty.get("req1");
      assert.equal(entry.normHistory.length, 16);
    });

    it("isolates different requests", () => {
      router.observeToken("req1", f32(1, 0));
      router.observeToken("req2", f32(2, 0));

      assert.equal(router.tokenDifficulty.get("req1").normHistory.length, 1);
      assert.equal(router.tokenDifficulty.get("req2").normHistory.length, 1);
    });
  });

  describe("clear()", () => {
    it("removes token tracking for a request", () => {
      router.observeToken("req1", f32(1, 2));
      assert.ok(router.tokenDifficulty.has("req1"));

      router.clear("req1");
      assert.ok(!router.tokenDifficulty.has("req1"));
    });
  });

  describe("getStats()", () => {
    it("computes skip rate", () => {
      router.stats.totalRouted = 10;
      router.stats.layersSkipped = 3;
      router.stats.layersProcessed = 7;

      const stats = router.getStats();
      assert.ok(Math.abs(stats.skipRate - 0.3) < 0.001);
    });

    it("returns zero skip rate with no routing", () => {
      const stats = router.getStats();
      assert.equal(stats.skipRate, 0);
    });

    it("includes layer difficulty array", () => {
      const stats = router.getStats();
      assert.equal(stats.layerDifficulty.length, 6);
    });
  });

  describe("reset()", () => {
    it("restores initial state", () => {
      router.enabled = true;
      router.observeToken("req1", f32(1, 2));
      router.stats.totalRouted = 50;
      router.layerDifficulty[2] = 0.99;

      router.reset();

      assert.equal(router.stats.totalRouted, 0);
      assert.equal(router.layerDifficulty[2], 0.5);
      assert.equal(router.tokenDifficulty.size, 0);
    });
  });

  describe("token difficulty estimation", () => {
    it("returns 0.5 for unknown tokens", () => {
      // Access private method through route behavior
      router.enabled = true;
      router.layerDifficulty[2] = 0.5;

      // No observations for req1 → difficulty defaults to 0.5
      const r = router.route(2, "unknown_req", f32(1, 2, 3));
      assert.ok(!r.skip); // medium difficulty token + medium layer → no skip at default capacity
    });

    it("classifies stable norms as easy", () => {
      // Stable norms → low coefficient of variation → easy token
      router.tokenDifficulty.set("stable", { normHistory: [10, 10, 10, 10, 10, 10] });

      router.enabled = true;
      router.capacity = 0.5;
      router.layerDifficulty[2] = 0.01; // unimportant layer

      const r = router.route(2, "stable", f32(1, 2, 3));
      assert.equal(r.skip, true, "Stable token + low-difficulty layer should skip");
    });

    it("classifies volatile norms as hard", () => {
      // Wildly varying norms → high CV → hard token
      router.tokenDifficulty.set("volatile", { normHistory: [1, 100, 2, 90, 3, 80] });

      router.enabled = true;
      router.capacity = 0.5;
      router.layerDifficulty[2] = 0.01;

      const r = router.route(2, "volatile", f32(1, 2, 3));
      // Hard token should not skip even with low-difficulty layer
      assert.equal(r.skip, false, "Volatile token should not skip");
    });
  });

  describe("capacity tuning", () => {
    it("higher capacity means fewer skips", () => {
      router.enabled = true;

      // Setup: easy token, unimportant layer
      router.tokenDifficulty.set("req1", { normHistory: [5, 5, 5, 5, 5] });
      router.layerDifficulty[2] = 0.01;

      // High capacity = almost everything processes
      router.capacity = 0.99;
      const r1 = router.route(2, "req1", f32(1, 2, 3));
      // threshold = 1 - 0.99 = 0.01, very hard to exceed
      assert.equal(r1.skip, false);
    });

    it("lower capacity means more skips", () => {
      router.enabled = true;

      router.tokenDifficulty.set("req1", { normHistory: [5, 5, 5, 5, 5] });
      router.layerDifficulty[2] = 0.01;

      // Low capacity = aggressive skipping
      router.capacity = 0.3;
      // Reset stats from previous route calls
      router.stats.totalRouted = 0;
      router.stats.layersSkipped = 0;
      router.stats.layersProcessed = 0;

      const r = router.route(2, "req1", f32(1, 2, 3));
      assert.equal(r.skip, true);
    });
  });
});
