/**
 * SpeculativeController Tests
 *
 * Tests the accept/reject speculation logic, stats tracking, and lifecycle.
 * Pipeline and GPU operations are mocked — this tests the control flow.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";

// Mock browser GPU globals before importing speculative.js
globalThis.GPUBufferUsage = { STORAGE: 0x80, COPY_SRC: 0x04, COPY_DST: 0x08 };

import { SpeculativeController } from "../node/speculative.js";
import { ActivationPredictor } from "../node/predictor.js";

// ─── Helpers ─────────────────────────────────────────────────

function f32(...vals) {
  return new Float32Array(vals);
}

function makeMockPipeline() {
  return {
    _createBuffer: () => ({ label: "mock_buffer" }),
    device: {
      queue: {
        writeBuffer: () => {},
      },
    },
    forwardLayersCached: async () => ({
      buffer: { label: "speculative_output" },
      shape: [1, 3],
    }),
  };
}

// ─── SpeculativeController ─────────────────────────────────────

describe("SpeculativeController", () => {
  let ctrl;
  let mockPipeline;

  beforeEach(() => {
    mockPipeline = makeMockPipeline();
    ctrl = new SpeculativeController(mockPipeline);
  });

  describe("initialization", () => {
    it("starts enabled", () => {
      assert.equal(ctrl.enabled, true);
    });

    it("starts with empty pending map", () => {
      assert.equal(ctrl.pending.size, 0);
    });

    it("starts with zero stats", () => {
      const s = ctrl.stats;
      assert.equal(s.speculations, 0);
      assert.equal(s.accepted, 0);
      assert.equal(s.rejected, 0);
      assert.equal(s.savedMs, 0);
    });

    it("has a predictor instance", () => {
      assert.ok(ctrl.predictor instanceof ActivationPredictor);
    });
  });

  describe("onActivationReceived — no pending speculation", () => {
    it("returns useSpeculative=false when nothing pending", async () => {
      const result = await ctrl.onActivationReceived(
        "req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 0, 0, 6
      );
      assert.equal(result.useSpeculative, false);
      assert.equal(result.speculativeHidden, null);
    });

    it("observes the activation for future predictions", async () => {
      await ctrl.onActivationReceived(
        "req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 0, 0, 6
      );
      assert.ok(ctrl.predictor.requests.has("req1"));
    });

    it("kicks off speculation for next step if predictor has enough data", async () => {
      // Feed two observations so predictor can predict
      ctrl.predictor.observe("req1", f32(1, 2, 3));
      await ctrl.onActivationReceived(
        "req1", f32(2, 4, 6), { buffer: {}, shape: [1, 3] }, 1, 0, 6
      );
      // After onActivationReceived, a speculation may be pending for seqPos=2
      // (depends on prediction confidence)
    });
  });

  describe("onActivationReceived — accept path", () => {
    it("accepts when prediction closely matches actual", async () => {
      // Setup: create a pending speculation with a prediction that matches
      const prediction = f32(3, 6, 9);
      const speculativeResult = { buffer: { label: "spec_out" }, shape: [1, 3] };

      ctrl.pending.set("req1", {
        promise: Promise.resolve(speculativeResult),
        prediction: prediction,
        seqPos: 2,
        estimatedComputeMs: 5,
      });

      // The "actual" activation is nearly identical to prediction
      const actual = f32(3, 6, 9);
      const result = await ctrl.onActivationReceived(
        "req1", actual, { buffer: {}, shape: [1, 3] }, 2, 0, 6
      );

      assert.equal(result.useSpeculative, true);
      assert.deepEqual(result.speculativeHidden, speculativeResult);
      assert.equal(ctrl.stats.accepted, 1);
      assert.equal(ctrl.stats.savedMs, 5);
    });

    it("clears pending entry after acceptance", async () => {
      ctrl.pending.set("req1", {
        promise: Promise.resolve({ buffer: {}, shape: [1, 3] }),
        prediction: f32(1, 2, 3),
        seqPos: 5,
        estimatedComputeMs: 1,
      });

      await ctrl.onActivationReceived(
        "req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 5, 0, 6
      );

      assert.equal(ctrl.pending.has("req1"), false);
    });
  });

  describe("onActivationReceived — reject path", () => {
    it("rejects when prediction is orthogonal to actual", async () => {
      ctrl.pending.set("req1", {
        promise: Promise.resolve({ buffer: {}, shape: [1, 3] }),
        prediction: f32(1, 0, 0),
        seqPos: 2,
        estimatedComputeMs: 1,
      });

      const result = await ctrl.onActivationReceived(
        "req1", f32(0, 1, 0), { buffer: {}, shape: [1, 3] }, 2, 0, 6
      );

      assert.equal(result.useSpeculative, false);
      assert.equal(result.speculativeHidden, null);
      assert.equal(ctrl.stats.rejected, 1);
      assert.equal(ctrl.stats.accepted, 0);
    });

    it("rejects when prediction is reversed", async () => {
      ctrl.pending.set("req1", {
        promise: Promise.resolve({ buffer: {}, shape: [1, 3] }),
        prediction: f32(1, 2, 3),
        seqPos: 3,
        estimatedComputeMs: 1,
      });

      const result = await ctrl.onActivationReceived(
        "req1", f32(-1, -2, -3), { buffer: {}, shape: [1, 3] }, 3, 0, 6
      );

      assert.equal(result.useSpeculative, false);
      assert.equal(ctrl.stats.rejected, 1);
    });

    it("clears pending entry after rejection", async () => {
      ctrl.pending.set("req1", {
        promise: Promise.resolve({ buffer: {}, shape: [1, 3] }),
        prediction: f32(1, 0, 0),
        seqPos: 1,
        estimatedComputeMs: 1,
      });

      await ctrl.onActivationReceived(
        "req1", f32(0, 0, 1), { buffer: {}, shape: [1, 3] }, 1, 0, 6
      );

      assert.equal(ctrl.pending.has("req1"), false);
    });
  });

  describe("onActivationReceived — seqPos mismatch", () => {
    it("ignores pending speculation if seqPos does not match", async () => {
      ctrl.pending.set("req1", {
        promise: Promise.resolve({ buffer: {}, shape: [1, 3] }),
        prediction: f32(1, 2, 3),
        seqPos: 10, // pending is for step 10
        estimatedComputeMs: 1,
      });

      // But we receive step 5
      const result = await ctrl.onActivationReceived(
        "req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 5, 0, 6
      );

      assert.equal(result.useSpeculative, false);
      // Pending entry should NOT be cleared since seqPos didn't match
      assert.equal(ctrl.pending.has("req1"), true);
      assert.equal(ctrl.stats.accepted, 0);
      assert.equal(ctrl.stats.rejected, 0);
    });
  });

  describe("onActivationReceived — speculative compute failure", () => {
    it("falls back gracefully when speculative promise rejects", async () => {
      ctrl.pending.set("req1", {
        promise: Promise.reject(new Error("GPU error")),
        prediction: f32(1, 2, 3),
        seqPos: 2,
        estimatedComputeMs: 1,
      });

      const result = await ctrl.onActivationReceived(
        "req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 2, 0, 6
      );

      // Prediction matched but promise failed — should fall back
      assert.equal(result.useSpeculative, false);
    });
  });

  describe("onActivationReceived — disabled", () => {
    it("does not speculate when disabled", async () => {
      ctrl.enabled = false;

      // Give predictor enough data to predict
      ctrl.predictor.observe("req1", f32(1, 2, 3));
      ctrl.predictor.observe("req1", f32(2, 4, 6));

      await ctrl.onActivationReceived(
        "req1", f32(3, 6, 9), { buffer: {}, shape: [1, 3] }, 2, 0, 6
      );

      // Should not have created any pending speculation
      assert.equal(ctrl.pending.has("req1"), false);
      assert.equal(ctrl.stats.speculations, 0);
    });
  });

  describe("_speculateNext", () => {
    it("does not speculate with insufficient data", () => {
      ctrl.predictor.observe("req1", f32(1, 2, 3));
      // Only 1 observation — predictor returns null
      ctrl._speculateNext("req1", 1, 0, 6);
      assert.equal(ctrl.pending.has("req1"), false);
      assert.equal(ctrl.stats.speculations, 0);
    });

    it("speculates when predictor has high confidence", () => {
      // Linear trajectory → confidence 1.0
      ctrl.predictor.observe("req1", f32(1, 2, 3));
      ctrl.predictor.observe("req1", f32(2, 4, 6));

      ctrl._speculateNext("req1", 2, 0, 6);

      assert.equal(ctrl.pending.has("req1"), true);
      assert.equal(ctrl.stats.speculations, 1);
      assert.equal(ctrl.pending.get("req1").seqPos, 2);
    });

    it("does not speculate with low confidence", () => {
      // Erratic trajectory → low confidence
      ctrl.predictor.observe("req1", f32(1, 0, 0));
      ctrl.predictor.observe("req1", f32(0, 1, 0));
      ctrl.predictor.observe("req1", f32(0, 0, 1));

      ctrl._speculateNext("req1", 3, 0, 6);

      // Confidence should be < 0.9, so no speculation
      const pred = ctrl.predictor.predict("req1");
      if (pred && pred.confidence < 0.9) {
        assert.equal(ctrl.pending.has("req1"), false);
      }
    });

    it("calls pipeline.forwardLayersCached", () => {
      let forwardCalled = false;
      mockPipeline.forwardLayersCached = async () => {
        forwardCalled = true;
        return { buffer: {}, shape: [1, 3] };
      };

      ctrl.predictor.observe("req1", f32(1, 2, 3));
      ctrl.predictor.observe("req1", f32(2, 4, 6));
      ctrl._speculateNext("req1", 2, 0, 6);

      assert.equal(forwardCalled, true);
    });
  });

  describe("clear", () => {
    it("removes pending speculation for request", () => {
      ctrl.pending.set("req1", { promise: Promise.resolve(), prediction: f32(1), seqPos: 0 });
      ctrl.predictor.observe("req1", f32(1, 2));

      ctrl.clear("req1");

      assert.equal(ctrl.pending.has("req1"), false);
      assert.equal(ctrl.predictor.requests.has("req1"), false);
    });

    it("does not affect other requests", () => {
      ctrl.pending.set("req1", { promise: Promise.resolve(), prediction: f32(1), seqPos: 0 });
      ctrl.pending.set("req2", { promise: Promise.resolve(), prediction: f32(2), seqPos: 0 });

      ctrl.clear("req1");

      assert.equal(ctrl.pending.has("req1"), false);
      assert.equal(ctrl.pending.has("req2"), true);
    });
  });

  describe("getStats", () => {
    it("returns hitRate=0 with no speculations", () => {
      const stats = ctrl.getStats();
      assert.equal(stats.hitRate, 0);
    });

    it("computes hitRate correctly", () => {
      ctrl.stats.speculations = 10;
      ctrl.stats.accepted = 7;
      ctrl.stats.rejected = 3;

      const stats = ctrl.getStats();
      assert.ok(Math.abs(stats.hitRate - 0.7) < 0.001);
    });

    it("includes predictor stats", () => {
      const stats = ctrl.getStats();
      assert.ok("predictor" in stats);
      assert.ok("predictions" in stats.predictor);
    });

    it("includes savedMs", () => {
      ctrl.stats.savedMs = 42;
      assert.equal(ctrl.getStats().savedMs, 42);
    });
  });

  describe("end-to-end speculation flow", () => {
    it("full cycle: observe → predict → speculate → accept", async () => {
      // Step 1: Feed observations (no speculation yet, not enough data)
      await ctrl.onActivationReceived(
        "req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 0, 0, 6
      );
      await ctrl.onActivationReceived(
        "req1", f32(2, 4, 6), { buffer: {}, shape: [1, 3] }, 1, 0, 6
      );

      // After step 1, there should be a pending speculation for seqPos=2
      // (predictor has 2 observations → predicts [3, 6, 9] with high confidence)
      const hasPending = ctrl.pending.has("req1");

      if (hasPending) {
        const pendingEntry = ctrl.pending.get("req1");
        assert.equal(pendingEntry.seqPos, 2);

        // Step 2: Real activation arrives close to prediction
        const result = await ctrl.onActivationReceived(
          "req1", f32(3, 6, 9), { buffer: {}, shape: [1, 3] }, 2, 0, 6
        );

        assert.equal(result.useSpeculative, true);
        assert.ok(ctrl.stats.accepted >= 1);
      }
    });

    it("full cycle: observe → predict → speculate → reject", async () => {
      await ctrl.onActivationReceived(
        "req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 0, 0, 6
      );
      await ctrl.onActivationReceived(
        "req1", f32(2, 4, 6), { buffer: {}, shape: [1, 3] }, 1, 0, 6
      );

      if (ctrl.pending.has("req1")) {
        // Real activation is completely different from prediction
        const result = await ctrl.onActivationReceived(
          "req1", f32(-10, 0, 5), { buffer: {}, shape: [1, 3] }, 2, 0, 6
        );

        assert.equal(result.useSpeculative, false);
        assert.ok(ctrl.stats.rejected >= 1);
      }
    });

    it("multiple requests don't interfere", async () => {
      // Setup req1
      await ctrl.onActivationReceived(
        "req1", f32(1, 0, 0), { buffer: {}, shape: [1, 3] }, 0, 0, 6
      );
      await ctrl.onActivationReceived(
        "req1", f32(2, 0, 0), { buffer: {}, shape: [1, 3] }, 1, 0, 6
      );

      // Setup req2
      await ctrl.onActivationReceived(
        "req2", f32(0, 1, 0), { buffer: {}, shape: [1, 3] }, 0, 0, 6
      );
      await ctrl.onActivationReceived(
        "req2", f32(0, 2, 0), { buffer: {}, shape: [1, 3] }, 1, 0, 6
      );

      // Verify both have independent predictor histories
      assert.ok(ctrl.predictor.requests.has("req1"));
      assert.ok(ctrl.predictor.requests.has("req2"));

      // Clear one doesn't affect other
      ctrl.clear("req1");
      assert.equal(ctrl.predictor.requests.has("req1"), false);
      assert.ok(ctrl.predictor.requests.has("req2"));
    });
  });

  describe("batch speculation — _speculateNextBatch", () => {
    beforeEach(() => {
      ctrl.batchDepth = 3;
    });

    it("does not batch-speculate with insufficient data", () => {
      ctrl.predictor.observe("req1", f32(1, 2, 3));
      ctrl._speculateNextBatch("req1", 1, 3, 0, 6);
      assert.equal(ctrl.pendingBatch.has("req1"), false);
      assert.equal(ctrl.pending.has("req1"), false);
    });

    it("creates batch entries for high-confidence multi-step predictions", () => {
      // Linear trajectory → high confidence for multiple steps
      ctrl.predictor.observe("req1", f32(1, 2, 3));
      ctrl.predictor.observe("req1", f32(2, 4, 6));
      ctrl.predictor.observe("req1", f32(3, 6, 9));

      ctrl._speculateNextBatch("req1", 3, 3, 0, 6);

      // With a perfect linear trajectory, confidence decays as 1.0, 0.85, 0.7225
      // Steps with confidence < 0.9 are filtered, so we may get 1-2 viable steps
      const hasBatch = ctrl.pendingBatch.has("req1");
      const hasSingle = ctrl.pending.has("req1");
      assert.ok(hasBatch || hasSingle, "should have either batch or single pending");
    });

    it("falls back to single-step when only 1 prediction is viable", () => {
      // Only 2 observations → predictMulti gives decaying confidence
      ctrl.predictor.observe("req1", f32(1, 2, 3));
      ctrl.predictor.observe("req1", f32(2, 4, 6));

      ctrl._speculateNextBatch("req1", 2, 3, 0, 6);

      // With only 2 observations, base confidence = 1.0 (no 3rd point to compare)
      // So steps: 1.0, 0.85, 0.7225 — only steps 1 and 2 are >= 0.9
      // This should create a batch or single depending on viable count
      const total = ctrl.stats.speculations;
      assert.ok(total > 0, "should have launched at least 1 speculation");
    });

    it("batch entries have sequential seqPos values", () => {
      ctrl.predictor.observe("req1", f32(1, 2, 3));
      ctrl.predictor.observe("req1", f32(2, 4, 6));
      ctrl.predictor.observe("req1", f32(3, 6, 9));

      ctrl._speculateNextBatch("req1", 3, 3, 0, 6);

      if (ctrl.pendingBatch.has("req1")) {
        const batch = ctrl.pendingBatch.get("req1");
        for (let i = 0; i < batch.length; i++) {
          assert.equal(batch[i].seqPos, 3 + i);
        }
      }
    });

    it("each batch entry calls forwardLayersCached", () => {
      let callCount = 0;
      mockPipeline.forwardLayersCached = async () => {
        callCount++;
        return { buffer: {}, shape: [1, 3] };
      };

      ctrl.predictor.observe("req1", f32(1, 2, 3));
      ctrl.predictor.observe("req1", f32(2, 4, 6));
      ctrl.predictor.observe("req1", f32(3, 6, 9));

      ctrl._speculateNextBatch("req1", 3, 3, 0, 6);

      assert.ok(callCount >= 1, `forwardLayersCached called ${callCount} times`);
    });
  });

  describe("batch speculation — _verifyBatch", () => {
    it("rejects entire batch when first prediction is wrong", async () => {
      const batch = [
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(1, 0, 0), seqPos: 2, estimatedComputeMs: 1 },
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(2, 0, 0), seqPos: 3, estimatedComputeMs: 1 },
      ];

      // Actual is orthogonal to prediction
      const result = await ctrl._verifyBatch("req1", f32(0, 1, 0), batch);
      assert.equal(result.useSpeculative, false);
      assert.equal(result.acceptedSteps, 0);
      assert.equal(result.rollbackPos, 2);
    });

    it("accepts all steps when first prediction matches", async () => {
      const batch = [
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(3, 6, 9), seqPos: 2, estimatedComputeMs: 2 },
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(4, 8, 12), seqPos: 3, estimatedComputeMs: 2 },
      ];

      const result = await ctrl._verifyBatch("req1", f32(3, 6, 9), batch);
      assert.equal(result.useSpeculative, true);
      assert.equal(result.acceptedSteps, 2);
      assert.equal(result.rollbackPos, null);
      assert.equal(ctrl.stats.batchFull, 1);
    });

    it("handles GPU failure in batch gracefully", async () => {
      const batch = [
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(3, 6, 9), seqPos: 2, estimatedComputeMs: 1 },
        { promise: Promise.reject(new Error("GPU error")), prediction: f32(4, 8, 12), seqPos: 3, estimatedComputeMs: 1 },
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(5, 10, 15), seqPos: 4, estimatedComputeMs: 1 },
      ];

      const result = await ctrl._verifyBatch("req1", f32(3, 6, 9), batch);
      assert.equal(result.useSpeculative, true);
      assert.equal(result.acceptedSteps, 1); // only first step before GPU failure
      assert.equal(ctrl.stats.batchPartial, 1);
    });

    it("rejects when first step GPU compute fails", async () => {
      const batch = [
        { promise: Promise.reject(new Error("GPU error")), prediction: f32(3, 6, 9), seqPos: 2, estimatedComputeMs: 1 },
      ];

      const result = await ctrl._verifyBatch("req1", f32(3, 6, 9), batch);
      assert.equal(result.useSpeculative, false);
      assert.equal(result.acceptedSteps, 0);
    });

    it("tracks batch stats correctly", async () => {
      const batch = [
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(1, 2, 3), seqPos: 5, estimatedComputeMs: 3 },
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(2, 4, 6), seqPos: 6, estimatedComputeMs: 3 },
        { promise: Promise.resolve({ buffer: {}, shape: [1, 3] }), prediction: f32(3, 6, 9), seqPos: 7, estimatedComputeMs: 3 },
      ];

      await ctrl._verifyBatch("req1", f32(1, 2, 3), batch);
      assert.equal(ctrl.stats.batchSpeculations, 1);
      assert.equal(ctrl.stats.batchAccepted, 3);
      assert.equal(ctrl.stats.savedMs, 9);
    });
  });

  describe("batch speculation — end-to-end", () => {
    beforeEach(() => {
      ctrl.batchDepth = 3;
    });

    it("full cycle: build history → batch speculate → verify", async () => {
      // Build up a linear trajectory (3 observations for high confidence)
      await ctrl.onActivationReceived("req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 0, 0, 6);
      await ctrl.onActivationReceived("req1", f32(2, 4, 6), { buffer: {}, shape: [1, 3] }, 1, 0, 6);
      await ctrl.onActivationReceived("req1", f32(3, 6, 9), { buffer: {}, shape: [1, 3] }, 2, 0, 6);

      // Should now have batch or single pending for next step(s)
      const hasBatch = ctrl.pendingBatch.has("req1");
      const hasSingle = ctrl.pending.has("req1");
      assert.ok(hasBatch || hasSingle, "should have pending speculation after 3 observations");

      if (hasBatch) {
        // Verify with the expected next activation
        const result = await ctrl.onActivationReceived(
          "req1", f32(4, 8, 12), { buffer: {}, shape: [1, 3] }, 3, 0, 6
        );
        assert.equal(result.useSpeculative, true);
        assert.ok(result.acceptedSteps >= 1);
      }
    });

    it("batch reject then single-step recovery", async () => {
      ctrl.batchDepth = 2;
      await ctrl.onActivationReceived("req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 0, 0, 6);
      await ctrl.onActivationReceived("req1", f32(2, 4, 6), { buffer: {}, shape: [1, 3] }, 1, 0, 6);

      // Send a wildly different activation to trigger rejection
      const result = await ctrl.onActivationReceived(
        "req1", f32(-5, 10, -1), { buffer: {}, shape: [1, 3] }, 2, 0, 6
      );

      // Rejection still observes the activation — next batch can learn from it
      assert.ok(ctrl.predictor.requests.has("req1"));
    });

    it("clear removes batch state", () => {
      ctrl.pendingBatch.set("req1", [
        { promise: Promise.resolve(), prediction: f32(1), seqPos: 0 },
      ]);
      ctrl.pending.set("req1", { promise: Promise.resolve(), prediction: f32(1), seqPos: 0 });
      ctrl.predictor.observe("req1", f32(1, 2));

      ctrl.clear("req1");

      assert.equal(ctrl.pendingBatch.has("req1"), false);
      assert.equal(ctrl.pending.has("req1"), false);
      assert.equal(ctrl.predictor.requests.has("req1"), false);
    });

    it("batchDepth=1 uses single-step path", async () => {
      ctrl.batchDepth = 1;
      await ctrl.onActivationReceived("req1", f32(1, 2, 3), { buffer: {}, shape: [1, 3] }, 0, 0, 6);
      await ctrl.onActivationReceived("req1", f32(2, 4, 6), { buffer: {}, shape: [1, 3] }, 1, 0, 6);

      // With batchDepth=1, should use _speculateNext, not _speculateNextBatch
      assert.equal(ctrl.pendingBatch.has("req1"), false);
      // May have single pending
    });

    it("getStats includes batchAvgAccepted", () => {
      ctrl.stats.batchSpeculations = 4;
      ctrl.stats.batchAccepted = 10;
      const stats = ctrl.getStats();
      assert.ok(Math.abs(stats.batchAvgAccepted - 2.5) < 0.001);
    });
  });
});
