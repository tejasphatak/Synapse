/**
 * Speculative Execution Controller
 *
 * Orchestrates prediction → speculative compute → verify → accept/recompute.
 *
 * The key insight: network latency (50ms) >> GPU compute (1ms).
 * If we can predict the next activation with >99.5% cosine similarity,
 * we start computing on the prediction immediately. When the real
 * activation arrives, we check: was the prediction good enough?
 *
 * If yes → keep speculative result, save 50ms
 * If no  → discard speculative result, recompute with real data (no worse than baseline)
 *
 * VLSI analogy: branch prediction. Wrong predictions cost one pipeline flush.
 * Right predictions hide the branch latency entirely.
 */

import { ActivationPredictor } from "./predictor.js";

export class SpeculativeController {
  constructor(pipeline) {
    this.pipeline = pipeline;
    this.predictor = new ActivationPredictor();

    // Pending speculative work per request
    // Key: requestId, Value: { promise, prediction, seqPos }
    this.pending = new Map();

    // Whether speculation is enabled (can be toggled at runtime)
    this.enabled = true;

    // Stats
    this.stats = {
      speculations: 0,
      accepted: 0,
      rejected: 0,
      savedMs: 0,
    };
  }

  /**
   * Called when a real activation arrives. Does three things:
   * 1. If there's a pending speculation, verify it
   * 2. Record this activation for future predictions
   * 3. Kick off speculation for the NEXT step
   *
   * @param {string} requestId
   * @param {Float32Array} activationFloat32 - the decoded real activation
   * @param {object} hidden - { buffer, shape } GPU tensor
   * @param {number} seqPos - current sequence position
   * @param {number} layerStart
   * @param {number} layerEnd
   * @returns {{ useSpeculative: boolean, speculativeHidden: object|null, computePromise: Promise|null }}
   */
  async onActivationReceived(requestId, activationFloat32, hidden, seqPos, layerStart, layerEnd) {
    let useSpeculative = false;
    let speculativeHidden = null;

    // Step 1: Check if we had a pending speculation for this step
    const pendingSpec = this.pending.get(requestId);
    if (pendingSpec && pendingSpec.seqPos === seqPos) {
      // Verify the prediction
      const verification = this.predictor.verify(pendingSpec.prediction, activationFloat32);

      if (verification.accept) {
        // Prediction was good — use the speculative result
        try {
          speculativeHidden = await pendingSpec.promise;
          useSpeculative = true;
          this.stats.accepted++;
          this.stats.savedMs += pendingSpec.estimatedComputeMs || 1;
        } catch (err) {
          // Speculative compute failed — fall back to real
          useSpeculative = false;
        }
      } else {
        // Prediction was bad — discard speculative result
        this.stats.rejected++;
        // KV cache for the speculative step needs to be rolled back
        // For now, we just overwrite it during the real compute
      }
      this.pending.delete(requestId);
    }

    // Step 2: Record observation for future predictions
    this.predictor.observe(requestId, activationFloat32);

    // Step 3: Predict next step and kick off speculative compute
    if (this.enabled) {
      this._speculateNext(requestId, seqPos + 1, layerStart, layerEnd);
    }

    return { useSpeculative, speculativeHidden };
  }

  /**
   * Predict the next activation and start computing on it speculatively.
   * This runs in the background — the result is checked when the real
   * activation arrives.
   */
  _speculateNext(requestId, nextSeqPos, layerStart, layerEnd) {
    const prediction = this.predictor.predict(requestId);
    if (!prediction || prediction.confidence < 0.9) {
      return; // not confident enough to speculate
    }

    this.stats.speculations++;

    // Upload prediction to GPU
    const size = prediction.prediction.length;
    const shape = [1, size];
    const buffer = this.pipeline._createBuffer(
      "speculative_input",
      prediction.prediction.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    this.pipeline.device.queue.writeBuffer(buffer, 0, prediction.prediction);

    const speculativeHidden = { buffer, shape };

    // Start computing — this runs concurrently with the network transfer
    const promise = this.pipeline.forwardLayersCached(
      speculativeHidden, layerStart, layerEnd, requestId, nextSeqPos
    );

    this.pending.set(requestId, {
      promise,
      prediction: prediction.prediction,
      seqPos: nextSeqPos,
      estimatedComputeMs: 1,
    });
  }

  /**
   * Clear state for a completed generation.
   */
  clear(requestId) {
    this.pending.delete(requestId);
    this.predictor.clear(requestId);
  }

  /**
   * Get combined stats.
   */
  getStats() {
    return {
      ...this.stats,
      hitRate: this.stats.speculations > 0
        ? this.stats.accepted / this.stats.speculations
        : 0,
      predictor: this.predictor.getStats(),
    };
  }
}
