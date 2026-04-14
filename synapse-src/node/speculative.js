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

    // Pending batch speculations per request
    // Key: requestId, Value: [{ promise, prediction, seqPos, estimatedComputeMs }, ...]
    this.pendingBatch = new Map();

    // Whether speculation is enabled (can be toggled at runtime)
    this.enabled = true;

    // Batch speculation depth (1 = single-step legacy, 2-4 = batch)
    this.batchDepth = 3;

    // Stats
    this.stats = {
      speculations: 0,
      accepted: 0,
      rejected: 0,
      savedMs: 0,
      batchSpeculations: 0,
      batchAccepted: 0,   // total steps accepted across all batches
      batchPartial: 0,    // batches where some but not all steps accepted
      batchFull: 0,       // batches where all K steps accepted
    };
  }

  /**
   * Called when a real activation arrives. Does three things:
   * 1. If there's a pending speculation (single or batch), verify it
   * 2. Record this activation for future predictions
   * 3. Kick off speculation for the NEXT step(s)
   *
   * @param {string} requestId
   * @param {Float32Array} activationFloat32 - the decoded real activation
   * @param {object} hidden - { buffer, shape } GPU tensor
   * @param {number} seqPos - current sequence position
   * @param {number} layerStart
   * @param {number} layerEnd
   * @returns {{ useSpeculative: boolean, speculativeHidden: object|null, acceptedSteps: number }}
   */
  async onActivationReceived(requestId, activationFloat32, hidden, seqPos, layerStart, layerEnd) {
    let useSpeculative = false;
    let speculativeHidden = null;
    let acceptedSteps = 0;
    let rollbackPos = null;

    // Step 1a: Check batch speculation first (takes priority over single)
    const batch = this.pendingBatch.get(requestId);
    if (batch && batch.length > 0 && batch[0].seqPos === seqPos) {
      const result = await this._verifyBatch(requestId, activationFloat32, batch);
      useSpeculative = result.useSpeculative;
      speculativeHidden = result.speculativeHidden;
      acceptedSteps = result.acceptedSteps;
      rollbackPos = result.rollbackPos;
      this.pendingBatch.delete(requestId);
    }
    // Step 1b: Fall back to single pending speculation
    else {
      const pendingSpec = this.pending.get(requestId);
      if (pendingSpec && pendingSpec.seqPos === seqPos) {
        const verification = this.predictor.verify(pendingSpec.prediction, activationFloat32);

        if (verification.accept) {
          try {
            speculativeHidden = await pendingSpec.promise;
            useSpeculative = true;
            acceptedSteps = 1;
            this.stats.accepted++;
            this.stats.savedMs += pendingSpec.estimatedComputeMs || 1;
          } catch (err) {
            useSpeculative = false;
            rollbackPos = seqPos;
          }
        } else {
          this.stats.rejected++;
          rollbackPos = seqPos;
        }
        this.pending.delete(requestId);
      }
    }

    // Step 2: Record observation for future predictions
    this.predictor.observe(requestId, activationFloat32);

    // Step 3: Predict next step(s) and kick off speculative compute
    if (this.enabled) {
      if (this.batchDepth > 1) {
        this._speculateNextBatch(requestId, seqPos + 1, this.batchDepth, layerStart, layerEnd);
      } else {
        this._speculateNext(requestId, seqPos + 1, layerStart, layerEnd);
      }
    }

    return { useSpeculative, speculativeHidden, acceptedSteps, rollbackPos };
  }

  /**
   * Verify a batch of speculative predictions against the real activation.
   * Accepts the longest correct prefix — if step 1 is wrong, all are rejected.
   * If step 1 is right but step 2 is wrong, only step 1 is accepted.
   *
   * @returns {{ useSpeculative: boolean, speculativeHidden: object|null, acceptedSteps: number, rollbackPos: number|null }}
   */
  async _verifyBatch(requestId, activationFloat32, batch) {
    // First entry must match the current activation
    const first = batch[0];
    const verification = this.predictor.verify(first.prediction, activationFloat32);

    if (!verification.accept) {
      // First step wrong — reject entire batch
      this.stats.rejected++;
      this.stats.batchSpeculations++;
      return { useSpeculative: false, speculativeHidden: null, acceptedSteps: 0, rollbackPos: first.seqPos };
    }

    // First step accepted — now check how many subsequent steps we can keep
    let acceptedSteps = 1;
    let lastGoodResult = null;

    try {
      lastGoodResult = await first.promise;
    } catch (err) {
      this.stats.rejected++;
      this.stats.batchSpeculations++;
      return { useSpeculative: false, speculativeHidden: null, acceptedSteps: 0, rollbackPos: first.seqPos };
    }

    // For subsequent steps, we can't verify against real data (it hasn't arrived yet).
    // We accept them if the GPU compute succeeded. The real verification happens
    // when those future activations arrive — if wrong, KV cache gets rolled back.
    for (let i = 1; i < batch.length; i++) {
      try {
        lastGoodResult = await batch[i].promise;
        acceptedSteps++;
      } catch (err) {
        break; // GPU error on this step — stop accepting
      }
    }

    this.stats.accepted++;
    this.stats.savedMs += batch.slice(0, acceptedSteps).reduce((sum, b) => sum + (b.estimatedComputeMs || 1), 0);
    this.stats.batchSpeculations++;
    this.stats.batchAccepted += acceptedSteps;
    if (acceptedSteps === batch.length) {
      this.stats.batchFull++;
    } else if (acceptedSteps > 0) {
      this.stats.batchPartial++;
    }

    return {
      useSpeculative: true,
      speculativeHidden: lastGoodResult,
      acceptedSteps,
      rollbackPos: acceptedSteps < batch.length ? first.seqPos + acceptedSteps : null,
    };
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
   * Predict K steps ahead and launch speculative compute for each.
   * Steps are computed sequentially (each builds on the previous KV cache state).
   * Uses predictMulti() from the predictor to get K predictions at once.
   *
   * @param {string} requestId
   * @param {number} nextSeqPos - seqPos for the first speculative step
   * @param {number} k - number of steps to speculate
   * @param {number} layerStart
   * @param {number} layerEnd
   */
  _speculateNextBatch(requestId, nextSeqPos, k, layerStart, layerEnd) {
    const predictions = this.predictor.predictMulti(requestId, k);
    if (!predictions || predictions.length === 0) {
      return;
    }

    // Filter to only high-confidence predictions
    const viable = predictions.filter(p => p.confidence >= 0.9);
    if (viable.length === 0) {
      return;
    }

    // If only 1 viable prediction, fall back to single-step
    if (viable.length === 1) {
      // Use the single-step path for efficiency
      const pred = viable[0];
      this.stats.speculations++;
      const size = pred.prediction.length;
      const buffer = this.pipeline._createBuffer(
        "speculative_input",
        pred.prediction.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      this.pipeline.device.queue.writeBuffer(buffer, 0, pred.prediction);
      const speculativeHidden = { buffer, shape: [1, size] };

      const promise = this.pipeline.forwardLayersCached(
        speculativeHidden, layerStart, layerEnd, requestId, nextSeqPos
      );
      this.pending.set(requestId, {
        promise,
        prediction: pred.prediction,
        seqPos: nextSeqPos,
        estimatedComputeMs: 1,
      });
      return;
    }

    // Launch K speculative compute steps sequentially
    const batchEntries = [];
    for (let i = 0; i < viable.length; i++) {
      const pred = viable[i];
      const seqPos = nextSeqPos + i;

      this.stats.speculations++;

      const size = pred.prediction.length;
      const buffer = this.pipeline._createBuffer(
        `speculative_batch_${i}`,
        pred.prediction.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      this.pipeline.device.queue.writeBuffer(buffer, 0, pred.prediction);
      const speculativeHidden = { buffer, shape: [1, size] };

      // Each step is launched independently — they write to different KV cache positions
      const promise = this.pipeline.forwardLayersCached(
        speculativeHidden, layerStart, layerEnd, requestId, seqPos
      );

      batchEntries.push({
        promise,
        prediction: pred.prediction,
        seqPos,
        estimatedComputeMs: 1,
      });
    }

    this.pendingBatch.set(requestId, batchEntries);
  }

  /**
   * Clear state for a completed generation.
   */
  clear(requestId) {
    this.pending.delete(requestId);
    this.pendingBatch.delete(requestId);
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
      batchAvgAccepted: this.stats.batchSpeculations > 0
        ? this.stats.batchAccepted / this.stats.batchSpeculations
        : 0,
      predictor: this.predictor.getStats(),
    };
  }
}
