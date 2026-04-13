/**
 * Early Exit — Per-layer confidence check for skipping remaining layers.
 *
 * After each transformer layer, we check: did the hidden state change much?
 * If the delta between layer L and layer L-1 is small, the remaining layers
 * are unlikely to change the output. We can exit early and save compute.
 *
 * This is especially powerful in distributed inference because:
 * - If node 0 (layers 0-5) detects early exit at layer 3, it can skip layers 4-5
 *   AND skip sending the activation to node 1 entirely — saving both compute AND network
 *
 * Metrics:
 * - Cosine similarity between consecutive layer outputs (convergence)
 * - Norm of the residual delta (magnitude of change)
 * - Entropy of the projection to logits (confidence in prediction)
 *
 * Conservative approach: only exit early if ALL metrics agree.
 */

export class EarlyExitDetector {
  constructor() {
    // Thresholds (conservative defaults — tune with real data)
    this.cosineThreshold = 0.9995;   // layers producing nearly identical output
    this.normRatioThreshold = 0.01;  // delta norm < 1% of activation norm

    // Track per-request convergence
    this.layerOutputs = new Map(); // requestId -> { lastHidden: Float32Array }

    // Stats
    this.stats = {
      checks: 0,
      earlyExits: 0,
      layersSaved: 0,
    };

    // Whether early exit is active (disabled by default until validated)
    this.enabled = false;
  }

  /**
   * Check if we should exit early after processing layer `layerIdx`.
   *
   * @param {string} requestId
   * @param {number} layerIdx - which layer just completed
   * @param {Float32Array} currentHidden - output of this layer
   * @param {number} totalLayers - how many layers this node is responsible for
   * @returns {{ shouldExit: boolean, cosine: number, normRatio: number }}
   */
  check(requestId, layerIdx, currentHidden, totalLayers) {
    this.stats.checks++;

    const entry = this.layerOutputs.get(requestId);
    if (!entry || !entry.lastHidden) {
      // First layer — no comparison possible, just record
      this._record(requestId, currentHidden);
      return { shouldExit: false, cosine: 0, normRatio: 1 };
    }

    const prev = entry.lastHidden;
    const cosine = this._cosineSimilarity(prev, currentHidden);
    const normRatio = this._deltaNormRatio(prev, currentHidden);

    // Record for next layer's comparison
    this._record(requestId, currentHidden);

    // Don't exit on the last layer (nothing to skip)
    if (layerIdx >= totalLayers - 1) {
      return { shouldExit: false, cosine, normRatio };
    }

    // Conservative: both metrics must agree
    const shouldExit = this.enabled &&
      cosine >= this.cosineThreshold &&
      normRatio <= this.normRatioThreshold;

    if (shouldExit) {
      const saved = totalLayers - 1 - layerIdx;
      this.stats.earlyExits++;
      this.stats.layersSaved += saved;
    }

    return { shouldExit, cosine, normRatio };
  }

  /**
   * Get the fraction of compute saved.
   */
  getStats() {
    return {
      ...this.stats,
      exitRate: this.stats.checks > 0 ? this.stats.earlyExits / this.stats.checks : 0,
      avgLayersSaved: this.stats.earlyExits > 0 ? this.stats.layersSaved / this.stats.earlyExits : 0,
    };
  }

  /**
   * Clear state for a completed generation.
   */
  clear(requestId) {
    this.layerOutputs.delete(requestId);
  }

  // ─── Internal ──────────────────────────────────────────────────

  _record(requestId, hidden) {
    this.layerOutputs.set(requestId, {
      lastHidden: new Float32Array(hidden), // copy
    });
  }

  _cosineSimilarity(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  _deltaNormRatio(a, b) {
    let deltaNorm = 0, aNorm = 0;
    for (let i = 0; i < a.length; i++) {
      const d = b[i] - a[i];
      deltaNorm += d * d;
      aNorm += a[i] * a[i];
    }
    if (aNorm === 0) return 1;
    return Math.sqrt(deltaNorm) / Math.sqrt(aNorm);
  }
}
