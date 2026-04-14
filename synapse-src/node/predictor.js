/**
 * Synapse Activation Predictor
 *
 * Predicts the next activation tensor based on recent history.
 * Used for speculative execution: the downstream node starts computing
 * on the predicted activation while waiting for the real one.
 *
 * Strategy (layered, increasing complexity):
 *   1. Linear extrapolation — predict from last 2 activations
 *   2. EMA (exponential moving average) — smooth prediction from history
 *   3. Learned MLP — small model trained on activation patterns (future)
 *
 * VLSI analogy: this is branch prediction for distributed inference.
 * A wrong prediction costs one recompute. A right prediction hides
 * the entire network latency.
 */

export class ActivationPredictor {
  constructor() {
    // Ring buffer of recent activations per request
    // Key: requestId, Value: { history: Float32Array[], writeIdx: number }
    this.requests = new Map();
    this.maxHistory = 4; // keep last 4 activations per request

    // Prediction accuracy tracking
    this.stats = {
      predictions: 0,
      hits: 0,        // cosine > threshold
      misses: 0,
      avgCosine: 0,
      totalCosine: 0,
    };

    // Cosine similarity threshold for "good enough" prediction
    this.acceptThreshold = 0.995;
  }

  /**
   * Record an observed activation (ground truth from the wire).
   * Call this every time a real activation arrives.
   *
   * @param {string} requestId
   * @param {Float32Array} activation - the real activation values
   */
  observe(requestId, activation) {
    if (!this.requests.has(requestId)) {
      this.requests.set(requestId, {
        history: [],
        shape: null,
      });
    }
    const entry = this.requests.get(requestId);
    entry.shape = [1, activation.length]; // assume single-token for cached steps

    if (entry.history.length >= this.maxHistory) {
      entry.history.shift(); // drop oldest
    }
    entry.history.push(new Float32Array(activation)); // copy to avoid mutation
  }

  /**
   * Predict the next activation for a given request.
   * Returns null if not enough history to predict.
   *
   * @param {string} requestId
   * @returns {{ prediction: Float32Array, confidence: number } | null}
   */
  predict(requestId) {
    const entry = this.requests.get(requestId);
    if (!entry || entry.history.length < 2) {
      return null; // need at least 2 observations for linear extrapolation
    }

    const history = entry.history;
    const n = history.length;
    const prev = history[n - 2];
    const curr = history[n - 1];
    const size = curr.length;

    // Linear extrapolation: next ≈ curr + (curr - prev)
    // This assumes the activation trajectory is locally linear
    const prediction = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      prediction[i] = curr[i] + (curr[i] - prev[i]);
    }

    // Confidence estimate: how similar were the last two deltas?
    // If the trajectory is smooth, confidence is high
    let confidence = 1.0;
    if (n >= 3) {
      const prevPrev = history[n - 3];
      // Compare delta(n-1, n-2) with delta(n-2, n-3)
      const cos = this._cosineSimilarity(
        this._delta(curr, prev),
        this._delta(prev, prevPrev)
      );
      confidence = Math.max(0, cos); // clamp to [0, 1]
    }

    this.stats.predictions++;
    return { prediction, confidence };
  }

  /**
   * Predict K steps ahead for batch speculative decoding.
   * Each step extrapolates further from the current trajectory.
   *
   * @param {string} requestId
   * @param {number} k - number of steps to predict (1-4)
   * @returns {Array<{ prediction: Float32Array, confidence: number, stepsAhead: number }> | null}
   */
  predictMulti(requestId, k = 3) {
    const entry = this.requests.get(requestId);
    if (!entry || entry.history.length < 2) {
      return null;
    }

    const history = entry.history;
    const n = history.length;
    const curr = history[n - 1];
    const prev = history[n - 2];
    const size = curr.length;

    // Compute the delta (velocity)
    const delta = this._delta(curr, prev);

    // Base confidence from trajectory smoothness
    let baseConfidence = 1.0;
    if (n >= 3) {
      const prevPrev = history[n - 3];
      baseConfidence = Math.max(0, this._cosineSimilarity(
        delta,
        this._delta(prev, prevPrev)
      ));
    }

    const results = [];
    for (let step = 1; step <= k; step++) {
      const prediction = new Float32Array(size);
      // Linear extrapolation: curr + step * delta
      for (let i = 0; i < size; i++) {
        prediction[i] = curr[i] + step * delta[i];
      }
      // Confidence decays with each step ahead — farther predictions are less reliable
      const confidence = baseConfidence * Math.pow(0.85, step - 1);
      results.push({ prediction, confidence, stepsAhead: step });

      this.stats.predictions++;
    }

    return results;
  }

  /**
   * Verify a prediction against ground truth.
   * Returns whether the prediction was good enough to keep.
   *
   * @param {string} requestId
   * @param {Float32Array} predicted
   * @param {Float32Array} actual
   * @returns {{ accept: boolean, cosine: number }}
   */
  verify(predicted, actual) {
    const cosine = this._cosineSimilarity(predicted, actual);

    this.stats.totalCosine += cosine;
    if (cosine >= this.acceptThreshold) {
      this.stats.hits++;
    } else {
      this.stats.misses++;
    }

    return {
      accept: cosine >= this.acceptThreshold,
      cosine,
    };
  }

  /**
   * Get prediction accuracy stats.
   */
  getStats() {
    const total = this.stats.hits + this.stats.misses;
    return {
      predictions: this.stats.predictions,
      hits: this.stats.hits,
      misses: this.stats.misses,
      hitRate: total > 0 ? this.stats.hits / total : 0,
      avgCosine: total > 0 ? this.stats.totalCosine / total : 0,
    };
  }

  /**
   * Clear state for a completed generation.
   */
  clear(requestId) {
    this.requests.delete(requestId);
  }

  /**
   * Clear all state.
   */
  clearAll() {
    this.requests.clear();
    this.stats = { predictions: 0, hits: 0, misses: 0, avgCosine: 0, totalCosine: 0 };
  }

  // ─── Internal helpers ──────────────────────────────────────────

  _delta(a, b) {
    const d = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      d[i] = a[i] - b[i];
    }
    return d;
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
}
