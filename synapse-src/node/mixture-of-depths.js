/**
 * Mixture-of-Depths Router
 *
 * Decides per-token, per-layer whether to apply the full transformer block
 * or skip it via a residual passthrough. "Easy" tokens that are already
 * well-represented skip layers, saving compute. "Hard" tokens (ambiguous,
 * high-entropy) get full processing.
 *
 * Based on: "Mixture-of-Depths" (Raposo et al., 2024)
 *
 * In distributed inference, this is especially powerful:
 * - Skip decisions reduce GPU compute on each node
 * - If ALL layers on a node are skipped, the activation passes through
 *   unchanged — we could even short-circuit the send to the next node
 *
 * Routing strategy (heuristic, no learned weights):
 * - Track activation norm stability across layers
 * - If the hidden state isn't changing much (low delta-norm ratio),
 *   the layer is doing little work — future similar tokens can skip it
 * - Use per-layer difficulty scores based on observed delta magnitudes
 *
 * This is the online/heuristic version. A future version can use
 * a learned 1-layer MLP router trained on real inference traces.
 */

export class MixtureOfDepthsRouter {
  constructor(numLayers) {
    this.numLayers = numLayers;

    // Per-layer difficulty profile: EMA of delta-norm ratios
    // Higher value = layer makes bigger changes = more important
    this.layerDifficulty = new Float32Array(numLayers).fill(0.5);

    // EMA decay for difficulty tracking
    this.alpha = 0.85;

    // Capacity ratio: fraction of tokens that go through each layer
    // 1.0 = all tokens processed (no skipping), 0.5 = half skipped
    this.capacity = 0.75;

    // Per-layer skip threshold derived from difficulty profile
    // Layers with difficulty below threshold get skipped for "easy" tokens
    this.skipThresholds = new Float32Array(numLayers).fill(0.1);

    // Whether routing is active
    this.enabled = false;

    // Minimum layers to always process (never skip first or last)
    this.protectedLayers = new Set([0, numLayers - 1]);

    // Stats
    this.stats = {
      totalRouted: 0,
      layersSkipped: 0,
      layersProcessed: 0,
    };

    // Per-request token difficulty tracking
    this.tokenDifficulty = new Map(); // requestId -> { normHistory: number[] }
  }

  /**
   * Decide whether to process or skip a layer for a given token.
   *
   * @param {number} layerIdx - relative layer index (0-based within shard)
   * @param {string} requestId - identifies the generation
   * @param {Float32Array} hidden - current hidden state
   * @returns {{ skip: boolean, reason: string }}
   */
  route(layerIdx, requestId, hidden) {
    this.stats.totalRouted++;

    // Always process protected layers
    if (this.protectedLayers.has(layerIdx)) {
      this.stats.layersProcessed++;
      return { skip: false, reason: "protected" };
    }

    if (!this.enabled) {
      this.stats.layersProcessed++;
      return { skip: false, reason: "disabled" };
    }

    // Get token difficulty estimate from recent history
    const tokenDiff = this._getTokenDifficulty(requestId, hidden);
    const layerDiff = this.layerDifficulty[layerIdx];

    // Skip decision: if the token is "easy" AND the layer has low difficulty
    // Easy token + unimportant layer = skip
    // Hard token OR important layer = process
    const skipScore = (1 - tokenDiff) * (1 - layerDiff);
    const threshold = this.capacity; // capacity 0.75 → threshold 0.75 (harder to skip)

    if (skipScore > threshold + 1e-6) {
      this.stats.layersSkipped++;
      return { skip: true, reason: `score=${skipScore.toFixed(3)}` };
    }

    this.stats.layersProcessed++;
    return { skip: false, reason: `score=${skipScore.toFixed(3)}` };
  }

  /**
   * After processing a layer, record how much the hidden state changed.
   * This updates the layer's difficulty profile.
   *
   * @param {number} layerIdx
   * @param {Float32Array} inputHidden - hidden state before the layer
   * @param {Float32Array} outputHidden - hidden state after the layer
   */
  recordLayerEffect(layerIdx, inputHidden, outputHidden) {
    if (layerIdx >= this.numLayers) return;

    const ratio = this._deltaNormRatio(inputHidden, outputHidden);

    // EMA update of layer difficulty
    this.layerDifficulty[layerIdx] =
      this.alpha * this.layerDifficulty[layerIdx] + (1 - this.alpha) * ratio;

    // Update skip threshold based on global difficulty distribution
    this._updateThresholds();
  }

  /**
   * Get routing decisions for all layers at once.
   * Returns a boolean mask: true = process, false = skip.
   *
   * @param {string} requestId
   * @param {Float32Array} hidden
   * @returns {boolean[]}
   */
  getLayerMask(requestId, hidden) {
    const mask = [];
    for (let l = 0; l < this.numLayers; l++) {
      const decision = this.route(l, requestId, hidden);
      // Undo the stats increments from route() — getLayerMask is a planning call
      this.stats.totalRouted--;
      if (decision.skip) this.stats.layersSkipped--;
      else this.stats.layersProcessed--;

      mask.push(!decision.skip);
    }
    return mask;
  }

  /**
   * Record token activation for difficulty estimation.
   */
  observeToken(requestId, hidden) {
    const norm = this._l2Norm(hidden);
    let entry = this.tokenDifficulty.get(requestId);
    if (!entry) {
      entry = { normHistory: [] };
      this.tokenDifficulty.set(requestId, entry);
    }
    entry.normHistory.push(norm);
    // Keep last 16 observations
    if (entry.normHistory.length > 16) {
      entry.normHistory.shift();
    }
  }

  /**
   * Clear state for a completed generation.
   */
  clear(requestId) {
    this.tokenDifficulty.delete(requestId);
  }

  getStats() {
    return {
      ...this.stats,
      skipRate: this.stats.totalRouted > 0
        ? this.stats.layersSkipped / this.stats.totalRouted
        : 0,
      enabled: this.enabled,
      layerDifficulty: Array.from(this.layerDifficulty).map(d => +d.toFixed(4)),
    };
  }

  reset() {
    this.layerDifficulty.fill(0.5);
    this.skipThresholds.fill(0.1);
    this.stats = { totalRouted: 0, layersSkipped: 0, layersProcessed: 0 };
    this.tokenDifficulty.clear();
  }

  // ─── Internal ──────────────────────────────────────────────────

  _getTokenDifficulty(requestId, hidden) {
    const entry = this.tokenDifficulty.get(requestId);
    if (!entry || entry.normHistory.length < 2) {
      return 0.5; // uncertain → assume medium difficulty
    }

    // Token difficulty = coefficient of variation of recent norms
    // High variance in norms across steps → hard token (changing a lot)
    // Low variance → easy token (stable representation)
    const norms = entry.normHistory;
    const mean = norms.reduce((s, v) => s + v, 0) / norms.length;
    if (mean === 0) return 0.5;

    const variance = norms.reduce((s, v) => s + (v - mean) ** 2, 0) / norms.length;
    const cv = Math.sqrt(variance) / mean;

    // Map CV to [0, 1] — CV of 0.5+ is very hard
    return Math.min(1, cv * 2);
  }

  _updateThresholds() {
    // Set skip thresholds based on layer difficulty distribution
    // Layers well below the mean difficulty are candidates for skipping
    const mean = this.layerDifficulty.reduce((s, v) => s + v, 0) / this.numLayers;
    for (let l = 0; l < this.numLayers; l++) {
      this.skipThresholds[l] = mean * 0.5;
    }
  }

  _deltaNormRatio(a, b) {
    let deltaNorm = 0, aNorm = 0;
    for (let i = 0; i < a.length; i++) {
      const d = b[i] - a[i];
      deltaNorm += d * d;
      aNorm += a[i] * a[i];
    }
    if (aNorm === 0) return 0;
    return Math.sqrt(deltaNorm) / Math.sqrt(aNorm);
  }

  _l2Norm(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i] * arr[i];
    }
    return Math.sqrt(sum);
  }
}
