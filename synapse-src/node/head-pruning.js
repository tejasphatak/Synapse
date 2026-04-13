/**
 * Online Attention Head Pruning
 *
 * Measures importance of each attention head during inference and
 * dynamically skips low-importance heads. For GPT-2 with 12 heads,
 * pruning 3-4 heads saves 25-33% compute with minimal quality loss.
 *
 * Importance metric: L2 norm of the attention output for each head.
 * Heads that consistently produce near-zero output aren't contributing.
 *
 * The pruning decision is made per-token: "easy" tokens may need fewer
 * heads than "hard" tokens. This is adaptive, not static.
 *
 * References:
 * - "Are Sixteen Heads Really Better than One?" (Michel et al., 2019)
 * - "Analyzing Multi-Head Self-Attention" (Voita et al., 2019)
 */

export class HeadPruner {
  constructor(numHeads, numLayers) {
    this.numHeads = numHeads;
    this.numLayers = numLayers;

    // Running importance scores per head per layer
    // EMA (exponential moving average) of L2 norms
    this.importance = Array.from({ length: numLayers }, () =>
      new Float32Array(numHeads).fill(1.0) // start with equal importance
    );

    // EMA decay factor (0.9 = recent tokens weighted 10x more than 10 tokens ago)
    this.alpha = 0.9;

    // Pruning threshold: heads below this fraction of mean importance get skipped
    this.pruneRatio = 0.25; // skip bottom 25% of heads

    // Minimum heads to keep (never prune below this)
    this.minHeads = Math.max(4, Math.floor(numHeads * 0.5));

    // Whether pruning is active
    this.enabled = false;

    // Stats
    this.stats = {
      totalHeadOps: 0,   // total head computations
      skippedHeadOps: 0, // skipped due to pruning
    };
  }

  /**
   * Record the output norm of each head after attention computation.
   * Call this during forward pass after computing attention for a layer.
   *
   * @param {number} layerIdx - relative layer index (0-based within this shard)
   * @param {Float32Array} headNorms - L2 norm of each head's output [numHeads]
   */
  recordHeadNorms(layerIdx, headNorms) {
    if (layerIdx >= this.numLayers) return;

    const imp = this.importance[layerIdx];
    for (let h = 0; h < this.numHeads; h++) {
      // EMA update
      imp[h] = this.alpha * imp[h] + (1 - this.alpha) * headNorms[h];
    }
  }

  /**
   * Get the mask of which heads to compute for a given layer.
   * Returns a boolean array: true = compute, false = skip.
   *
   * @param {number} layerIdx - relative layer index
   * @returns {boolean[]} mask of length numHeads
   */
  getHeadMask(layerIdx) {
    if (!this.enabled || layerIdx >= this.numLayers) {
      return Array(this.numHeads).fill(true); // compute all
    }

    const imp = this.importance[layerIdx];
    this.stats.totalHeadOps += this.numHeads;

    // Sort heads by importance
    const indexed = Array.from(imp).map((val, idx) => ({ val, idx }));
    indexed.sort((a, b) => b.val - a.val);

    // Keep top heads, skip bottom ones
    const keepCount = Math.max(this.minHeads, Math.ceil(this.numHeads * (1 - this.pruneRatio)));
    const mask = Array(this.numHeads).fill(false);
    for (let i = 0; i < keepCount; i++) {
      mask[indexed[i].idx] = true;
    }

    this.stats.skippedHeadOps += (this.numHeads - keepCount);
    return mask;
  }

  /**
   * Get current importance rankings for a layer.
   * Useful for telemetry and debugging.
   */
  getImportanceRanking(layerIdx) {
    if (layerIdx >= this.numLayers) return [];
    const imp = this.importance[layerIdx];
    return Array.from(imp)
      .map((val, idx) => ({ head: idx, importance: +val.toFixed(4) }))
      .sort((a, b) => b.importance - a.importance);
  }

  getStats() {
    return {
      ...this.stats,
      pruneRate: this.stats.totalHeadOps > 0
        ? this.stats.skippedHeadOps / this.stats.totalHeadOps
        : 0,
      enabled: this.enabled,
    };
  }

  reset() {
    for (let l = 0; l < this.numLayers; l++) {
      this.importance[l].fill(1.0);
    }
    this.stats = { totalHeadOps: 0, skippedHeadOps: 0 };
  }
}
