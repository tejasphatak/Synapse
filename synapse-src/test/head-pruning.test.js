/**
 * Head Pruning Tests — Online attention head importance tracking and pruning.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { HeadPruner } from "../node/head-pruning.js";

describe("HeadPruner", () => {
  it("initializes with correct defaults", () => {
    const hp = new HeadPruner(12, 6);
    assert.equal(hp.numHeads, 12);
    assert.equal(hp.numLayers, 6);
    assert.equal(hp.enabled, false);
    assert.equal(hp.pruneRatio, 0.25);
    assert.equal(hp.minHeads, 6); // max(4, floor(12 * 0.5))
    assert.equal(hp.importance.length, 6);
    assert.equal(hp.importance[0].length, 12);
  });

  it("starts with uniform importance of 1.0", () => {
    const hp = new HeadPruner(12, 6);
    for (let l = 0; l < 6; l++) {
      for (let h = 0; h < 12; h++) {
        assert.equal(hp.importance[l][h], 1.0);
      }
    }
  });

  it("returns all-true mask when disabled", () => {
    const hp = new HeadPruner(12, 6);
    const mask = hp.getHeadMask(0);
    assert.equal(mask.length, 12);
    assert.ok(mask.every(v => v === true));
  });

  it("returns all-true mask for out-of-range layer", () => {
    const hp = new HeadPruner(12, 6);
    hp.enabled = true;
    const mask = hp.getHeadMask(10);
    assert.equal(mask.length, 12);
    assert.ok(mask.every(v => v === true));
  });

  it("prunes low-importance heads when enabled", () => {
    const hp = new HeadPruner(12, 6);
    hp.enabled = true;

    // Set head 0-2 to low importance, rest high
    for (let i = 0; i < 12; i++) {
      hp.importance[0][i] = i < 3 ? 0.01 : 1.0;
    }

    const mask = hp.getHeadMask(0);
    const kept = mask.filter(v => v).length;
    // pruneRatio = 0.25 → keep ceil(12 * 0.75) = 9, skip 3
    assert.equal(kept, 9);

    // The 3 low-importance heads should be pruned
    assert.equal(mask[0], false);
    assert.equal(mask[1], false);
    assert.equal(mask[2], false);
  });

  it("never prunes below minHeads", () => {
    const hp = new HeadPruner(12, 6);
    hp.enabled = true;
    hp.pruneRatio = 0.9; // try to prune 90%

    const mask = hp.getHeadMask(0);
    const kept = mask.filter(v => v).length;
    assert.ok(kept >= hp.minHeads, `kept ${kept} < minHeads ${hp.minHeads}`);
  });

  it("recordHeadNorms updates importance via EMA", () => {
    const hp = new HeadPruner(12, 6);
    const norms = new Float32Array(12);
    norms[0] = 10.0; // head 0 very important
    norms[11] = 0.01; // head 11 unimportant

    hp.recordHeadNorms(0, norms);
    // EMA: 0.9 * 1.0 + 0.1 * 10.0 = 1.9 for head 0
    assert.ok(Math.abs(hp.importance[0][0] - 1.9) < 1e-5);
    // EMA: 0.9 * 1.0 + 0.1 * 0.01 = 0.901 for head 11
    assert.ok(Math.abs(hp.importance[0][11] - 0.901) < 1e-5);
  });

  it("recordHeadNorms ignores out-of-range layers", () => {
    const hp = new HeadPruner(12, 6);
    // Should not throw
    hp.recordHeadNorms(10, new Float32Array(12));
  });

  it("tracks stats correctly", () => {
    const hp = new HeadPruner(12, 6);
    hp.enabled = true;

    // Make some heads low importance
    for (let i = 0; i < 3; i++) hp.importance[0][i] = 0.01;
    hp.getHeadMask(0);

    assert.equal(hp.stats.totalHeadOps, 12);
    assert.equal(hp.stats.skippedHeadOps, 3);
  });

  it("getStats returns pruneRate", () => {
    const hp = new HeadPruner(12, 6);
    hp.enabled = true;
    for (let i = 0; i < 3; i++) hp.importance[0][i] = 0.01;
    hp.getHeadMask(0);

    const stats = hp.getStats();
    assert.ok(Math.abs(stats.pruneRate - 3 / 12) < 1e-6);
    assert.equal(stats.enabled, true);
  });

  it("getImportanceRanking returns sorted heads", () => {
    const hp = new HeadPruner(12, 6);
    hp.importance[0][5] = 10.0;
    hp.importance[0][0] = 0.01;

    const ranking = hp.getImportanceRanking(0);
    assert.equal(ranking.length, 12);
    assert.equal(ranking[0].head, 5); // most important
    assert.equal(ranking[ranking.length - 1].head, 0); // least important
  });

  it("getImportanceRanking returns empty for invalid layer", () => {
    const hp = new HeadPruner(12, 6);
    assert.deepEqual(hp.getImportanceRanking(10), []);
  });

  it("reset restores initial state", () => {
    const hp = new HeadPruner(12, 6);
    hp.enabled = true;
    hp.importance[0][0] = 999;
    hp.getHeadMask(0);

    hp.reset();
    assert.equal(hp.stats.totalHeadOps, 0);
    assert.equal(hp.stats.skippedHeadOps, 0);
    for (let l = 0; l < 6; l++) {
      for (let h = 0; h < 12; h++) {
        assert.equal(hp.importance[l][h], 1.0);
      }
    }
  });

  it("handles small number of heads", () => {
    const hp = new HeadPruner(4, 2);
    assert.equal(hp.minHeads, 4); // max(4, floor(4*0.5)) = max(4, 2) = 4
    hp.enabled = true;
    // With minHeads = 4 and numHeads = 4, can never prune
    const mask = hp.getHeadMask(0);
    assert.equal(mask.filter(v => v).length, 4);
  });

  it("convergence: many EMA updates drive importance to observed norm", () => {
    const hp = new HeadPruner(12, 6);
    const norms = new Float32Array(12);
    norms[0] = 5.0;

    // After many updates, importance[0] should converge near 5.0
    for (let i = 0; i < 100; i++) {
      hp.recordHeadNorms(0, norms);
    }
    assert.ok(Math.abs(hp.importance[0][0] - 5.0) < 0.01,
      `expected ~5.0, got ${hp.importance[0][0]}`);
  });
});
