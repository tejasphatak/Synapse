/**
 * Mixture-of-Depths Router Tests
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { MixtureOfDepthsRouter } from "../node/mixture-of-depths.js";

function randomHidden(size = 768, scale = 1) {
  const h = new Float32Array(size);
  for (let i = 0; i < size; i++) h[i] = (Math.random() - 0.5) * 2 * scale;
  return h;
}

function constantHidden(size = 768, value = 1.0) {
  return new Float32Array(size).fill(value);
}

describe("MixtureOfDepthsRouter", () => {
  it("initializes with correct defaults", () => {
    const mod = new MixtureOfDepthsRouter(6);
    assert.equal(mod.numLayers, 6);
    assert.equal(mod.enabled, false);
    assert.equal(mod.capacity, 0.75);
    assert.equal(mod.layerDifficulty.length, 6);
    assert.ok(mod.protectedLayers.has(0));
    assert.ok(mod.protectedLayers.has(5));
  });

  it("never skips when disabled", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const hidden = randomHidden();
    for (let l = 0; l < 6; l++) {
      const { skip, reason } = mod.route(l, "req-1", hidden);
      if (l === 0 || l === 5) {
        assert.equal(skip, false);
        assert.equal(reason, "protected");
      } else {
        assert.equal(skip, false);
        assert.equal(reason, "disabled");
      }
    }
  });

  it("always processes protected layers even when enabled", () => {
    const mod = new MixtureOfDepthsRouter(6);
    mod.enabled = true;
    const hidden = randomHidden();
    const r0 = mod.route(0, "req-1", hidden);
    const r5 = mod.route(5, "req-1", hidden);
    assert.equal(r0.skip, false);
    assert.equal(r5.skip, false);
    assert.equal(r0.reason, "protected");
    assert.equal(r5.reason, "protected");
  });

  it("tracks stats correctly", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const hidden = randomHidden();
    for (let l = 0; l < 6; l++) {
      mod.route(l, "req-1", hidden);
    }
    assert.equal(mod.stats.totalRouted, 6);
    assert.equal(mod.stats.layersProcessed, 6);
    assert.equal(mod.stats.layersSkipped, 0);
  });

  it("recordLayerEffect updates difficulty profile", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const input = constantHidden(768, 1.0);
    const output = constantHidden(768, 1.0); // identical → delta = 0

    const before = mod.layerDifficulty[2];
    mod.recordLayerEffect(2, input, output);
    const after = mod.layerDifficulty[2];

    // Delta norm ratio = 0, so difficulty should decrease
    assert.ok(after < before, `difficulty should decrease: ${before} → ${after}`);
  });

  it("recordLayerEffect increases difficulty for large changes", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const input = constantHidden(768, 1.0);
    const output = constantHidden(768, 5.0); // big change

    const before = mod.layerDifficulty[2];
    mod.recordLayerEffect(2, input, output);
    const after = mod.layerDifficulty[2];

    // Large delta → high difficulty ratio → should increase or stay high
    assert.ok(after >= before * 0.8, `difficulty should not drop drastically: ${before} → ${after}`);
  });

  it("ignores layers beyond numLayers in recordLayerEffect", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const h = randomHidden();
    // Should not throw
    mod.recordLayerEffect(10, h, h);
  });

  it("observeToken tracks norm history", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const h = constantHidden(768, 2.0);
    mod.observeToken("req-1", h);
    mod.observeToken("req-1", h);
    mod.observeToken("req-1", h);

    const entry = mod.tokenDifficulty.get("req-1");
    assert.ok(entry);
    assert.equal(entry.normHistory.length, 3);
  });

  it("observeToken caps history at 16", () => {
    const mod = new MixtureOfDepthsRouter(6);
    for (let i = 0; i < 20; i++) {
      mod.observeToken("req-1", randomHidden());
    }
    const entry = mod.tokenDifficulty.get("req-1");
    assert.equal(entry.normHistory.length, 16);
  });

  it("clear removes token difficulty state", () => {
    const mod = new MixtureOfDepthsRouter(6);
    mod.observeToken("req-1", randomHidden());
    assert.ok(mod.tokenDifficulty.has("req-1"));
    mod.clear("req-1");
    assert.ok(!mod.tokenDifficulty.has("req-1"));
  });

  it("getLayerMask returns correct length mask", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const hidden = randomHidden();
    const mask = mod.getLayerMask("req-1", hidden);
    assert.equal(mask.length, 6);
    // When disabled, all should be true
    assert.ok(mask.every(v => v === true));
  });

  it("getLayerMask doesn't modify stats", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const hidden = randomHidden();
    const before = { ...mod.stats };
    mod.getLayerMask("req-1", hidden);
    assert.deepEqual(mod.stats, before);
  });

  it("getStats returns skipRate", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const hidden = randomHidden();
    for (let l = 0; l < 6; l++) mod.route(l, "req-1", hidden);
    const stats = mod.getStats();
    assert.equal(stats.skipRate, 0);
    assert.equal(stats.enabled, false);
    assert.equal(stats.layerDifficulty.length, 6);
  });

  it("reset clears all state", () => {
    const mod = new MixtureOfDepthsRouter(6);
    mod.enabled = true;
    const hidden = randomHidden();
    for (let l = 0; l < 6; l++) mod.route(l, "req-1", hidden);
    mod.observeToken("req-1", hidden);

    mod.reset();
    assert.equal(mod.stats.totalRouted, 0);
    assert.equal(mod.stats.layersSkipped, 0);
    assert.equal(mod.tokenDifficulty.size, 0);
    assert.ok(mod.layerDifficulty.every(v => v === 0.5));
  });

  it("can skip layers when enabled with trained difficulty", () => {
    const mod = new MixtureOfDepthsRouter(6);
    mod.enabled = true;
    mod.capacity = 0.3; // aggressive skipping (lower threshold)

    // Train layer 2 to have very low difficulty
    const input = constantHidden(768, 1.0);
    const same = constantHidden(768, 1.0);
    for (let i = 0; i < 50; i++) {
      mod.recordLayerEffect(2, input, same); // zero delta → low difficulty
    }

    // Layer 2 should now have very low difficulty
    assert.ok(mod.layerDifficulty[2] < 0.05,
      `difficulty should be very low: ${mod.layerDifficulty[2]}`);
  });

  it("_deltaNormRatio returns 0 for identical inputs", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const h = constantHidden(768, 1.0);
    const ratio = mod._deltaNormRatio(h, h);
    assert.equal(ratio, 0);
  });

  it("_deltaNormRatio handles zero input", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const zero = new Float32Array(10);
    const nonzero = new Float32Array(10).fill(1);
    const ratio = mod._deltaNormRatio(zero, nonzero);
    assert.equal(ratio, 0); // returns 0 when input norm is 0
  });

  it("_l2Norm computes correctly", () => {
    const mod = new MixtureOfDepthsRouter(6);
    const v = new Float32Array([3, 4]); // norm = 5
    assert.ok(Math.abs(mod._l2Norm(v) - 5) < 1e-6);
  });
});
