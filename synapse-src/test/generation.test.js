/**
 * Generation Tests — Autoregressive generation state machine
 *
 * Tests the Generation and GenerationManager classes that track
 * autoregressive token generation state, EOS detection, timeout,
 * and statistics.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { Generation, GenerationManager } from "../coordinator/generation.js";

describe("Generation", () => {
  const prompt = [1, 2, 3, 4, 5];

  it("initializes with prompt tokens", () => {
    const gen = new Generation("gen-1", prompt, 10);
    assert.equal(gen.id, "gen-1");
    assert.equal(gen.promptLen, 5);
    assert.equal(gen.maxTokens, 10);
    assert.deepEqual(gen.generatedTokens, []);
    assert.deepEqual(gen.tokenIds, [1, 2, 3, 4, 5]);
    assert.equal(gen.prefillDone, false);
  });

  it("does not share array reference with input", () => {
    const tokens = [10, 20, 30];
    const gen = new Generation("gen-2", tokens, 5);
    tokens.push(99);
    assert.equal(gen.tokenIds.length, 3); // not affected
  });

  it("addToken accumulates tokens and returns not-done", () => {
    const gen = new Generation("gen-1", prompt, 10);
    const result = gen.addToken(100);
    assert.equal(result.done, false);
    assert.equal(result.reason, null);
    assert.equal(result.seqPos, 5); // tokenIds = [1,2,3,4,5,100], seqPos = length-1 = 5
    assert.deepEqual(gen.generatedTokens, [100]);
    assert.deepEqual(gen.tokenIds, [1, 2, 3, 4, 5, 100]);
    assert.equal(gen.prefillDone, true);
  });

  it("addToken detects EOS token (50256)", () => {
    const gen = new Generation("gen-1", prompt, 100);
    gen.addToken(42);
    const result = gen.addToken(50256);
    assert.equal(result.done, true);
    assert.equal(result.reason, "eos");
    assert.equal(gen.generatedTokens.length, 2);
  });

  it("addToken detects max tokens reached", () => {
    const gen = new Generation("gen-1", prompt, 3);
    gen.addToken(10);
    gen.addToken(20);
    const result = gen.addToken(30);
    assert.equal(result.done, true);
    assert.equal(result.reason, "max_tokens");
  });

  it("EOS takes priority when it coincides with max tokens", () => {
    const gen = new Generation("gen-1", prompt, 1);
    const result = gen.addToken(50256);
    assert.equal(result.done, true);
    assert.equal(result.reason, "eos"); // EOS checked first
  });

  it("isTimedOut detects stalled generation", () => {
    const gen = new Generation("gen-1", prompt, 10, 5000);
    const start = gen.startTime;
    assert.equal(gen.isTimedOut(start + 4999), false);
    assert.equal(gen.isTimedOut(start + 5001), true);
  });

  it("isTimedOut resets after token is generated", () => {
    const gen = new Generation("gen-1", prompt, 10, 5000);
    const start = gen.startTime;
    // Would be timed out based on startTime
    gen.addToken(42);
    const tokenTime = gen._lastTokenTime;
    // Not timed out relative to last token
    assert.equal(gen.isTimedOut(tokenTime + 4999), false);
    assert.equal(gen.isTimedOut(tokenTime + 5001), true);
  });

  it("getStats computes correct statistics", () => {
    const gen = new Generation("gen-1", prompt, 10);
    gen.addToken(10);
    gen.addToken(20);
    gen.addToken(30);
    const stats = gen.getStats(gen.startTime + 1000);
    assert.equal(stats.totalTokens, 3);
    assert.equal(stats.elapsedMs, 1000);
    assert.equal(stats.tokensPerSecond, 3.0);
    assert.equal(stats.promptLen, 5);
  });

  it("getStats handles zero elapsed time", () => {
    const gen = new Generation("gen-1", prompt, 10);
    const stats = gen.getStats(gen.startTime);
    assert.equal(stats.tokensPerSecond, 0);
  });

  it("nextSeqPos tracks position correctly", () => {
    const gen = new Generation("gen-1", [1, 2, 3], 10);
    assert.equal(gen.nextSeqPos, 2); // 3 tokens, pos 2
    gen.addToken(10);
    assert.equal(gen.nextSeqPos, 3); // 4 tokens, pos 3
    gen.addToken(20);
    assert.equal(gen.nextSeqPos, 4);
  });

  it("lastToken returns most recent generated token", () => {
    const gen = new Generation("gen-1", prompt, 10);
    assert.equal(gen.lastToken, null);
    gen.addToken(42);
    assert.equal(gen.lastToken, 42);
    gen.addToken(99);
    assert.equal(gen.lastToken, 99);
  });

  it("binaryReqId is settable", () => {
    const gen = new Generation("gen-1", prompt, 10);
    assert.equal(gen._binaryReqId, null);
    gen._binaryReqId = 7;
    assert.equal(gen._binaryReqId, 7);
  });
});

describe("GenerationManager", () => {
  it("creates and retrieves generations", () => {
    const mgr = new GenerationManager();
    const gen = mgr.create("gen-1", [1, 2, 3], 10);
    assert.equal(gen.id, "gen-1");
    assert.equal(mgr.get("gen-1"), gen);
    assert.equal(mgr.size, 1);
  });

  it("returns undefined for unknown generation", () => {
    const mgr = new GenerationManager();
    assert.equal(mgr.get("nonexistent"), undefined);
  });

  it("removes generations", () => {
    const mgr = new GenerationManager();
    mgr.create("gen-1", [1], 5);
    mgr.create("gen-2", [2], 5);
    mgr.remove("gen-1");
    assert.equal(mgr.get("gen-1"), undefined);
    assert.equal(mgr.size, 1);
  });

  it("removes nonexistent generation without error", () => {
    const mgr = new GenerationManager();
    mgr.remove("nope"); // no throw
    assert.equal(mgr.size, 0);
  });

  it("getTimedOut finds stalled generations", () => {
    const mgr = new GenerationManager(5000);
    const g1 = mgr.create("gen-1", [1], 10);
    const g2 = mgr.create("gen-2", [2], 10);
    // g1 started 10s ago, g2 just started
    g1.startTime = Date.now() - 10000;

    const timedOut = mgr.getTimedOut();
    assert.equal(timedOut.length, 1);
    assert.equal(timedOut[0].id, "gen-1");
  });

  it("sweepTimedOut removes and returns stalled generations", () => {
    const mgr = new GenerationManager(5000);
    const g1 = mgr.create("gen-1", [1], 10);
    mgr.create("gen-2", [2], 10);
    g1.startTime = Date.now() - 10000;

    const swept = mgr.sweepTimedOut();
    assert.equal(swept.length, 1);
    assert.equal(swept[0].id, "gen-1");
    assert.equal(mgr.size, 1);
    assert.equal(mgr.get("gen-1"), undefined);
    assert.ok(mgr.get("gen-2"));
  });

  it("sweepTimedOut returns empty array when nothing timed out", () => {
    const mgr = new GenerationManager(60000);
    mgr.create("gen-1", [1], 10);
    const swept = mgr.sweepTimedOut();
    assert.deepEqual(swept, []);
    assert.equal(mgr.size, 1);
  });

  it("multiple generations track independently", () => {
    const mgr = new GenerationManager();
    const g1 = mgr.create("gen-1", [1, 2], 5);
    const g2 = mgr.create("gen-2", [10, 20, 30], 3);

    g1.addToken(100);
    g2.addToken(200);
    g2.addToken(300);

    assert.equal(g1.generatedTokens.length, 1);
    assert.equal(g2.generatedTokens.length, 2);
    assert.equal(g1.promptLen, 2);
    assert.equal(g2.promptLen, 3);
  });

  it("generation completion flow", () => {
    const mgr = new GenerationManager();
    const gen = mgr.create("gen-1", [1, 2, 3], 3);

    // Generate 3 tokens
    let result;
    result = gen.addToken(10); assert.equal(result.done, false);
    result = gen.addToken(20); assert.equal(result.done, false);
    result = gen.addToken(30); assert.equal(result.done, true);

    // Stats
    const stats = gen.getStats(gen.startTime + 500);
    assert.equal(stats.totalTokens, 3);

    // Cleanup
    mgr.remove("gen-1");
    assert.equal(mgr.size, 0);
  });

  it("EOS mid-generation stops early", () => {
    const mgr = new GenerationManager();
    const gen = mgr.create("gen-1", [1, 2], 100);

    gen.addToken(10);
    gen.addToken(20);
    const result = gen.addToken(50256); // EOS
    assert.equal(result.done, true);
    assert.equal(result.reason, "eos");
    assert.equal(gen.generatedTokens.length, 3); // stopped at 3, not 100
  });
});
