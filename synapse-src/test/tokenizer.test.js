import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TOK_DIR = resolve(__dirname, "..", "model", "shards", "tokenizer");
const gemmaTokenizerPresent = existsSync(join(TOK_DIR, "tokenizer.json"));

describe("tokenizer facade (arch-aware)", () => {
  test("exports init + encode + decode + getArch", async () => {
    const tk = await import("../coordinator/tokenizer.js");
    assert.equal(typeof tk.initTokenizer, "function");
    assert.equal(typeof tk.encodeText, "function");
    assert.equal(typeof tk.decodeTokens, "function");
    assert.equal(typeof tk.getArch, "function");
    assert.equal(typeof tk.getSpecialTokens, "function");
  });

  test("Gemma path: encode → decode round-trips", {
    skip: gemmaTokenizerPresent ? false : "tokenizer.json not present (run split_gemma.py first)",
  }, async () => {
    const tk = await import("../coordinator/tokenizer.js");
    await tk.initTokenizer();
    // Only meaningful if current manifest is gemma; otherwise skip via arch check
    if (tk.getArch() !== "gemma") return;
    const text = "Hello Synapse! Distributed WebGPU is fun.";
    const ids = tk.encodeText(text);
    assert.ok(Array.isArray(ids), "encode returns array");
    assert.ok(ids.length > 5, "non-trivial encoding");
    const round = tk.decodeTokens(ids, { skipSpecial: true });
    assert.equal(round, text);
    const { bos, eos } = tk.getSpecialTokens();
    assert.equal(typeof bos, "number");
    assert.equal(typeof eos, "number");
  });
});
