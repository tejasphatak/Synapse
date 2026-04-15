/**
 * Arch-aware tokenizer facade for the coordinator.
 *
 * GPT-2 path uses `gpt-tokenizer` (r50k_base / text-davinci-001 encoding),
 * synchronous, small. Gemma path uses `@huggingface/transformers` AutoTokenizer
 * loaded from local files under model/shards/tokenizer/, async to initialise
 * then synchronous to call.
 *
 * The arch is read from manifest.json at startup. No HF network access is
 * required at runtime — tokenizer.json is shipped alongside the shards by
 * split_gemma.py's companion step (see shards/tokenizer/).
 */

import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { encode as gptEncode, decode as gptDecode } from "gpt-tokenizer/model/text-davinci-001";

const __dirname = dirname(fileURLToPath(import.meta.url));
const SHARDS_DIR = resolve(__dirname, "..", "model", "shards");
const MANIFEST_PATH = join(SHARDS_DIR, "manifest.json");
const GEMMA_TOKENIZER_DIR = join(SHARDS_DIR, "tokenizer");

let state = {
  arch: "gpt2",
  gemmaTokenizer: null, // populated lazily for Gemma
  ready: false,
};

function readArch() {
  if (!existsSync(MANIFEST_PATH)) return "gpt2";
  try {
    const m = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
    return m.arch || "gpt2";
  } catch {
    return "gpt2";
  }
}

/**
 * Must be awaited once at coordinator boot. GPT-2 returns immediately;
 * Gemma loads the local tokenizer.json into memory.
 */
export async function initTokenizer() {
  state.arch = readArch();
  if (state.arch === "gemma") {
    if (!existsSync(join(GEMMA_TOKENIZER_DIR, "tokenizer.json"))) {
      throw new Error(
        `Gemma tokenizer files missing: expected ${GEMMA_TOKENIZER_DIR}/tokenizer.json. ` +
        `Run split_gemma.py with --include-tokenizer or copy them manually.`
      );
    }
    const { AutoTokenizer, env } = await import("@huggingface/transformers");
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = SHARDS_DIR + "/";
    state.gemmaTokenizer = await AutoTokenizer.from_pretrained("tokenizer");
  }
  state.ready = true;
  return { arch: state.arch };
}

export function encodeText(text) {
  if (!state.ready) throw new Error("tokenizer not initialised — call initTokenizer() first");
  if (state.arch === "gemma") {
    // Returns a Tensor / TypedArray — normalise to plain number[]
    const out = state.gemmaTokenizer.encode(text);
    return Array.from(out);
  }
  return gptEncode(text);
}

export function decodeTokens(tokenIds, opts = {}) {
  if (!state.ready) throw new Error("tokenizer not initialised — call initTokenizer() first");
  if (state.arch === "gemma") {
    return state.gemmaTokenizer.decode(tokenIds, { skip_special_tokens: opts.skipSpecial ?? true });
  }
  return gptDecode(tokenIds);
}

export function getArch() {
  return state.arch;
}

export function getSpecialTokens() {
  if (state.arch === "gemma" && state.gemmaTokenizer) {
    return {
      bos: state.gemmaTokenizer.bos_token_id,
      eos: state.gemmaTokenizer.eos_token_id,
      pad: state.gemmaTokenizer.pad_token_id ?? null,
    };
  }
  return { bos: null, eos: 50256, pad: null }; // GPT-2: <|endoftext|> is both bos and eos
}
