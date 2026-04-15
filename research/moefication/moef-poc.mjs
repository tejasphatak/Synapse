#!/usr/bin/env node
/**
 * MoEfication proof-of-concept, Node.js native.
 *
 * Loads Gemma 3 1B-it weights, picks one FFN layer, and demonstrates:
 *
 *   1. FFN output Y = up_proj(act(gate_proj(X))) * ... down_proj  (the full dense FFN)
 *      can be exactly decomposed as Y = sum_i Y_i  where Y_i is computed from a
 *      COLUMN SLICE of gate_proj / up_proj and a ROW SLICE of down_proj.
 *
 *   2. For a sparse top-k approximation Y ≈ sum_{i ∈ TopK} Y_i we measure
 *      cosine similarity to the dense output on random input. This is the
 *      number that predicts whether MoEfication is viable for this layer.
 *
 * No calibration dataset, no activation-based clustering — this POC uses
 * a simple structural split. Real MoEfication adds co-activation K-means,
 * which needs a forward pass and labeled data. Shipping that is a week 2
 * task; this script is week 0, validating the DECOMPOSITION MATH.
 *
 * Outputs:
 *   - perplexity-free cosine stat per k ∈ {1,2,3,4,8}
 *   - per-cluster slice sizes
 *   - total bytes per cluster (what a phone would download)
 */

import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

// ─── safetensors loader ─────────────────────────────────────────────

/**
 * Parse a .safetensors file. Format is: u64 LE header_len, JSON header,
 * then raw tensor bytes at the offsets the header gives.
 */
function loadSafetensors(path) {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const headerJson = buf.subarray(8, 8 + headerLen).toString("utf8");
  const header = JSON.parse(headerJson);
  const dataStart = 8 + headerLen;

  // Return a map: tensor name -> { dtype, shape, offset, byteLen, fetch() }
  const tensors = {};
  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [startOff, endOff] = meta.data_offsets;
    tensors[name] = {
      dtype: meta.dtype,
      shape: meta.shape,
      byteLen: endOff - startOff,
      fetch: () => buf.subarray(dataStart + startOff, dataStart + endOff),
    };
  }
  return tensors;
}

// ─── bf16 / fp16 → f32 ──────────────────────────────────────────────

/**
 * bfloat16 → float32. bf16 is the top 16 bits of an fp32, so promotion
 * is a zero-extended u16 shift.
 */
function bf16ToF32(buf) {
  const n = buf.length / 2;
  const out = new Float32Array(n);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < n; i++) {
    const lo = buf[i * 2];
    const hi = buf[i * 2 + 1];
    u32[i] = ((hi << 8) | lo) << 16;
  }
  return out;
}

function fp16ToF32(buf) {
  // IEEE 754 half-precision → single-precision.
  const n = buf.length / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const h = buf[i * 2] | (buf[i * 2 + 1] << 8);
    const s = (h >> 15) & 0x1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    let f;
    if (e === 0) {
      f = (m === 0) ? 0 : Math.pow(2, -14) * (m / 1024);
    } else if (e === 0x1f) {
      f = (m === 0) ? Infinity : NaN;
    } else {
      f = Math.pow(2, e - 15) * (1 + m / 1024);
    }
    out[i] = s ? -f : f;
  }
  return out;
}

function loadTensorF32(tensor) {
  const raw = tensor.fetch();
  if (tensor.dtype === "BF16") return bf16ToF32(raw);
  if (tensor.dtype === "F16") return fp16ToF32(raw);
  if (tensor.dtype === "F32") return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLen / 4);
  throw new Error(`Unsupported dtype: ${tensor.dtype}`);
}

// ─── Minimal dense math (CPU, float32) ──────────────────────────────

/** A[M,K] @ B[K,N] → C[M,N], row-major. */
function matmul(A, B, M, K, N) {
  const C = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const a = A[i * K + k];
      for (let j = 0; j < N; j++) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
  return C;
}

/** Column slice of a [M,N] matrix — cols [j0, j1). Returns [M, j1-j0]. */
function colSlice(W, M, N, j0, j1) {
  const w = j1 - j0;
  const out = new Float32Array(M * w);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < w; j++) {
      out[i * w + j] = W[i * N + (j0 + j)];
    }
  }
  return out;
}

/** Row slice of a [M,N] matrix — rows [i0, i1). Returns [i1-i0, N]. */
function rowSlice(W, M, N, i0, i1) {
  const h = i1 - i0;
  return new Float32Array(W.buffer, W.byteOffset + i0 * N * 4, h * N);
}

function gelu(x) {
  // gelu_pytorch_tanh per Gemma config
  const out = new Float32Array(x.length);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
  }
  return out;
}

function elementwiseMul(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i];
  return out;
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ─── Main experiment ────────────────────────────────────────────────

const MODEL_PATH = resolve(process.argv[2] || "../../synapse-src/model/gemma3-1b-it/model.safetensors");
const LAYER_IDX = parseInt(process.argv[3] || "12", 10);   // pick a mid-layer
const N_EXPERTS = parseInt(process.argv[4] || "8", 10);

console.log(`=== MoEfication POC ===`);
console.log(`  model:     ${MODEL_PATH}`);
console.log(`  file size: ${(statSync(MODEL_PATH).size / 1024 / 1024).toFixed(1)} MB`);
console.log(`  layer:     ${LAYER_IDX}`);
console.log(`  experts:   ${N_EXPERTS}`);
console.log();

console.time("  safetensors header parse");
const tensors = loadSafetensors(MODEL_PATH);
console.timeEnd("  safetensors header parse");
console.log(`  tensors in model: ${Object.keys(tensors).length}`);

// Locate the FFN weights for the chosen layer. Gemma 3's naming:
//   model.layers.L.mlp.gate_proj.weight   [ffn_hidden, hidden]
//   model.layers.L.mlp.up_proj.weight     [ffn_hidden, hidden]
//   model.layers.L.mlp.down_proj.weight   [hidden, ffn_hidden]
const gateKey = `model.layers.${LAYER_IDX}.mlp.gate_proj.weight`;
const upKey   = `model.layers.${LAYER_IDX}.mlp.up_proj.weight`;
const downKey = `model.layers.${LAYER_IDX}.mlp.down_proj.weight`;

for (const k of [gateKey, upKey, downKey]) {
  if (!tensors[k]) { console.error(`missing tensor: ${k}`); process.exit(1); }
  const t = tensors[k];
  console.log(`  ${k.padEnd(60)} shape=${JSON.stringify(t.shape)} dtype=${t.dtype}`);
}

console.time("  load FFN tensors to f32");
const gateW = loadTensorF32(tensors[gateKey]);  // [ffn, hidden]
const upW   = loadTensorF32(tensors[upKey]);    // [ffn, hidden]
const downW = loadTensorF32(tensors[downKey]);  // [hidden, ffn]
console.timeEnd("  load FFN tensors to f32");

const [FFN, HIDDEN] = tensors[gateKey].shape;
console.log(`  dims: hidden=${HIDDEN}  ffn_hidden=${FFN}`);
console.log();

// Gemma's weight layout for a linear is [out, in] (standard PyTorch). To multiply
// X @ W.T we treat weight as [out, in], output = sum_k X[k] * W[j, k] for output j.
// Our matmul(A,B,M,K,N) does A[M,K] @ B[K,N] — so we transpose weights on the fly
// via a view or materialise. For POC simplicity: materialise W.T once.

function transpose(W, M, N) {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      out[j * M + i] = W[i * N + j];
    }
  }
  return out;
}

console.time("  transpose weights");
const gateT = transpose(gateW, FFN, HIDDEN); // [HIDDEN, FFN]
const upT   = transpose(upW,   FFN, HIDDEN); // [HIDDEN, FFN]
const downT = transpose(downW, HIDDEN, FFN); // [FFN, HIDDEN]
console.timeEnd("  transpose weights");

// Generate a random input of SEQLEN=1 (one token's hidden state).
// Real calibration would use actual activations, but for the decomposition
// math test, any fixed input works.
const SEQLEN = 1;
const X = new Float32Array(SEQLEN * HIDDEN);
// Deterministic pseudo-random: seed = 42
let seed = 42;
function rand() { seed = (seed * 1664525 + 1013904223) | 0; return ((seed >>> 0) / 4294967296) * 2 - 1; }
for (let i = 0; i < X.length; i++) X[i] = rand() * 0.5; // scale like real RMSNorm output

// Dense forward: gemma-3 gated MLP = down_proj( gelu(gate_proj(X)) * up_proj(X) )
console.time("  dense FFN forward");
const gate = matmul(X, gateT, SEQLEN, HIDDEN, FFN);   // [1, FFN]
const up   = matmul(X, upT,   SEQLEN, HIDDEN, FFN);   // [1, FFN]
const gated = elementwiseMul(gelu(gate), up);         // [1, FFN]
const denseOut = matmul(gated, downT, SEQLEN, FFN, HIDDEN); // [1, HIDDEN]
console.timeEnd("  dense FFN forward");

// MoE decomposition: split FFN dim into N_EXPERTS equal contiguous chunks.
// Each expert i owns:
//   gateT_i = gateT[:, i*chunk:(i+1)*chunk]   [HIDDEN, chunk]
//   upT_i   = upT[:, i*chunk:(i+1)*chunk]     [HIDDEN, chunk]
//   downT_i = downT[i*chunk:(i+1)*chunk, :]   [chunk, HIDDEN]
// Expert output Y_i = matmul(gated_i, downT_i) where gated_i = gelu(gate_i) * up_i.
// Sum of all Y_i exactly equals denseOut (modulo fp rounding).
const chunk = Math.floor(FFN / N_EXPERTS);

console.time(`  compute ${N_EXPERTS} experts`);
const expertOutputs = [];
const expertMagnitudes = [];
for (let e = 0; e < N_EXPERTS; e++) {
  const j0 = e * chunk;
  const j1 = (e === N_EXPERTS - 1) ? FFN : (e + 1) * chunk;
  const cw = j1 - j0;

  const gateT_e = colSlice(gateT, HIDDEN, FFN, j0, j1);
  const upT_e   = colSlice(upT,   HIDDEN, FFN, j0, j1);
  // downT slice: rows j0..j1 × full HIDDEN cols
  const downT_e = new Float32Array(downT.buffer, downT.byteOffset + j0 * HIDDEN * 4, cw * HIDDEN);

  const g = matmul(X, gateT_e, SEQLEN, HIDDEN, cw);
  const u = matmul(X, upT_e,   SEQLEN, HIDDEN, cw);
  const gated_e = elementwiseMul(gelu(g), u);
  const y_e = matmul(gated_e, downT_e, SEQLEN, cw, HIDDEN);

  expertOutputs.push(y_e);
  // magnitude = L2 norm, a proxy for "activation" of this expert
  let sum2 = 0;
  for (let v of y_e) sum2 += v * v;
  expertMagnitudes.push(Math.sqrt(sum2));
}
console.timeEnd(`  compute ${N_EXPERTS} experts`);

console.log();
console.log(`  expert magnitudes: ${expertMagnitudes.map(m => m.toFixed(3)).join(", ")}`);

// Sanity: full sum reconstructs dense exactly (within fp rounding).
const fullSum = new Float32Array(HIDDEN);
for (const y of expertOutputs) {
  for (let i = 0; i < HIDDEN; i++) fullSum[i] += y[i];
}
const fullCos = cosine(denseOut, fullSum);
console.log(`  full-sum cosine vs dense: ${fullCos.toFixed(6)}   (should be >0.9999)`);

// The real test: sparse top-k approximation. Rank experts by magnitude,
// sum only the top-k. Measure cosine.
const ranked = expertOutputs.map((y, i) => ({ i, mag: expertMagnitudes[i], y }))
  .sort((a, b) => b.mag - a.mag);
console.log();
console.log(`  top-K sparse approximation (cosine vs dense):`);
for (const k of [1, 2, 3, 4, Math.floor(N_EXPERTS / 2), N_EXPERTS]) {
  const topk = new Float32Array(HIDDEN);
  for (let r = 0; r < k; r++) {
    const y = ranked[r].y;
    for (let i = 0; i < HIDDEN; i++) topk[i] += y[i];
  }
  const cos = cosine(denseOut, topk);
  console.log(`    k=${k.toString().padStart(2)} / ${N_EXPERTS}: cosine=${cos.toFixed(4)}   active=${(100 * k / N_EXPERTS).toFixed(0)}% compute`);
}

// Per-expert bytes (what a phone would download for this one layer):
const bytesPerExpert = chunk * HIDDEN * 4 * 3; // gate + up + down weight slices
console.log();
console.log(`  bytes per expert (this layer): ${(bytesPerExpert / 1024 / 1024).toFixed(1)} MB  (fp32 in memory)`);
console.log(`  if shipped as bf16:            ${(bytesPerExpert / 2 / 1024 / 1024).toFixed(1)} MB per expert per layer`);
console.log(`  26 layers × 8 experts × 1.5 MB ≈ ${(26 * 8 * bytesPerExpert / 2 / 1024 / 1024).toFixed(0)} MB total fleet capacity`);
