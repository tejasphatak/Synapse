#!/usr/bin/env node
/**
 * MoEfication v2 — clustered experts.
 *
 * Extends moef-poc.mjs with two clustering strategies:
 *
 *   (a) WEIGHT-clustering — no calibration data needed. K-means on
 *       the (gate_col, up_col) weight signature per FFN neuron.
 *       Neurons with similar weights get grouped.
 *
 *   (b) ACTIVATION-clustering (mini calibration) — feed N random-but-
 *       varied inputs through the FFN, record each neuron's activation
 *       pattern (gate*up post-GELU), K-means cluster neurons by their
 *       activation vectors over the input batch. Proxy for real
 *       MoEfication.
 *
 * Compares (a) and (b) against the baseline sequential split at the
 * same top-k sparsity levels.
 *
 * Usage:
 *   node --max-old-space-size=6144 moef-v2-cluster.mjs [path] [layer] [n_experts]
 */

import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

// ─── Shared with v1 POC (inlined for self-contained file) ──────────

function loadSafetensors(path) {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString("utf8"));
  const dataStart = 8 + headerLen;
  const tensors = {};
  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [s, e] = meta.data_offsets;
    tensors[name] = {
      dtype: meta.dtype, shape: meta.shape, byteLen: e - s,
      fetch: () => buf.subarray(dataStart + s, dataStart + e),
    };
  }
  return tensors;
}

function bf16ToF32(buf) {
  const n = buf.length / 2;
  const out = new Float32Array(n);
  const u32 = new Uint32Array(out.buffer);
  for (let i = 0; i < n; i++) u32[i] = ((buf[i * 2 + 1] << 8) | buf[i * 2]) << 16;
  return out;
}

function loadTensorF32(t) {
  const r = t.fetch();
  if (t.dtype === "BF16") return bf16ToF32(r);
  if (t.dtype === "F32") return new Float32Array(r.buffer, r.byteOffset, r.byteLen / 4);
  throw new Error(`dtype ${t.dtype}`);
}

function matmul(A, B, M, K, N) {
  const C = new Float32Array(M * N);
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < K; k++) {
      const a = A[i * K + k];
      if (a === 0) continue;
      for (let j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
    }
  }
  return C;
}

function transpose(W, M, N) {
  const out = new Float32Array(M * N);
  for (let i = 0; i < M; i++)
    for (let j = 0; j < N; j++) out[j * M + i] = W[i * N + j];
  return out;
}

function gelu(x) {
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
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return (na && nb) ? d / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

// ─── K-means (simple, pure JS) ─────────────────────────────────────

/**
 * vectors: Float32Array of shape [N_items, dim], flattened
 * k: number of clusters
 * maxIter: default 30
 * Returns: Int32Array of length N_items with cluster assignments [0..k-1]
 */
function kmeans(vectors, nItems, dim, k, maxIter = 30, seed = 42) {
  // Random init: pick k random items as centroids
  let s = seed;
  const rand = () => { s = (s * 1664525 + 1013904223) | 0; return (s >>> 0) / 4294967296; };
  const centroids = new Float32Array(k * dim);
  const used = new Set();
  for (let c = 0; c < k; c++) {
    let idx;
    do { idx = Math.floor(rand() * nItems); } while (used.has(idx));
    used.add(idx);
    for (let j = 0; j < dim; j++) centroids[c * dim + j] = vectors[idx * dim + j];
  }

  const assign = new Int32Array(nItems);
  const counts = new Int32Array(k);
  const newCentroids = new Float32Array(k * dim);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = 0;
    // assign each item to nearest centroid (L2)
    for (let i = 0; i < nItems; i++) {
      let bestC = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        let d = 0;
        for (let j = 0; j < dim; j++) {
          const diff = vectors[i * dim + j] - centroids[c * dim + j];
          d += diff * diff;
        }
        if (d < bestD) { bestD = d; bestC = c; }
      }
      if (assign[i] !== bestC) changed++;
      assign[i] = bestC;
    }
    if (changed === 0 && iter > 0) break;

    // update centroids = mean of assigned items
    newCentroids.fill(0);
    counts.fill(0);
    for (let i = 0; i < nItems; i++) {
      const c = assign[i];
      counts[c]++;
      for (let j = 0; j < dim; j++) newCentroids[c * dim + j] += vectors[i * dim + j];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c] > 0) {
        for (let j = 0; j < dim; j++) centroids[c * dim + j] = newCentroids[c * dim + j] / counts[c];
      }
    }
  }
  return { assign, counts };
}

// ─── Main ───────────────────────────────────────────────────────────

const MODEL_PATH = resolve(process.argv[2] || "../../synapse-src/model/gemma3-1b-it/model.safetensors");
const LAYER_IDX = parseInt(process.argv[3] || "12", 10);
const N_EXPERTS = parseInt(process.argv[4] || "8", 10);

console.log(`=== MoEfication v2 (clustered) ===`);
console.log(`  layer ${LAYER_IDX}, ${N_EXPERTS} experts`);

const tensors = loadSafetensors(MODEL_PATH);
const gateW = loadTensorF32(tensors[`model.layers.${LAYER_IDX}.mlp.gate_proj.weight`]);
const upW   = loadTensorF32(tensors[`model.layers.${LAYER_IDX}.mlp.up_proj.weight`]);
const downW = loadTensorF32(tensors[`model.layers.${LAYER_IDX}.mlp.down_proj.weight`]);

const [FFN, HIDDEN] = tensors[`model.layers.${LAYER_IDX}.mlp.gate_proj.weight`].shape;
console.log(`  dims: hidden=${HIDDEN}  ffn=${FFN}`);

const gateT = transpose(gateW, FFN, HIDDEN); // [HIDDEN, FFN]
const upT   = transpose(upW,   FFN, HIDDEN); // [HIDDEN, FFN]
const downT = transpose(downW, HIDDEN, FFN); // [FFN, HIDDEN]
console.log(`  weights loaded + transposed`);

// ─── Build neuron signatures for weight-clustering ───────────────

// Each FFN neuron i has gate_col = gateW[i, :] (row of original weight)
// We cluster in (gate || up) space. dim per neuron = 2 * HIDDEN.
console.time("  build weight signatures");
const signatures = new Float32Array(FFN * 2 * HIDDEN);
for (let i = 0; i < FFN; i++) {
  // normalize each row (cosine clustering rather than magnitude)
  let gn = 0, un = 0;
  for (let j = 0; j < HIDDEN; j++) {
    gn += gateW[i * HIDDEN + j] ** 2;
    un += upW[i * HIDDEN + j] ** 2;
  }
  gn = Math.sqrt(gn) || 1;
  un = Math.sqrt(un) || 1;
  for (let j = 0; j < HIDDEN; j++) {
    signatures[i * 2 * HIDDEN + j]          = gateW[i * HIDDEN + j] / gn;
    signatures[i * 2 * HIDDEN + HIDDEN + j] = upW[i * HIDDEN + j] / un;
  }
}
console.timeEnd("  build weight signatures");

console.time(`  k-means on weight signatures (${FFN} items × ${2 * HIDDEN}-dim)`);
const { assign: weightClusters, counts: weightCounts } =
  kmeans(signatures, FFN, 2 * HIDDEN, N_EXPERTS, 15);
console.timeEnd(`  k-means on weight signatures (${FFN} items × ${2 * HIDDEN}-dim)`);
console.log(`  weight-cluster sizes: ${Array.from(weightCounts).join(", ")}`);

// Free signatures, large buffer
// (no explicit free in JS, but dropping reference helps GC)

// ─── Evaluate dense vs sequential vs weight-clustered ─────────

const SEQLEN = 1;
// Try a few different random seeds to average
const N_TRIALS = 5;

function computeFFN(X) {
  const gate = matmul(X, gateT, SEQLEN, HIDDEN, FFN);
  const up   = matmul(X, upT,   SEQLEN, HIDDEN, FFN);
  const gated = elementwiseMul(gelu(gate), up);
  return { gated, out: matmul(gated, downT, SEQLEN, FFN, HIDDEN) };
}

function expertOutputForNeuronSet(X, neuronIds) {
  // Compute Y_e = downT[neuronIds, :] @ (gelu(gate[neuronIds]) * up[neuronIds])
  // Build slice matrices
  const m = neuronIds.length;
  const gateT_e = new Float32Array(HIDDEN * m);
  const upT_e   = new Float32Array(HIDDEN * m);
  const downT_e = new Float32Array(m * HIDDEN);
  for (let idx = 0; idx < m; idx++) {
    const n = neuronIds[idx];
    for (let h = 0; h < HIDDEN; h++) {
      gateT_e[h * m + idx] = gateT[h * FFN + n];
      upT_e[h * m + idx]   = upT[h * FFN + n];
    }
    for (let h = 0; h < HIDDEN; h++) {
      downT_e[idx * HIDDEN + h] = downT[n * HIDDEN + h];
    }
  }
  const g = matmul(X, gateT_e, SEQLEN, HIDDEN, m);
  const u = matmul(X, upT_e,   SEQLEN, HIDDEN, m);
  const gated = elementwiseMul(gelu(g), u);
  return matmul(gated, downT_e, SEQLEN, m, HIDDEN);
}

function topkEval(X, expertNeuronLists, k) {
  // Compute all expert outputs, rank by L2 magnitude, sum top-k, compare to dense.
  const outputs = expertNeuronLists.map(list => expertOutputForNeuronSet(X, list));
  const mags = outputs.map(y => {
    let s = 0; for (const v of y) s += v * v; return Math.sqrt(s);
  });
  const rank = outputs.map((_, i) => i).sort((a, b) => mags[b] - mags[a]);
  const acc = new Float32Array(HIDDEN);
  for (let r = 0; r < k; r++) {
    const y = outputs[rank[r]];
    for (let i = 0; i < HIDDEN; i++) acc[i] += y[i];
  }
  return acc;
}

function buildNeuronLists(assign, k) {
  const lists = Array.from({ length: k }, () => []);
  for (let n = 0; n < assign.length; n++) lists[assign[n]].push(n);
  return lists;
}

function buildSequentialLists(n_items, k) {
  const lists = Array.from({ length: k }, () => []);
  const chunk = Math.ceil(n_items / k);
  for (let n = 0; n < n_items; n++) lists[Math.min(k - 1, Math.floor(n / chunk))].push(n);
  return lists;
}

const seqLists = buildSequentialLists(FFN, N_EXPERTS);
const wcLists  = buildNeuronLists(weightClusters, N_EXPERTS);

// Run trials
const KS = [1, 2, 3, 4];
const results = { seq: {}, wc: {} };
for (const k of KS) { results.seq[k] = []; results.wc[k] = []; }

let seed = 99;
function rand() { seed = (seed * 1664525 + 1013904223) | 0; return ((seed >>> 0) / 4294967296) * 2 - 1; }

console.log(`\n  running ${N_TRIALS} trials with different random inputs`);
for (let t = 0; t < N_TRIALS; t++) {
  const X = new Float32Array(SEQLEN * HIDDEN);
  for (let i = 0; i < X.length; i++) X[i] = rand() * 0.5;
  const { out: dense } = computeFFN(X);
  for (const k of KS) {
    const seqOut = topkEval(X, seqLists, k);
    const wcOut  = topkEval(X, wcLists,  k);
    results.seq[k].push(cosine(dense, seqOut));
    results.wc[k].push(cosine(dense, wcOut));
  }
  process.stdout.write(`.`);
}
console.log();

function avg(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }

console.log(`\n  cosine vs dense (averaged over ${N_TRIALS} random inputs):`);
console.log(`    k      sequential       weight-clustered    delta`);
for (const k of KS) {
  const s = avg(results.seq[k]);
  const w = avg(results.wc[k]);
  const d = w - s;
  console.log(`    ${k}/${N_EXPERTS}    ${s.toFixed(4)}           ${w.toFixed(4)}              ${d >= 0 ? "+" : ""}${d.toFixed(4)}`);
}
