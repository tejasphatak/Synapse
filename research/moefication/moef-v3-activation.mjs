#!/usr/bin/env node
/**
 * MoEfication v3 — activation-clustered experts.
 *
 * v2 clustered FFN neurons by their weight signatures. That helped modestly
 * (+3% cosine) but didn't reach published MoEfication quality because
 * WEIGHT similarity isn't the same as ACTIVATION similarity.
 *
 * v3: cluster by actual activation patterns observed over many diverse
 * inputs. Feed N random-magnitude-realistic inputs through the FFN's
 * gate_proj + up_proj path, record each neuron's post-gelu*up activation
 * across the input batch, then K-means cluster neurons in this
 * "co-activation space". Neurons that fire together on similar inputs
 * get grouped.
 *
 * This is the single-layer approximation of MoEfication. True
 * MoEfication uses activations recorded from FULL-MODEL forward passes
 * on natural text; we approximate with random inputs into a single
 * FFN. Still directly tests "does activation-based clustering beat
 * weight-based?".
 *
 * Runs 3-way comparison: sequential | weight-clustered | activation-clustered.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// ─── boilerplate (duplicated from v2 for self-containment) ────────

function loadSafetensors(path) {
  const buf = readFileSync(path);
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.subarray(8, 8 + headerLen).toString("utf8"));
  const dataStart = 8 + headerLen;
  const tensors = {};
  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [s, e] = meta.data_offsets;
    tensors[name] = { dtype: meta.dtype, shape: meta.shape,
      fetch: () => buf.subarray(dataStart + s, dataStart + e) };
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
function loadTensorF32(t) { return t.dtype === "BF16" ? bf16ToF32(t.fetch()) : new Float32Array(t.fetch().buffer, t.fetch().byteOffset, t.fetch().byteLen / 4); }
function matmul(A, B, M, K, N) {
  const C = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let k = 0; k < K; k++) {
    const a = A[i * K + k]; if (a === 0) continue;
    for (let j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
  }
  return C;
}
function transpose(W, M, N) {
  const o = new Float32Array(M * N);
  for (let i = 0; i < M; i++) for (let j = 0; j < N; j++) o[j * M + i] = W[i * N + j];
  return o;
}
function gelu(x) {
  const o = new Float32Array(x.length);
  const c = Math.sqrt(2 / Math.PI);
  for (let i = 0; i < x.length; i++) { const v = x[i]; o[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v))); }
  return o;
}
function emul(a, b) {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] * b[i];
  return o;
}
function cosine(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return (na && nb) ? d / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}

function kmeans(vectors, nItems, dim, k, maxIter = 20, seed = 42) {
  let s = seed;
  const rand = () => { s = (s * 1664525 + 1013904223) | 0; return (s >>> 0) / 4294967296; };
  const centroids = new Float32Array(k * dim);
  const used = new Set();
  for (let c = 0; c < k; c++) {
    let idx; do { idx = Math.floor(rand() * nItems); } while (used.has(idx));
    used.add(idx);
    for (let j = 0; j < dim; j++) centroids[c * dim + j] = vectors[idx * dim + j];
  }
  const assign = new Int32Array(nItems);
  const counts = new Int32Array(k);
  const newC = new Float32Array(k * dim);
  for (let iter = 0; iter < maxIter; iter++) {
    let ch = 0;
    for (let i = 0; i < nItems; i++) {
      let bestC = 0, bestD = Infinity;
      for (let c = 0; c < k; c++) {
        let d = 0;
        for (let j = 0; j < dim; j++) { const df = vectors[i * dim + j] - centroids[c * dim + j]; d += df * df; }
        if (d < bestD) { bestD = d; bestC = c; }
      }
      if (assign[i] !== bestC) ch++;
      assign[i] = bestC;
    }
    if (ch === 0 && iter > 0) break;
    newC.fill(0); counts.fill(0);
    for (let i = 0; i < nItems; i++) {
      const c = assign[i]; counts[c]++;
      for (let j = 0; j < dim; j++) newC[c * dim + j] += vectors[i * dim + j];
    }
    for (let c = 0; c < k; c++) if (counts[c] > 0)
      for (let j = 0; j < dim; j++) centroids[c * dim + j] = newC[c * dim + j] / counts[c];
  }
  return { assign, counts };
}

// ─── Main ──────────────────────────────────────────────────────────

const MODEL_PATH = resolve(process.argv[2] || "../../synapse-src/model/gemma3-1b-it/model.safetensors");
const LAYER_IDX = parseInt(process.argv[3] || "12", 10);
const N_EXPERTS = parseInt(process.argv[4] || "8", 10);
const N_CALIB = 64;   // calibration inputs for activation recording
const N_EVAL  = 10;   // eval trials (separate from calibration set)

console.log(`=== MoEfication v3 (activation-clustered) ===`);
console.log(`  layer ${LAYER_IDX}, ${N_EXPERTS} experts, calib=${N_CALIB}, eval=${N_EVAL}`);

const tensors = loadSafetensors(MODEL_PATH);
const gateW = loadTensorF32(tensors[`model.layers.${LAYER_IDX}.mlp.gate_proj.weight`]);
const upW   = loadTensorF32(tensors[`model.layers.${LAYER_IDX}.mlp.up_proj.weight`]);
const downW = loadTensorF32(tensors[`model.layers.${LAYER_IDX}.mlp.down_proj.weight`]);

const [FFN, HIDDEN] = tensors[`model.layers.${LAYER_IDX}.mlp.gate_proj.weight`].shape;
console.log(`  dims: hidden=${HIDDEN}  ffn=${FFN}`);

const gateT = transpose(gateW, FFN, HIDDEN);
const upT   = transpose(upW,   FFN, HIDDEN);
const downT = transpose(downW, HIDDEN, FFN);

// ─── Record activations across N_CALIB random inputs ─────────────

let seed = 42;
function rand() { seed = (seed * 1664525 + 1013904223) | 0; return ((seed >>> 0) / 4294967296) * 2 - 1; }

console.time("  calibration: record FFN activations");
// For each neuron n, we'll have a N_CALIB-dim signature. Store as
// [FFN, N_CALIB] — rows=neurons, cols=calibration samples.
const activationSig = new Float32Array(FFN * N_CALIB);

for (let t = 0; t < N_CALIB; t++) {
  const X = new Float32Array(HIDDEN);
  for (let i = 0; i < HIDDEN; i++) X[i] = rand() * 0.5;
  // Compute gate(X) and up(X) → gelu(gate) * up → neuron activations
  const gate = matmul(X, gateT, 1, HIDDEN, FFN);
  const up   = matmul(X, upT,   1, HIDDEN, FFN);
  const act  = emul(gelu(gate), up);
  // Store this sample's activation for each neuron
  for (let n = 0; n < FFN; n++) activationSig[n * N_CALIB + t] = act[n];
  if ((t + 1) % 16 === 0) process.stdout.write(".");
}
console.log();
console.timeEnd("  calibration: record FFN activations");

// Normalize each neuron's signature to unit L2 so clustering is by PATTERN not MAGNITUDE
for (let n = 0; n < FFN; n++) {
  let s = 0;
  for (let t = 0; t < N_CALIB; t++) s += activationSig[n * N_CALIB + t] ** 2;
  s = Math.sqrt(s) || 1;
  for (let t = 0; t < N_CALIB; t++) activationSig[n * N_CALIB + t] /= s;
}

console.time(`  k-means on activation signatures (${FFN} × ${N_CALIB})`);
const { assign: actClusters, counts: actCounts } =
  kmeans(activationSig, FFN, N_CALIB, N_EXPERTS, 20);
console.timeEnd(`  k-means on activation signatures (${FFN} × ${N_CALIB})`);
console.log(`  activation-cluster sizes: ${Array.from(actCounts).join(", ")}`);

// ─── Build neuron lists and evaluate ────────────────────────────

function buildLists(assign, k) {
  const l = Array.from({ length: k }, () => []);
  for (let n = 0; n < assign.length; n++) l[assign[n]].push(n);
  return l;
}
function buildSeq(n, k) {
  const chunk = Math.ceil(n / k);
  const l = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) l[Math.min(k - 1, Math.floor(i / chunk))].push(i);
  return l;
}
function denseFFN(X) {
  const gate = matmul(X, gateT, 1, HIDDEN, FFN);
  const up   = matmul(X, upT,   1, HIDDEN, FFN);
  const gated = emul(gelu(gate), up);
  return matmul(gated, downT, 1, FFN, HIDDEN);
}
function expertOut(X, neuronIds) {
  const m = neuronIds.length;
  const gT = new Float32Array(HIDDEN * m), uT = new Float32Array(HIDDEN * m);
  const dT = new Float32Array(m * HIDDEN);
  for (let idx = 0; idx < m; idx++) {
    const n = neuronIds[idx];
    for (let h = 0; h < HIDDEN; h++) {
      gT[h * m + idx] = gateT[h * FFN + n];
      uT[h * m + idx] = upT[h * FFN + n];
      dT[idx * HIDDEN + h] = downT[n * HIDDEN + h];
    }
  }
  const g = matmul(X, gT, 1, HIDDEN, m);
  const u = matmul(X, uT, 1, HIDDEN, m);
  const gated = emul(gelu(g), u);
  return matmul(gated, dT, 1, m, HIDDEN);
}
function topkEval(X, lists, k) {
  const outs = lists.map(l => expertOut(X, l));
  const mags = outs.map(y => { let s = 0; for (const v of y) s += v * v; return Math.sqrt(s); });
  const rank = outs.map((_, i) => i).sort((a, b) => mags[b] - mags[a]);
  const acc = new Float32Array(HIDDEN);
  for (let r = 0; r < k; r++) { const y = outs[rank[r]]; for (let i = 0; i < HIDDEN; i++) acc[i] += y[i]; }
  return acc;
}

const seqLists = buildSeq(FFN, N_EXPERTS);
const actLists = buildLists(actClusters, N_EXPERTS);

// Also build weight-clustered for comparison
console.time(`  building weight cluster signatures`);
const wsig = new Float32Array(FFN * 2 * HIDDEN);
for (let i = 0; i < FFN; i++) {
  let gn = 0, un = 0;
  for (let j = 0; j < HIDDEN; j++) { gn += gateW[i * HIDDEN + j] ** 2; un += upW[i * HIDDEN + j] ** 2; }
  gn = Math.sqrt(gn) || 1; un = Math.sqrt(un) || 1;
  for (let j = 0; j < HIDDEN; j++) {
    wsig[i * 2 * HIDDEN + j]          = gateW[i * HIDDEN + j] / gn;
    wsig[i * 2 * HIDDEN + HIDDEN + j] = upW[i * HIDDEN + j] / un;
  }
}
const { assign: wClusters } = kmeans(wsig, FFN, 2 * HIDDEN, N_EXPERTS, 15);
const wLists = buildLists(wClusters, N_EXPERTS);
console.timeEnd(`  building weight cluster signatures`);

// Eval: separate random inputs not used in calibration
const KS = [1, 2, 3, 4];
const res = { seq: {}, w: {}, act: {} };
for (const k of KS) { res.seq[k] = []; res.w[k] = []; res.act[k] = []; }

seed = 77777;  // different seed, different inputs from calibration
for (let t = 0; t < N_EVAL; t++) {
  const X = new Float32Array(HIDDEN);
  for (let i = 0; i < HIDDEN; i++) X[i] = rand() * 0.5;
  const dense = denseFFN(X);
  for (const k of KS) {
    res.seq[k].push(cosine(dense, topkEval(X, seqLists, k)));
    res.w[k].push(cosine(dense, topkEval(X, wLists,  k)));
    res.act[k].push(cosine(dense, topkEval(X, actLists, k)));
  }
  process.stdout.write(".");
}
console.log();

function avg(a) { return a.reduce((x, y) => x + y, 0) / a.length; }

console.log(`\n  cosine vs dense, 3-way comparison (avg of ${N_EVAL} held-out inputs):`);
console.log(`    k      sequential     weight-clust    activation-clust    act vs seq`);
for (const k of KS) {
  const s = avg(res.seq[k]), w = avg(res.w[k]), a = avg(res.act[k]);
  console.log(`    ${k}/${N_EXPERTS}    ${s.toFixed(4)}         ${w.toFixed(4)}          ${a.toFixed(4)}              ${((a - s) >= 0 ? "+" : "")}${(a - s).toFixed(4)}`);
}
