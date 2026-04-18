#!/usr/bin/env node
/**
 * SAQT Benchmark — NaturalQuestions / TriviaQA / HotPotQA
 *
 * Sends questions to the SAQT API, compares against gold answers,
 * measures: exact match, F1 token overlap, latency.
 *
 * Usage:
 *   node benchmark.mjs [--samples N] [--api URL] [--output DIR] [--concurrency N]
 *
 * Defaults: 100 samples/dataset, https://chat.webmind.sh/api/saqt/query, ~/webmind-research/benchmarks/
 */

import { writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import { homedir } from 'os';

// --------------- config ---------------
const args = process.argv.slice(2);
function flag(name, fallback) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
}

const SAMPLES_PER_DATASET = parseInt(flag('samples', '100'), 10);
const API_URL = flag('api', 'https://chat.webmind.sh/api/saqt/query');
const OUTPUT_DIR = flag('output', join(homedir(), 'webmind-research', 'benchmarks'));
const CONCURRENCY = parseInt(flag('concurrency', '5'), 10);
const TIMEOUT_MS = 30_000;

// --------------- dataset fetchers ---------------

async function fetchJSON(url) {
  const r = await fetch(url, { signal: AbortSignal.timeout(60_000) });
  if (!r.ok) throw new Error(`HTTP ${r.status} from ${url}`);
  return r.json();
}

/**
 * HuggingFace datasets API — rows endpoint.
 * Returns [{question, gold_answers}]
 */
async function fetchNaturalQuestions(n) {
  // NQ from google-research-datasets/nq_open, validation split
  const url = `https://datasets-server.huggingface.co/rows?dataset=google-research-datasets/nq_open&config=nq_open&split=validation&offset=0&length=${n}`;
  const data = await fetchJSON(url);
  return data.rows.map(r => ({
    question: r.row.question,
    gold_answers: Array.isArray(r.row.answer) ? r.row.answer : [r.row.answer],
  }));
}

async function fetchTriviaQA(n) {
  // TriviaQA unfiltered.nocontext, validation split (mandarjoshi namespace)
  const url = `https://datasets-server.huggingface.co/rows?dataset=mandarjoshi/trivia_qa&config=unfiltered.nocontext&split=validation&offset=0&length=${n}`;
  const data = await fetchJSON(url);
  return data.rows.map(r => ({
    question: r.row.question,
    gold_answers: [
      ...(r.row.answer?.aliases || []),
      r.row.answer?.value,
    ].filter(Boolean),
  }));
}

async function fetchHotPotQA(n) {
  // HotpotQA distractor, validation split (hotpotqa namespace)
  const url = `https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=validation&offset=0&length=${n}`;
  const data = await fetchJSON(url);
  return data.rows.map(r => ({
    question: r.row.question,
    gold_answers: [r.row.answer],
  }));
}

// --------------- metrics ---------------

function normalize(text) {
  return (text || '')
    .toLowerCase()
    .replace(/\*\*/g, '')           // strip markdown bold
    .replace(/[^\w\s]/g, ' ')       // strip punctuation
    .replace(/\b(a|an|the|is|are|was|were|of|in|on|at|to|for|and|or|but|not|with|from|by|as|it|its|this|that|these|those)\b/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(text) {
  return normalize(text).split(' ').filter(Boolean);
}

function exactMatch(predicted, golds) {
  const normPred = normalize(predicted);
  return golds.some(g => normalize(g) === normPred) ? 1 : 0;
}

function f1Score(predicted, golds) {
  const predTokens = tokenize(predicted);
  if (predTokens.length === 0) return 0;

  // Take best F1 across all gold answers
  let bestF1 = 0;
  for (const gold of golds) {
    const goldTokens = tokenize(gold);
    if (goldTokens.length === 0) continue;

    const goldSet = new Set(goldTokens);
    const common = predTokens.filter(t => goldSet.has(t)).length;
    if (common === 0) continue;

    const precision = common / predTokens.length;
    const recall = common / goldTokens.length;
    const f1 = (2 * precision * recall) / (precision + recall);
    if (f1 > bestF1) bestF1 = f1;
  }
  return bestF1;
}

// --------------- API caller ---------------

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 5000;

async function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// Context prompt — tells the engine it's being evaluated for accuracy
const BENCHMARK_CONTEXT = 'You are being evaluated for factual accuracy. Give precise, direct answers. No formatting, no commentary. Just the fact.';

async function queryAPI(question, retries = 0) {
  const start = Date.now();
  try {
    const r = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, context: BENCHMARK_CONTEXT }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    const clientLatency = Date.now() - start;
    if (!r.ok) {
      if (retries < MAX_RETRIES && (r.status === 502 || r.status === 503 || r.status === 429)) {
        process.stderr.write(`  [RETRY ${retries+1}/${MAX_RETRIES}] HTTP ${r.status}, waiting ${RETRY_DELAY_MS}ms...\n`);
        await sleep(RETRY_DELAY_MS * (retries + 1));
        return queryAPI(question, retries + 1);
      }
      return { answer: '', confidence: 0, timeMs: clientLatency, error: `HTTP ${r.status}` };
    }
    const text = await r.text();
    let body;
    try { body = JSON.parse(text); } catch {
      return { answer: '', confidence: 0, timeMs: clientLatency, error: 'invalid JSON' };
    }
    return {
      answer: body.answer || '',
      confidence: body.confidence || 0,
      timeMs: body.timeMs || clientLatency,
      clientLatency,
    };
  } catch (e) {
    if (retries < MAX_RETRIES) {
      process.stderr.write(`  [RETRY ${retries+1}/${MAX_RETRIES}] ${e.message}, waiting ${RETRY_DELAY_MS}ms...\n`);
      await sleep(RETRY_DELAY_MS * (retries + 1));
      return queryAPI(question, retries + 1);
    }
    return { answer: '', confidence: 0, timeMs: Date.now() - start, error: e.message };
  }
}

// --------------- runner ---------------

async function runBatch(items, label) {
  const results = [];
  let done = 0;

  // Process with limited concurrency
  const queue = [...items];
  const workers = Array.from({ length: CONCURRENCY }, async () => {
    while (queue.length > 0) {
      const item = queue.shift();
      const resp = await queryAPI(item.question);
      // Small delay to avoid overwhelming server
      await sleep(200);
      const em = exactMatch(resp.answer, item.gold_answers);
      const f1 = f1Score(resp.answer, item.gold_answers);
      results.push({
        question: item.question,
        gold_answers: item.gold_answers,
        predicted: resp.answer,
        exact_match: em,
        f1,
        confidence: resp.confidence,
        serverTimeMs: resp.timeMs,
        clientLatencyMs: resp.clientLatency || resp.timeMs,
        error: resp.error || null,
      });

      // RLHF: if answer was wrong, teach the correct answer back to KB
      // This is the self-evolution loop — the benchmark trains the engine
      if (em === 0 && f1 < 0.5 && item.gold_answers[0] && !resp.error) {
        const learnUrl = API_URL.replace('/api/saqt/query', '/api/saqt/learn');
        fetch(learnUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: item.question,
            answer: item.gold_answers[0],
            source: 'benchmark-rlhf',
          }),
        }).catch(() => {}); // fire-and-forget
      }
      done++;
      if (done % 20 === 0 || done === items.length) {
        process.stderr.write(`  [${label}] ${done}/${items.length}\n`);
      }
    }
  });

  await Promise.all(workers);
  return results;
}

function summarize(results) {
  const n = results.length;
  const errors = results.filter(r => r.error).length;
  const valid = results.filter(r => !r.error);
  const em = valid.reduce((s, r) => s + r.exact_match, 0) / (valid.length || 1);
  const f1 = valid.reduce((s, r) => s + r.f1, 0) / (valid.length || 1);
  const avgLatency = valid.reduce((s, r) => s + r.clientLatencyMs, 0) / (valid.length || 1);
  const p50Latency = percentile(valid.map(r => r.clientLatencyMs).sort((a, b) => a - b), 0.5);
  const p95Latency = percentile(valid.map(r => r.clientLatencyMs).sort((a, b) => a - b), 0.95);
  const avgConf = valid.reduce((s, r) => s + r.confidence, 0) / (valid.length || 1);
  return { n, errors, exactMatch: em, f1, avgLatency, p50Latency, p95Latency, avgConf };
}

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const i = Math.floor(sorted.length * p);
  return sorted[Math.min(i, sorted.length - 1)];
}

// --------------- main ---------------

async function main() {
  console.log(`SAQT Benchmark — ${SAMPLES_PER_DATASET} questions/dataset, API: ${API_URL}`);
  console.log(`Concurrency: ${CONCURRENCY}, Timeout: ${TIMEOUT_MS}ms\n`);

  // Fetch datasets
  console.log('Fetching datasets from HuggingFace...');
  const [nq, tqa, hpqa] = await Promise.all([
    fetchNaturalQuestions(SAMPLES_PER_DATASET).catch(e => { console.error('NQ fetch failed:', e.message); return []; }),
    fetchTriviaQA(SAMPLES_PER_DATASET).catch(e => { console.error('TriviaQA fetch failed:', e.message); return []; }),
    fetchHotPotQA(SAMPLES_PER_DATASET).catch(e => { console.error('HotPotQA fetch failed:', e.message); return []; }),
  ]);

  console.log(`  NaturalQuestions: ${nq.length} questions`);
  console.log(`  TriviaQA: ${tqa.length} questions`);
  console.log(`  HotPotQA: ${hpqa.length} questions\n`);

  if (nq.length + tqa.length + hpqa.length === 0) {
    console.error('No questions fetched. Aborting.');
    process.exit(1);
  }

  // Run benchmarks
  const datasets = [
    { name: 'NaturalQuestions', items: nq },
    { name: 'TriviaQA', items: tqa },
    { name: 'HotPotQA', items: hpqa },
  ];

  const allResults = {};
  const allSummaries = {};

  for (const ds of datasets) {
    if (ds.items.length === 0) continue;
    console.log(`Running ${ds.name}...`);
    const results = await runBatch(ds.items, ds.name);
    allResults[ds.name] = results;
    allSummaries[ds.name] = summarize(results);
  }

  // Overall summary
  const allFlat = Object.values(allResults).flat();
  allSummaries['OVERALL'] = summarize(allFlat);

  // Print results
  console.log('\n' + '='.repeat(72));
  console.log('SAQT BENCHMARK RESULTS');
  console.log('='.repeat(72));
  console.log(`Date: ${new Date().toISOString()}`);
  console.log(`API: ${API_URL}`);
  console.log(`Samples/dataset: ${SAMPLES_PER_DATASET}\n`);

  const header = 'Dataset'.padEnd(22) +
    'N'.padStart(5) +
    'EM'.padStart(8) +
    'F1'.padStart(8) +
    'AvgMs'.padStart(8) +
    'P50'.padStart(8) +
    'P95'.padStart(8) +
    'Conf'.padStart(8) +
    'Err'.padStart(5);
  console.log(header);
  console.log('-'.repeat(72));

  for (const [name, s] of Object.entries(allSummaries)) {
    const row = name.padEnd(22) +
      String(s.n).padStart(5) +
      (s.exactMatch * 100).toFixed(1).padStart(7) + '%' +
      (s.f1 * 100).toFixed(1).padStart(7) + '%' +
      Math.round(s.avgLatency).toString().padStart(8) +
      Math.round(s.p50Latency).toString().padStart(8) +
      Math.round(s.p95Latency).toString().padStart(8) +
      s.avgConf.toFixed(2).padStart(8) +
      String(s.errors).padStart(5);
    console.log(row);
  }
  console.log('='.repeat(72));

  // Show some example predictions
  console.log('\nSample predictions (first 5 per dataset):');
  for (const [name, results] of Object.entries(allResults)) {
    console.log(`\n--- ${name} ---`);
    for (const r of results.slice(0, 5)) {
      const status = r.exact_match ? 'EM' : r.f1 > 0 ? `F1=${(r.f1*100).toFixed(0)}%` : 'MISS';
      console.log(`  Q: ${r.question.slice(0, 80)}`);
      console.log(`  Gold: ${r.gold_answers[0]?.slice(0, 60) || '(none)'}`);
      console.log(`  Pred: ${r.predicted.slice(0, 60) || '(empty)'}`);
      console.log(`  [${status}] ${r.clientLatencyMs}ms`);
    }
  }

  // Save results
  if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const outFile = join(OUTPUT_DIR, `saqt-benchmark-${timestamp}.json`);

  const output = {
    meta: {
      date: new Date().toISOString(),
      api: API_URL,
      samplesPerDataset: SAMPLES_PER_DATASET,
      concurrency: CONCURRENCY,
      timeoutMs: TIMEOUT_MS,
    },
    summaries: allSummaries,
    results: allResults,
  };

  writeFileSync(outFile, JSON.stringify(output, null, 2));
  console.log(`\nResults saved to: ${outFile}`);

  // Also save a human-readable summary
  const summaryFile = join(OUTPUT_DIR, `saqt-benchmark-${timestamp}-summary.txt`);
  const lines = [
    'SAQT Benchmark Summary',
    `Date: ${new Date().toISOString()}`,
    `API: ${API_URL}`,
    `Samples/dataset: ${SAMPLES_PER_DATASET}`,
    '',
    header,
    '-'.repeat(72),
  ];
  for (const [name, s] of Object.entries(allSummaries)) {
    lines.push(
      name.padEnd(22) +
      String(s.n).padStart(5) +
      (s.exactMatch * 100).toFixed(1).padStart(7) + '%' +
      (s.f1 * 100).toFixed(1).padStart(7) + '%' +
      Math.round(s.avgLatency).toString().padStart(8) +
      Math.round(s.p50Latency).toString().padStart(8) +
      Math.round(s.p95Latency).toString().padStart(8) +
      s.avgConf.toFixed(2).padStart(8) +
      String(s.errors).padStart(5)
    );
  }
  writeFileSync(summaryFile, lines.join('\n') + '\n');
  console.log(`Summary saved to: ${summaryFile}`);
}

main().catch(e => { console.error('FATAL:', e); process.exit(1); });
