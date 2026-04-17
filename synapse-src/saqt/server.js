/**
 * SAQT API — Distributed Knowledge Retrieval
 * Loads chunks.jsonl, serves queries via text similarity + multi-hop traversal.
 */

import { readFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

let chunks = [];
let ready = false;

function loadKnowledge() {
  // Try trained_model first, then data dir
  const paths = [
    join(__dirname, '../../webmind-research/trained_model/chunks.jsonl'),
    process.env.SAQT_CHUNKS || '',
  ];

  for (const p of paths) {
    if (p && existsSync(p)) {
      const lines = readFileSync(p, 'utf-8').trim().split('\n');
      chunks = lines.map(line => JSON.parse(line));
      console.log(`[saqt] Loaded ${chunks.length} chunks from ${p}`);
      ready = true;
      return;
    }
  }

  // Fall back: load all .jsonl from data dir
  const dataDir = join(__dirname, '../../webmind-research/data');
  if (existsSync(dataDir)) {
    for (const f of readdirSync(dataDir).filter(f => f.endsWith('.jsonl')).sort()) {
      const lines = readFileSync(join(dataDir, f), 'utf-8').trim().split('\n');
      for (const line of lines) {
        const row = JSON.parse(line);
        const text = row.text || row.question || '';
        if (text.length > 10) {
          chunks.push({ text: text.substring(0, 500), topic: row.topic || row.category || 'general',
            source: row.source || f, answer: row.answer || '' });
        }
      }
    }
    console.log(`[saqt] Loaded ${chunks.length} chunks from ${dataDir}`);
    ready = true;
  }
}

function similarity(query, text) {
  const q = query.toLowerCase();
  const t = text.toLowerCase();

  // Word overlap
  const qWords = new Set(q.split(/\s+/).filter(w => w.length > 2));
  const tWords = new Set(t.split(/\s+/).filter(w => w.length > 2));
  let wordOverlap = 0;
  for (const w of qWords) if (tWords.has(w)) wordOverlap++;
  const wordScore = wordOverlap / Math.max(qWords.size, 1);

  // Trigram overlap
  const trigrams = (s) => {
    const set = new Set();
    for (let i = 0; i <= s.length - 3; i++) set.add(s.substring(i, i + 3));
    return set;
  };
  const qTri = trigrams(q);
  const tTri = trigrams(t);
  let triOverlap = 0;
  for (const ng of qTri) if (tTri.has(ng)) triOverlap++;
  const triScore = triOverlap / Math.max(qTri.size, 1);

  return 0.4 * wordScore + 0.6 * triScore;
}

function search(query, topK = 5) {
  const scored = chunks.map((chunk, idx) => ({
    chunk, idx,
    score: similarity(query, chunk.text + ' ' + (chunk.answer || ''))
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topK);
}

function saqtQuery(question, maxHops = 5) {
  const t0 = Date.now();
  const facts = [];
  const answers = [];
  const trace = [];
  let currentQuery = question;

  for (let hop = 0; hop < maxHops; hop++) {
    const results = search(currentQuery, 3);
    for (const { chunk, score } of results) {
      if (score > 0.05 && !facts.includes(chunk.text)) {
        facts.push(chunk.text);
        if (chunk.answer) answers.push(chunk.answer);
      }
    }
    trace.push({ hop, topScore: results[0]?.score || 0, facts: facts.length });
    const topFact = results[0]?.chunk?.text || '';
    currentQuery = `${question} ${topFact}`.substring(0, 300);
  }

  return {
    question, facts: facts.slice(0, 10), answers: [...new Set(answers)].slice(0, 5),
    trace, hops: maxHops, totalFacts: facts.length, timeMs: Date.now() - t0
  };
}

export function registerSAQT(app) {
  try { loadKnowledge(); } catch (e) {
    console.error('[saqt] Load failed:', e.message);
  }

  app.post('/api/saqt/query', (req, res) => {
    if (!ready) return res.status(503).json({ error: 'Not loaded' });
    const { question, hops = 5 } = req.body;
    if (!question) return res.status(400).json({ error: 'Missing question' });
    res.json(saqtQuery(question, Math.min(hops, 10)));
  });

  app.get('/api/saqt/stats', (_req, res) => {
    res.json({ ready, chunks: chunks.length,
      sources: [...new Set(chunks.map(c => c.source))],
      topics: [...new Set(chunks.map(c => c.topic))].length });
  });

  app.get('/api/saqt/search', (req, res) => {
    if (!ready) return res.status(503).json({ error: 'Not loaded' });
    const q = req.query.q;
    if (!q) return res.status(400).json({ error: 'Missing ?q=' });
    res.json(search(q, parseInt(req.query.k) || 5).map(r => ({
      text: r.chunk.text, answer: r.chunk.answer,
      topic: r.chunk.topic, source: r.chunk.source, score: r.score
    })));
  });

  console.log('[saqt] API ready: /api/saqt/query, /api/saqt/stats, /api/saqt/search');
}
