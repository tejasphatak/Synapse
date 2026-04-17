/**
 * SAQT Standalone Server
 * ======================
 * Serves the SAQT chat UI + knowledge API.
 * Light green theme. No GPU required.
 */

import { createServer } from 'http';
import { readFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname, extname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PORT = parseInt(process.env.SAQT_PORT || '3000', 10);

// ─── Knowledge Base ──────────────────────────────────────
let chunks = [];
let ready = false;

function loadKnowledge() {
  const home = process.env.HOME || '/home/tejasphatak';
  const paths = [
    process.env.SAQT_CHUNKS || '',
    join(home, 'webmind-research/trained_model/chunks.jsonl'),
    join(__dirname, '../../webmind-research/trained_model/chunks.jsonl'),
  ];

  for (const p of paths) {
    if (p && existsSync(p)) {
      const lines = readFileSync(p, 'utf-8').trim().split('\n');
      chunks = lines.map(line => JSON.parse(line));
      console.log(`[saqt] Loaded ${chunks.length.toLocaleString()} chunks from ${p}`);
      ready = true;
      return;
    }
  }

  const dataDir = existsSync(join(home, 'webmind-research/data'))
    ? join(home, 'webmind-research/data')
    : join(__dirname, '../../webmind-research/data');
  if (existsSync(dataDir)) {
    for (const f of readdirSync(dataDir).filter(f => f.endsWith('.jsonl')).sort()) {
      const lines = readFileSync(join(dataDir, f), 'utf-8').trim().split('\n');
      for (const line of lines) {
        try {
          const row = JSON.parse(line);
          const text = row.text || row.question || '';
          if (text.length > 10) {
            chunks.push({
              text: text.substring(0, 500),
              topic: row.topic || row.category || 'general',
              source: row.source || f.replace('.jsonl', ''),
              answer: row.answer || ''
            });
          }
        } catch {}
      }
    }
    console.log(`[saqt] Loaded ${chunks.length.toLocaleString()} chunks from data/`);
    ready = true;
  }
}

function similarity(query, text) {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const qWords = new Set(q.split(/\s+/).filter(w => w.length > 2));
  const tWords = new Set(t.split(/\s+/).filter(w => w.length > 2));
  let wordOverlap = 0;
  for (const w of qWords) if (tWords.has(w)) wordOverlap++;
  const wordScore = wordOverlap / Math.max(qWords.size, 1);
  const trigrams = (s) => {
    const set = new Set();
    for (let i = 0; i <= s.length - 3; i++) set.add(s.substring(i, i + 3));
    return set;
  };
  const qTri = trigrams(q), tTri = trigrams(t);
  let triOverlap = 0;
  for (const ng of qTri) if (tTri.has(ng)) triOverlap++;
  const triScore = triOverlap / Math.max(qTri.size, 1);
  return 0.4 * wordScore + 0.6 * triScore;
}

function search(query, topK = 5) {
  return chunks
    .map((chunk, idx) => ({ chunk, idx, score: similarity(query, chunk.text + ' ' + (chunk.answer || '')) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function saqtQuery(question, maxHops = 5) {
  const t0 = Date.now();
  const facts = [], answers = [], trace = [];
  let currentQuery = question;
  for (let hop = 0; hop < maxHops; hop++) {
    const results = search(currentQuery, 3);
    for (const { chunk, score } of results) {
      if (score > 0.05 && !facts.includes(chunk.text)) {
        facts.push(chunk.text);
        if (chunk.answer && !answers.includes(chunk.answer)) answers.push(chunk.answer);
      }
    }
    trace.push({ hop, topScore: results[0]?.score || 0, facts: facts.length });
    currentQuery = `${question} ${results[0]?.chunk?.text || ''}`.substring(0, 300);
  }
  return { question, facts: facts.slice(0, 10), answers: [...new Set(answers)].slice(0, 5),
    trace, hops: maxHops, totalFacts: facts.length, timeMs: Date.now() - t0 };
}

// ─── Answer Synthesis (no LLM — facts are the answer) ────

function synthesizeAnswer(question, facts, answers) {
  // Direct answer if available
  if (answers.length > 0) {
    return { answer: answers[0], confidence: 'direct', otherAnswers: answers.slice(1) };
  }
  // Best matching fact
  if (facts.length > 0) {
    return { answer: facts[0], confidence: 'retrieved', otherAnswers: [] };
  }
  return { answer: "I don't have enough information in my knowledge base to answer that.", confidence: 'none', otherAnswers: [] };
}

// ─── HTTP Server ──────────────────────────────────────────

const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css',
  '.json': 'application/json', '.png': 'image/png', '.svg': 'image/svg+xml' };

function handleRequest(req, res) {
  const cors = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST',
    'Access-Control-Allow-Headers': 'Content-Type' };

  if (req.method === 'OPTIONS') { res.writeHead(204, cors); res.end(); return; }

  // API routes
  if (req.url === '/api/saqt/query' && req.method === 'POST') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', () => {
      try {
        const { question, hops = 5 } = JSON.parse(body);
        if (!question) { res.writeHead(400, cors); res.end(JSON.stringify({ error: 'Missing question' })); return; }
        const result = saqtQuery(question, Math.min(hops, 10));
        const synth = synthesizeAnswer(question, result.facts, result.answers);
        result.answer = synth.answer;
        result.confidence = synth.confidence;
        result.otherAnswers = synth.otherAnswers;
        res.writeHead(200, { ...cors, 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (e) { res.writeHead(500, cors); res.end(JSON.stringify({ error: e.message })); }
    });
    return;
  }

  if (req.url === '/api/saqt/stats') {
    res.writeHead(200, { ...cors, 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ready, chunks: chunks.length,
      sources: [...new Set(chunks.map(c => c.source))],
      topics: [...new Set(chunks.map(c => c.topic))].length }));
    return;
  }

  // Serve static files
  let filePath = req.url === '/' ? '/chat.html' : req.url;
  const fullPath = join(__dirname, 'public', filePath);
  if (existsSync(fullPath)) {
    const ext = extname(fullPath);
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'text/plain' });
    res.end(readFileSync(fullPath));
    return;
  }

  res.writeHead(404); res.end('Not found');
}

// ─── Start ────────────────────────────────────────────────
loadKnowledge();
const server = createServer(handleRequest);
server.listen(PORT, '0.0.0.0', () => {
  console.log(`[saqt] SAQT server running on http://0.0.0.0:${PORT}`);
  console.log(`[saqt] Chat UI: http://localhost:${PORT}/`);
  console.log(`[saqt] API: http://localhost:${PORT}/api/saqt/query`);
  console.log(`[saqt] Stats: http://localhost:${PORT}/api/saqt/stats`);
});
