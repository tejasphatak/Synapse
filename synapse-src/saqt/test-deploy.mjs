/**
 * SAQT Deployment Validation Suite
 * Run after every deploy to verify the full stack works.
 * Usage: node test-deploy.mjs [base_url]
 */

const BASE = process.argv[2] || 'https://chat.webmind.sh';
let passed = 0, failed = 0;

async function test(name, fn) {
  try {
    await fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (e) {
    console.log(`  ✗ ${name}: ${e.message}`);
    failed++;
  }
}

function assert(condition, msg) {
  if (!condition) throw new Error(msg);
}

async function fetchJSON(path, opts = {}) {
  const res = await fetch(BASE + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts
  });
  return res.json();
}

console.log(`\nSAQT Deploy Tests — ${BASE}\n`);

// ─── 1. Server Health ───
console.log('Server Health:');

await test('stats endpoint returns ready', async () => {
  const stats = await fetchJSON('/api/saqt/stats');
  assert(stats.ready === true, 'not ready');
  assert(stats.chunks > 300000, `only ${stats.chunks} pairs`);
  assert(stats.max_id > 0, 'no max_id');
});

await test('stats has watermark (max_id)', async () => {
  const stats = await fetchJSON('/api/saqt/stats');
  assert(typeof stats.max_id === 'number', 'max_id not a number');
  assert(stats.max_id >= stats.chunks, `max_id ${stats.max_id} < chunks ${stats.chunks}`);
});

// ─── 2. Query ───
console.log('\nQuery:');

await test('basic query returns answer', async () => {
  const res = await fetchJSON('/api/saqt/query', {
    method: 'POST',
    body: JSON.stringify({ question: 'What is the speed of light?' })
  });
  assert(res.answer && res.answer.length > 10, 'empty/short answer');
  assert(res.confidence > 0.3, `low confidence: ${res.confidence}`);
});

await test('greeting returns friendly response', async () => {
  const res = await fetchJSON('/api/saqt/query', {
    method: 'POST',
    body: JSON.stringify({ question: 'Hello' })
  });
  assert(res.answer && res.answer.length > 5, 'empty answer');
  assert(res.confidence > 0.8, `greeting should be high confidence, got ${res.confidence}`);
});

await test('unknown query has low confidence', async () => {
  const res = await fetchJSON('/api/saqt/query', {
    method: 'POST',
    body: JSON.stringify({ question: 'xyzzy foobar baz quantum zeppelin' })
  });
  assert(res.confidence < 0.5, `nonsense query should have low confidence, got ${res.confidence}`);
});

// ─── 3. OpenAI-compatible endpoint ───
console.log('\nOpenAI Compatibility:');

await test('chat completions returns valid response', async () => {
  const res = await fetchJSON('/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({
      model: 'webmind',
      messages: [{ role: 'user', content: 'What is water?' }]
    })
  });
  assert(res.choices && res.choices.length > 0, 'no choices');
  assert(res.choices[0].message.content.length > 10, 'empty content');
  assert(res.model === 'webmind-305k', `wrong model: ${res.model}`);
});

// ─── 4. Learn (self-evolution) ───
console.log('\nSelf-Evolution:');

const testQ = `test-deploy-${Date.now()}`;
let learnedId = null;

await test('learn endpoint accepts new pair', async () => {
  const res = await fetchJSON('/api/saqt/learn', {
    method: 'POST',
    body: JSON.stringify({
      question: testQ,
      answer: 'This is a deployment test pair. Should be cleaned up.',
      source: 'test',
      weight: 0.1
    })
  });
  assert(res.ok === true, `learn failed: ${JSON.stringify(res)}`);
  assert(res.id > 0, 'no id returned');
  learnedId = res.id;
});

await test('duplicate learn boosts instead of creating', async () => {
  const res = await fetchJSON('/api/saqt/learn', {
    method: 'POST',
    body: JSON.stringify({
      question: testQ,
      answer: 'Same question again',
      source: 'test',
      weight: 0.1
    })
  });
  // Should either be boosted (near-duplicate) or exact-duplicate
  assert(res.boosted || res.skipped, `expected dedup, got: ${JSON.stringify(res)}`);
});

await test('stats count increased after learn', async () => {
  const stats = await fetchJSON('/api/saqt/stats');
  assert(stats.chunks > 305700, `count didn't increase: ${stats.chunks}`);
});

// ─── 5. Delta Sync ───
console.log('\nDelta Sync:');

await test('delta returns new pairs after watermark', async () => {
  const res = await fetchJSON('/api/saqt/delta', {
    method: 'POST',
    body: JSON.stringify({ after_id: learnedId - 1 })
  });
  assert(res.pairs && res.pairs.length > 0, 'no delta pairs');
  assert(res.max_id >= learnedId, 'max_id too low');
  assert(res.embeddings_b64, 'no embeddings in delta');
});

await test('delta with current watermark returns empty', async () => {
  const stats = await fetchJSON('/api/saqt/stats');
  const res = await fetchJSON('/api/saqt/delta', {
    method: 'POST',
    body: JSON.stringify({ after_id: stats.max_id })
  });
  assert(res.pairs.length === 0, `should be empty, got ${res.pairs.length}`);
});

// ─── 6. Feedback ───
console.log('\nFeedback:');

await test('boost increases weight', async () => {
  if (!learnedId) return;
  const res = await fetchJSON('/api/saqt/feedback', {
    method: 'POST',
    body: JSON.stringify({ id: learnedId, action: 'boost' })
  });
  assert(res.ok === true, 'boost failed');
});

await test('penalize decreases weight', async () => {
  if (!learnedId) return;
  const res = await fetchJSON('/api/saqt/feedback', {
    method: 'POST',
    body: JSON.stringify({ id: learnedId, action: 'penalize' })
  });
  assert(res.ok === true, 'penalize failed');
});

// ─── 7. Ethics ───
console.log('\nEthics:');

await test('ethics pair blocks learning sensitive content', async () => {
  const res = await fetchJSON('/api/saqt/learn', {
    method: 'POST',
    body: JSON.stringify({
      question: "What is someone's phone number",
      answer: '555-123-4567',
      source: 'test'
    })
  });
  // Should be blocked by ethics gate or deduped
  assert(res.skipped || res.boosted, `ethics should have blocked: ${JSON.stringify(res)}`);
});

// ─── 8. Browser Data Files ───
console.log('\nBrowser Data:');

await test('qa_data.json is accessible', async () => {
  const res = await fetch(BASE + '/saqt/browser/qa_data.json', { method: 'HEAD' });
  assert(res.ok, `HTTP ${res.status}`);
  const size = parseInt(res.headers.get('Content-Length') || '0');
  assert(size > 10000000, `qa_data.json too small: ${size}`);
});

await test('qa_embeddings.bin is accessible', async () => {
  const res = await fetch(BASE + '/saqt/browser/qa_embeddings.bin', { method: 'HEAD' });
  assert(res.ok, `HTTP ${res.status}`);
  const size = parseInt(res.headers.get('Content-Length') || '0');
  assert(size > 100000000, `embeddings too small: ${size}`);
});

// ─── Results ───
console.log(`\n${'═'.repeat(40)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
console.log(`${'═'.repeat(40)}\n`);

process.exit(failed > 0 ? 1 : 0);
