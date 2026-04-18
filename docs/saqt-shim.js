/**
 * SAQT Browser Engine + Service Worker Backend
 *
 * 1. Registers sw-backend.js as a Service Worker (intercepts all /api/* calls)
 * 2. Loads sentence transformer + Q&A knowledge base
 * 3. Communicates with SW via MessageChannel for query handling
 *
 * Zero dependencies. Pure browser APIs.
 */
(async function() {
  // Loading overlay
  const overlay = document.createElement('div');
  overlay.id = 'saqt-overlay';
  overlay.innerHTML = `
    <div style="position:fixed;inset:0;background:#fff;z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:-apple-system,sans-serif;">
      <div style="width:48px;height:48px;background:#2d8a4e;border-radius:12px;display:flex;align-items:center;justify-content:center;color:white;font-size:22px;font-weight:700;margin-bottom:12px;">W</div>
      <div style="font-size:18px;font-weight:700;margin-bottom:4px;">Webmind <span style="background:#ff9800;color:#fff;font-size:9px;padding:1px 5px;border-radius:3px;">ALPHA</span></div>
      <div id="saqt-status" style="font-size:12px;color:#888;margin-bottom:12px;">Starting...</div>
      <div style="width:240px;height:3px;background:#eee;border-radius:2px;overflow:hidden;"><div id="saqt-progress" style="height:100%;background:#2d8a4e;width:0%;transition:width 0.3s;"></div></div>
      <div id="saqt-detail" style="font-size:10px;color:#aaa;margin-top:8px;"></div>
    </div>`;
  document.documentElement.appendChild(overlay);

  const setStatus = (s, d, p) => {
    const el = document.getElementById('saqt-status'); if (el) el.textContent = s;
    const dl = document.getElementById('saqt-detail'); if (dl) dl.textContent = d || '';
    const pr = document.getElementById('saqt-progress'); if (pr) pr.style.width = p + '%';
  };

  // Pre-set auth token
  if (!localStorage.getItem('token')) localStorage.setItem('token', 'webmind-local-token');

  try {
    // Step 1: Register Service Worker
    setStatus('Registering backend...', 'Service Worker', 5);
    const reg = await navigator.serviceWorker.register('/sw-backend.js', { scope: '/' });

    // Wait for SW to be active
    const sw = reg.active || reg.waiting || reg.installing;
    if (sw.state !== 'activated') {
      await new Promise((resolve) => {
        sw.addEventListener('statechange', () => {
          if (sw.state === 'activated') resolve();
        });
        if (sw.state === 'activated') resolve();
      });
    }
    // Claim this page
    await navigator.serviceWorker.ready;
    console.log('[webmind] Service Worker active');

    // If SW isn't controlling yet (first install), reload to get control
    if (!navigator.serviceWorker.controller) {
      setStatus('Activating backend...', 'First install — reloading', 10);
      setTimeout(() => location.reload(), 500);
      return;
    }

    // Step 2: Set up MessageChannel for SAQT queries
    const channel = new MessageChannel();
    function sendPort() {
      if (navigator.serviceWorker.controller) {
        navigator.serviceWorker.controller.postMessage({ type: 'saqt-port' }, [channel.port2]);
        console.log('[webmind] SAQT MessageChannel established');
      }
    }
    sendPort();
    // Re-send port if SW changes (update, new activation)
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      const newChannel = new MessageChannel();
      navigator.serviceWorker.controller.postMessage({ type: 'saqt-port' }, [newChannel.port2]);
      console.log('[webmind] Re-sent SAQT port to new SW controller');
    });

    // Step 3: Load SAQT engine
    setStatus('Loading AI model...', 'Sentence transformer (80MB)', 15);
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
    if (env.backends?.onnx?.webgpu) env.backends.onnx.webgpu.enabled = false;
    env.allowLocalModels = false;

    const encoder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      progress_callback: (p) => {
        if (p.progress) setStatus('Loading AI model...', Math.round(p.progress) + '%', 15 + p.progress * 0.25);
      }
    });

    // Step 4: Load knowledge base (with IndexedDB cache + version check)
    setStatus('Loading knowledge base...', '', 40);
    const VM_BASE = 'https://chat.webmind.sh/saqt/browser';

    // IndexedDB helpers for caching large data
    function openCache() {
      return new Promise((resolve, reject) => {
        const req = indexedDB.open('webmind-saqt-cache', 1);
        req.onupgradeneeded = () => req.result.createObjectStore('data');
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }
    function idbGet(db, key) {
      return new Promise((resolve) => {
        const tx = db.transaction('data', 'readonly');
        const req = tx.objectStore('data').get(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => resolve(null);
      });
    }
    function idbPut(db, key, val) {
      return new Promise((resolve) => {
        const tx = db.transaction('data', 'readwrite');
        tx.objectStore('data').put(val, key);
        tx.oncomplete = () => resolve();
        tx.onerror = () => resolve();
      });
    }

    let qaData, qaEmbeddings;
    const cacheDB = await openCache().catch(() => null);

    // Strategy: load from cache FIRST (instant), then sync from remote in background.
    // User can start chatting immediately with cached data.
    let loadedFromCache = false;

    if (cacheDB) {
      const cachedData = await idbGet(cacheDB, 'qaData');
      const cachedEmb = await idbGet(cacheDB, 'qaEmbeddings');
      if (cachedData && cachedEmb) {
        setStatus('Loading from cache...', '', 45);
        qaData = cachedData;
        qaEmbeddings = new Float32Array(cachedEmb);
        loadedFromCache = true;
        setStatus('Ready', qaData.length.toLocaleString() + ' pairs', 85);
        console.log('[webmind] Cache hit — ' + qaData.length.toLocaleString() + ' pairs (syncing in background)');
      }
    }

    if (!loadedFromCache) {
      // No cache — must download before user can chat
      setStatus('Downloading knowledge base...', 'First load — this takes a moment', 40);
      const dataResp = await fetch(VM_BASE + '/qa_data.json');
      qaData = await dataResp.json();
      setStatus('Loading embeddings...', qaData.length.toLocaleString() + ' pairs', 55);

      const embResp = await fetch(VM_BASE + '/qa_embeddings.bin');
      const cl = embResp.headers.get('Content-Length');
      if (cl && parseInt(cl) > 1000000) {
        const reader = embResp.body.getReader();
        const total = parseInt(cl);
        let received = 0;
        const chunks = [];
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.length;
          setStatus('Downloading embeddings...', Math.round(received/1024/1024) + 'MB / ' + Math.round(total/1024/1024) + 'MB', 55 + (received/total)*30);
        }
        qaEmbeddings = new Float32Array(await new Blob(chunks).arrayBuffer());
      } else {
        qaEmbeddings = new Float32Array(await embResp.arrayBuffer());
      }

      // Cache for next time + set watermark
      if (cacheDB) {
        setStatus('Caching for next visit...', '', 86);
        await idbPut(cacheDB, 'qaData', qaData);
        await idbPut(cacheDB, 'qaEmbeddings', qaEmbeddings.buffer);
        // Set watermark from server stats
        try {
          const statsResp = await fetch(VM_BASE.replace('/saqt/browser', '') + '/api/saqt/stats');
          const stats = await statsResp.json();
          await idbPut(cacheDB, 'watermark', stats.max_id || qaData.length);
        } catch(e) {
          await idbPut(cacheDB, 'watermark', qaData.length);
        }
        console.log('[webmind] Cached ' + qaData.length.toLocaleString() + ' pairs to IndexedDB');
      }
    }

    const DIM = 384;

    // Background delta sync — watermark-based, like git pull
    // Local stores a watermark (last synced max_id from server).
    // On boot: compare local watermark vs remote max_id.
    // If remote is ahead: fetch only pairs after our watermark.
    // Watermark is the server's pair ID, not array length — handles gaps/deletes.
    const DELTA_API = VM_BASE.replace('/saqt/browser', '') + '/api/saqt/delta';
    const STATS_API = VM_BASE.replace('/saqt/browser', '') + '/api/saqt/stats';
    if (loadedFromCache) {
      (async () => {
        try {
          // Get local watermark (stored from last sync)
          const localWatermark = cacheDB ? (await idbGet(cacheDB, 'watermark') || 0) : 0;

          // Check remote HEAD
          const statsResp = await fetch(STATS_API);
          const stats = await statsResp.json();
          const remoteHead = stats.max_id || 0;

          if (remoteHead <= localWatermark) {
            console.log(`[webmind] Up to date (watermark: ${localWatermark})`);
            return;
          }

          // Fetch delta — all pairs after our watermark
          const behind = remoteHead - localWatermark;
          console.log(`[webmind] Behind by ~${behind} pairs (local: ${localWatermark}, remote: ${remoteHead}). Syncing...`);

          // Paginate if needed (server caps at 5000 per request)
          let afterId = localWatermark;
          let totalAdded = 0;

          while (afterId < remoteHead) {
            const deltaResp = await fetch(DELTA_API, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ after_id: afterId })
            });
            const delta = await deltaResp.json();

            if (!delta.pairs || delta.pairs.length === 0) break;

            // Append new pairs
            for (const p of delta.pairs) {
              qaData.push({ question: p.question, answer: p.answer, source: p.source, weight: p.weight });
            }

            // Append new embeddings
            if (delta.embeddings_b64) {
              const newEmbBytes = Uint8Array.from(atob(delta.embeddings_b64), c => c.charCodeAt(0));
              const newEmb = new Float32Array(newEmbBytes.buffer);
              const merged = new Float32Array(qaEmbeddings.length + newEmb.length);
              merged.set(qaEmbeddings);
              merged.set(newEmb, qaEmbeddings.length);
              qaEmbeddings = merged;
            }

            // Advance watermark to the last pair we received
            afterId = delta.pairs[delta.pairs.length - 1].id;
            totalAdded += delta.pairs.length;

            if (!delta.has_more) break;
          }

          // Persist updated data + watermark
          if (totalAdded > 0 && cacheDB) {
            await idbPut(cacheDB, 'qaData', qaData);
            await idbPut(cacheDB, 'qaEmbeddings', qaEmbeddings.buffer);
            await idbPut(cacheDB, 'watermark', afterId);
            console.log(`[webmind] Synced ${totalAdded} new pairs. Watermark: ${afterId}. Total: ${qaData.length.toLocaleString()}`);
          }
        } catch(e) {
          console.log('[webmind] Sync failed (offline?) — using cached data');
        }
      })();
    }

    // ─── Google CSE search (CX from Programmable Search Engine) ───
    const GOOGLE_CSE_CX = 'c4ba99d848f5d433b';
    let googleCSEReady = false;
    let googleCSEPendingResolve = null;

    // Load Google CSE JS
    window.__gcse = {
      parsetags: 'explicit',
      callback: () => { googleCSEReady = true; }
    };
    const cseScript = document.createElement('script');
    cseScript.async = true;
    cseScript.src = `https://cse.google.com/cse.js?cx=${GOOGLE_CSE_CX}`;
    document.head.appendChild(cseScript);

    // Hidden container for CSE results
    const cseDiv = document.createElement('div');
    cseDiv.id = 'webmind-cse-results';
    cseDiv.style.cssText = 'position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;overflow:hidden;';
    document.body?.appendChild(cseDiv) || document.documentElement.appendChild(cseDiv);

    async function searchGoogle(query) {
      if (!googleCSEReady) {
        // Wait up to 5s for CSE to load
        await new Promise(r => {
          const check = setInterval(() => {
            if (googleCSEReady || typeof google !== 'undefined' && google.search?.cse) {
              googleCSEReady = true;
              clearInterval(check);
              r();
            }
          }, 200);
          setTimeout(() => { clearInterval(check); r(); }, 5000);
        });
      }

      if (!googleCSEReady || typeof google === 'undefined') return [];

      return new Promise((resolve) => {
        const timeout = setTimeout(() => resolve([]), 8000);

        try {
          // Set up callback to capture results
          window.__webmindCSECallback = (results) => {
            clearTimeout(timeout);
            resolve(results);
          };

          // Render CSE element if not already
          if (!cseDiv.querySelector('.gsc-control')) {
            google.search.cse.element.render({
              div: 'webmind-cse-results',
              tag: 'searchresults-only',
              attributes: { enableHistory: false }
            });
          }

          // Execute search
          const element = google.search.cse.element.getElement('webmind-cse-results');
          if (element) {
            // Override the result rendering to capture data
            const origCallback = element.resultSetCallback;
            element.resultSetCallback = function(gname, q, promos, results) {
              const parsed = [];
              if (results) {
                for (const r of results) {
                  parsed.push({
                    source: 'Google',
                    title: r.titleNoFormatting || r.title?.replace(/<[^>]+>/g, '') || '',
                    text: r.contentNoFormatting || r.content?.replace(/<[^>]+>/g, '') || '',
                    url: r.unescapedUrl || r.url || ''
                  });
                }
              }
              window.__webmindCSECallback?.(parsed);
              if (origCallback) origCallback.call(this, gname, q, promos, results);
            };
            element.execute(query);
          } else {
            clearTimeout(timeout);
            resolve([]);
          }
        } catch(e) {
          clearTimeout(timeout);
          resolve([]);
        }
      });
    }

    // ─── Multi-source web search (shared by tool code + low-confidence fallback) ───
    async function searchWebMulti(query) {
      const results = [];

      // Source 0: Google CSE (best quality, try first)
      try {
        const googleResults = await searchGoogle(query);
        for (const r of googleResults.slice(0, 5)) {
          if (r.text || r.title) {
            results.push({ source: 'Google', text: `**${r.title}**\n${r.text}`, url: r.url });
          }
        }
      } catch(e) {}

      // Source 1: Wikipedia direct
      try {
        const q = encodeURIComponent(query);
        const r = await fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${q}`);
        if (r.ok) {
          const d = await r.json();
          if (d.extract) results.push({ source: 'Wikipedia', text: d.extract, url: d.content_urls?.desktop?.page || '' });
        }
      } catch(e) {}

      // Source 2: DuckDuckGo Instant Answers
      try {
        const q = encodeURIComponent(query);
        const r = await fetch(`https://api.duckduckgo.com/?q=${q}&format=json&no_html=1&skip_disambig=1`);
        if (r.ok) {
          const d = await r.json();
          const text = d.AbstractText || d.Answer || '';
          if (text) results.push({ source: d.AbstractSource || 'DuckDuckGo', text, url: d.AbstractURL || '' });
          if (d.RelatedTopics?.length) {
            const related = d.RelatedTopics.slice(0, 3).map(t => t.Text).filter(Boolean).join('\n');
            if (related && !text) results.push({ source: 'DuckDuckGo', text: related, url: '' });
          }
        }
      } catch(e) {}

      // Source 3: Wikipedia search fallback
      if (!results.some(r => r.source === 'Wikipedia')) {
        try {
          const q = encodeURIComponent(query);
          const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=${q}&format=json&origin=*&srlimit=3`);
          if (r.ok) {
            const d = await r.json();
            const hits = d.query?.search || [];
            if (hits.length) {
              const snippet = hits.map(h => h.snippet.replace(/<[^>]+>/g, '')).join(' ');
              results.push({ source: 'Wikipedia Search', text: snippet, url: `https://en.wikipedia.org/wiki/${encodeURIComponent(hits[0].title)}` });
            }
          }
        } catch(e) {}
      }

      // Source 4: Hacker News Algolia (tech/science/current)
      try {
        const q = encodeURIComponent(query);
        const r = await fetch(`https://hn.algolia.com/api/v1/search?query=${q}&tags=story&hitsPerPage=5`);
        if (r.ok) {
          const d = await r.json();
          const hits = d.hits || [];
          if (hits.length > 0) {
            const text = hits.slice(0, 3).map(h =>
              `**${h.title}**${h.points ? ` (${h.points} pts)` : ''}${h.url ? ` — [link](${h.url})` : ''}`
            ).join('\n');
            results.push({ source: 'Hacker News', text, url: 'https://news.ycombinator.com' });
          }
        }
      } catch(e) {}

      // Deduplicate
      const seen = new Set();
      return results.filter(r => {
        if (seen.has(r.text.substring(0, 50))) return false;
        seen.add(r.text.substring(0, 50));
        return true;
      });
    }

    // String version for tool code (returns text, not array)
    async function searchWebText(query) {
      const results = await searchWebMulti(query);
      if (results.length === 0) return '';
      return results.map(r => r.text).join('\n\n');
    }

    setStatus('Engine ready', qaData.length.toLocaleString() + ' pairs', 95);

    // ─── Python → JS transpiler for tool code ───
    function pythonToJS(code) {
      // Already JS? (has const/let/var/await/fetch/localStorage)
      if (/\b(const |let |var |await |fetch\(|localStorage|document\.)/.test(code)) return code;

      let js = code;

      // Remove Python imports (handle their functionality inline)
      js = js.replace(/^import datetime;?\s*/gm, '');
      js = js.replace(/^import math;?\s*/gm, '');
      js = js.replace(/^import re;?\s*/gm, '');
      js = js.replace(/^from \w+ import \*;?\s*/gm, '');
      js = js.replace(/^import \w+;?\s*/gm, '');

      // datetime replacements
      js = js.replace(/datetime\.date\.today\(\)\.strftime\(["']%A["']\)/g,
        'new Date().toLocaleDateString("en-US",{weekday:"long"})');
      js = js.replace(/datetime\.datetime\.now\(\)\.strftime\(["']%H:%M:%S["']\)/g,
        'new Date().toLocaleTimeString("en-US",{hour12:false})');
      js = js.replace(/datetime\.datetime\.now\(\)\.strftime\(["']([^"']+)["']\)/g,
        'new Date().toLocaleString()');
      js = js.replace(/datetime\.date\.today\(\)/g,
        'new Date().toISOString().split("T")[0]');
      js = js.replace(/datetime\.datetime\.now\(\)/g, 'new Date().toISOString()');

      // math replacements
      js = js.replace(/math\.sqrt\(/g, 'Math.sqrt(');
      js = js.replace(/math\.pi/g, 'Math.PI');
      js = js.replace(/math\.e\b/g, 'Math.E');
      js = js.replace(/math\.floor\(/g, 'Math.floor(');
      js = js.replace(/math\.ceil\(/g, 'Math.ceil(');
      js = js.replace(/math\.log\(/g, 'Math.log(');
      js = js.replace(/math\.sin\(/g, 'Math.sin(');
      js = js.replace(/math\.cos\(/g, 'Math.cos(');

      // Python round() → JS toFixed or Math.round
      js = js.replace(/round\(([^,]+),\s*(\d+)\)/g, '(($1).toFixed($2))');

      // Python ** → JS ** (both support it)
      // Python // → JS Math.floor(a/b)
      js = js.replace(/(\w+)\s*\/\/\s*(\w+)/g, 'Math.floor($1/$2)');

      // sorted([...]) → [...].sort((a,b)=>a-b)
      js = js.replace(/sorted\((\[.*?\])\)/g, '$1.sort((a,b)=>a-b)');

      // Python string reverse [::-1]
      js = js.replace(/["'](\w+)["']\[::-1\]/g, '"$1".split("").reverse().join("")');

      // re.findall(pattern, string) → string.match(pattern) || []
      js = js.replace(/re\.findall\(r?["']([^"']+)["'],\s*(\w+)\)/g,
        '($2.match(/$1/g) || [])');

      // re.sub(pattern, replacement, string) → string.replace(pattern, replacement)
      js = js.replace(/re\.sub\(r?["']([^"']+)["'],\s*["']([^"']*)["'],\s*(\w+)\)/g,
        '$3.replace(/$1/gi, "$2")');

      // Python f-string and {QUERY} → QUERY variable
      js = js.replace(/['"]?\{QUERY\}['"]?/g, 'QUERY');

      // len(x) → x.length
      js = js.replace(/len\((\w+)\)/g, '$1.length');

      // float(x) → parseFloat(x)
      js = js.replace(/float\(([^)]+)\)/g, 'parseFloat($1)');

      // int(x) → parseInt(x)
      js = js.replace(/\bint\(([^)]+)\)/g, 'parseInt($1)');

      // Python True/False/None → JS
      js = js.replace(/\bTrue\b/g, 'true');
      js = js.replace(/\bFalse\b/g, 'false');
      js = js.replace(/\bNone\b/g, 'null');

      // Python elif → else if
      js = js.replace(/\belif\b/g, 'else if');

      // Handle multiline: Python uses indentation, JS uses braces
      // For simple single-line expressions, this works as-is

      return js;
    }

    // No hardcoded formatting — responses pass through as-is from KB.
    // Web search results are formatted as markdown since we construct them.
    // To make KB answers markdown: update the Q&A pairs in the knowledge base.

    // ─── Two-pass query understanding ───

    // Pass 1: Extract topic from format requests
    // "Can you create a markdown for current affairs?" → "current affairs"
    // No hardcoded extractTopic() — the KB learns intent from its own data.
    // Format/action requests ("create a list about X") are handled by teaching pairs.
    async function searchKB(queryText) {
      const output = await encoder(queryText, { pooling: 'mean', normalize: true });
      const qEmb = Array.from(output.data);

      let bestIdx = 0, bestScore = -1;
      for (let i = 0; i < qaData.length; i++) {
        let dot = 0;
        const off = i * DIM;
        for (let d = 0; d < DIM; d++) dot += qEmb[d] * qaEmbeddings[off + d];
        if (dot > bestScore) { bestScore = dot; bestIdx = i; }
      }
      return { bestIdx, bestScore };
    }

    // ─── Status emitter for thinking/hop display ───
    function emitStatus(chatId, messageId, action, description, done = false, extra = {}) {
      if (!chatId || !messageId) return;
      channel.port1.postMessage({
        status: {
          chatId, messageId,
          data: { action, description, done, ...extra }
        }
      });
    }

    // Step 5: Handle queries from Service Worker
    // Derive thresholds from data distribution (computed once at load time)
    // p25 of answer lengths = what counts as "short" for this KB
    const sortedLens = qaData.map(q => q.answer.length).sort((a, b) => a - b);
    const dataP25 = sortedLens[Math.floor(sortedLens.length * 0.25)];
    const dataMedian = sortedLens[Math.floor(sortedLens.length * 0.5)];
    console.log(`[webmind] Answer length distribution: p25=${dataP25}, median=${dataMedian}`);

    channel.port1.onmessage = async (event) => {
      const { id, question, chatId, messageId } = event.data;
      try {
        // ─── Pure search engine: no hardcoded intent detection ───
        // The KB teaches intent via weighted pairs (tool calls, system prompts).
        // The engine just searches, composes, and lets the data decide.

        const thinking = [];
        const t0 = Date.now();
        const think = (line) => thinking.push(line);

        think(`Query: "${question}"`);

        // Step 1: Search KB — fire in parallel with web (web results used only if KB is weak)
        emitStatus(chatId, messageId, 'queries_generated', `Searching "${question.substring(0, 50)}"`, false, { queries: [question.substring(0, 60)] });

        const kbPromise = searchKB(question);
        const webPromise = searchWebMulti(question); // always fire — data decides if we use it

        const [{ bestIdx: firstIdx, bestScore: firstScore }, webResults] = await Promise.all([kbPromise, webPromise]);

        let answer = '';
        let bestOverallScore = firstScore;
        const visited = new Set();
        const facts = [];
        let usedWeb = false;

        think(`KB: "${qaData[firstIdx].question.substring(0, 80)}" → ${(firstScore * 100).toFixed(1)}%`);

        // Use top match — the score IS the confidence (no hardcoded threshold needed,
        // but we need a floor below which "no answer" is better than noise.
        // Use 1/sqrt(N) as the random-chance baseline for this embedding space)
        const noiseFloor = 1 / Math.sqrt(qaData.length);

        if (firstScore > noiseFloor) {
          answer = qaData[firstIdx].answer;
          visited.add(firstIdx);
          facts.push(qaData[firstIdx].question);
          think(`✓ Above noise floor (${(noiseFloor * 100).toFixed(2)}%). Using as primary.`);
          emitStatus(chatId, messageId, 'sources_retrieved', `Matched: "${qaData[firstIdx].question.substring(0, 60)}" (${(firstScore * 100).toFixed(0)}%)`, true, { count: 1 });

          // If KB answer contains <tool> — it's a learned behavior (web search, compute, etc.)
          // The KB itself taught the engine when to use tools. No hardcoded check needed.

        } else {
          think(`✗ Below noise floor. KB has nothing relevant.`);
          emitStatus(chatId, messageId, 'sources_retrieved', `No confident match`, true, { count: 0 });
        }

        // Step 2: Use web results if KB didn't produce a strong answer
        // "Strong" = top match is well above noise. Let the score gap decide, not a constant.
        const kbWeak = firstScore < noiseFloor * 10; // ~1.8% for 305K pairs — KB has weak/no coverage
        if (kbWeak && webResults.length > 0) {
          usedWeb = true;
          think(`KB weak (${(firstScore * 100).toFixed(1)}%). Using ${webResults.length} web results:`);
          webResults.slice(0, 3).forEach((r, i) => think(`  ${i + 1}. ${r.text.substring(0, 80)}`));
          const webText = webResults.map(r => r.text).join(' ').substring(0, 500);
          facts.push(...webResults.map(r => r.text.substring(0, 100)));
          emitStatus(chatId, messageId, 'web_search', `Searched ${webResults.length} sites`, true, { urls: webResults.filter(r => r.url).map(r => r.url) });

          if (!answer) {
            // No KB answer at all — web is the answer
            answer = webResults.map(r => {
              let md = r.text;
              if (r.url) md += `\n\n[Source](${r.url})`;
              return md;
            }).join('\n\n---\n\n').substring(0, 2000);
            think(`Using web as primary answer.`);
          }

          // Re-search KB with web context — web might help find a better KB match
          const context = `${question} ${webText}`;
          emitStatus(chatId, messageId, 'queries_generated', 'Re-searching with context', false, { queries: [question.substring(0, 40) + ' + web'] });
          const { bestIdx: webIdx, bestScore: webScore } = await searchKB(context);
          think(`Re-search: "${qaData[webIdx].question.substring(0, 60)}" → ${(webScore * 100).toFixed(1)}%`);
          if (webScore > bestOverallScore && webScore > noiseFloor && !visited.has(webIdx)) {
            answer = qaData[webIdx].answer;
            bestOverallScore = webScore;
            visited.add(webIdx);
            facts.push(qaData[webIdx].question);
            think(`✓ Better match via web-augmented search.`);
            emitStatus(chatId, messageId, 'sources_retrieved', `Found: "${qaData[webIdx].question.substring(0, 60)}" (${(webScore * 100).toFixed(0)}%)`, true, { count: visited.size });
          }
        }

        // Step 3: Multi-hop — keep searching while finding new relevant info
        // No fixed hop count. Stop when the search stops finding new things.
        if (answer) {
          let hopsWithoutNew = 0;
          for (let hop = 1; hopsWithoutNew < 2; hop++) { // stop after 2 consecutive dry hops
            const ctx = `${question} ${answer.substring(0, 200)}`;
            const { bestIdx, bestScore } = await searchKB(ctx);

            if (bestScore > noiseFloor && !visited.has(bestIdx)) {
              visited.add(bestIdx);
              hopsWithoutNew = 0;
              if (!facts.includes(qaData[bestIdx].question)) facts.push(qaData[bestIdx].question);
              think(`Hop ${hop}: "${qaData[bestIdx].question.substring(0, 60)}" → ${(bestScore * 100).toFixed(1)}%`);
              emitStatus(chatId, messageId, 'sources_retrieved', `Hop ${hop}: "${qaData[bestIdx].question.substring(0, 50)}" (${(bestScore * 100).toFixed(0)}%)`, true, { count: visited.size });

              if (bestScore > bestOverallScore) {
                answer = qaData[bestIdx].answer;
                bestOverallScore = bestScore;
                think(`✓ Better answer found (${(bestOverallScore * 100).toFixed(0)}%).`);
              }
            } else {
              hopsWithoutNew++;
            }
          }
        }

        // Step 4: Compose — if answer is shorter than the data's own p25,
        // it's unusually terse for this KB. Enrich with related entries.
        if (answer && answer.length < dataP25 && visited.size > 0) {
          think(`Answer (${answer.length} chars) below KB p25 (${dataP25}). Enriching...`);

          // Search with answer as context to find elaborations
          const searches = await Promise.all([
            searchKB(`${question} ${answer}`),
            searchKB(answer)
          ]);

          const composed = [answer];
          for (const { bestIdx: idx, bestScore: score } of searches) {
            if (score > noiseFloor && !visited.has(idx) && qaData[idx].answer !== answer) {
              const s = qaData[idx].answer;
              if (s.length > 30 && !composed.some(c => c.includes(s.substring(0, 30)))) {
                composed.push(s);
                visited.add(idx);
                think(`  +related: "${qaData[idx].question.substring(0, 50)}" (${(score * 100).toFixed(0)}%)`);
              }
            }
          }
          if (composed.length > 1) {
            answer = composed.join('\n\n');
            think(`  Enriched: ${answer.length} chars from ${composed.length} sources.`);
          }
        }

        if (!answer) {
          think(`No relevant matches found.`);
          answer = "I don't have enough information to answer that question.";
        }

        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        think(`\nSources: ${visited.size} KB, ${usedWeb ? webResults.length + ' web' : 'no web'}. Time: ${elapsed}s.`);

        // Step 5: Learn — send web-sourced answers to server
        // No client-side filtering. The server handles dedup (semantic similarity)
        // and weight management. Everything starts at low weight and earns its way up
        // through retrieval. The data decides what's valuable, not hardcoded rules.
        const LEARN_API = VM_BASE.replace('/saqt/browser', '') + '/api/saqt/learn';
        if (usedWeb && answer) {
          const cleanAnswer = answer.replace(/\n\n\[Source\]\([^)]+\)/g, '').replace(/\n\n---\n\n/g, '\n\n').substring(0, 800);
          think(`Learning: sending to KB. Server will dedup + set weight.`);
          fetch(LEARN_API, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, answer: cleanAnswer, source: 'web-learned' })
          }).catch(() => {});
        }

        const thinkingBlock = `<details type="reasoning" done="true" duration="${elapsed}">\n${thinking.join('\n')}\n</details>\n\n`;
        answer = thinkingBlock + answer;

        // Tool execution on final answer
        const toolMatch = answer.match(/<tool>([\s\S]*?)<\/tool>/);
        if (toolMatch) {
          let toolCode = pythonToJS(toolMatch[1].trim());
          try {
            const out = [];
            const print = (...a) => out.push(a.join(' '));
            const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
            const fn = new AsyncFunction('print', 'QUERY', 'searchWeb', toolCode);
            await fn(print, question, searchWebText);
            if (out.length) answer = out.join('\n');
          } catch(e) { /* tool failed, return raw answer */ }
        }

        channel.port1.postMessage({ id, answer });
      } catch(e) {
        channel.port1.postMessage({ id, answer: 'Error: ' + e.message });
      }
    };

    setStatus('Ready!', qaData.length.toLocaleString() + ' pairs loaded', 100);

    // Remove overlay + clean up UI (remove redundant branding)
    setTimeout(() => {
      const el = document.getElementById('saqt-overlay');
      if (el) { el.style.opacity = '0'; el.style.transition = 'opacity 0.5s'; setTimeout(() => el.remove(), 500); }

      // Clean up redundant UI elements
      setTimeout(() => {
        // Remove "Set as default" buttons
        document.querySelectorAll('button').forEach(b => {
          if (b.textContent.trim() === 'Set as default') b.style.display = 'none';
        });
        // Remove footer version "Webmind · v0.8.12"
        document.querySelectorAll('div').forEach(d => {
          const t = d.textContent.trim();
          if (t.match(/^Webmind\s*[·‧]\s*v\d/) && d.children.length === 0) d.style.display = 'none';
        });
      }, 500);
    }, 1000);

  } catch(e) {
    setStatus('Error: ' + e.message, '', 0);
    console.error('[webmind]', e);
  }
})();
