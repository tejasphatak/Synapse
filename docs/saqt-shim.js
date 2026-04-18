/**
 * SAQT Browser Engine + Service Worker Backend
 *
 * 1. Registers sw-backend.js as a Service Worker (intercepts all /api/* calls)
 * 2. Loads sentence transformer + 305K Q&A pairs
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
    navigator.serviceWorker.controller.postMessage({ type: 'saqt-port' }, [channel.port2]);
    console.log('[webmind] SAQT MessageChannel established');

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

    // Step 4: Load knowledge base
    setStatus('Loading knowledge base...', '', 40);
    const VM_BASE = 'https://chat.webmind.sh/saqt/browser';
    const dataResp = await fetch(VM_BASE + '/qa_data.json');
    const qaData = await dataResp.json();
    setStatus('Loading embeddings...', qaData.length.toLocaleString() + ' pairs', 55);

    const embResp = await fetch(VM_BASE + '/qa_embeddings.bin');
    const cl = embResp.headers.get('Content-Length');
    let qaEmbeddings;
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

    const DIM = 384;

    // ─── Multi-source web search (shared by tool code + low-confidence fallback) ───
    async function searchWebMulti(query) {
      const results = [];

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
    function extractTopic(query) {
      const patterns = [
        // "create/make/write a markdown/list/table about/for/of TOPIC"
        /(?:can you |please |could you |i need |i want )?(?:create|make|write|generate|draft|prepare|give me|provide|build|compose|put together|show me)(?:\s+me)?\s+(?:a|an|the|some)?\s*(?:markdown|md|list|table|document|doc|summary|report|essay|article|outline|presentation|slides?|spreadsheet|csv|json|html|text|paragraph|bullets?|overview|brief|writeup|write-up|notes?|chart|graph|diagram)\s*(?:about|for|of|on|regarding|related to|covering|explaining|describing|summarizing|detailing)\s+(.+)/i,
        // "TOPIC in markdown/list format"
        /(.+?)\s+(?:in|using|as|formatted as|formatted in)\s+(?:a\s+)?(?:markdown|md|list|table|document|summary|report|essay|article|outline|bullets?|html|text|paragraph)\s*(?:format)?$/i,
        // "summarize/explain/describe TOPIC"
        /(?:can you |please |could you )?(?:summarize|explain|describe|elaborate on|tell me about|give me info on|give me information about|what do you know about)\s+(.+)/i,
      ];

      for (const pattern of patterns) {
        const match = query.match(pattern);
        if (match && match[1]) {
          const topic = match[1].replace(/[?.!,]+$/, '').trim();
          if (topic.length > 2) return topic;
        }
      }
      return null;
    }

    // Pass 2: Search with topic-focused embedding
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

    // Step 5: Handle queries from Service Worker
    channel.port1.onmessage = async (event) => {
      const { id, question } = event.data;
      try {
        // Two-pass: extract topic if this is a format/action request
        const topic = extractTopic(question);
        let searchQuery = question;
        let usedTopic = false;

        if (topic) {
          // Search both interpretations to detect ambiguity
          const topicResult = await searchKB(topic);
          const fullResult = await searchKB(question);

          const AMBIGUITY_THRESHOLD = 0.5;
          const topicStrong = topicResult.bestScore >= AMBIGUITY_THRESHOLD;
          const fullStrong = fullResult.bestScore >= AMBIGUITY_THRESHOLD;

          // Check if matches are about DIFFERENT topics by comparing their embeddings
          let matchSimilarity = 0;
          if (topicResult.bestIdx !== fullResult.bestIdx) {
            const offA = topicResult.bestIdx * DIM, offB = fullResult.bestIdx * DIM;
            for (let d = 0; d < DIM; d++) matchSimilarity += qaEmbeddings[offA + d] * qaEmbeddings[offB + d];
          } else {
            matchSimilarity = 1.0; // same pair = not ambiguous
          }

          // Both strong + matches about different topics (low similarity) → genuinely ambiguous
          if (topicStrong && fullStrong && matchSimilarity < 0.5) {
            const topicQ = qaData[topicResult.bestIdx].question;
            const fullQ = qaData[fullResult.bestIdx].question;
            const clarification = `I found strong matches for different interpretations of your question:\n\n` +
              `1. **${topic}** — "${topicQ.substring(0, 80)}"\n` +
              `2. **${question}** — "${fullQ.substring(0, 80)}"\n\n` +
              `Could you clarify what you're looking for?`;
            channel.port1.postMessage({ id, answer: clarification });
            return;
          }

          // Only one interpretation is strong — use it
          if (topicStrong) {
            searchQuery = topic;
            usedTopic = true;
          }
          // else: topic weak, use full query as-is
        }

        // ─── Multi-hop search with web as a hop ───
        const MAX_HOPS = 5;
        const CONFIDENCE_THRESHOLD = 0.35;
        const WEB_HOP_THRESHOLD = 0.5; // try web if below this
        let answer = '';
        let facts = [];
        let visited = new Set();
        let context = searchQuery;
        let bestOverallScore = 0;
        let usedWeb = false;

        for (let hop = 0; hop < MAX_HOPS; hop++) {
          const { bestIdx, bestScore } = await searchKB(context);

          // Hop 0: initial search
          if (hop === 0) {
            bestOverallScore = bestScore;

            if (bestScore >= CONFIDENCE_THRESHOLD) {
              answer = qaData[bestIdx].answer;
              visited.add(bestIdx);
              facts.push(qaData[bestIdx].question);
            }

            // Low confidence or moderate confidence — try web search as next hop
            if (bestScore < WEB_HOP_THRESHOLD && !usedWeb) {
              usedWeb = true;
              const webResults = await searchWebMulti(searchQuery);
              if (webResults.length > 0) {
                const webText = webResults.map(r => r.text).join(' ').substring(0, 500);
                facts.push(...webResults.map(r => r.text.substring(0, 100)));

                if (bestScore < CONFIDENCE_THRESHOLD) {
                  // KB had nothing — use web as primary answer, formatted as markdown
                  answer = webResults.map(r => {
                    let md = r.text;
                    if (r.url) md += `\n\n[Source](${r.url})`;
                    return md;
                  }).join('\n\n---\n\n').substring(0, 2000);
                }

                // Re-search KB with web context for better matches
                context = `${searchQuery} ${webText}`;
                continue; // next hop with enriched context
              }
            }

            if (bestScore < CONFIDENCE_THRESHOLD && !answer) {
              answer = "I don't have enough confidence to answer that.";
              break;
            }
          } else {
            // Subsequent hops: look for new relevant info
            if (bestScore >= CONFIDENCE_THRESHOLD && !visited.has(bestIdx)) {
              visited.add(bestIdx);
              const newAnswer = qaData[bestIdx].answer;
              if (!facts.includes(qaData[bestIdx].question)) {
                facts.push(qaData[bestIdx].question);
              }
              // If this is a better match than what we had, use it
              if (bestScore > bestOverallScore) {
                answer = newAnswer;
                bestOverallScore = bestScore;
              }
            }

            // Try web on later hops if still below threshold
            if (bestOverallScore < WEB_HOP_THRESHOLD && !usedWeb) {
              usedWeb = true;
              const webResults = await searchWebMulti(context);
              if (webResults.length > 0) {
                const webText = webResults.map(r => r.text).join(' ').substring(0, 500);
                context = `${searchQuery} ${webText}`;
                continue;
              }
            }

            // Enrich context for next hop
            context = `${searchQuery} ${answer.substring(0, 200)}`;
          }

          // Stop if we have high confidence
          if (bestOverallScore > 0.8) break;
        }

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

        // Format as markdown if not already
        channel.port1.postMessage({ id, answer });
      } catch(e) {
        channel.port1.postMessage({ id, answer: 'Error: ' + e.message });
      }
    };

    setStatus('Ready!', qaData.length.toLocaleString() + ' pairs loaded', 100);

    // Remove overlay
    setTimeout(() => {
      const el = document.getElementById('saqt-overlay');
      if (el) { el.style.opacity = '0'; el.style.transition = 'opacity 0.5s'; setTimeout(() => el.remove(), 500); }
    }, 1000);

  } catch(e) {
    setStatus('Error: ' + e.message, '', 0);
    console.error('[webmind]', e);
  }
})();
