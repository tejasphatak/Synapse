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
    setStatus('Engine ready', qaData.length.toLocaleString() + ' pairs', 95);

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

        // Search with best interpretation
        const { bestIdx, bestScore } = await searchKB(searchQuery);

        let answer;
        if (bestScore < 0.35) {
          answer = "I don't have enough confidence to answer that.";
        } else {
          answer = qaData[bestIdx].answer;
          // Tool execution
          const toolMatch = answer.match(/<tool>([\s\S]*?)<\/tool>/);
          if (toolMatch) {
            try {
              const out = [];
              const print = (...a) => out.push(a.join(' '));
              const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
              const fn = new AsyncFunction('print', 'QUERY', toolMatch[1].trim());
              await fn(print, question);
              if (out.length) answer = out.join('\n');
            } catch(e) { /* tool failed, return raw answer */ }
          }
        }

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
