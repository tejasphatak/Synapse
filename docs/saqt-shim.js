/**
 * SAQT Browser Backend — almostnode + Express + SAQT engine
 *
 * Boots an Express server inside the browser via almostnode.
 * Service Worker intercepts fetch calls → routes to Express.
 * SAQT engine provides the intelligence (sentence transformer + 305K Q&A pairs).
 */
(async function() {
  // Loading overlay
  const overlay = document.createElement('div');
  overlay.id = 'saqt-overlay';
  overlay.innerHTML = `
    <div style="position:fixed;inset:0;background:#fff;z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:-apple-system,sans-serif;">
      <div style="width:48px;height:48px;background:#2d8a4e;border-radius:12px;display:flex;align-items:center;justify-content:center;color:white;font-size:22px;font-weight:700;margin-bottom:12px;">W</div>
      <div style="font-size:18px;font-weight:700;margin-bottom:4px;">Webmind <span style="background:#ff9800;color:#fff;font-size:9px;padding:1px 5px;border-radius:3px;">ALPHA</span></div>
      <div id="saqt-status" style="font-size:12px;color:#888;margin-bottom:12px;">Starting backend...</div>
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
    // Step 1: Import almostnode
    setStatus('Loading runtime...', 'almostnode', 5);
    const { createContainer, getServerBridge } = await import('/almostnode.bundle.js');

    // Step 2: Create container
    setStatus('Booting backend...', 'Express server', 10);
    const container = createContainer({
      onServerReady: (port, url) => {
        console.log('[webmind] Server ready on virtual port', port, url);
      }
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

    // Step 4: Load knowledge base
    setStatus('Loading knowledge base...', '', 40);
    const VM_BASE = 'https://chat.webmind.sh/saqt/browser';
    const dataResp = await fetch(VM_BASE + '/qa_data.json');
    const qaData = await dataResp.json();
    setStatus('Loading embeddings...', qaData.length.toLocaleString() + ' pairs', 55);

    const embResp = await fetch(VM_BASE + '/qa_embeddings.bin');
    const contentLength = embResp.headers.get('Content-Length');
    let qaEmbeddings;
    if (contentLength && parseInt(contentLength) > 1000000) {
      const reader = embResp.body.getReader();
      const total = parseInt(contentLength);
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

    // SAQT query function
    function saqtQuery(question) {
      try {
        // Synchronous search (encoder is async but we pre-encode)
        // For the Express handler, we need sync. Use pre-computed embeddings.
        const n = qaData.length;
        // We need async encode — but Express handler is sync in almostnode
        // So we'll expose an async version via globalThis
        return 'Searching...'; // placeholder — real query is async
      } catch(e) {
        return 'Error: ' + e.message;
      }
    }

    // Expose async query function globally for Express to use
    globalThis._saqtQueryAsync = async function(question) {
      const output = await encoder(question, { pooling: 'mean', normalize: true });
      const qEmb = Array.from(output.data);
      // Cosine similarity search
      let bestIdx = 0, bestScore = -1;
      for (let i = 0; i < qaData.length; i++) {
        let dot = 0;
        const off = i * DIM;
        for (let d = 0; d < DIM; d++) dot += qEmb[d] * qaEmbeddings[off + d];
        if (dot > bestScore) { bestScore = dot; bestIdx = i; }
      }
      if (bestScore < 0.35) return "I don't have enough confidence to answer that.";
      let answer = qaData[bestIdx].answer;
      // Tool execution
      const toolMatch = answer.match(/<tool>([\s\S]*?)<\/tool>/);
      if (toolMatch) {
        try {
          const out = [];
          const print = (...a) => out.push(a.join(' '));
          const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
          const fn = new AsyncFunction('print', 'QUERY', 'fetch', toolMatch[1].trim());
          await fn(print, question, fetch);
          if (out.length) answer = out.join('\n');
        } catch(e) { /* tool failed */ }
      }
      return answer;
    };

    // Synchronous wrapper that returns a promise result (for SSE streaming)
    globalThis._saqtQuery = function(question) {
      // This is called from Express which is sync in almostnode
      // We return a placeholder and the actual response is handled via async
      return "Processing...";
    };

    setStatus('Starting server...', 'Express + SAQT', 90);

    // Step 5: Install Express and start server
    await container.npm.install('express');

    // Step 6: Mount and run server code
    const { SERVER_CODE } = await import('/saqt-backend.js');
    container.execute(SERVER_CODE, 'server.js');

    // Step 7: Set up Service Worker to intercept fetch
    const bridge = getServerBridge();
    await bridge.initServiceWorker({ swUrl: '/__sw__.js' });

    // Step 8: Override fetch to route API calls through almostnode
    const originalFetch = window.fetch;
    const bridgeFetch = bridge.createFetchHandler();
    window.fetch = async function(url, opts) {
      const urlStr = typeof url === 'string' ? url : url?.url || '';
      // Route API calls through almostnode's virtual server
      if (urlStr.includes('/api/') || urlStr.includes('/openai/') || urlStr.includes('/ollama/')) {
        // Rewrite URL to almostnode's virtual server format
        const virtualUrl = '/__virtual__/3000' + (urlStr.startsWith('/') ? urlStr : new URL(urlStr).pathname);
        try {
          const resp = await bridgeFetch(new Request(virtualUrl, opts));
          return resp;
        } catch(e) {
          console.log('[webmind] bridge fetch failed, falling back:', e.message);
        }
      }
      return originalFetch.apply(this, arguments);
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
