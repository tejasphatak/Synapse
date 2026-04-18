/**
 * SAQT Browser Shim — intercepts Open WebUI API calls and serves locally
 * Loads sentence transformer + 305K Q&A pairs in-browser
 * No server needed. Everything runs in the browser tab.
 */
(function() {
  let encoder = null, qaData = [], qaEmbeddings = null, saqtReady = false;
  const DIM = 384;
  const VM_BASE = 'https://chat.webmind.sh/saqt/browser';

  // Loading UI
  const overlay = document.createElement('div');
  overlay.id = 'saqt-overlay';
  overlay.innerHTML = `
    <div style="position:fixed;inset:0;background:#fff;z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:-apple-system,sans-serif;">
      <div style="width:48px;height:48px;background:#2d8a4e;border-radius:12px;display:flex;align-items:center;justify-content:center;color:white;font-size:22px;font-weight:700;margin-bottom:12px;">W</div>
      <div style="font-size:18px;font-weight:700;margin-bottom:4px;">Webmind <span style="background:#ff9800;color:#fff;font-size:9px;padding:1px 5px;border-radius:3px;">ALPHA</span></div>
      <div id="saqt-status" style="font-size:12px;color:#888;margin-bottom:12px;">Initializing...</div>
      <div style="width:240px;height:3px;background:#eee;border-radius:2px;overflow:hidden;"><div id="saqt-progress" style="height:100%;background:#2d8a4e;width:0%;transition:width 0.3s;"></div></div>
      <div id="saqt-detail" style="font-size:10px;color:#aaa;margin-top:8px;"></div>
    </div>`;
  document.documentElement.appendChild(overlay);

  const setStatus = (s, d, p) => {
    const el = document.getElementById('saqt-status'); if (el) el.textContent = s;
    const dl = document.getElementById('saqt-detail'); if (dl) dl.textContent = d || '';
    const pr = document.getElementById('saqt-progress'); if (pr) pr.style.width = p + '%';
  };

  // Encode text to embedding
  async function encode(text) {
    const output = await encoder(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data);
  }

  // Cosine search
  function search(queryEmb, topK) {
    const n = qaData.length, scores = [];
    for (let i = 0; i < n; i++) {
      let dot = 0; const off = i * DIM;
      for (let d = 0; d < DIM; d++) dot += queryEmb[d] * qaEmbeddings[off + d];
      scores.push({ idx: i, score: dot });
    }
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, topK);
  }

  // Query
  async function query(question) {
    const t0 = performance.now();
    const qEmb = await encode(question);
    const results = search(qEmb, 3);
    const best = results[0];
    if (!best || best.score < 0.35)
      return { answer: "I don't have enough confidence to answer that.", confidence: 0, timeMs: Math.round(performance.now() - t0) };
    let answer = qaData[best.idx].answer;
    // Tool execution
    const toolMatch = answer.match(/<tool>([\s\S]*?)<\/tool>/);
    if (toolMatch) {
      try {
        const output = [];
        const print = (...a) => output.push(a.join(' '));
        const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
        const fn = new AsyncFunction('print', 'QUERY', 'fetch', toolMatch[1].trim());
        await fn(print, question, window._originalFetch || fetch);
        if (output.length) answer = output.join('\n');
      } catch(e) { /* tool failed */ }
    }
    return { answer, confidence: best.score, timeMs: Math.round(performance.now() - t0) };
  }

  // Helper: create JSON response
  function jsonResponse(data, status = 200) {
    return new Response(JSON.stringify(data), {
      status,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  // Helper: create streaming chat response
  function streamResponse(text) {
    const chunk = JSON.stringify({
      id: 'wmind-' + Date.now(),
      object: 'chat.completion.chunk',
      created: Math.floor(Date.now() / 1000),
      model: 'webmind-305k',
      choices: [{ index: 0, delta: { role: 'assistant', content: text }, finish_reason: null }]
    });
    const done = JSON.stringify({
      id: 'wmind-' + Date.now(),
      object: 'chat.completion.chunk',
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
    });
    const body = `data: ${chunk}\n\ndata: ${done}\n\ndata: [DONE]\n\n`;
    return new Response(body, {
      status: 200,
      headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' }
    });
  }

  // Fake user for auth bypass
  const FAKE_USER = {
    id: 'webmind-local',
    name: 'User',
    email: 'user@webmind.sh',
    role: 'admin',
    profile_image_url: '',
    token: 'webmind-local-token',
    permissions: { workspace: { models: true, knowledge: true, prompts: true, tools: true } }
  };

  // Pre-set auth token and skip onboarding
  if (!localStorage.getItem('token')) {
    localStorage.setItem('token', 'webmind-local-token');
  }
  // Mark onboarding/changelog as seen
  localStorage.setItem('dismissedChangelog', 'true');
  localStorage.setItem('onboarding', 'false');
  localStorage.setItem('version', '0.8.12');

  // Suppress WebSocket errors — we don't need real-time updates
  const OrigWebSocket = window.WebSocket;
  window.WebSocket = function(url, protocols) {
    if (url.includes('/ws/socket.io')) {
      // Return a fake WebSocket that does nothing
      return { close(){}, send(){}, addEventListener(){}, removeEventListener(){},
        readyState: 3, CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 };
    }
    return new OrigWebSocket(url, protocols);
  };
  window.WebSocket.prototype = OrigWebSocket.prototype;

  // Intercept fetch
  const originalFetch = window.fetch;
  window._originalFetch = originalFetch;
  window.fetch = async function(url, opts) {
    const urlStr = typeof url === 'string' ? url : url?.url || '';
    const method = (opts?.method || 'GET').toUpperCase();

    // Profile image — return a placeholder
    if (urlStr.includes('/profile/image') || urlStr.includes('/user.png'))
      return originalFetch('data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="50" fill="#2d8a4e"/><text x="50" y="65" text-anchor="middle" fill="white" font-size="40" font-family="sans-serif">W</text></svg>'));

    // Only intercept API calls
    const isApiCall = urlStr.includes('/api/') || urlStr.includes('/openai/') || urlStr.includes('/ollama/');
    if (!isApiCall) return originalFetch.apply(this, arguments);

    // Parse body if POST
    let body = {};
    if (opts?.body) {
      try { body = JSON.parse(opts.body); } catch(e) {}
    }

    // --- AUTH (must return user object for getSessionUser) ---
    if (urlStr.includes('/auths')) {
      console.log('[saqt-shim] AUTH:', method, urlStr, '→ returning user');
      return jsonResponse(FAKE_USER);
    }

    // --- CONFIG ---
    if (urlStr.endsWith('/api/config') || urlStr.includes('/api/config?'))
      return jsonResponse({
        status: true, name: 'Webmind', version: '0.8.12',
        default_locale: 'en-US', default_models: 'webmind-305k',
        default_prompt_suggestions: [],
        features: { auth: false, auth_trusted_header: false,
          enable_signup: false, enable_login_form: true,
          enable_websocket: false, enable_direct_connections: false,
          enable_web_search: false, enable_image_generation: false,
          enable_community_sharing: false, enable_admin_export: false,
          enable_admin_chat_access: false },
        onboarding: false,
        permissions: { workspace: { models: true, knowledge: true, prompts: true, tools: true },
          chat: { file_upload: false, delete: true, edit: true, temporary: true } },
        oauth: { providers: {} }
      });
    if (urlStr.includes('/api/version'))
      return jsonResponse({ version: '0.8.12', deployment_id: null });

    // --- MODELS ---
    if (urlStr.includes('/api/models') || urlStr.includes('/api/v1/models'))
      return jsonResponse({ data: [{
        id: 'webmind-305k', name: 'Webmind 305K', object: 'model', owned_by: 'webmind',
        info: { id: 'webmind-305k', name: 'Webmind 305K', meta: { description: '305K Q&A pairs. No LLM. Runs in your browser.', profile_image_url: '' } },
        preset: true, actions: [], arena: false
      }]});
    if (urlStr.includes('/openai/models'))
      return jsonResponse({ data: [{ id: 'webmind-305k', object: 'model', owned_by: 'webmind' }]});

    // --- CHAT COMPLETIONS (main chat endpoint) ---
    if (urlStr.includes('/chat/completions')) {
      if (!saqtReady) return jsonResponse({ error: { message: 'SAQT engine still loading. Please wait for the knowledge base to finish downloading.' }}, 503);
      const messages = body.messages || [];
      let q = '';
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === 'user') {
          q = typeof messages[i].content === 'string' ? messages[i].content :
              (Array.isArray(messages[i].content) ? messages[i].content.map(c => c.text || '').join(' ') : '');
          break;
        }
      }
      if (!q) return jsonResponse({ error: { message: 'No user message' }}, 400);
      const result = await query(q);
      // Always return SSE stream format (Open WebUI expects it)
      return streamResponse(result.answer);
    }

    // --- TASKS (title gen, tags, follow-ups, emoji) — return empty/defaults ---
    if (urlStr.includes('/api/v1/tasks/title'))
      return jsonResponse({ choices: [{ message: { content: JSON.stringify({ title: body.prompt?.substring(0, 40) || 'Chat' }) } }]});
    if (urlStr.includes('/api/v1/tasks/tags'))
      return jsonResponse({ choices: [{ message: { content: '{"tags": []}' } }]});
    if (urlStr.includes('/api/v1/tasks/emoji'))
      return jsonResponse({ choices: [{ message: { content: '"💬"' } }]});
    if (urlStr.includes('/api/v1/tasks/follow_ups'))
      return jsonResponse({ choices: [{ message: { content: '{"follow_ups": []}' } }]});
    if (urlStr.includes('/api/v1/tasks/auto/completions'))
      return jsonResponse({ choices: [{ message: { content: '{"text": ""}' } }]});
    if (urlStr.includes('/api/v1/tasks/config'))
      return jsonResponse({ TASK_MODEL: 'webmind-305k', TASK_MODEL_EXTERNAL: 'webmind-305k' });

    // --- CHATS (local storage) ---
    if (urlStr.includes('/api/v1/chats')) {
      if (method === 'GET') return jsonResponse({ data: [] });
      if (method === 'POST') return jsonResponse({ id: 'local-' + Date.now(), chat: body });
      return jsonResponse({});
    }

    // --- USERS ---
    if (urlStr.includes('/api/v1/users/settings'))
      return jsonResponse({ ui: {} });
    if (urlStr.includes('/api/v1/users'))
      return jsonResponse(FAKE_USER);
    if (urlStr.includes('/api/v1/configs/banners'))
      return jsonResponse([]);

    // --- KNOWLEDGE/MEMORIES/TOOLS/FUNCTIONS/etc — return empty ---
    if (urlStr.includes('/api/v1/knowledge')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/memories')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/tools')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/functions')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/prompts')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/folders')) return jsonResponse([]);
    if (urlStr.includes('/api/v1/channels')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/groups')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/evaluations')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/configs')) return jsonResponse({});
    if (urlStr.includes('/api/v1/notes')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/usage')) return jsonResponse({});
    if (urlStr.includes('/api/v1/analytics')) return jsonResponse({});
    if (urlStr.includes('/api/changelog')) return jsonResponse([]);
    if (urlStr.includes('/api/v1/terminals')) return jsonResponse([]);
    if (urlStr.includes('/api/v1/skills')) return jsonResponse({ data: [] });
    if (urlStr.includes('/api/v1/banners')) return jsonResponse([]);
    if (urlStr.includes('/api/v1/tags')) return jsonResponse([]);
    if (urlStr.includes('/api/events')) return jsonResponse([]);
    if (urlStr.includes('/api/community')) return jsonResponse([]);

    // --- OLLAMA ---
    if (urlStr.includes('/ollama/'))
      return jsonResponse({ models: [] });

    // --- OPENAI config ---
    if (urlStr.includes('/openai/config'))
      return jsonResponse({ ENABLE_OPENAI_API: true, OPENAI_API_BASE_URLS: [''], OPENAI_API_KEYS: [''], OPENAI_API_CONFIGS: {} });

    // --- Pipelines ---
    if (urlStr.includes('/api/v1/pipelines'))
      return jsonResponse({ data: [] });

    // --- Catch-all: return empty success for any API call ---
    console.log('[saqt-shim] catch-all:', method, urlStr.replace(location.origin, ''));
    if (method === 'GET') return jsonResponse([]);
    return jsonResponse({});
  };

  // Boot SAQT engine
  async function boot() {
    try {
      setStatus('Loading AI model...', 'Sentence transformer (80MB)', 10);
      const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
      if (env.backends?.onnx?.webgpu) env.backends.onnx.webgpu.enabled = false;
      // Force model loading from HuggingFace CDN, not local origin
      env.allowLocalModels = false;
      encoder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        progress_callback: (p) => { if (p.progress) setStatus('Loading AI model...', Math.round(p.progress) + '%', 10 + p.progress * 0.3); }
      });
      setStatus('Loading knowledge base...', '', 40);

      // Load Q&A data from VM
      const dr = await originalFetch(VM_BASE + '/qa_data.json');
      if (!dr.ok) throw new Error('Failed to load Q&A data');
      qaData = await dr.json();
      setStatus('Loading embeddings...', qaData.length.toLocaleString() + ' pairs', 60);

      const er = await originalFetch(VM_BASE + '/qa_embeddings.bin');
      if (!er.ok) throw new Error('Failed to load embeddings');
      const cl = er.headers.get('Content-Length');
      if (cl && parseInt(cl) > 1000000) {
        const reader = er.body.getReader();
        const total = parseInt(cl);
        let received = 0;
        const chunks = [];
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.length;
          setStatus('Downloading embeddings...', Math.round(received/1024/1024) + 'MB / ' + Math.round(total/1024/1024) + 'MB', 60 + (received/total)*35);
        }
        qaEmbeddings = new Float32Array(await new Blob(chunks).arrayBuffer());
      } else {
        qaEmbeddings = new Float32Array(await er.arrayBuffer());
      }

      saqtReady = true;
      setStatus('Ready!', qaData.length.toLocaleString() + ' pairs', 100);
      setTimeout(() => {
        const el = document.getElementById('saqt-overlay');
        if (el) { el.style.opacity = '0'; el.style.transition = 'opacity 0.5s'; setTimeout(() => el.remove(), 500); }
      }, 800);
    } catch(e) {
      setStatus('Error: ' + e.message, '', 0);
      console.error('[saqt]', e);
    }
  }

  boot();
})();
