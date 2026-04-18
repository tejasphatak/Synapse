/**
 * Webmind Service Worker Backend
 * Intercepts ALL fetch requests at the network level.
 * No monkey-patching, no almostnode, no dependencies.
 * The browser's built-in Service Worker API IS the server.
 */

const USER = {
  id: 'webmind-local', name: 'User', email: 'user@webmind.sh',
  role: 'admin', profile_image_url: '', token: 'webmind-local-token',
  permissions: { workspace: { models: true, knowledge: true, prompts: true, tools: true } }
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }
  });
}

function sse(text) {
  const chunk = JSON.stringify({
    id: 'wmind-' + Date.now(), object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000), model: 'W',
    choices: [{ index: 0, delta: { role: 'assistant', content: text }, finish_reason: null }]
  });
  const done = JSON.stringify({
    id: 'wmind-' + Date.now(), object: 'chat.completion.chunk',
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
  });
  return new Response(`data: ${chunk}\n\ndata: ${done}\n\ndata: [DONE]\n\n`, {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Access-Control-Allow-Origin': '*' }
  });
}

// SAQT query — uses message channel to main thread
let queryPort = null;
let queryId = 0;
const pendingQueries = new Map();

self.addEventListener('message', (event) => {
  console.log('[sw] message received:', event.data?.type, 'ports:', event.ports?.length);
  if (event.data?.type === 'saqt-port') {
    queryPort = event.ports[0];
    console.log('[sw] SAQT port received');
    queryPort.onmessage = (e) => {
      console.log('[sw] Got answer from main thread:', e.data?.id);
      const { id, answer } = e.data;
      const resolve = pendingQueries.get(id);
      if (resolve) { resolve(answer); pendingQueries.delete(id); }
    };
  }
});

function saqtQuery(question) {
  console.log('[sw] saqtQuery called, port exists:', !!queryPort, 'question:', question?.substring(0,30));
  if (!queryPort) return Promise.resolve("SAQT engine not ready. Please refresh the page.");
  return new Promise((resolve) => {
    const id = ++queryId;
    pendingQueries.set(id, resolve);
    queryPort.postMessage({ id, question });
    // Timeout after 30s
    setTimeout(() => { if (pendingQueries.has(id)) { pendingQueries.delete(id); resolve("Query timed out."); } }, 30000);
  });
}

// Route table
async function handleAPI(request) {
  const url = new URL(request.url);
  const path = url.pathname;
  const method = request.method;

  if (method === 'OPTIONS')
    return new Response(null, { status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type, Authorization' } });

  let body = {};
  if (method === 'POST' || method === 'PUT') {
    try { body = await request.json(); } catch(e) {}
  }

  // Config
  if (path === '/api/config')
    return json({ status: true, name: 'Webmind', version: '0.8.12', default_locale: 'en-US', default_models: 'W', default_prompt_suggestions: [], features: { auth: false, auth_trusted_header: false, enable_signup: false, enable_login_form: true, enable_websocket: false, enable_direct_connections: false, enable_web_search: false, enable_image_generation: false, enable_community_sharing: false, enable_admin_export: false, enable_admin_chat_access: false }, onboarding: false, permissions: { workspace: { models: true, knowledge: true, prompts: true, tools: true }, chat: { file_upload: false, delete: true, edit: true, temporary: true } }, oauth: { providers: {} } });

  if (path === '/api/version')
    return json({ version: '0.8.12', deployment_id: null });

  // Auth
  if (path.includes('/auths'))
    return json(USER);

  // Models
  if (path === '/api/models' || path.startsWith('/api/models'))
    return json({ data: [{ id: 'W', name: 'W', object: 'model', owned_by: 'webmind', info: { id: 'W', name: 'W', meta: { description: '305K answers. Zero hallucinations. 100% private.', profile_image_url: '' } }, preset: true, actions: [], arena: false }] });

  if (path.startsWith('/openai/models'))
    return json({ data: [{ id: 'W', object: 'model', owned_by: 'webmind' }] });

  // User settings
  if (path.match(/\/users\/.*\/settings/))
    return json({ ui: { version: '0.8.12', showChangelog: false } });

  if (path.includes('/users/'))
    return json(USER);

  // Chat completions
  if (path.includes('/chat/completions')) {
    const messages = body.messages || [];
    let q = '';
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        q = typeof messages[i].content === 'string' ? messages[i].content :
            (Array.isArray(messages[i].content) ? messages[i].content.map(c => c.text || '').join(' ') : '');
        break;
      }
    }
    if (!q) return json({ error: { message: 'No user message' } }, 400);
    const answer = await saqtQuery(q);
    return sse(answer);
  }

  // Tasks
  if (path.includes('/tasks/title')) return json({ choices: [{ message: { content: JSON.stringify({ title: (body.prompt || 'Chat').substring(0, 40) }) } }] });
  if (path.includes('/tasks/tags')) return json({ choices: [{ message: { content: '{"tags": []}' } }] });
  if (path.includes('/tasks/emoji')) return json({ choices: [{ message: { content: '"💬"' } }] });
  if (path.includes('/tasks/follow_ups')) return json({ choices: [{ message: { content: '{"follow_ups": []}' } }] });
  if (path.includes('/tasks/auto')) return json({ choices: [{ message: { content: '{"text": ""}' } }] });
  if (path.includes('/tasks/config')) return json({ TASK_MODEL: 'W', TASK_MODEL_EXTERNAL: 'W' });

  // Empty collections
  if (path.includes('/configs/banners')) return json([]);
  if (path.includes('/tools')) return json([]);
  if (path.includes('/chats') && method === 'GET') return json({ data: [] });
  if (path.includes('/chats') && method === 'POST') return json({ id: 'local-' + Date.now(), chat: body });
  if (path.includes('/knowledge')) return json({ data: [] });
  if (path.includes('/memories')) return json({ data: [] });
  if (path.includes('/functions')) return json({ data: [] });
  if (path.includes('/prompts')) return json({ data: [] });
  if (path.includes('/folders')) return json([]);
  if (path.includes('/channels')) return json({ data: [] });
  if (path.includes('/groups')) return json({ data: [] });
  if (path.includes('/evaluations')) return json({ data: [] });
  if (path.includes('/notes')) return json({ data: [] });
  if (path.includes('/terminals')) return json([]);
  if (path.includes('/skills')) return json({ data: [] });
  if (path.includes('/banners')) return json([]);
  if (path.includes('/tags')) return json([]);
  if (path.includes('/pipelines')) return json({ data: [] });
  if (path === '/api/usage') return json({});
  if (path.includes('/analytics')) return json({});
  if (path === '/api/changelog') return json([]);
  if (path.includes('/community')) return json([]);
  if (path.includes('/events')) return json([]);
  if (path.startsWith('/ollama/')) return json({ models: [] });
  if (path.startsWith('/openai/config')) return json({ ENABLE_OPENAI_API: true, OPENAI_API_BASE_URLS: [''], OPENAI_API_KEYS: [''], OPENAI_API_CONFIGS: {} });

  // Catch-all
  return json(method === 'GET' ? [] : {});
}

// Intercept fetch events
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  // Only intercept API calls to same origin
  if (url.origin === self.location.origin &&
      (url.pathname.startsWith('/api/') || url.pathname.startsWith('/openai/') || url.pathname.startsWith('/ollama/'))) {
    event.respondWith(handleAPI(event.request));
  }
  // Everything else (static files, CDN) passes through normally
});

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));
