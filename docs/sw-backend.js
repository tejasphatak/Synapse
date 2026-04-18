/**
 * Webmind Service Worker Backend
 * Intercepts ALL fetch requests at the network level.
 * No monkey-patching, no almostnode, no dependencies.
 * The browser's built-in Service Worker API IS the server.
 *
 * Implements:
 * - REST API endpoints (auth, models, chats, config, etc.)
 * - Socket.io polling transport (engine.io v4) for streaming chat responses
 * - MessageChannel to main thread for SAQT query execution
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

// ─── Socket.io polling transport (engine.io v4) ───

const SIO_SID = 'wmind-' + Math.random().toString(36).substring(2, 14);
const SIO_NS_SID = 'wmind-ns-' + Math.random().toString(36).substring(2, 14);
let sioConnected = false;
let eventQueue = [];
let pollResolvers = [];

function textResponse(body) {
  return new Response(body, {
    status: 200,
    headers: {
      'Content-Type': 'text/plain; charset=UTF-8',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Credentials': 'true'
    }
  });
}

function queueSocketEvent(eventName, data) {
  const payload = '42' + JSON.stringify([eventName, data]);
  eventQueue.push(payload);
  // Wake any waiting long-poll
  if (pollResolvers.length > 0) {
    const resolve = pollResolvers.shift();
    const events = eventQueue.splice(0);
    resolve(textResponse(events.join('\x1e')));
  }
}

function handleSocketIO(request) {
  const url = new URL(request.url);
  const sid = url.searchParams.get('sid');
  const method = request.method;

  if (method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        'Access-Control-Allow-Credentials': 'true'
      }
    });
  }

  // Handshake (no sid yet)
  if (method === 'GET' && !sid) {
    sioConnected = false;
    const handshake = '0' + JSON.stringify({
      sid: SIO_SID,
      upgrades: [],
      pingInterval: 25000,
      pingTimeout: 20000,
      maxPayload: 1000000
    });
    return textResponse(handshake);
  }

  // POST — client sends connect packet or ping
  if (method === 'POST' && sid) {
    // Client sends '40' (connect) or '2' (ping) or '42[...]' (event)
    // We just acknowledge
    return textResponse('ok');
  }

  // GET with sid — poll for events
  if (method === 'GET' && sid) {
    // First poll after handshake: send namespace connect ack
    if (!sioConnected) {
      sioConnected = true;
      const connectAck = '40' + JSON.stringify({ sid: SIO_NS_SID });
      if (eventQueue.length > 0) {
        const events = eventQueue.splice(0);
        return textResponse(connectAck + '\x1e' + events.join('\x1e'));
      }
      return textResponse(connectAck);
    }

    // Return queued events if any
    if (eventQueue.length > 0) {
      const events = eventQueue.splice(0);
      return textResponse(events.join('\x1e'));
    }

    // Long-poll: wait for events or timeout with pong
    return new Promise((resolve) => {
      pollResolvers.push(resolve);
      setTimeout(() => {
        const idx = pollResolvers.indexOf(resolve);
        if (idx !== -1) {
          pollResolvers.splice(idx, 1);
          resolve(textResponse('3')); // pong
        }
      }, 25000);
    });
  }

  return textResponse('ok');
}

// ─── SAQT query — uses message channel to main thread ───

let queryPort = null;
let queryId = 0;
const pendingQueries = new Map();

self.addEventListener('message', (event) => {
  if (event.data?.type === 'saqt-port') {
    queryPort = event.ports[0];
    console.log('[sw] SAQT port received');
    queryPort.onmessage = (e) => {
      const { id, answer } = e.data;
      const resolve = pendingQueries.get(id);
      if (resolve) { resolve(answer); pendingQueries.delete(id); }
    };
  }
});

function saqtQuery(question) {
  if (!queryPort) return Promise.resolve("SAQT engine not ready. Please refresh the page.");
  return new Promise((resolve) => {
    const id = ++queryId;
    pendingQueries.set(id, resolve);
    queryPort.postMessage({ id, question });
    setTimeout(() => { if (pendingQueries.has(id)) { pendingQueries.delete(id); resolve("Query timed out."); } }, 30000);
  });
}

// ─── REST API Route table ───

async function handleAPI(request) {
  const url = new URL(request.url);
  const path = url.pathname;
  const method = request.method;

  // Socket.io polling
  if (path.startsWith('/ws/socket.io')) return handleSocketIO(request);

  if (method === 'OPTIONS')
    return new Response(null, { status: 204, headers: { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type, Authorization' } });

  let body = {};
  if (method === 'POST' || method === 'PUT') {
    try { body = await request.json(); } catch(e) {}
  }

  // Config
  if (path === '/api/config')
    return json({ status: true, name: 'Webmind', version: '0.8.12', default_locale: 'en-US', default_models: 'W', default_prompt_suggestions: [], features: { auth: false, auth_trusted_header: false, enable_signup: false, enable_login_form: true, enable_websocket: false, enable_direct_connections: false, enable_web_search: false, enable_image_generation: false, enable_community_sharing: false, enable_admin_export: false, enable_admin_chat_access: false }, onboarding: false, permissions: { workspace: { models: true, knowledge: true, prompts: true, tools: true }, chat: { file_upload: false, delete: true, edit: true, temporary: true } }, oauth: { providers: {} } });

  if (path === '/api/version/updates')
    return json({ current: '0.8.12', latest: '0.8.12' });
  if (path === '/api/version')
    return json({ version: '0.8.12', deployment_id: null });

  // Auth
  if (path.includes('/auths'))
    return json(USER);

  // Models
  if (path === '/api/models' || path.startsWith('/api/models'))
    return json({ data: [{ id: 'W', name: 'W', object: 'model', owned_by: 'webmind', info: { id: 'W', name: 'W', meta: { description: '305K answers. Zero hallucinations. 100% private.', profile_image_url: '' }, params: {} }, preset: true, actions: [], arena: false, tags: [], urlIdx: 0 }] });

  if (path.startsWith('/openai/models'))
    return json({ data: [{ id: 'W', object: 'model', owned_by: 'webmind' }] });

  // User settings
  if (path.match(/\/users\/.*\/settings/))
    return json({ ui: { version: '0.8.12', showChangelog: false } });

  if (path.includes('/users/'))
    return json(USER);

  // Chat completions — async: return task_id, stream answer via socket.io
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

    const taskId = 'task-' + Date.now();
    const responseMessageId = body.id;
    const chatId = body.chat_id;

    // Fire SAQT query asynchronously, deliver answer via socket.io events
    saqtQuery(q).then((answer) => {
      // Send non-streaming completion via socket event
      queueSocketEvent('events', {
        chat_id: chatId,
        message_id: responseMessageId,
        data: {
          type: 'chat:completion',
          data: {
            id: taskId,
            done: false,
            choices: [{ message: { content: answer } }]
          }
        }
      });
      // Send done event
      queueSocketEvent('events', {
        chat_id: chatId,
        message_id: responseMessageId,
        data: {
          type: 'chat:completion',
          data: {
            id: taskId,
            done: true,
            choices: [{ delta: {}, finish_reason: 'stop' }]
          }
        }
      });
    });

    return json({ task_id: taskId });
  }

  // Tasks
  if (path.includes('/tasks/stop')) return json({ status: true });
  if (path.includes('/tasks/chat/')) return json([]);
  if (path.includes('/tasks/title')) return json({ choices: [{ message: { content: JSON.stringify({ title: (body.prompt || body.messages?.[body.messages?.length - 1]?.content || 'Chat').substring(0, 40) }) } }] });
  if (path.includes('/tasks/tags')) return json({ choices: [{ message: { content: '{"tags": []}' } }] });
  if (path.includes('/tasks/emoji')) return json({ choices: [{ message: { content: '"💬"' } }] });
  if (path.includes('/tasks/follow_ups')) return json({ choices: [{ message: { content: '{"follow_ups": []}' } }] });
  if (path.includes('/tasks/auto')) return json({ choices: [{ message: { content: '{"text": ""}' } }] });
  if (path.includes('/tasks/config')) return json({ TASK_MODEL: 'W', TASK_MODEL_EXTERNAL: 'W' });

  // Empty collections
  if (path.includes('/configs/banners')) return json([]);
  if (path.includes('/tools')) return json([]);
  if (path.includes('/chats/tags')) return json([]);
  if (path.includes('/chats/list')) return json([]);
  if (path.includes('/chats/search')) return json([]);
  // Individual chat by ID
  if (path.match(/\/chats\/[^/]+$/) && method === 'GET') return json({ id: path.split('/').pop(), title: 'Chat', models: ['W'], tags: [], history: { messages: {}, currentId: null }, messages: [], chat: {}, updated_at: new Date().toISOString() });
  // Chat list
  if (path.includes('/chats') && method === 'GET') return json([]);
  if (path.match(/\/chats\/new/) && method === 'POST') return json({ id: 'local-' + Date.now(), title: 'Chat', models: ['W'], tags: [], history: { messages: {}, currentId: null }, messages: [], chat: body, updated_at: new Date().toISOString() });
  if (path.includes('/chats') && method === 'POST') return json({ id: path.split('/').pop() || ('local-' + Date.now()), title: 'Chat', models: ['W'], tags: [], history: body?.chat?.history || { messages: {}, currentId: null }, messages: body?.chat?.messages || [], chat: body?.chat || body, updated_at: new Date().toISOString() });
  if (path.includes('/chats') && method === 'DELETE') return json({ success: true });
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

// ─── Intercept fetch events ───

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (url.origin === self.location.origin &&
      (url.pathname.startsWith('/api/') || url.pathname.startsWith('/openai/') || url.pathname.startsWith('/ollama/') || url.pathname.startsWith('/ws/'))) {
    event.respondWith(handleAPI(event.request));
  }
});

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));
