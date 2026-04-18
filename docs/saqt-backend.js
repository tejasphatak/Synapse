/**
 * SAQT In-Browser Backend via almostnode
 *
 * Runs an Express server INSIDE the browser using almostnode.
 * The Service Worker intercepts fetch calls and routes them to Express.
 * SAQT engine provides the intelligence.
 */

// Express server code that runs inside almostnode's runtime
const SERVER_CODE = `
const express = require('express');
const app = express();
app.use(express.json());

// CORS
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  if (req.method === 'OPTIONS') return res.status(204).end();
  next();
});

// Fake user
const USER = {
  id: 'webmind-local', name: 'User', email: 'user@webmind.sh',
  role: 'admin', profile_image_url: '', token: 'webmind-local-token',
  permissions: { workspace: { models: true, knowledge: true, prompts: true, tools: true } }
};

// Config
app.get('/api/config', (req, res) => res.json({
  status: true, name: 'Webmind', version: '0.8.12',
  default_locale: 'en-US', default_models: 'webmind-305k',
  default_prompt_suggestions: [],
  features: { auth: false, auth_trusted_header: false, enable_signup: false,
    enable_login_form: true, enable_websocket: false, enable_direct_connections: false,
    enable_web_search: false, enable_image_generation: false,
    enable_community_sharing: false, enable_admin_export: false,
    enable_admin_chat_access: false },
  onboarding: false,
  permissions: { workspace: { models: true, knowledge: true, prompts: true, tools: true },
    chat: { file_upload: false, delete: true, edit: true, temporary: true } },
  oauth: { providers: {} }
}));

// Version
app.get('/api/version', (req, res) => res.json({ version: '0.8.12', deployment_id: null }));

// Auth
app.get('/api/v1/auths/*', (req, res) => res.json(USER));
app.post('/api/v1/auths/*', (req, res) => res.json(USER));

// Models
app.get('/api/models', (req, res) => res.json({ data: [{
  id: 'webmind-305k', name: 'Webmind 305K', object: 'model', owned_by: 'webmind',
  info: { id: 'webmind-305k', name: 'Webmind 305K',
    meta: { description: '305K Q&A pairs. Runs in your browser.', profile_image_url: '' } },
  preset: true, actions: [], arena: false
}]}));
app.get('/openai/models', (req, res) => res.json({ data: [{ id: 'webmind-305k', object: 'model', owned_by: 'webmind' }]}));

// User settings
app.get('/api/v1/users/*/settings', (req, res) => res.json({ ui: { version: '0.8.12', showChangelog: false } }));
app.get('/api/v1/users/*', (req, res) => res.json(USER));

// Chat completions — SSE stream
app.post('/api/chat/completions', (req, res) => {
  const messages = req.body?.messages || [];
  let q = '';
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === 'user') {
      q = typeof messages[i].content === 'string' ? messages[i].content :
          (Array.isArray(messages[i].content) ? messages[i].content.map(c => c.text || '').join(' ') : '');
      break;
    }
  }

  // Call SAQT engine (exposed as global)
  const answer = globalThis._saqtQuery ? globalThis._saqtQuery(q) : 'SAQT engine not loaded yet. Please wait.';

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');

  const chunk = JSON.stringify({
    id: 'wmind-' + Date.now(), object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000), model: 'webmind-305k',
    choices: [{ index: 0, delta: { role: 'assistant', content: answer }, finish_reason: null }]
  });
  const done = JSON.stringify({
    id: 'wmind-' + Date.now(), object: 'chat.completion.chunk',
    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
  });

  res.write('data: ' + chunk + '\\n\\n');
  res.write('data: ' + done + '\\n\\n');
  res.write('data: [DONE]\\n\\n');
  res.end();
});

// Tasks (title, tags, emoji, follow-ups, auto-complete)
app.post('/api/v1/tasks/title/*', (req, res) => res.json({ choices: [{ message: { content: JSON.stringify({ title: (req.body?.prompt || 'Chat').substring(0, 40) }) } }]}));
app.post('/api/v1/tasks/tags/*', (req, res) => res.json({ choices: [{ message: { content: '{"tags": []}' } }]}));
app.post('/api/v1/tasks/emoji/*', (req, res) => res.json({ choices: [{ message: { content: '"💬"' } }]}));
app.post('/api/v1/tasks/follow_ups/*', (req, res) => res.json({ choices: [{ message: { content: '{"follow_ups": []}' } }]}));
app.post('/api/v1/tasks/auto/*', (req, res) => res.json({ choices: [{ message: { content: '{"text": ""}' } }]}));
app.get('/api/v1/tasks/config', (req, res) => res.json({ TASK_MODEL: 'webmind-305k', TASK_MODEL_EXTERNAL: 'webmind-305k' }));

// Empty collections
app.get('/api/v1/chats', (req, res) => res.json({ data: [] }));
app.post('/api/v1/chats/*', (req, res) => res.json({ id: 'local-' + Date.now(), chat: req.body }));
app.get('/api/v1/knowledge', (req, res) => res.json({ data: [] }));
app.get('/api/v1/memories', (req, res) => res.json({ data: [] }));
app.get('/api/v1/tools', (req, res) => res.json([]));
app.get('/api/v1/tools/*', (req, res) => res.json([]));
app.get('/api/v1/functions', (req, res) => res.json({ data: [] }));
app.get('/api/v1/prompts', (req, res) => res.json({ data: [] }));
app.get('/api/v1/folders', (req, res) => res.json([]));
app.get('/api/v1/channels', (req, res) => res.json({ data: [] }));
app.get('/api/v1/groups', (req, res) => res.json({ data: [] }));
app.get('/api/v1/evaluations', (req, res) => res.json({ data: [] }));
app.get('/api/v1/configs/*', (req, res) => res.json([]));
app.get('/api/v1/notes', (req, res) => res.json({ data: [] }));
app.get('/api/usage', (req, res) => res.json({}));
app.get('/api/v1/analytics', (req, res) => res.json({}));
app.get('/api/changelog', (req, res) => res.json([]));
app.get('/api/community', (req, res) => res.json([]));
app.get('/api/v1/terminals', (req, res) => res.json([]));
app.get('/api/v1/skills', (req, res) => res.json({ data: [] }));
app.get('/api/v1/banners', (req, res) => res.json([]));
app.get('/api/v1/tags', (req, res) => res.json([]));
app.get('/api/events', (req, res) => res.json([]));
app.get('/ollama/*', (req, res) => res.json({ models: [] }));
app.get('/openai/config', (req, res) => res.json({ ENABLE_OPENAI_API: true, OPENAI_API_BASE_URLS: [''], OPENAI_API_KEYS: [''], OPENAI_API_CONFIGS: {} }));
app.get('/api/v1/pipelines', (req, res) => res.json({ data: [] }));

// Catch-all
app.all('*', (req, res) => {
  console.log('[webmind-api] unhandled:', req.method, req.path);
  res.json(req.method === 'GET' ? [] : {});
});

app.listen(3000, () => console.log('[webmind] Express server running on virtual port 3000'));
`;

export { SERVER_CODE };
