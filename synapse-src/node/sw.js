// Minimal service worker for PWA install + offline caching of app shell
const CACHE = 'synapse-node-v1';
const SHELL = ['/node/index.html', '/node/node.js', '/node/pipeline.js', '/node/kv-cache.js', '/node/shard-loader.js', '/node/predictor.js', '/node/speculative.js', '/node/early-exit.js', '/node/p2p.js', '/node/head-pruning.js'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(clients.claim());
});

self.addEventListener('fetch', e => {
  // Network-first for everything — cache is just fallback
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
