/**
 * Synapse Headless Compute Node
 *
 * Launches headless Chrome with WebGPU on a T4 GPU,
 * opens the Synapse node UI, and connects to the coordinator.
 *
 * Usage:
 *   COORDINATOR_URL=http://<ip>:8080 node headless-node.js
 *   COORDINATOR_URL=http://<ip>:8080 NUM_NODES=2 node headless-node.js
 *
 * Works on: GCP N1+T4, Google Colab, any Linux+NVIDIA GPU
 * Auto-installs puppeteer-core if missing (Colab-friendly).
 */

import { execSync } from 'child_process';
import { createRequire } from 'module';

// Auto-install puppeteer-core if missing
let puppeteer;
try {
  puppeteer = (await import('puppeteer-core')).default;
} catch {
  console.log('puppeteer-core not found — installing...');
  execSync('npm install puppeteer-core@^22.0.0 --no-save', { stdio: 'inherit' });
  puppeteer = (await import('puppeteer-core')).default;
}

const COORDINATOR_URL = process.env.COORDINATOR_URL || 'http://localhost:8080';
const NUM_NODES = parseInt(process.env.NUM_NODES || '2', 10);
const CHROME_PATH = process.env.CHROME_PATH || '/usr/bin/google-chrome-stable';

const CHROME_FLAGS = [
  '--no-sandbox',
  '--headless=new',
  '--use-angle=vulkan',
  '--enable-features=Vulkan',
  '--disable-vulkan-surface',
  '--enable-unsafe-webgpu',
  '--no-first-run',
  '--no-default-browser-check',
  '--disable-features=Translate',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--disable-backgrounding-occluded-windows',
  '--window-size=1280,720',
];

async function launchNode(nodeIndex, browser) {
  const page = await browser.newPage();

  // Capture console logs from the browser
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('[Synapse]') || text.includes('WebGPU') || text.includes('shard') || text.includes('connected')) {
      console.log(`[Node ${nodeIndex}] ${text}`);
    }
  });

  page.on('pageerror', err => {
    console.error(`[Node ${nodeIndex}] PAGE ERROR: ${err.message}`);
  });

  const nodeUrl = `${COORDINATOR_URL}/node/index.html`;
  console.log(`[Node ${nodeIndex}] Navigating to ${nodeUrl}`);

  await page.goto(nodeUrl, {
    waitUntil: 'networkidle2',
    timeout: 60000
  });

  // Check WebGPU availability
  const hasWebGPU = await page.evaluate(async () => {
    if (!navigator.gpu) return { ok: false, reason: 'navigator.gpu not available' };
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) return { ok: false, reason: 'no adapter' };
      const info = await adapter.requestAdapterInfo();
      return { ok: true, adapter: info.description || info.device || 'unknown' };
    } catch (e) {
      return { ok: false, reason: e.message };
    }
  });

  if (hasWebGPU.ok) {
    console.log(`[Node ${nodeIndex}] WebGPU OK — adapter: ${hasWebGPU.adapter}`);
  } else {
    console.error(`[Node ${nodeIndex}] WebGPU FAILED — ${hasWebGPU.reason}`);
    return null;
  }

  // Wait for the node to connect and get shard assignment
  console.log(`[Node ${nodeIndex}] Waiting for shard assignment...`);

  await page.waitForFunction(() => {
    // Check if the node UI shows a connected/assigned state
    const status = document.querySelector('#status, .status, [data-status]');
    if (status && (status.textContent.includes('assigned') || status.textContent.includes('ready'))) {
      return true;
    }
    // Also check for WebSocket connection via window state
    return window.__synapse_connected === true;
  }, { timeout: 120000 }).catch(() => {
    console.log(`[Node ${nodeIndex}] Timed out waiting for assignment (may still connect)`);
  });

  return page;
}

async function main() {
  console.log('=== Synapse Headless GPU Node Launcher ===');
  console.log(`Coordinator: ${COORDINATOR_URL}`);
  console.log(`Nodes to launch: ${NUM_NODES}`);
  console.log(`Chrome: ${CHROME_PATH}`);
  console.log('');

  const browser = await puppeteer.launch({
    executablePath: CHROME_PATH,
    headless: 'new',
    args: CHROME_FLAGS,
  });

  console.log('Chrome launched with WebGPU flags');

  // Verify GPU at chrome://gpu
  const gpuPage = await browser.newPage();
  await gpuPage.goto('chrome://gpu', { waitUntil: 'networkidle2' });
  const gpuStatus = await gpuPage.evaluate(() => {
    const text = document.body.innerText;
    const lines = text.split('\n').filter(l =>
      l.includes('WebGPU') || l.includes('Vulkan') || l.includes('GL_RENDERER')
    );
    return lines.join('\n');
  });
  console.log('GPU Status:\n' + gpuStatus);
  await gpuPage.close();

  // Launch compute nodes
  const pages = [];
  for (let i = 0; i < NUM_NODES; i++) {
    const page = await launchNode(i, browser);
    if (page) pages.push(page);
    // Stagger launches slightly
    if (i < NUM_NODES - 1) await new Promise(r => setTimeout(r, 2000));
  }

  console.log(`\n${pages.length}/${NUM_NODES} nodes launched successfully`);
  console.log('Nodes are running. Press Ctrl+C to stop.\n');

  // Keep alive and log periodic status
  const statusInterval = setInterval(async () => {
    for (let i = 0; i < pages.length; i++) {
      try {
        const status = await pages[i].evaluate(() => {
          // Try to get node status from the page
          const el = document.querySelector('#status, .status, [data-status]');
          return el ? el.textContent.trim() : 'running';
        });
        console.log(`[Node ${i}] Status: ${status}`);
      } catch (e) {
        console.log(`[Node ${i}] Status check failed: ${e.message}`);
      }
    }
  }, 30000);

  // Graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\nShutting down nodes...');
    clearInterval(statusInterval);
    await browser.close();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    clearInterval(statusInterval);
    await browser.close();
    process.exit(0);
  });

  // Keep process alive
  await new Promise(() => {});
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
