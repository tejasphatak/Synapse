/**
 * Headless Chrome WebGPU inference test.
 * Starts coordinator locally, launches Chrome with WebGPU flags,
 * loads the diag page, and captures results.
 *
 * Works with software WebGPU (SwiftShader/Dawn) — no real GPU needed.
 */
import puppeteer from 'puppeteer-core';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, dirname, extname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');

// Minimal static file server (no WS needed for diag page)
const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.wgsl': 'text/plain', '.json': 'application/json', '.bin': 'application/octet-stream' };

const server = createServer((req, res) => {
  // Tokenize API
  if (req.url === '/api/tokenize' && req.method === 'POST') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', async () => {
      const { encode } = await import('gpt-tokenizer/model/text-davinci-001');
      const { text } = JSON.parse(body);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ tokenIds: encode(text) }));
    });
    return;
  }
  if (req.url === '/api/detokenize' && req.method === 'POST') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', async () => {
      const { decode } = await import('gpt-tokenizer/model/text-davinci-001');
      const { tokenIds } = JSON.parse(body);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ text: decode(tokenIds) }));
    });
    return;
  }

  let filePath = req.url.split('?')[0];
  if (filePath === '/') filePath = '/ui/diag.html';

  // Handle /shards/ path
  if (filePath.startsWith('/shards/')) {
    filePath = '/model' + filePath;
  }

  const fullPath = join(ROOT, filePath);
  if (existsSync(fullPath)) {
    const ext = extname(fullPath);
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(readFileSync(fullPath));
  } else {
    res.writeHead(404);
    res.end('Not found: ' + filePath);
  }
});

const PORT = 9222;
server.listen(PORT, async () => {
  console.log(`Test server on http://localhost:${PORT}`);

  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/google-chrome-stable',
    headless: 'new',
    args: [
      '--no-sandbox',
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--use-angle=swiftshader',  // Software WebGPU — no real GPU needed
      '--disable-gpu-sandbox',
    ],
  });

  const page = await browser.newPage();

  // Capture console output
  const output = [];
  page.on('console', msg => {
    const text = msg.text();
    output.push(text);
    process.stdout.write(`[browser] ${text}\n`);
  });

  page.on('pageerror', err => {
    console.error(`[PAGE ERROR] ${err.message}`);
  });

  console.log('Navigating to diag page...');
  await page.goto(`http://localhost:${PORT}/ui/diag.html`, {
    waitUntil: 'networkidle2',
    timeout: 300000, // 5 min — shard loading is slow on software GPU
  });

  // Wait for completion
  console.log('Waiting for diag to complete...');
  try {
    await page.waitForFunction(
      () => document.getElementById('log')?.textContent?.includes('=== DONE ==='),
      { timeout: 600000 } // 10 min max
    );
  } catch (e) {
    console.error('Timeout waiting for diag completion');
  }

  const result = await page.evaluate(() => document.getElementById('log')?.textContent);
  console.log('\n=== DIAG RESULTS ===\n');
  console.log(result);

  await browser.close();
  server.close();
  process.exit(0);
});
