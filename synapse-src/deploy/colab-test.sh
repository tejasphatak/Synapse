#!/bin/bash
# Synapse Colab WebGPU Test — paste into a Colab cell: !bash colab-test.sh
set -e

echo "=== Installing deps ==="
apt-get update -qq
apt-get install -y -qq vulkan-tools 2>/dev/null || true
wget -qO /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && apt-get install -y /tmp/chrome.deb 2>/dev/null || true
npm install -g puppeteer-core@^22.0.0 2>&1 | tail -1

echo ""
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: No GPU"

echo ""
echo "=== Testing WebGPU ==="
node --input-type=module << 'JS'
import puppeteer from 'puppeteer-core';
const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome-stable',
  headless: 'new',
  args: ['--no-sandbox','--use-angle=vulkan','--enable-features=Vulkan','--disable-vulkan-surface','--enable-unsafe-webgpu'],
});
const page = await browser.newPage();
const r = await page.evaluate(async () => {
  if (!navigator.gpu) return {ok:false, reason:'no navigator.gpu'};
  const a = await navigator.gpu.requestAdapter();
  if (!a) return {ok:false, reason:'no adapter'};
  const info = await a.requestAdapterInfo();
  return {ok:true, gpu:info.description||info.device, maxBuf:a.limits.maxBufferSize};
});
console.log(JSON.stringify(r, null, 2));
await browser.close();
JS
