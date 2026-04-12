#!/bin/bash
set -e

echo "=== Synapse Startup ==="

# Install Node.js dependencies
if [ ! -d "node_modules" ]; then
  echo "[1/3] Installing Node.js dependencies..."
  npm install
else
  echo "[1/3] Node dependencies already installed"
fi

# Check if model shards exist
if [ ! -f "model/shards/manifest.json" ]; then
  echo "[2/3] Model shards not found — downloading and splitting GPT-2..."
  echo "       (This takes 2-5 minutes on first run)"

  # Install Python dependencies if needed
  pip install transformers torch --quiet 2>/dev/null || pip3 install transformers torch --quiet 2>/dev/null

  # Run the split script (default: GPT-2 small, float32, 2 shards)
  # Change this line to use a different model/quantization:
  #   python3 model/split.py --model gpt2-medium --dtype int8
  #   python3 model/split.py --model gpt2-xl --dtype int4 --num-shards 4
  python3 model/split.py
else
  echo "[2/3] Model shards found — skipping download"
fi

echo "[3/3] Starting Synapse coordinator..."
echo ""
echo "  Open these URLs on your devices:"
echo "  - Prompt UI:  https://your-repl-url/"
echo "  - Node (Tab 1): https://your-repl-url/node/index.html"
echo "  - Node (Tab 2): https://your-repl-url/node/index.html"
echo "  - Dashboard:  https://your-repl-url/ui/dashboard.html"
echo ""

node coordinator/index.js
