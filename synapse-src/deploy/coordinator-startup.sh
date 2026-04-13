#!/bin/bash
# Synapse Coordinator VM startup script
# Runs on first boot of the GCP instance

set -euo pipefail
exec > /var/log/synapse-startup.log 2>&1

echo "=== Synapse Coordinator Startup $(date) ==="

# Install Node.js 20 LTS
if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

echo "Node version: $(node --version)"

# Install Python 3 + pip for model splitting
apt-get update -y
apt-get install -y python3 python3-pip git

# Create app directory
mkdir -p /opt/synapse
cd /opt/synapse

# Clone or copy the code (we'll use gcloud scp from the dev machine)
# This script assumes code is already at /opt/synapse/synapse-src/

# If code exists, install deps and start
if [ -d "/opt/synapse/synapse-src" ]; then
  cd /opt/synapse/synapse-src
  npm install --production

  # Install Python deps for model splitting
  pip3 install transformers torch --break-system-packages 2>/dev/null || pip3 install transformers torch

  # Split model if shards don't exist
  if [ ! -f "model/shards/manifest.json" ]; then
    echo "Splitting model..."
    python3 model/split.py --model gpt2 --dtype float16 --num-shards 2
  fi

  # Start coordinator
  echo "Starting coordinator on port 8080..."
  PORT=8080 node coordinator/index.js &
  echo "Coordinator PID: $!"
else
  echo "WARNING: /opt/synapse/synapse-src not found. Waiting for code deployment via SCP."
fi

echo "=== Startup complete $(date) ==="
