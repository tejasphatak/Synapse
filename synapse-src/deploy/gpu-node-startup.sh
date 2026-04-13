#!/bin/bash
# Synapse GPU Node startup script
# Works on: GCP N1+T4, Google Colab T4, any Linux + NVIDIA GPU
# Installs headless Chrome with WebGPU support, then runs Synapse compute nodes

set -euo pipefail

echo "=== Synapse GPU Node Setup $(date) ==="

# Detect environment
if [ -d "/content" ] && [ -f "/proc/driver/nvidia/version" ]; then
  ENV="colab"
  WORKDIR="/content/synapse"
elif command -v nvidia-smi &>/dev/null; then
  ENV="gcp"
  WORKDIR="/opt/synapse"
else
  ENV="cpu-only"
  WORKDIR="/opt/synapse"
fi

echo "Environment: $ENV"

# --- NVIDIA driver setup (GCP only — Colab already has drivers) ---
if [ "$ENV" = "gcp" ]; then
  if ! nvidia-smi &>/dev/null; then
    echo "Installing NVIDIA drivers..."
    curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py -o /tmp/install_gpu_driver.py
    python3 /tmp/install_gpu_driver.py
  fi
fi

nvidia-smi 2>/dev/null && echo "GPU OK" || echo "WARNING: No GPU detected"

# --- Install Vulkan + headless Chrome dependencies ---
echo "Installing Vulkan and Chrome dependencies..."
apt-get update -y -qq
apt-get install -y -qq vulkan-tools libnvidia-gl-535 2>/dev/null || \
  apt-get install -y -qq vulkan-tools 2>/dev/null || true

# Install Chrome
if ! command -v google-chrome-stable &>/dev/null; then
  echo "Installing Chrome..."
  wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
  apt-get install -y /tmp/chrome.deb 2>/dev/null || apt-get install -y -f
  rm -f /tmp/chrome.deb
fi

# Install Node.js
if ! command -v node &>/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -qq nodejs
fi

echo "Node: $(node --version)"
echo "Chrome: $(google-chrome-stable --version 2>/dev/null || echo 'not found')"

# --- Install Puppeteer ---
mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -f "package.json" ]; then
  cat > package.json << 'PKGJSON'
{
  "name": "synapse-gpu-node",
  "type": "module",
  "dependencies": {
    "puppeteer-core": "^22.0.0"
  }
}
PKGJSON
  npm install 2>&1
fi

echo "=== Setup complete ==="
echo ""
echo "To launch headless Synapse nodes:"
echo "  COORDINATOR_URL=http://<coordinator-ip>:8080 node headless-node.js"
