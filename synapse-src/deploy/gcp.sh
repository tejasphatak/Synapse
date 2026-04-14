#!/bin/bash
# Synapse GCP Deployment Script
# Usage:
#   ./gcp.sh coord-up   — Create coordinator VM only (~$0.03/hr)
#   ./gcp.sh coord-down — Delete coordinator VM
#   ./gcp.sh gpu-up     — Create GPU VM (N1+T4, ~$0.38/hr)
#   ./gcp.sh gpu-down   — Delete GPU VM
#   ./gcp.sh up         — Create both VMs
#   ./gcp.sh down       — Delete all VMs (stop billing)
#   ./gcp.sh status     — Check VM status
#   ./gcp.sh deploy     — Push latest code to coordinator
#   ./gcp.sh ssh-coord  — SSH into coordinator
#   ./gcp.sh ssh-gpu    — SSH into GPU node
#   ./gcp.sh logs       — Tail coordinator logs
#   ./gcp.sh ip         — Print coordinator external IP

set -euo pipefail

PROJECT="directed-cove-493200-b8"
ZONE="us-west1-b"
REGION="us-west1"

COORD_VM="synapse-coordinator"
COORD_TYPE="e2-medium"            # 2 vCPU, 4GB — plenty for coordinator
COORD_IMAGE_FAMILY="debian-12"
COORD_IMAGE_PROJECT="debian-cloud"

GPU_VM="synapse-gpu-node"
GPU_TYPE="n1-standard-1"          # 1 vCPU, 3.75GB — cheapest N1 with T4 (~$0.38/hr)
GPU_ACCEL="type=nvidia-tesla-t4,count=1"
GPU_IMAGE_FAMILY="debian-12"
GPU_IMAGE_PROJECT="debian-cloud"

SYNAPSE_SRC="$(cd "$(dirname "$0")/.." && pwd)"
FIREWALL_RULE="synapse-allow-8080"

# ---------- helpers ----------

log() { echo ">>> $*"; }

get_coord_ip() {
  gcloud compute instances describe "$COORD_VM" --zone="$ZONE" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)" \
    --project="$PROJECT" 2>/dev/null
}

wait_for_ssh() {
  local vm=$1
  log "Waiting for SSH on $vm..."
  for i in $(seq 1 30); do
    if gcloud compute ssh "$vm" --zone="$ZONE" --project="$PROJECT" \
      --command="echo ok" --strict-host-key-checking=no 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: SSH timeout for $vm"
  return 1
}

ensure_firewall() {
  if ! gcloud compute firewall-rules describe "$FIREWALL_RULE" --project="$PROJECT" &>/dev/null; then
    log "Creating firewall rule for port 8080..."
    gcloud compute firewall-rules create "$FIREWALL_RULE" \
      --allow=tcp:8080 \
      --target-tags=synapse \
      --description="Allow Synapse coordinator port" \
      --project="$PROJECT" 2>&1
  fi
}

# ---------- coordinator ----------

cmd_coord_up() {
  ensure_firewall

  log "Creating coordinator VM: $COORD_VM ($COORD_TYPE) ~\$0.03/hr..."
  gcloud compute instances create "$COORD_VM" \
    --zone="$ZONE" \
    --machine-type="$COORD_TYPE" \
    --image-family="$COORD_IMAGE_FAMILY" \
    --image-project="$COORD_IMAGE_PROJECT" \
    --boot-disk-size=30GB \
    --tags=synapse \
    --metadata-from-file=startup-script="$SYNAPSE_SRC/deploy/coordinator-startup.sh" \
    --project="$PROJECT" 2>&1

  wait_for_ssh "$COORD_VM"
  cmd_deploy

  COORD_IP=$(get_coord_ip)
  log "=== COORDINATOR UP ==="
  log "IP: $COORD_IP"
  log "Prompt UI:  http://$COORD_IP:8080/"
  log "Node UI:    http://$COORD_IP:8080/node/index.html"
  log "Dashboard:  http://$COORD_IP:8080/ui/dashboard.html"
  log ""
  log "For Colab GPU nodes, use:"
  log "  COORDINATOR_URL=http://$COORD_IP:8080 node headless-node.js"
  log ""
  log "To stop billing: ./gcp.sh coord-down"
}

cmd_coord_down() {
  log "Deleting coordinator VM..."
  gcloud compute instances delete "$COORD_VM" --zone="$ZONE" --project="$PROJECT" --quiet 2>&1 || true
  log "Coordinator deleted."
}

# ---------- gpu node ----------

cmd_gpu_up() {
  ensure_firewall

  log "Creating GPU VM: $GPU_VM ($GPU_TYPE + T4) ~\$0.55/hr..."
  gcloud compute instances create "$GPU_VM" \
    --zone="$ZONE" \
    --machine-type="$GPU_TYPE" \
    --accelerator="$GPU_ACCEL" \
    --maintenance-policy=TERMINATE \
    --image-family="$GPU_IMAGE_FAMILY" \
    --image-project="$GPU_IMAGE_PROJECT" \
    --boot-disk-size=50GB \
    --tags=synapse \
    --metadata-from-file=startup-script="$SYNAPSE_SRC/deploy/gpu-node-startup.sh" \
    --project="$PROJECT" 2>&1

  wait_for_ssh "$GPU_VM"

  # Copy headless node script
  gcloud compute scp \
    "$SYNAPSE_SRC/deploy/headless-node.js" \
    "${GPU_VM}:/opt/synapse/" \
    --zone="$ZONE" --project="$PROJECT" --strict-host-key-checking=no 2>&1 || true

  GPU_IP=$(gcloud compute instances describe "$GPU_VM" --zone="$ZONE" \
    --format="get(networkInterfaces[0].accessConfigs[0].natIP)" --project="$PROJECT")
  COORD_IP=$(get_coord_ip 2>/dev/null || echo "<coordinator-ip>")

  log "=== GPU NODE UP ==="
  log "IP: $GPU_IP"
  log "To start nodes:"
  log "  ./gcp.sh ssh-gpu"
  log "  cd /opt/synapse && COORDINATOR_URL=http://$COORD_IP:8080 node headless-node.js"
  log ""
  log "To stop billing: ./gcp.sh gpu-down"
}

cmd_gpu_down() {
  log "Deleting GPU VM..."
  gcloud compute instances delete "$GPU_VM" --zone="$ZONE" --project="$PROJECT" --quiet 2>&1 || true
  log "GPU node deleted."
}

# ---------- deploy code ----------

cmd_deploy() {
  log "Deploying code to coordinator..."

  # Create the directory structure on the VM
  gcloud compute ssh "$COORD_VM" --zone="$ZONE" --project="$PROJECT" \
    --command="sudo mkdir -p /opt/synapse/synapse-src/model" \
    --strict-host-key-checking=no 2>&1

  # SCP each directory individually (avoids issues with --exclude)
  for dir in coordinator node protocol ui test deploy; do
    log "  Copying $dir/..."
    gcloud compute scp --recurse --compress \
      "$SYNAPSE_SRC/$dir" \
      "${COORD_VM}:/opt/synapse/synapse-src/" \
      --zone="$ZONE" --project="$PROJECT" --strict-host-key-checking=no 2>&1
  done

  # Copy individual files
  for f in package.json package-lock.json start.sh; do
    if [ -f "$SYNAPSE_SRC/$f" ]; then
      gcloud compute scp --compress \
        "$SYNAPSE_SRC/$f" \
        "${COORD_VM}:/opt/synapse/synapse-src/" \
        --zone="$ZONE" --project="$PROJECT" --strict-host-key-checking=no 2>&1
    fi
  done

  # Copy model splitter (not the binary shards)
  gcloud compute scp --compress \
    "$SYNAPSE_SRC/model/split.py" \
    "${COORD_VM}:/opt/synapse/synapse-src/model/" \
    --zone="$ZONE" --project="$PROJECT" --strict-host-key-checking=no 2>&1

  log "Installing deps and starting coordinator..."
  gcloud compute ssh "$COORD_VM" --zone="$ZONE" --project="$PROJECT" \
    --strict-host-key-checking=no \
    --command="cd /opt/synapse/synapse-src && npm install --production 2>&1 | tail -3" 2>&1

  log "Running validation tests..."
  gcloud compute ssh "$COORD_VM" --zone="$ZONE" --project="$PROJECT" \
    --strict-host-key-checking=no \
    --command="cd /opt/synapse/synapse-src && npm test 2>&1 | tail -8" 2>&1

  log "Starting coordinator..."
  gcloud compute ssh "$COORD_VM" --zone="$ZONE" --project="$PROJECT" \
    --strict-host-key-checking=no \
    --command="cd /opt/synapse/synapse-src && pkill -f 'node coordinator' 2>/dev/null; PORT=8080 nohup node coordinator/index.js > /tmp/synapse.log 2>&1 &" 2>&1

  sleep 3
  gcloud compute ssh "$COORD_VM" --zone="$ZONE" --project="$PROJECT" \
    --strict-host-key-checking=no \
    --command="cat /tmp/synapse.log && curl -s http://localhost:8080/api/topology" 2>&1
}

# ---------- utility commands ----------

cmd_up() {
  cmd_coord_up
  echo ""
  cmd_gpu_up
}

cmd_down() {
  log "Tearing down all Synapse VMs..."
  cmd_coord_down
  cmd_gpu_down
  log "=== All VMs deleted. Billing stopped. ==="
}

cmd_status() {
  log "VM Status:"
  gcloud compute instances list --filter="name~synapse" --project="$PROJECT" \
    --format="table(name,zone,machineType.basename(),status,networkInterfaces[0].accessConfigs[0].natIP)" 2>&1

  echo ""
  log "Firewall:"
  gcloud compute firewall-rules list --filter="name~synapse" --project="$PROJECT" \
    --format="table(name,allowed,targetTags)" 2>&1
}

cmd_ip() {
  get_coord_ip
}

cmd_ssh_coord() {
  gcloud compute ssh "$COORD_VM" --zone="$ZONE" --project="$PROJECT" --strict-host-key-checking=no
}

cmd_ssh_gpu() {
  gcloud compute ssh "$GPU_VM" --zone="$ZONE" --project="$PROJECT" --strict-host-key-checking=no
}

cmd_logs() {
  gcloud compute ssh "$COORD_VM" --zone="$ZONE" --project="$PROJECT" \
    --strict-host-key-checking=no \
    "sudo tail -f /var/log/synapse-coordinator.log /var/log/synapse-startup.log 2>/dev/null"
}

# ---------- main ----------

case "${1:-}" in
  coord-up)   cmd_coord_up ;;
  coord-down) cmd_coord_down ;;
  gpu-up)     cmd_gpu_up ;;
  gpu-down)   cmd_gpu_down ;;
  up)         cmd_up ;;
  down)       cmd_down ;;
  status)     cmd_status ;;
  deploy)     cmd_deploy ;;
  ssh-coord)  cmd_ssh_coord ;;
  ssh-gpu)    cmd_ssh_gpu ;;
  ip)         cmd_ip ;;
  logs)       cmd_logs ;;
  *)
    echo "Synapse GCP Deploy — hybrid coordinator + free Colab GPU nodes"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "  Coordinator (~\$0.03/hr):"
    echo "    coord-up    Create coordinator VM, deploy code, start server"
    echo "    coord-down  Delete coordinator VM"
    echo "    deploy      Push latest code to coordinator"
    echo "    logs        Tail coordinator logs"
    echo "    ssh-coord   SSH into coordinator"
    echo "    ip          Print coordinator external IP"
    echo ""
    echo "  GPU Node (~\$0.55/hr) — or use free Colab instead:"
    echo "    gpu-up      Create N1+T4 GPU VM"
    echo "    gpu-down    Delete GPU VM"
    echo "    ssh-gpu     SSH into GPU node"
    echo ""
    echo "  Both:"
    echo "    up          Create coordinator + GPU VMs"
    echo "    down        Delete all VMs (stop billing)"
    echo "    status      Show VM status"
    exit 1
    ;;
esac
