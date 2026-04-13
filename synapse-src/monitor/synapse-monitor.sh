#!/bin/bash
# synapse-monitor.sh — continuous monitoring, alerts, and perf data capture
# Cron: */15 * * * * (every 15 minutes)
set -uo pipefail

LOG="$HOME/.claude/monitor.log"
DATA_DIR="$HOME/Synapse/synapse-src/monitor/data"
ALERT_FILE="$HOME/.claude/monitor-alerts"
COORDINATOR_URL="http://34.82.32.123:8080"
DISCORD_WEBHOOK_FILE="$HOME/.claude/discord-webhook-url"

mkdir -p "$DATA_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
alert() {
  local msg="$1"
  log "ALERT: $msg"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg" >> "$ALERT_FILE"
  # Post to Discord if webhook exists
  if [ -f "$DISCORD_WEBHOOK_FILE" ]; then
    local webhook
    webhook=$(cat "$DISCORD_WEBHOOK_FILE")
    curl -s -X POST "$webhook" -H "Content-Type: application/json" \
      -d "{\"content\":\"**MONITOR ALERT:** $msg\"}" > /dev/null 2>&1
  fi
}

# ─── 1. GCP Instance Count & Cost ───────────────────────────────────

log "=== MONITOR CHECK ==="

VM_LIST=$(gcloud compute instances list --format="csv[no-heading](name,zone,status,machineType)" 2>/dev/null)
VM_COUNT=$(echo "$VM_LIST" | grep -c "RUNNING" || true)
# GPU VMs: machine types that indicate accelerator-attached instances
GPU_VMS=$(echo "$VM_LIST" | grep "RUNNING" | grep -cE "n1-|a2-|g2-" || true)

log "Running VMs: $VM_COUNT (GPU VMs: $GPU_VMS)"

# Cost estimate using python (bc not available on this VM)
read -r COST_PER_HOUR DAILY_COST <<< $(python3 -c "
costs = {'e2-medium': 0.03, 'n1-standard-1': 0.38, 'n1-standard-4': 0.60}
total = 0
for line in '''$VM_LIST'''.strip().split('\n'):
    if not line or ',RUNNING' not in line: continue
    parts = line.split(',')
    mtype = parts[-1].split('/')[-1] if '/' in parts[-1] else parts[-1]
    for k, v in costs.items():
        if k in mtype:
            total += v
            break
    else:
        total += 0.10
print(f'{total:.2f} {total*24:.2f}')
" 2>/dev/null || echo "0.00 0.00")

log "Estimated cost: \$$COST_PER_HOUR/hr (\$$DAILY_COST/day)"

# Alert if GPU VMs are running (they're expensive)
if [ "$GPU_VMS" -gt 0 ]; then
  alert "GPU VM(s) running! Count: $GPU_VMS. Estimated \$$COST_PER_HOUR/hr. Shut down if idle."
fi

# Alert if more than 2 VMs total (unexpected)
if [ "$VM_COUNT" -gt 2 ]; then
  alert "Unexpected VM count: $VM_COUNT running. Expected max 2 (home + coordinator)."
fi

# Save to time series
echo "$(date +%s),$VM_COUNT,$GPU_VMS,$COST_PER_HOUR" >> "$DATA_DIR/vm-costs.csv"

# ─── 2. Coordinator Health ──────────────────────────────────────────

COORD_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$COORDINATOR_URL/api/topology" 2>/dev/null || echo "000")

if [ "$COORD_STATUS" = "200" ]; then
  log "Coordinator: UP (HTTP $COORD_STATUS)"

  # Grab topology — how many nodes connected?
  TOPOLOGY=$(curl -s --connect-timeout 5 "$COORDINATOR_URL/api/topology" 2>/dev/null)
  NODE_COUNT=$(echo "$TOPOLOGY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('nodes',[])))" 2>/dev/null || echo "?")
  READY_NODES=$(echo "$TOPOLOGY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(sum(1 for n in d.get('nodes',[]) if n.get('status')=='ready'))" 2>/dev/null || echo "?")
  log "Nodes: $NODE_COUNT total, $READY_NODES ready"

  echo "$(date +%s),up,$NODE_COUNT,$READY_NODES" >> "$DATA_DIR/coordinator-health.csv"

  # Grab perf data if available
  PERF=$(curl -s --connect-timeout 5 "$COORDINATOR_URL/api/perf" 2>/dev/null)
  if [ -n "$PERF" ] && [ "$PERF" != "{}" ]; then
    echo "$(date +%s),$PERF" >> "$DATA_DIR/perf-snapshots.jsonl"
    log "Perf data captured"
  fi

  # Grab speculation stats from logs
  SPEC_LOGS=$(curl -s --connect-timeout 5 "$COORDINATOR_URL/api/logs?event=speculation_accepted&since=$(date -d '15 minutes ago' +%s000 2>/dev/null || echo 0)" 2>/dev/null)
  SPEC_ACCEPTED=$(echo "$SPEC_LOGS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('logs',[])))" 2>/dev/null || echo "0")

  SPEC_REJ_LOGS=$(curl -s --connect-timeout 5 "$COORDINATOR_URL/api/logs?event=speculation_rejected&since=$(date -d '15 minutes ago' +%s000 2>/dev/null || echo 0)" 2>/dev/null)
  SPEC_REJECTED=$(echo "$SPEC_REJ_LOGS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('logs',[])))" 2>/dev/null || echo "0")

  if [ "$SPEC_ACCEPTED" -gt 0 ] || [ "$SPEC_REJECTED" -gt 0 ]; then
    TOTAL=$((SPEC_ACCEPTED + SPEC_REJECTED))
    HIT_RATE=$(python3 -c "print(f'{$SPEC_ACCEPTED/$TOTAL:.4f}')" 2>/dev/null || echo "?")
    log "Speculation: $SPEC_ACCEPTED accepted, $SPEC_REJECTED rejected (hit rate: $HIT_RATE)"
    echo "$(date +%s),$SPEC_ACCEPTED,$SPEC_REJECTED,$HIT_RATE" >> "$DATA_DIR/speculation-stats.csv"
  fi

else
  log "Coordinator: DOWN (HTTP $COORD_STATUS)"
  echo "$(date +%s),down,0,0" >> "$DATA_DIR/coordinator-health.csv"
  # Don't alert every 15 min — only if it was up last check
  LAST_STATUS=$(tail -2 "$DATA_DIR/coordinator-health.csv" 2>/dev/null | head -1 | cut -d',' -f2)
  if [ "$LAST_STATUS" = "up" ]; then
    alert "Coordinator went DOWN! Was up at last check."
  fi
fi

# ─── 3. Disk & Resource Check ──────────────────────────────────────

DISK_PCT=$(df -h / | awk 'NR==2{print $5}' | tr -d '%')
MEM_PCT=$(free | awk '/Mem:/{printf "%.0f", $3/$2*100}')
log "Resources: disk ${DISK_PCT}%, memory ${MEM_PCT}%"

if [ "$DISK_PCT" -gt 85 ]; then
  alert "Disk usage at ${DISK_PCT}%!"
fi

log "=== MONITOR DONE ==="
