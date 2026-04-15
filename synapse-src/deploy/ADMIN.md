# Synapse Coordinator — admin endpoints

Admin-only HTTP endpoints for operators running a Synapse coordinator.
All endpoints require `X-Admin-Token` header matching the `NEX_ADMIN_TOKEN`
environment variable on the coordinator process. If `NEX_ADMIN_TOKEN` is
unset, admin endpoints return HTTP 503 (fail closed).

## Setup

Generate a strong random token:
```bash
ADMIN_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
```

Set on the coordinator via a systemd drop-in (recommended for production):
```bash
sudo mkdir -p /etc/systemd/system/synapse-coordinator.service.d
sudo tee /etc/systemd/system/synapse-coordinator.service.d/admin-token.conf > /dev/null <<EOF
[Service]
Environment="NEX_ADMIN_TOKEN=$ADMIN_TOKEN"
EOF
sudo chmod 600 /etc/systemd/system/synapse-coordinator.service.d/admin-token.conf
sudo systemctl daemon-reload
sudo systemctl restart synapse-coordinator
```

Store the token locally (not in the repo) — e.g. `~/.synapse-admin-token`
with `chmod 600`. Anyone holding the token can reassign shards.

## `POST /api/assign`

Explicitly assign a shard to a currently-connected node, or release a
shard from a node.

### Assign a shard
```bash
curl -X POST https://<coord>/api/assign \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -d '{"nodeId":"node-abc12345","shardId":0}'
```

Preconditions:
- `nodeId` must be currently connected to the coordinator.
- If the node already holds a different shard, the call fails with
  `"node already holds shard N; unassign first"`. (Use the unassign form
  below to free it.)

On success, the coordinator:
1. Records the shard assignment in the topology.
2. Sends an `ASSIGN_SHARD` WebSocket message to the node with shard URL.
3. Broadcasts the updated topology to all prompt clients.
4. Returns `{"ok":true,"action":"assign","nodeId":"...","shardId":N,"topology":{...}}`.

The node will download the shard, load it onto GPU, and report `NODE_READY`.
Pipeline becomes active once all expected shards are covered by ready nodes.

### Release a shard
```bash
curl -X POST https://<coord>/api/assign \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: $ADMIN_TOKEN" \
  -d '{"nodeId":"node-abc12345","unassign":true}'
```

On success:
1. The node's `shardId`, `layerStart`, `layerEnd` are cleared.
2. Status returns to `"connected"`.
3. Pipeline rebuilds (this node is removed from the active pipeline).
4. Topology is broadcast.

The node keeps its WebSocket connection — it can be reassigned later
without reconnecting. If no other nodes hold that shard, the pipeline
is incomplete until a replacement is assigned.

## Use cases

**Promote a late-arriving better node.** When a desktop GPU (Nvidia, Intel)
joins after mobile nodes already hold shards, auto-assign does nothing
because no shards are free. Unassign a mobile holder, assign the shard
to the desktop node. Phones can continue connected as redundancy.

**Rebalance after drift.** If a node's telemetry (perf logs) shows it's
bottlenecking the pipeline, unassign it and assign the shard to a
better-suited idle node.

**Controlled demo.** For reproducible benchmarks, explicitly assign the
same node IDs to the same shards every run.

## What this endpoint does NOT do

- **Does not disconnect nodes.** Unassigned nodes remain connected as
  idle redundancy. If you want a node fully removed, close its WebSocket
  (e.g., have it reload the node page or kill the browser tab).
- **Does not migrate state.** When a shard moves to a new node, the new
  node loads the shard from scratch — existing KV caches on the old
  holder are dropped. For in-flight generations, expect an `INFER_ERROR`
  during the handoff.
- **Does not check capability.** You can assign a shard to a tiny
  underpowered node and it will fail at runtime. Check
  `GET /api/topology` for node capabilities before assigning:

```bash
curl -s https://<coord>/api/topology | jq '.nodes[] | {nodeId, shardId, status, capabilities}'
```

## Security model

The admin token is the sole protection. It's not part of the WebSocket
protocol or node-to-coord auth. Treat it like a root password:
- Don't commit it to git.
- Don't log it.
- Rotate by changing `NEX_ADMIN_TOKEN` and reloading the systemd unit.
- Anyone with the token can reassign shards and disrupt live inference.

If the threat model requires finer granularity (per-shard permissions,
audit log, etc.), layer that on top — the endpoint is minimal
on purpose.

## See also

- `coordinator/index.js:244` — `/api/assign` implementation
- `coordinator/topology.js:41` — `assignShard()` function
- `deploy/cli-infer.mjs` — client CLI (does not require admin token)
- `deploy/bench.mjs` — throughput benchmark (does not require admin token)
