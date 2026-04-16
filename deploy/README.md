# Synapse two-fleet deploy

## Topology

- **prod** — `synapse.webmind.sh` (port 8080 on coord VM)
  - Code: `/opt/synapse` on coord VM
  - Service: `synapse-coordinator.service`
  - Updated only by explicit operator command (`sudo /opt/deploy-prod.sh [ref]`)
  - Intended for demo-ready use. Stable fleet.

- **alpha** — `lab.webmind.sh` (port 8081 on coord VM, via caddy HTTPS)
  - Code: `/opt/synapse-alpha` on coord VM (symlinks to prod for shards + node_modules — zero extra disk)
  - Service: `synapse-coordinator-alpha.service`
  - Auto-pulls `main` every 3 minutes via `synapse-alpha-deploy.timer`
  - Hot-reload broadcast to all alpha-connected nodes on each update

## Operator commands

```bash
# Promote a commit (or tag, or branch) from main to prod:
gcloud compute ssh synapse-coordinator --zone=us-west1-b --command='sudo /opt/deploy-prod.sh <ref>'

# Force an immediate alpha deploy (don't wait for the 3-min timer):
gcloud compute ssh synapse-coordinator --zone=us-west1-b --command='sudo /opt/deploy-alpha.sh'

# Watch alpha deploys:
gcloud compute ssh synapse-coordinator --zone=us-west1-b --command='sudo journalctl -u synapse-alpha-deploy.service -f'
```

## Nodes

- Open `https://synapse.webmind.sh` → joins prod fleet (dedicated phones + laptop)
- Open `https://lab.webmind.sh` → joins alpha fleet (test devices)
- Both served by the same coord VM on different ports; different shards dir symlink-shared.

## HTTPS

- Caddy on coord VM terminates TLS for lab.webmind.sh via Let's Encrypt.
- synapse.webmind.sh HTTPS handled by its existing port-8443 cert.
