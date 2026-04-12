# Synapse

## Overview

Distributed Browser LLM Network — runs transformer models across multiple browser tabs using WebGPU.

## Architecture

- **Synapse Coordinator** (`synapse-src/`) — Node.js WebSocket server that assigns model shards to browser nodes, routes activation tensors between nodes, and serves the web UI
- **API Server** (`artifacts/api-server/`) — Express 5 backend (TypeScript) for any additional REST endpoints
- **Model Splitter** (`synapse-src/model/split.py`) — Python script to download and split HuggingFace models into shards

## Stack

- **Coordinator**: Node.js ESM, `ws` WebSocket library
- **Browser Nodes**: WebGPU, vanilla JS
- **API Server**: Express 5, TypeScript, Drizzle ORM, PostgreSQL
- **Monorepo**: pnpm workspaces, Node.js 24

## How It Works

1. Open `/node/index.html` in 2+ browser tabs — each tab loads a model shard using WebGPU
2. Tabs connect to the coordinator via WebSocket and download their assigned shard
3. Once all shards are loaded, the pipeline is ready
4. Send a prompt at `/` — tokens flow through the distributed pipeline
5. Watch the network at `/ui/dashboard.html`

## URLs

- `/` — Prompt UI
- `/node/index.html` — Compute node (open multiple tabs)
- `/ui/dashboard.html` — Network dashboard

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas
- `pnpm --filter @workspace/db run push` — push DB schema changes

## Synapse Source

Cloned from `https://github.com/tejasphatak/Synapse` branch `claude/synapse-poc-phase-1-P5z9o` into `synapse-src/`.
