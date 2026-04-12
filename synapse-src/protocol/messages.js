/**
 * Synapse Protocol — Message Schema
 *
 * All messages between nodes and coordinator are JSON over WebSocket.
 * This module defines message types, constructors, and validation.
 */

// ─── Message Types ────────────────────────────────────────────────

export const MessageType = {
  // Node → Coordinator
  JOIN: "JOIN",
  PING: "PING",

  // Coordinator → Node
  ASSIGN_SHARD: "ASSIGN_SHARD",
  PONG: "PONG",
  TOPOLOGY_UPDATE: "TOPOLOGY_UPDATE",
  INFERENCE_REQUEST: "INFERENCE_REQUEST",

  // Node ↔ Coordinator (routed)
  ACTIVATION: "ACTIVATION",

  // Node → Coordinator (shard loaded)
  NODE_READY: "NODE_READY",

  // Node → Coordinator → Dashboard
  OUTPUT: "OUTPUT",

  // Error
  ERROR: "ERROR",
};

// ─── Message Constructors ─────────────────────────────────────────

/**
 * Node joins the network, advertising its capabilities.
 */
export function createJoinMessage(nodeId, capabilities = {}) {
  return {
    type: MessageType.JOIN,
    nodeId,
    capabilities: {
      webgpu: capabilities.webgpu ?? false,
      maxLayers: capabilities.maxLayers ?? 6,
      ...capabilities,
    },
    timestamp: Date.now(),
  };
}

/**
 * Coordinator assigns a model shard to a node.
 */
export function createAssignShardMessage(shardId, layerStart, layerEnd, shardUrl, sharedUrl) {
  return {
    type: MessageType.ASSIGN_SHARD,
    shardId,
    layerStart,
    layerEnd,
    shardUrl,
    sharedUrl,
  };
}

/**
 * Activation tensor being routed between nodes.
 */
export function createActivationMessage(fromNode, toNode, layer, requestId, data, shape) {
  return {
    type: MessageType.ACTIVATION,
    fromNode,
    toNode,
    layer,
    requestId,
    tensor: {
      shape,
      dtype: "float32",
      data, // base64 encoded Float32Array
    },
    timestamp: Date.now(),
  };
}

/**
 * Final output tokens from the last node in the pipeline.
 */
export function createOutputMessage(requestId, tokens, text) {
  return {
    type: MessageType.OUTPUT,
    requestId,
    tokens,
    text,
    timestamp: Date.now(),
  };
}

/**
 * Topology update broadcast to all connected clients (nodes + dashboard).
 */
export function createTopologyUpdateMessage(nodes, pipeline) {
  return {
    type: MessageType.TOPOLOGY_UPDATE,
    nodes, // array of { nodeId, shardId, layerStart, layerEnd, status }
    pipeline, // ordered array of nodeIds representing the inference pipeline
    timestamp: Date.now(),
  };
}

/**
 * Coordinator sends an inference request to the first node in the pipeline.
 */
export function createInferenceRequestMessage(requestId, tokenIds) {
  return {
    type: MessageType.INFERENCE_REQUEST,
    requestId,
    tokenIds,
    timestamp: Date.now(),
  };
}

/**
 * Heartbeat messages.
 */
export function createPingMessage() {
  return { type: MessageType.PING, timestamp: Date.now() };
}

export function createPongMessage() {
  return { type: MessageType.PONG, timestamp: Date.now() };
}

/**
 * Node reports that its shard is loaded and ready for inference.
 */
export function createNodeReadyMessage(nodeId, shardId) {
  return {
    type: MessageType.NODE_READY,
    nodeId,
    shardId,
    timestamp: Date.now(),
  };
}

/**
 * Error message.
 */
export function createErrorMessage(code, message, details = null) {
  return {
    type: MessageType.ERROR,
    code,
    message,
    details,
    timestamp: Date.now(),
  };
}

// ─── Validation ───────────────────────────────────────────────────

const REQUIRED_FIELDS = {
  [MessageType.JOIN]: ["nodeId", "capabilities"],
  [MessageType.ASSIGN_SHARD]: ["shardId", "layerStart", "layerEnd", "shardUrl", "sharedUrl"],
  [MessageType.ACTIVATION]: ["fromNode", "toNode", "layer", "requestId", "tensor"],
  [MessageType.OUTPUT]: ["requestId", "tokens"],
  [MessageType.TOPOLOGY_UPDATE]: ["nodes", "pipeline"],
  [MessageType.INFERENCE_REQUEST]: ["requestId", "tokenIds"],
  [MessageType.NODE_READY]: ["nodeId", "shardId"],
  [MessageType.PING]: [],
  [MessageType.PONG]: [],
  [MessageType.ERROR]: ["code", "message"],
};

/**
 * Validate a parsed message object. Returns { valid, error }.
 */
export function validateMessage(msg) {
  if (!msg || typeof msg !== "object") {
    return { valid: false, error: "Message must be a non-null object" };
  }

  if (!msg.type || !REQUIRED_FIELDS[msg.type]) {
    return { valid: false, error: `Unknown message type: ${msg.type}` };
  }

  const required = REQUIRED_FIELDS[msg.type];
  for (const field of required) {
    if (msg[field] === undefined || msg[field] === null) {
      return { valid: false, error: `Missing required field '${field}' for ${msg.type}` };
    }
  }

  // Type-specific validation
  if (msg.type === MessageType.ACTIVATION && msg.tensor) {
    if (!msg.tensor.shape || !Array.isArray(msg.tensor.shape)) {
      return { valid: false, error: "ACTIVATION tensor must have a shape array" };
    }
    if (!msg.tensor.data || typeof msg.tensor.data !== "string") {
      return { valid: false, error: "ACTIVATION tensor data must be a base64 string" };
    }
  }

  if (msg.type === MessageType.OUTPUT) {
    if (!Array.isArray(msg.tokens)) {
      return { valid: false, error: "OUTPUT tokens must be an array" };
    }
  }

  return { valid: true, error: null };
}

/**
 * Safely parse a WebSocket message string into a validated message object.
 */
export function parseMessage(raw) {
  try {
    const msg = JSON.parse(raw);
    const validation = validateMessage(msg);
    if (!validation.valid) {
      return { msg: null, error: validation.error };
    }
    return { msg, error: null };
  } catch (e) {
    return { msg: null, error: `Invalid JSON: ${e.message}` };
  }
}
