/**
 * Synapse Binary Wire Protocol v1
 *
 * Binary message format for high-performance activation transfer.
 * Control messages (JOIN, PING, etc.) remain JSON — this only optimizes
 * the hot path (ACTIVATION, OUTPUT).
 *
 * Wire format (24-byte header + payload):
 *   Bytes 0-3:   Magic 0x53594E31 ("SYN1")
 *   Byte 4:      Message type
 *   Byte 5:      Flags [quant:2 | compressed:1 | predicted:1 | early_exit:1 | delta:1 | reserved:2]
 *   Bytes 6-7:   Sequence position (uint16)
 *   Bytes 8-11:  Request ID (uint32)
 *   Bytes 12-15: Payload size in bytes (uint32)
 *   Bytes 16-19: Shape dim0 (uint32)
 *   Bytes 20-23: Shape dim1 (uint32)
 *   Bytes 24+:   Raw tensor data
 */

// ─── Constants ───────────────────────────────────────────────────

export const MAGIC = 0x53594E31; // "SYN1"
export const HEADER_SIZE = 24;

export const BinaryMsgType = {
  ACTIVATION: 0x01,
  OUTPUT: 0x02,
  KV_APPEND: 0x03,
  PREDICT: 0x04,
  EARLY_EXIT: 0x05,
};

// Flag bit positions
export const Flags = {
  QUANT_MASK: 0b00000011,   // bits 0-1: quantization (0=none, 1=int8, 2=int4)
  COMPRESSED: 0b00000100,   // bit 2: compressed
  PREDICTED: 0b00001000,    // bit 3: predicted activation
  EARLY_EXIT: 0b00010000,   // bit 4: early exit signal
  DELTA: 0b00100000,        // bit 5: delta-encoded
};

export const QuantMode = {
  NONE: 0,
  INT8: 1,
  INT4: 2,
};

// ─── Quantization Helpers ────────────────────────────────────────

/**
 * Get the quantization mode from flags byte.
 */
export function getQuantMode(flags) {
  return flags & Flags.QUANT_MASK;
}

/**
 * Set the quantization mode in flags byte.
 */
export function setQuantFlags(flags, quantMode) {
  return (flags & ~Flags.QUANT_MASK) | (quantMode & Flags.QUANT_MASK);
}

// ─── Request ID Mapping ──────────────────────────────────────────

const _requestIdMap = new Map();  // string -> uint32
const _reverseIdMap = new Map();  // uint32 -> string
let _nextId = 1;

/**
 * Convert a string request ID to a uint32 for the binary header.
 * Maintains a bidirectional mapping.
 */
export function requestIdToUint32(stringId) {
  if (_requestIdMap.has(stringId)) return _requestIdMap.get(stringId);
  const id = _nextId++;
  _requestIdMap.set(stringId, id);
  _reverseIdMap.set(id, stringId);
  return id;
}

/**
 * Convert a uint32 request ID back to the original string.
 */
export function uint32ToRequestId(numId) {
  return _reverseIdMap.get(numId) || `req-${numId}`;
}

/**
 * Register a known string <-> uint32 mapping (used by coordinator).
 */
export function registerRequestId(stringId, numId) {
  _requestIdMap.set(stringId, numId);
  _reverseIdMap.set(numId, stringId);
}

// ─── Encode ──────────────────────────────────────────────────────

/**
 * Encode a binary message with the 24-byte header + raw tensor payload.
 *
 * @param {number} type - BinaryMsgType value
 * @param {number} flags - Bitfield flags
 * @param {number} seqPos - Sequence position (uint16)
 * @param {number} requestId - Numeric request ID (uint32)
 * @param {number[]} shape - [dim0, dim1]
 * @param {ArrayBuffer|Uint8Array|Int8Array} tensorData - Raw tensor bytes
 * @returns {ArrayBuffer}
 */
export function encodeBinaryMessage(type, flags, seqPos, requestId, shape, tensorData) {
  const payload = tensorData instanceof ArrayBuffer ? tensorData : tensorData.buffer.slice(
    tensorData.byteOffset, tensorData.byteOffset + tensorData.byteLength
  );
  const payloadSize = payload.byteLength;
  const totalSize = HEADER_SIZE + payloadSize;

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);

  // Header
  view.setUint32(0, MAGIC, false);              // big-endian magic
  view.setUint8(4, type);
  view.setUint8(5, flags);
  view.setUint16(6, seqPos, true);               // little-endian
  view.setUint32(8, requestId, true);
  view.setUint32(12, payloadSize, true);
  view.setUint32(16, shape[0] || 0, true);
  view.setUint32(20, shape[1] || 0, true);

  // Payload
  new Uint8Array(buffer, HEADER_SIZE).set(new Uint8Array(payload));

  return buffer;
}

/**
 * Encode an OUTPUT message (token IDs, no tensor).
 */
export function encodeBinaryOutput(requestId, tokenIds, seqPos = 0) {
  const tokenData = new Uint32Array(tokenIds);
  return encodeBinaryMessage(
    BinaryMsgType.OUTPUT,
    0,
    seqPos,
    requestId,
    [tokenIds.length, 1],
    tokenData.buffer
  );
}

// ─── Decode ──────────────────────────────────────────────────────

/**
 * Decode a binary message. Returns header fields + a view into the payload
 * (zero-copy — the payload references the original buffer).
 *
 * @param {ArrayBuffer|Buffer} data - Raw binary frame
 * @returns {{ type, flags, seqPos, requestId, payloadSize, shape, payload }}
 */
export function decodeBinaryMessage(data) {
  // Normalize Node.js Buffer to ArrayBuffer
  const arrayBuf = data instanceof ArrayBuffer ? data : data.buffer.slice(
    data.byteOffset, data.byteOffset + data.byteLength
  );

  if (arrayBuf.byteLength < HEADER_SIZE) {
    throw new Error(`Binary message too short: ${arrayBuf.byteLength} bytes`);
  }

  const view = new DataView(arrayBuf);
  const magic = view.getUint32(0, false);
  if (magic !== MAGIC) {
    throw new Error(`Invalid magic: 0x${magic.toString(16)} (expected 0x${MAGIC.toString(16)})`);
  }

  const type = view.getUint8(4);
  const flags = view.getUint8(5);
  const seqPos = view.getUint16(6, true);
  const requestId = view.getUint32(8, true);
  const payloadSize = view.getUint32(12, true);
  const shape = [view.getUint32(16, true), view.getUint32(20, true)];

  // Payload is a zero-copy view into the original buffer
  const payload = new Uint8Array(arrayBuf, HEADER_SIZE, payloadSize);

  return { type, flags, seqPos, requestId, payloadSize, shape, payload };
}

/**
 * Decode OUTPUT message tokens from payload.
 */
export function decodeOutputTokens(payload) {
  return Array.from(new Uint32Array(payload.buffer, payload.byteOffset, payload.byteLength / 4));
}

// ─── Detection ───────────────────────────────────────────────────

/**
 * Check if a WebSocket message is a binary frame (vs JSON string).
 * Works in both browser (ArrayBuffer) and Node.js (Buffer) contexts.
 */
export function isBinaryMessage(data) {
  // Check for SYN1 magic bytes (0x53594E31) — not just buffer type,
  // because Node.js WebSocket delivers all messages as Buffers.
  if (data instanceof ArrayBuffer && data.byteLength >= 4) {
    return new DataView(data).getUint32(0, false) === MAGIC;
  }
  if (typeof Buffer !== "undefined" && Buffer.isBuffer(data) && data.length >= 4) {
    return data.readUInt32BE(0) === MAGIC;
  }
  return false;
}

/**
 * Extract the string request ID from a binary message without full decode.
 * Useful for coordinator routing where only the request ID is needed.
 */
export function peekRequestId(data) {
  const arrayBuf = data instanceof ArrayBuffer ? data : data.buffer.slice(
    data.byteOffset, data.byteOffset + data.byteLength
  );
  const view = new DataView(arrayBuf);
  return view.getUint32(8, true);
}

/**
 * Extract message type from a binary message without full decode.
 */
export function peekMessageType(data) {
  const arrayBuf = data instanceof ArrayBuffer ? data : data.buffer.slice(
    data.byteOffset, data.byteOffset + data.byteLength
  );
  const view = new DataView(arrayBuf);
  return view.getUint8(4);
}

/**
 * Extract flags byte from a binary message without full decode.
 */
export function peekFlags(data) {
  const arrayBuf = data instanceof ArrayBuffer ? data : data.buffer.slice(
    data.byteOffset, data.byteOffset + data.byteLength
  );
  const view = new DataView(arrayBuf);
  return view.getUint8(5);
}
