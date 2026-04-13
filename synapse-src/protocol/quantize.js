/**
 * Int8 Activation Quantization for Synapse Wire Protocol
 *
 * Quantizes float32 activation tensors to int8 before network transfer (4x reduction),
 * then dequantizes back to float32 on the receiving side.
 *
 * Uses per-tensor symmetric quantization:
 *   scale = max(|tensor|) / 127
 *   quantized[i] = clamp(round(tensor[i] / scale), -127, 127)
 *   dequantized[i] = quantized[i] * scale
 *
 * Typical error: <0.5% relative for transformer activations.
 */

/**
 * Quantize a Float32Array to Int8Array with per-tensor scale.
 *
 * @param {Float32Array} float32Data - Input activation tensor
 * @returns {{ data: Int8Array, scale: number }}
 */
export function quantizeInt8(float32Data) {
  // Find absolute maximum for scale
  let absMax = 0;
  for (let i = 0; i < float32Data.length; i++) {
    const abs = Math.abs(float32Data[i]);
    if (abs > absMax) absMax = abs;
  }

  // Avoid division by zero for all-zero tensors
  const scale = absMax > 0 ? absMax / 127 : 1;
  const invScale = 1 / scale;

  // Quantize
  const int8Data = new Int8Array(float32Data.length);
  for (let i = 0; i < float32Data.length; i++) {
    let val = Math.round(float32Data[i] * invScale);
    // Clamp to [-127, 127] (reserve -128 to keep symmetric range)
    if (val > 127) val = 127;
    else if (val < -127) val = -127;
    int8Data[i] = val;
  }

  return { data: int8Data, scale };
}

/**
 * Dequantize an Int8Array back to Float32Array.
 *
 * @param {Int8Array} int8Data - Quantized tensor
 * @param {number} scale - Per-tensor scale factor
 * @returns {Float32Array}
 */
export function dequantizeInt8(int8Data, scale) {
  const float32Data = new Float32Array(int8Data.length);
  for (let i = 0; i < int8Data.length; i++) {
    float32Data[i] = int8Data[i] * scale;
  }
  return float32Data;
}

/**
 * Pack quantized int8 data + scale into a single ArrayBuffer for wire transfer.
 * Layout: [int8_data...] [float32_scale] (scale appended as last 4 bytes)
 *
 * @param {Int8Array} int8Data
 * @param {number} scale
 * @returns {ArrayBuffer}
 */
export function packQuantized(int8Data, scale) {
  const totalBytes = int8Data.length + 4; // int8 data + 4-byte scale
  const buffer = new ArrayBuffer(totalBytes);

  // Copy int8 data
  new Int8Array(buffer, 0, int8Data.length).set(int8Data);

  // Append scale as float32 at the end
  new DataView(buffer).setFloat32(int8Data.length, scale, true); // little-endian

  return buffer;
}

/**
 * Unpack quantized data from wire format back to int8 + scale.
 *
 * @param {Uint8Array|ArrayBuffer} packed - Wire format data
 * @returns {{ int8Data: Int8Array, scale: number }}
 */
export function unpackQuantized(packed) {
  const buf = packed instanceof ArrayBuffer ? packed
    : packed.buffer.slice(packed.byteOffset, packed.byteOffset + packed.byteLength);

  const int8Len = buf.byteLength - 4;
  const int8Data = new Int8Array(buf, 0, int8Len);
  const scale = new DataView(buf).getFloat32(int8Len, true);

  return { int8Data, scale };
}

// ─── Delta Encoding ─────────────────────────────────────────────

/**
 * Compute delta between current and previous activation tensors.
 * delta[i] = current[i] - previous[i]
 *
 * Consecutive autoregressive activations typically differ by ~5-10%,
 * making deltas highly compressible via int8 quantization.
 *
 * @param {Float32Array} current
 * @param {Float32Array} previous
 * @returns {Float32Array}
 */
export function computeDelta(current, previous) {
  const delta = new Float32Array(current.length);
  for (let i = 0; i < current.length; i++) {
    delta[i] = current[i] - previous[i];
  }
  return delta;
}

/**
 * Reconstruct current activation from delta + previous.
 * current[i] = previous[i] + delta[i]
 *
 * @param {Float32Array} delta
 * @param {Float32Array} previous
 * @returns {Float32Array}
 */
export function applyDelta(delta, previous) {
  const current = new Float32Array(delta.length);
  for (let i = 0; i < delta.length; i++) {
    current[i] = previous[i] + delta[i];
  }
  return current;
}

/**
 * Measure delta sparsity — fraction of values below a threshold.
 * Useful for telemetry: high sparsity means delta encoding is effective.
 *
 * @param {Float32Array} delta
 * @param {number} threshold - Values with |delta[i]| < threshold are "near-zero"
 * @returns {number} Fraction in [0, 1]
 */
export function deltaSparsity(delta, threshold = 0.01) {
  let nearZero = 0;
  for (let i = 0; i < delta.length; i++) {
    if (Math.abs(delta[i]) < threshold) nearZero++;
  }
  return nearZero / delta.length;
}
