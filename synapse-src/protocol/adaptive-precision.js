/**
 * Adaptive Precision for Synapse Wire Protocol
 *
 * Selects per-layer quantization mode (NONE/INT8/INT4) based on online
 * measurement of quantization error. Layers where activations are robust
 * to quantization get more aggressive compression; sensitive layers stay
 * at higher precision.
 *
 * Key insight from transformer inference: early layers and attention-heavy
 * layers tend to be more sensitive to quantization error (they establish
 * token representations), while later MLP-heavy layers are robust (they
 * refine already-stable representations).
 *
 * Uses the same EMA profiling pattern as MixtureOfDepthsRouter.
 */

import { QuantMode } from "./binary.js";

/**
 * Quantize float32 to int4 (symmetric, packed 2 values per byte).
 * Range: [-7, 7] with scale = max(|tensor|) / 7
 *
 * @param {Float32Array} float32Data
 * @returns {{ data: Uint8Array, scale: number, length: number }}
 */
export function quantizeInt4(float32Data) {
  let absMax = 0;
  for (let i = 0; i < float32Data.length; i++) {
    const abs = Math.abs(float32Data[i]);
    if (abs > absMax) absMax = abs;
  }

  const scale = absMax > 0 ? absMax / 7 : 1;
  const invScale = 1 / scale;

  // Pack two int4 values per byte: low nibble = even index, high nibble = odd index
  const packedLen = Math.ceil(float32Data.length / 2);
  const packed = new Uint8Array(packedLen);

  for (let i = 0; i < float32Data.length; i += 2) {
    let lo = Math.round(float32Data[i] * invScale);
    if (lo > 7) lo = 7;
    else if (lo < -7) lo = -7;

    let hi = 0;
    if (i + 1 < float32Data.length) {
      hi = Math.round(float32Data[i + 1] * invScale);
      if (hi > 7) hi = 7;
      else if (hi < -7) hi = -7;
    }

    // Store as unsigned nibbles: value + 8 maps [-7,7] to [1,15], 0 stays 0 → [0,15]
    // Actually simpler: store signed 4-bit as offset binary
    packed[i >> 1] = ((lo + 8) & 0x0F) | (((hi + 8) & 0x0F) << 4);
  }

  return { data: packed, scale, length: float32Data.length };
}

/**
 * Dequantize int4 packed data back to float32.
 *
 * @param {Uint8Array} packedData
 * @param {number} scale
 * @param {number} originalLength
 * @returns {Float32Array}
 */
export function dequantizeInt4(packedData, scale, originalLength) {
  const result = new Float32Array(originalLength);

  for (let i = 0; i < originalLength; i += 2) {
    const byte = packedData[i >> 1];
    const lo = (byte & 0x0F) - 8;
    result[i] = lo * scale;

    if (i + 1 < originalLength) {
      const hi = ((byte >> 4) & 0x0F) - 8;
      result[i + 1] = hi * scale;
    }
  }

  return result;
}

/**
 * Pack int4 quantized data for wire transfer.
 * Layout: [packed_nibbles...] [float32 scale] [uint32 originalLength]
 *
 * @param {Uint8Array} packedData - Output of quantizeInt4
 * @param {number} scale
 * @param {number} originalLength - Number of float32 values
 * @returns {ArrayBuffer}
 */
export function packInt4(packedData, scale, originalLength) {
  const totalBytes = packedData.length + 4 + 4; // data + scale + length
  const buffer = new ArrayBuffer(totalBytes);
  const view = new DataView(buffer);

  new Uint8Array(buffer, 0, packedData.length).set(packedData);
  view.setFloat32(packedData.length, scale, true);
  view.setUint32(packedData.length + 4, originalLength, true);

  return buffer;
}

/**
 * Unpack int4 wire format back to packed data + metadata.
 *
 * @param {Uint8Array|ArrayBuffer} packed
 * @returns {{ packedData: Uint8Array, scale: number, originalLength: number }}
 */
export function unpackInt4(packed) {
  const buf = packed instanceof ArrayBuffer ? packed
    : packed.buffer.slice(packed.byteOffset, packed.byteOffset + packed.byteLength);
  const view = new DataView(buf);

  const originalLength = view.getUint32(buf.byteLength - 4, true);
  const scale = view.getFloat32(buf.byteLength - 8, true);
  const packedData = new Uint8Array(buf, 0, buf.byteLength - 8);

  return { packedData, scale, originalLength };
}

// ─── Adaptive Precision Selector ────────────────────────────────

/**
 * Tracks per-layer quantization error and selects the optimal precision
 * mode for each layer. Uses EMA of relative error to adapt over time.
 */
export class AdaptivePrecisionSelector {
  /**
   * @param {number} numLayers
   * @param {Object} opts
   * @param {number} opts.int8Threshold - Max relative error for int8 (default 0.01 = 1%)
   * @param {number} opts.int4Threshold - Max relative error for int4 (default 0.05 = 5%)
   * @param {number} opts.emaAlpha - EMA smoothing (default 0.1)
   * @param {number} opts.warmupSteps - Steps before enabling adaptive (default 5)
   */
  constructor(numLayers, opts = {}) {
    this.numLayers = numLayers;
    this.int8Threshold = opts.int8Threshold ?? 0.01;
    this.int4Threshold = opts.int4Threshold ?? 0.05;
    this.emaAlpha = opts.emaAlpha ?? 0.1;
    this.warmupSteps = opts.warmupSteps ?? 5;

    // Per-layer tracking
    this.int8Error = new Float64Array(numLayers);  // EMA of int8 relative error
    this.int4Error = new Float64Array(numLayers);  // EMA of int4 relative error
    this.stepCount = new Uint32Array(numLayers);
    this.currentMode = new Uint8Array(numLayers);  // starts as NONE (0)
  }

  /**
   * Measure quantization error for a layer's activation and update EMA.
   * Call this during inference to profile each layer.
   *
   * @param {number} layer - Relative layer index
   * @param {Float32Array} activation - Original float32 activation
   */
  observe(layer, activation) {
    if (layer >= this.numLayers) return;

    const norm = l2Norm(activation);
    if (norm === 0) return;

    // Measure int8 error
    const { data: int8Data, scale: int8Scale } = quantizeInt8Inline(activation);
    const int8Err = dequantErrorNorm(activation, int8Data, int8Scale) / norm;

    // Measure int4 error
    const { data: int4Data, scale: int4Scale } = quantizeInt4Inline(activation);
    const int4Err = dequantInt4ErrorNorm(activation, int4Data, int4Scale) / norm;

    const alpha = this.emaAlpha;
    const step = this.stepCount[layer];

    if (step === 0) {
      this.int8Error[layer] = int8Err;
      this.int4Error[layer] = int4Err;
    } else {
      this.int8Error[layer] = alpha * int8Err + (1 - alpha) * this.int8Error[layer];
      this.int4Error[layer] = alpha * int4Err + (1 - alpha) * this.int4Error[layer];
    }

    this.stepCount[layer]++;
    this._updateMode(layer);
  }

  /**
   * Get the recommended quantization mode for a layer.
   * Returns QuantMode.NONE during warmup.
   *
   * @param {number} layer - Relative layer index
   * @returns {number} QuantMode value (NONE=0, INT8=1, INT4=2)
   */
  getMode(layer) {
    if (layer >= this.numLayers) return QuantMode.NONE;
    if (this.stepCount[layer] < this.warmupSteps) return QuantMode.INT8; // safe default
    return this.currentMode[layer];
  }

  /**
   * Get a diagnostic snapshot of all layers.
   * @returns {Array<{ layer: number, mode: string, int8Error: number, int4Error: number, steps: number }>}
   */
  snapshot() {
    const modeNames = ["NONE", "INT8", "INT4"];
    const result = [];
    for (let l = 0; l < this.numLayers; l++) {
      result.push({
        layer: l,
        mode: modeNames[this.currentMode[l]],
        int8Error: this.int8Error[l],
        int4Error: this.int4Error[l],
        steps: this.stepCount[l],
      });
    }
    return result;
  }

  /** @private */
  _updateMode(layer) {
    if (this.stepCount[layer] < this.warmupSteps) return;

    if (this.int4Error[layer] <= this.int4Threshold) {
      this.currentMode[layer] = QuantMode.INT4;
    } else if (this.int8Error[layer] <= this.int8Threshold) {
      this.currentMode[layer] = QuantMode.INT8;
    } else {
      this.currentMode[layer] = QuantMode.NONE;
    }
  }
}

// ─── Inline helpers (avoid circular imports with quantize.js) ───

function l2Norm(arr) {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
  return Math.sqrt(sum);
}

function quantizeInt8Inline(data) {
  let absMax = 0;
  for (let i = 0; i < data.length; i++) {
    const abs = Math.abs(data[i]);
    if (abs > absMax) absMax = abs;
  }
  const scale = absMax > 0 ? absMax / 127 : 1;
  const invScale = 1 / scale;
  const int8 = new Int8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    let v = Math.round(data[i] * invScale);
    if (v > 127) v = 127;
    else if (v < -127) v = -127;
    int8[i] = v;
  }
  return { data: int8, scale };
}

function dequantErrorNorm(original, int8Data, scale) {
  let sum = 0;
  for (let i = 0; i < original.length; i++) {
    const diff = original[i] - int8Data[i] * scale;
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function quantizeInt4Inline(data) {
  let absMax = 0;
  for (let i = 0; i < data.length; i++) {
    const abs = Math.abs(data[i]);
    if (abs > absMax) absMax = abs;
  }
  const scale = absMax > 0 ? absMax / 7 : 1;
  const invScale = 1 / scale;
  const int4 = new Int8Array(data.length); // store unpacked for error measurement
  for (let i = 0; i < data.length; i++) {
    let v = Math.round(data[i] * invScale);
    if (v > 7) v = 7;
    else if (v < -7) v = -7;
    int4[i] = v;
  }
  return { data: int4, scale };
}

function dequantInt4ErrorNorm(original, int4Data, scale) {
  let sum = 0;
  for (let i = 0; i < original.length; i++) {
    const diff = original[i] - int4Data[i] * scale;
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}
