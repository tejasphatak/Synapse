/**
 * KVCache — Per-generation GPU-backed key/value cache for transformer attention.
 *
 * Caches K and V tensors from previous tokens so that each autoregressive step
 * only computes the NEW token's Q, K, V instead of the full sequence.
 *
 * Reduces attention from O(seq^2) per step to O(seq) per step.
 *
 * Memory per node (GPT-2 small, 6 layers):
 *   6 layers x 2 (K,V) x maxSeqLen x hiddenSize x 4 bytes
 *   = 6 x 2 x 1024 x 768 x 4 = ~36 MB
 */

export class KVCache {
  /**
   * @param {GPUDevice} device - WebGPU device
   * @param {number} numLayers - Number of layers this node owns
   * @param {number} layerStart - First layer index this node owns
   * @param {number} hiddenSize - Hidden dimension (768 for GPT-2 small)
   * @param {number} maxSeqLen - Maximum sequence length (1024 for GPT-2)
   * @param {number} [kvHiddenSize=hiddenSize] - K/V width. For MHA equals
   *   hiddenSize; for GQA/MQA equals numKvHeads * headDim (e.g. Gemma 3 1B:
   *   1 kv head * 256 = 256, vs hidden=1152). Separate so the KV buffers
   *   don't waste memory on K/V shapes narrower than the residual stream.
   */
  constructor(device, numLayers, layerStart, hiddenSize, maxSeqLen, kvHiddenSize = null) {
    this.device = device;
    this.numLayers = numLayers;
    this.layerStart = layerStart;
    this.hiddenSize = hiddenSize;
    this.kvHiddenSize = kvHiddenSize ?? hiddenSize;
    this.maxSeqLen = maxSeqLen;
    this.seqLen = 0; // current number of cached positions

    // Pre-allocate K and V buffers for each layer.
    // Shape: [maxSeqLen, kvHiddenSize] in float32.
    const bufferSize = maxSeqLen * this.kvHiddenSize * 4;
    this.kBuffers = new Map(); // layerIdx -> GPUBuffer
    this.vBuffers = new Map(); // layerIdx -> GPUBuffer

    for (let l = 0; l < numLayers; l++) {
      const layerIdx = layerStart + l;

      this.kBuffers.set(layerIdx, device.createBuffer({
        label: `kv_cache_k_layer${layerIdx}`,
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }));

      this.vBuffers.set(layerIdx, device.createBuffer({
        label: `kv_cache_v_layer${layerIdx}`,
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      }));
    }
  }

  /**
   * Append new K and V for a single token at the given sequence position.
   *
   * @param {number} layerIdx - Layer index
   * @param {GPUBuffer} newK - New key tensor [1, hiddenSize]
   * @param {GPUBuffer} newV - New value tensor [1, hiddenSize]
   * @param {number} seqPos - Position in the sequence (0-indexed)
   */
  append(layerIdx, newK, newV, seqPos) {
    const offset = seqPos * this.kvHiddenSize * 4; // byte offset
    const size = this.kvHiddenSize * 4; // bytes for one position

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(newK, 0, this.kBuffers.get(layerIdx), offset, size);
    encoder.copyBufferToBuffer(newV, 0, this.vBuffers.get(layerIdx), offset, size);
    this.device.queue.submit([encoder.finish()]);

    // Track the furthest position written
    if (seqPos + 1 > this.seqLen) {
      this.seqLen = seqPos + 1;
    }
  }

  /**
   * Append K and V for multiple tokens (used during prefill).
   *
   * @param {number} layerIdx - Layer index
   * @param {GPUBuffer} keys - Key tensor [numTokens, hiddenSize]
   * @param {GPUBuffer} values - Value tensor [numTokens, hiddenSize]
   * @param {number} startPos - Starting position in the sequence
   * @param {number} numTokens - Number of tokens to cache
   */
  appendBatch(layerIdx, keys, values, startPos, numTokens) {
    const offset = startPos * this.kvHiddenSize * 4;
    const size = numTokens * this.kvHiddenSize * 4;

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(keys, 0, this.kBuffers.get(layerIdx), offset, size);
    encoder.copyBufferToBuffer(values, 0, this.vBuffers.get(layerIdx), offset, size);
    this.device.queue.submit([encoder.finish()]);

    const newLen = startPos + numTokens;
    if (newLen > this.seqLen) {
      this.seqLen = newLen;
    }
  }

  /**
   * Get the cached K and V buffers for a layer.
   * The caller uses seqLen to know how much of the buffer is valid.
   *
   * @param {number} layerIdx - Layer index
   * @returns {{ kBuffer: GPUBuffer, vBuffer: GPUBuffer, seqLen: number }}
   */
  getKV(layerIdx) {
    return {
      kBuffer: this.kBuffers.get(layerIdx),
      vBuffer: this.vBuffers.get(layerIdx),
      seqLen: this.seqLen,
    };
  }

  /**
   * Roll back the cache to a previous sequence position.
   * Used when a speculative computation is rejected — the speculative
   * step appended KV entries that need to be discarded.
   *
   * @param {number} seqPos - The position to roll back to (exclusive upper bound)
   */
  rollback(seqPos) {
    if (seqPos < this.seqLen) {
      this.seqLen = seqPos;
    }
  }

  /**
   * Reset the cache (new generation or generation complete).
   */
  reset() {
    this.seqLen = 0;
    // No need to zero the buffers — seqLen controls valid region
  }

  /**
   * Destroy all GPU buffers and free memory.
   */
  destroy() {
    for (const buf of this.kBuffers.values()) buf.destroy();
    for (const buf of this.vBuffers.values()) buf.destroy();
    this.kBuffers.clear();
    this.vBuffers.clear();
    this.seqLen = 0;
  }
}
