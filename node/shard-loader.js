/**
 * Shard Loader — Fetches model shard binary, parses manifest,
 * and uploads weight tensors into WebGPU buffers.
 */

export class ShardLoader {
  constructor(device) {
    this.device = device;
    this.manifest = null;
    this.buffers = new Map(); // tensor name → GPUBuffer
    this.metadata = new Map(); // tensor name → { shape, dtype, offset, size }
  }

  /**
   * Load the manifest from the coordinator.
   */
  async loadManifest(baseUrl) {
    const resp = await fetch(`${baseUrl}/shards/manifest.json`);
    if (!resp.ok) throw new Error(`Failed to fetch manifest: ${resp.status}`);
    this.manifest = await resp.json();
    return this.manifest;
  }

  /**
   * Download a shard binary file and the shared weights file,
   * then upload all relevant tensors to GPU buffers.
   *
   * @param {number} shardId - Which shard to load (0 or 1)
   * @param {string} shardUrl - URL to the shard binary (e.g. /shards/shard_0.bin)
   * @param {string} sharedUrl - URL to the shared weights (e.g. /shards/shared.bin)
   * @param {Function} onProgress - Progress callback (loaded, total, phase)
   */
  async loadShard(shardId, shardUrl, sharedUrl, onProgress = null) {
    if (!this.manifest) {
      throw new Error("Manifest not loaded — call loadManifest() first");
    }

    const shardConfig = this.manifest.shard_layout[String(shardId)];
    if (!shardConfig) throw new Error(`Unknown shard ID: ${shardId}`);

    // Download both files in parallel
    const [shardData, sharedData] = await Promise.all([
      this._fetchBinary(shardUrl, (loaded, total) =>
        onProgress?.(loaded, total, "shard")
      ),
      this._fetchBinary(sharedUrl, (loaded, total) =>
        onProgress?.(loaded, total, "shared")
      ),
    ]);

    // Parse tensors from manifest and upload to GPU
    const shardFile = shardConfig.file;
    let uploadCount = 0;
    const totalTensors = this.manifest.tensors.filter(
      (t) => t.file === shardFile || t.file === "shared.bin"
    ).length;

    for (const entry of this.manifest.tensors) {
      let sourceData;
      if (entry.file === shardFile) {
        sourceData = shardData;
      } else if (entry.file === "shared.bin") {
        sourceData = sharedData;
      } else {
        continue; // Skip tensors from other shards
      }

      // Dequantize to float32 for GPU upload based on dtype
      const tensorData = this._dequantizeTensor(sourceData, entry);

      const gpuBuffer = this._createGPUBuffer(entry.name, tensorData, entry.shape);
      this.buffers.set(entry.name, gpuBuffer);
      this.metadata.set(entry.name, {
        shape: entry.shape,
        dtype: entry.dtype,
        originalDtype: entry.dtype,
        offset: entry.offset,
        size: entry.size,
      });

      uploadCount++;
      onProgress?.(uploadCount, totalTensors, "upload");
    }

    return {
      shardId,
      layerStart: shardConfig.layer_start,
      layerEnd: shardConfig.layer_end,
      tensorCount: uploadCount,
    };
  }

  /**
   * Get a GPU buffer by tensor name.
   */
  getBuffer(name) {
    return this.buffers.get(name) || null;
  }

  /**
   * Get tensor metadata.
   */
  getTensorMeta(name) {
    return this.metadata.get(name) || null;
  }

  /**
   * Get all tensor names matching a prefix.
   */
  getTensorsByPrefix(prefix) {
    const result = [];
    for (const [name, buffer] of this.buffers) {
      if (name.startsWith(prefix)) {
        result.push({ name, buffer, meta: this.metadata.get(name) });
      }
    }
    return result;
  }

  /**
   * Get the model config from the manifest.
   */
  getModelConfig() {
    if (!this.manifest) return null;
    return {
      model: this.manifest.model,
      arch: this.manifest.arch || "gpt2",
      numLayers: this.manifest.num_layers,
      hiddenSize: this.manifest.hidden_size,
      numHeads: this.manifest.num_heads,
      headDim: this.manifest.head_dim,
      vocabSize: this.manifest.vocab_size,
      maxSeqLen: this.manifest.max_seq_len,
      dtype: this.manifest.dtype || "float32",
      numShards: this.manifest.num_shards || 2,
    };
  }

  /**
   * Fetch a binary file as an ArrayBuffer with progress tracking.
   */
  async _fetchBinary(url, onProgress = null) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);

    const contentLength = parseInt(resp.headers.get("Content-Length") || "0", 10);
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.(loaded, contentLength);
    }

    // Merge chunks into a single ArrayBuffer
    const result = new ArrayBuffer(loaded);
    const view = new Uint8Array(result);
    let offset = 0;
    for (const chunk of chunks) {
      view.set(chunk, offset);
      offset += chunk.byteLength;
    }

    return result;
  }

  /**
   * Dequantize a tensor from its stored format to Float32Array for GPU upload.
   * Handles float32, float16, int8, and int4 formats.
   */
  _dequantizeTensor(sourceData, entry) {
    const dtype = entry.dtype;
    const raw = new Uint8Array(sourceData, entry.offset, entry.size);

    if (dtype === "float32") {
      return new Float32Array(sourceData, entry.offset, entry.size / 4);
    }

    if (dtype === "float16") {
      // Dequantize float16 → float32
      const numElements = entry.size / 2;
      const f16 = new Uint16Array(sourceData, entry.offset, numElements);
      const f32 = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        f32[i] = this._float16ToFloat32(f16[i]);
      }
      return f32;
    }

    if (dtype === "int8") {
      // Dequantize int8 → float32 using per-tensor scale
      const quant = entry.quant;
      const dataBytes = new Int8Array(sourceData, entry.offset, quant.scale_offset);
      const scaleView = new DataView(sourceData, entry.offset + quant.scale_offset, 4);
      const scale = scaleView.getFloat32(0, true);

      const f32 = new Float32Array(dataBytes.length);
      for (let i = 0; i < dataBytes.length; i++) {
        f32[i] = dataBytes[i] * scale;
      }
      return f32;
    }

    if (dtype === "int4") {
      // Dequantize int4 → float32 using per-group scales
      const quant = entry.quant;
      const packedData = new Uint8Array(sourceData, entry.offset, quant.packed_size);
      const scalesData = new Float32Array(
        sourceData, entry.offset + quant.scales_offset, quant.scales_size / 4
      );

      const numElements = quant.original_numel;
      const groupSize = quant.group_size;
      const f32 = new Float32Array(numElements);

      // Unpack: each byte holds two int4 values
      let elemIdx = 0;
      for (let i = 0; i < packedData.length && elemIdx < numElements; i++) {
        const byte = packedData[i];
        // Low nibble (even index)
        let val0 = byte & 0x0F;
        if (val0 > 7) val0 -= 16; // sign-extend from 4-bit
        // High nibble (odd index)
        let val1 = (byte >> 4) & 0x0F;
        if (val1 > 7) val1 -= 16;

        const groupIdx0 = Math.floor(elemIdx / groupSize);
        const scale0 = scalesData[groupIdx0] || 1.0;
        f32[elemIdx] = val0 * scale0;
        elemIdx++;

        if (elemIdx < numElements) {
          const groupIdx1 = Math.floor(elemIdx / groupSize);
          const scale1 = scalesData[groupIdx1] || 1.0;
          f32[elemIdx] = val1 * scale1;
          elemIdx++;
        }
      }
      return f32;
    }

    // Fallback: treat as float32
    console.warn(`[shard-loader] Unknown dtype "${dtype}", treating as float32`);
    return new Float32Array(sourceData, entry.offset, entry.size / 4);
  }

  /**
   * Convert a float16 (stored as uint16) to float32.
   */
  _float16ToFloat32(h) {
    const sign = (h >> 15) & 0x1;
    const exponent = (h >> 10) & 0x1F;
    const mantissa = h & 0x3FF;

    if (exponent === 0) {
      if (mantissa === 0) return sign ? -0 : 0;
      // Subnormal
      const val = (mantissa / 1024) * Math.pow(2, -14);
      return sign ? -val : val;
    }

    if (exponent === 31) {
      return mantissa ? NaN : (sign ? -Infinity : Infinity);
    }

    const val = Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
    return sign ? -val : val;
  }

  /**
   * Create a WebGPU storage buffer from a Float32Array.
   */
  _createGPUBuffer(name, data, shape) {
    // Align to 4 bytes (Float32 already aligned)
    const buffer = this.device.createBuffer({
      label: name,
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();

    return buffer;
  }

  /**
   * Clean up all GPU buffers.
   */
  destroy() {
    for (const buffer of this.buffers.values()) {
      buffer.destroy();
    }
    this.buffers.clear();
    this.metadata.clear();
  }
}
