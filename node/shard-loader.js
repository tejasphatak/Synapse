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

      const tensorData = new Float32Array(
        sourceData,
        entry.offset,
        entry.size / 4 // float32 = 4 bytes
      );

      const gpuBuffer = this._createGPUBuffer(entry.name, tensorData, entry.shape);
      this.buffers.set(entry.name, gpuBuffer);
      this.metadata.set(entry.name, {
        shape: entry.shape,
        dtype: entry.dtype,
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
      numLayers: this.manifest.num_layers,
      hiddenSize: this.manifest.hidden_size,
      numHeads: this.manifest.num_heads,
      headDim: this.manifest.head_dim,
      vocabSize: this.manifest.vocab_size,
      maxSeqLen: this.manifest.max_seq_len,
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
