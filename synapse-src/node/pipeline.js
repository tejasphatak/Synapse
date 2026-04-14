/**
 * Pipeline — Forward pass orchestration for GPT-2 transformer layers.
 *
 * Manages WebGPU compute shader dispatches for:
 *   embed → [layernorm → attention → residual → layernorm → FFN → residual] × N → final_ln → lm_head
 *
 * Each browser node runs a subset of layers (e.g., layers 0-5 or 6-11).
 */

import { KVCache } from "./kv-cache.js";
import { EarlyExitDetector } from "./early-exit.js";
import { MixtureOfDepthsRouter } from "./mixture-of-depths.js";
import { HeadPruner } from "./head-pruning.js";
import { quantizeInt8, dequantizeInt8, packQuantized, unpackQuantized, quantizeInt8PerChannel, dequantizeInt8PerChannel, packQuantizedPerChannel, unpackQuantizedPerChannel, computeDelta, applyDelta, deltaSparsity } from "../protocol/quantize.js";

export class Pipeline {
  constructor(device, shardLoader) {
    this.device = device;
    this.loader = shardLoader;
    this.config = shardLoader.getModelConfig();
    this.pipelines = {};   // cached compute pipelines
    this.shaderModules = {}; // cached shader modules
    this._initialized = false;
    this._tempBuffers = []; // track temporary buffers for cleanup
    this.kvCaches = new Map(); // requestId -> KVCache
    this.earlyExit = new EarlyExitDetector(); // disabled by default, tracks metrics
    this.modRouter = null; // MixtureOfDepths — initialized when layer range is known
    this.headPruner = null; // HeadPruner — initialized when layer range is known
  }

  /**
   * Initialize compute pipelines by loading and compiling all WGSL shaders.
   */
  async init() {
    const shaderNames = [
      "matmul", "matmul_transB", "attention", "attention_cached", "layernorm", "gelu",
      "residual_add", "embed", "bias_add", "head_slice", "head_concat",
    ];

    for (const name of shaderNames) {
      const code = await (await fetch(`/node/kernels/${name}.wgsl?v=${Date.now()}`)).text();
      this.shaderModules[name] = this.device.createShaderModule({
        label: name,
        code,
      });
    }

    this._initialized = true;
  }

  /**
   * Run the embedding step: token IDs → hidden state [seq_len, hidden_size].
   * Only called on the first node in the pipeline.
   */
  async embed(tokenIds) {
    const { hiddenSize, vocabSize, maxSeqLen } = this.config;
    const seqLen = tokenIds.length;

    // Upload token IDs to GPU
    const tokenBuf = this._createBuffer("token_ids", seqLen * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(tokenBuf, 0, new Uint32Array(tokenIds));

    // Get embedding weight buffers
    const wte = this.loader.getBuffer("transformer.wte.weight");
    const wpe = this.loader.getBuffer("transformer.wpe.weight");

    // Output buffer
    const outputSize = seqLen * hiddenSize * 4;
    const outputBuf = this._createBuffer("embed_output", outputSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // Uniform params
    const params = new Uint32Array([seqLen, hiddenSize, 0, 0]);
    const paramBuf = this._createBuffer("embed_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, params);

    // Dispatch
    const pipeline = this._getOrCreatePipeline("embed", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: tokenBuf } },
        { binding: 2, resource: { buffer: wte } },
        { binding: 3, resource: { buffer: wpe } },
        { binding: 4, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil((seqLen * hiddenSize) / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return { buffer: outputBuf, shape: [seqLen, hiddenSize] };
  }

  /**
   * Run a single transformer layer's forward pass.
   *
   * For layer `l`:
   *   h = h + Attention(LayerNorm(h))   [pre-attention norm]
   *   h = h + MLP(LayerNorm(h))         [pre-FFN norm]
   */
  async forwardLayer(hidden, layerIdx, relLayer) {
    const { hiddenSize, numHeads, headDim } = this.config;
    const seqLen = hidden.shape[0];
    const prefix = `transformer.h.${layerIdx}`;
    const headMask = this.headPruner ? this.headPruner.getHeadMask(relLayer ?? 0) : null;

    // ─── Pre-Attention LayerNorm ───────────────────────────
    const ln1Out = await this._layerNorm(
      hidden.buffer, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.ln_1.weight`),
      this.loader.getBuffer(`${prefix}.ln_1.bias`)
    );

    // ─── Multi-Head Self-Attention ────────────────────────
    // c_attn projects input to Q, K, V concatenated: [seq, 3*hidden]
    const qkvOut = await this._matmul(
      ln1Out, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.attn.c_attn.weight`), hiddenSize, 3 * hiddenSize
    );
    // Add bias
    await this._addBias(qkvOut, seqLen, 3 * hiddenSize,
      this.loader.getBuffer(`${prefix}.attn.c_attn.bias`)
    );

    // Split Q, K, V and compute multi-head attention
    const attnOut = await this._multiHeadAttention(qkvOut, seqLen, numHeads, headDim, headMask);

    // c_proj: project attention output back to hidden_size
    const projOut = await this._matmul(
      attnOut, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.attn.c_proj.weight`), hiddenSize, hiddenSize
    );
    await this._addBias(projOut, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.attn.c_proj.bias`)
    );

    // Residual connection: h = h + attention(ln(h))
    const residual1 = await this._residualAdd(hidden.buffer, projOut, seqLen * hiddenSize);

    // ─── Pre-FFN LayerNorm ────────────────────────────────
    const ln2Out = await this._layerNorm(
      residual1, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.ln_2.weight`),
      this.loader.getBuffer(`${prefix}.ln_2.bias`)
    );

    // ─── Feed-Forward Network ─────────────────────────────
    // mlp.c_fc: [hidden_size, 4*hidden_size]
    const ffnInnerDim = 4 * hiddenSize;
    const fcOut = await this._matmul(
      ln2Out, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.mlp.c_fc.weight`), hiddenSize, ffnInnerDim
    );
    await this._addBias(fcOut, seqLen, ffnInnerDim,
      this.loader.getBuffer(`${prefix}.mlp.c_fc.bias`)
    );

    // GELU activation
    const geluOut = await this._gelu(fcOut, seqLen * ffnInnerDim);

    // mlp.c_proj: [4*hidden_size, hidden_size]
    const ffnOut = await this._matmul(
      geluOut, seqLen, ffnInnerDim,
      this.loader.getBuffer(`${prefix}.mlp.c_proj.weight`), ffnInnerDim, hiddenSize
    );
    await this._addBias(ffnOut, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.mlp.c_proj.bias`)
    );

    // Residual connection: h = h + ffn(ln(h))
    const residual2 = await this._residualAdd(residual1, ffnOut, seqLen * hiddenSize);

    return { buffer: residual2, shape: [seqLen, hiddenSize] };
  }

  /**
   * Run all assigned layers sequentially, freeing temp buffers between layers.
   */
  async forwardLayers(hidden, layerStart, layerEnd) {
    // Build a set of weight buffers once so we never destroy them
    const weightBuffers = new Set(this.loader.buffers.values());
    const numLayers = layerEnd - layerStart + 1;

    // Initialize MoD router on first call (now we know layer count)
    if (!this.modRouter) {
      this.modRouter = new MixtureOfDepthsRouter(numLayers);
    }
    // Initialize head pruner on first call
    if (!this.headPruner) {
      this.headPruner = new HeadPruner(this.config.numHeads, numLayers);
    }

    let h = hidden;
    for (let l = layerStart; l <= layerEnd; l++) {
      const relLayer = l - layerStart;

      // MoD routing for prefill path (uses a synthetic requestId since prefill is one-shot)
      const byteSize = h.shape.reduce((a, b) => a * b, 1) * 4;
      const preHidden = new Float32Array(await this._readBuffer(h.buffer, 0, byteSize));
      this.modRouter.observeToken("prefill", preHidden);

      const routeDecision = this.modRouter.route(relLayer, "prefill", preHidden);
      if (routeDecision.skip) {
        continue;
      }

      h = await this.forwardLayer(h, l, relLayer);

      // Record layer effect for difficulty profiling
      const postByteSize = h.shape.reduce((a, b) => a * b, 1) * 4;
      const postHidden = new Float32Array(await this._readBuffer(h.buffer, 0, postByteSize));
      this.modRouter.recordLayerEffect(relLayer, preHidden, postHidden);

      // Free all temp buffers except the output we just produced and weight buffers
      const keep = h.buffer;
      const surviving = [];
      for (const buf of this._tempBuffers) {
        if (buf === keep || weightBuffers.has(buf)) {
          surviving.push(buf);
        } else {
          buf.destroy();
        }
      }
      this._tempBuffers = surviving;
    }
    this.modRouter.clear("prefill");
    return h;
  }

  /**
   * Run the final output head: layernorm → lm_head projection → logits.
   * Only called on the last node in the pipeline.
   */
  async outputHead(hidden) {
    const { hiddenSize, vocabSize } = this.config;
    const seqLen = hidden.shape[0];

    // Final layer norm
    const lnOut = await this._layerNorm(
      hidden.buffer, seqLen, hiddenSize,
      this.loader.getBuffer("transformer.ln_f.weight"),
      this.loader.getBuffer("transformer.ln_f.bias")
    );

    // lm_head: weight is [vocab_size, hidden_size] (nn.Linear format)
    // Need C = hidden × W^T, so use transposed matmul
    const logits = await this._matmulTransB(
      lnOut, seqLen, hiddenSize,
      this.loader.getBuffer("lm_head.weight"), vocabSize
    );

    return { buffer: logits, shape: [seqLen, vocabSize] };
  }

  /**
   * Sample the next token from logits using temperature sampling.
   * Reads the last position's logits from GPU, applies softmax on CPU, samples.
   */
  async sampleToken(logitsTensor, temperature = 0.8) {
    const { vocabSize } = this.config;
    const seqLen = logitsTensor.shape[0];

    // Read only the last position's logits from GPU
    const offset = (seqLen - 1) * vocabSize * 4;
    const logits = await this._readBuffer(logitsTensor.buffer, offset, vocabSize * 4);
    const logitsF32 = new Float32Array(logits);

    // Temperature scaling
    for (let i = 0; i < logitsF32.length; i++) {
      logitsF32[i] /= temperature;
    }

    // Softmax
    let maxLogit = -Infinity;
    for (let i = 0; i < logitsF32.length; i++) {
      if (logitsF32[i] > maxLogit) maxLogit = logitsF32[i];
    }
    let sumExp = 0;
    const probs = new Float32Array(logitsF32.length);
    for (let i = 0; i < logitsF32.length; i++) {
      probs[i] = Math.exp(logitsF32[i] - maxLogit);
      sumExp += probs[i];
    }
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
    }

    // Multinomial sample
    const r = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) return i;
    }
    return probs.length - 1;
  }

  // ─── KV Cache Methods ────────────────────────────────────────

  /**
   * Get or create a KV cache for a generation request.
   */
  getOrCreateKVCache(requestId, layerStart, numLayers) {
    if (!this.kvCaches.has(requestId)) {
      const { hiddenSize, maxSeqLen } = this.config;
      this.kvCaches.set(requestId, new KVCache(
        this.device, numLayers, layerStart, hiddenSize, maxSeqLen
      ));
    }
    return this.kvCaches.get(requestId);
  }

  /**
   * Clear KV cache for a completed generation.
   */
  clearCache(requestId) {
    const cache = this.kvCaches.get(requestId);
    if (cache) {
      cache.destroy();
      this.kvCaches.delete(requestId);
    }
    this.earlyExit?.clear(requestId);
    this.modRouter?.clear(requestId);
  }

  /**
   * Embed a single token at a specific sequence position.
   * Used during KV-cached autoregressive decoding.
   */
  async embedSingle(tokenId, seqPos) {
    const { hiddenSize } = this.config;

    // Upload single token ID
    const tokenBuf = this._createBuffer("token_single", 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(tokenBuf, 0, new Uint32Array([tokenId]));

    const wte = this.loader.getBuffer("transformer.wte.weight");
    const wpe = this.loader.getBuffer("transformer.wpe.weight");

    const outputBuf = this._createBuffer("embed_single_out", hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // Use seqPos for positional embedding
    const params = new Uint32Array([1, hiddenSize, seqPos, 0]);
    const paramBuf = this._createBuffer("embed_single_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this._getOrCreatePipeline("embed", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: tokenBuf } },
        { binding: 2, resource: { buffer: wte } },
        { binding: 3, resource: { buffer: wpe } },
        { binding: 4, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(hiddenSize / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return { buffer: outputBuf, shape: [1, hiddenSize] };
  }

  /**
   * Run a single transformer layer with KV cache (single-token forward).
   * Only processes the NEW token — K,V from previous tokens are cached.
   */
  async forwardLayerCached(hidden, layerIdx, kvCache, seqPos, relLayer) {
    const { hiddenSize, numHeads, headDim } = this.config;
    const seqLen = 1; // always 1 token in cached mode
    const prefix = `transformer.h.${layerIdx}`;

    // ─── Pre-Attention LayerNorm ─────────────────────
    const ln1Out = await this._layerNorm(
      hidden.buffer, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.ln_1.weight`),
      this.loader.getBuffer(`${prefix}.ln_1.bias`)
    );

    // ─── QKV Projection for single token ─────────────
    const qkvOut = await this._matmul(
      ln1Out, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.attn.c_attn.weight`), hiddenSize, 3 * hiddenSize
    );
    await this._addBias(qkvOut, seqLen, 3 * hiddenSize,
      this.loader.getBuffer(`${prefix}.attn.c_attn.bias`)
    );

    // Extract K_new and V_new (full hidden size) and append to KV cache
    // K is at offset [hiddenSize..2*hiddenSize], V at [2*hiddenSize..3*hiddenSize]
    const kNewBuf = this._createBuffer("k_new", hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const vNewBuf = this._createBuffer("v_new", hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // Copy K and V slices from QKV buffer
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(qkvOut, hiddenSize * 4, kNewBuf, 0, hiddenSize * 4);
    enc.copyBufferToBuffer(qkvOut, 2 * hiddenSize * 4, vNewBuf, 0, hiddenSize * 4);
    this.device.queue.submit([enc.finish()]);

    // Append new K,V to cache
    kvCache.append(layerIdx, kNewBuf, vNewBuf, seqPos);

    // ─── Cached Multi-Head Attention ─────────────────
    const cacheLen = seqPos + 1; // includes current token
    const headMask = this.headPruner ? this.headPruner.getHeadMask(relLayer ?? 0) : null;
    const attnOut = await this._multiHeadAttentionCached(
      qkvOut, numHeads, headDim, kvCache, layerIdx, cacheLen, headMask
    );

    // c_proj: project attention output back to hidden_size
    const projOut = await this._matmul(
      attnOut, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.attn.c_proj.weight`), hiddenSize, hiddenSize
    );
    await this._addBias(projOut, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.attn.c_proj.bias`)
    );

    // Residual connection
    const residual1 = await this._residualAdd(hidden.buffer, projOut, seqLen * hiddenSize);

    // ─── Pre-FFN LayerNorm ───────────────────────────
    const ln2Out = await this._layerNorm(
      residual1, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.ln_2.weight`),
      this.loader.getBuffer(`${prefix}.ln_2.bias`)
    );

    // ─── Feed-Forward Network ────────────────────────
    const ffnInnerDim = 4 * hiddenSize;
    const fcOut = await this._matmul(
      ln2Out, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.mlp.c_fc.weight`), hiddenSize, ffnInnerDim
    );
    await this._addBias(fcOut, seqLen, ffnInnerDim,
      this.loader.getBuffer(`${prefix}.mlp.c_fc.bias`)
    );

    const geluOut = await this._gelu(fcOut, seqLen * ffnInnerDim);

    const ffnOut = await this._matmul(
      geluOut, seqLen, ffnInnerDim,
      this.loader.getBuffer(`${prefix}.mlp.c_proj.weight`), ffnInnerDim, hiddenSize
    );
    await this._addBias(ffnOut, seqLen, hiddenSize,
      this.loader.getBuffer(`${prefix}.mlp.c_proj.bias`)
    );

    const residual2 = await this._residualAdd(residual1, ffnOut, seqLen * hiddenSize);

    return { buffer: residual2, shape: [seqLen, hiddenSize] };
  }

  /**
   * Run all assigned layers with KV cache (single-token path).
   */
  async forwardLayersCached(hidden, layerStart, layerEnd, requestId, seqPos) {
    const numLayers = layerEnd - layerStart + 1;
    const kvCache = this.getOrCreateKVCache(requestId, layerStart, numLayers);
    const weightBuffers = new Set(this.loader.buffers.values());

    // Initialize MoD router on first call (now we know layer count)
    if (!this.modRouter) {
      this.modRouter = new MixtureOfDepthsRouter(numLayers);
    }
    // Initialize head pruner on first call
    if (!this.headPruner) {
      this.headPruner = new HeadPruner(this.config.numHeads, numLayers);
    }

    let h = hidden;
    for (let l = layerStart; l <= layerEnd; l++) {
      const relLayer = l - layerStart;

      // Mixture-of-Depths: check if this layer should be skipped
      // Read hidden state for routing decision
      const byteSize = h.shape.reduce((a, b) => a * b, 1) * 4;
      const preHidden = new Float32Array(await this._readBuffer(h.buffer, 0, byteSize));

      // Observe token for difficulty estimation
      this.modRouter.observeToken(requestId, preHidden);

      const routeDecision = this.modRouter.route(relLayer, requestId, preHidden);
      if (routeDecision.skip) {
        // Residual passthrough — hidden state passes unchanged, but we MUST
        // still populate the KV cache so future tokens can attend to this position.
        // Compute K,V from the current hidden state via LayerNorm + QKV projection.
        const { hiddenSize, numHeads, headDim } = this.config;
        const prefix = `transformer.h.${l}`;

        const ln1Out = await this._layerNorm(
          h.buffer, 1, hiddenSize,
          this.loader.getBuffer(`${prefix}.ln_1.weight`),
          this.loader.getBuffer(`${prefix}.ln_1.bias`)
        );

        const qkvOut = await this._matmul(
          ln1Out, 1, hiddenSize,
          this.loader.getBuffer(`${prefix}.attn.c_attn.weight`), hiddenSize, 3 * hiddenSize
        );
        await this._addBias(qkvOut, 1, 3 * hiddenSize,
          this.loader.getBuffer(`${prefix}.attn.c_attn.bias`)
        );

        // Extract K and V from QKV and append to cache
        const kNewBuf = this._createBuffer("mod_skip_k", hiddenSize * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        const vNewBuf = this._createBuffer("mod_skip_v", hiddenSize * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(qkvOut, hiddenSize * 4, kNewBuf, 0, hiddenSize * 4);
        enc.copyBufferToBuffer(qkvOut, 2 * hiddenSize * 4, vNewBuf, 0, hiddenSize * 4);
        this.device.queue.submit([enc.finish()]);

        kvCache.append(l, kNewBuf, vNewBuf, seqPos);
        continue;
      }

      h = await this.forwardLayerCached(h, l, kvCache, seqPos, relLayer);

      // Record layer effect for MoD difficulty profiling
      const postHidden = new Float32Array(await this._readBuffer(h.buffer, 0, byteSize));
      this.modRouter.recordLayerEffect(relLayer, preHidden, postHidden);

      // Early exit check: read hidden state and check convergence
      if (this.earlyExit && l < layerEnd) {
        const exitResult = this.earlyExit.check(requestId, relLayer, postHidden, numLayers);
        if (exitResult.shouldExit) {
          // Skip remaining layers — this token is already converged
          break;
        }
      }

      // Free temp buffers between layers (same as non-cached path)
      const keep = h.buffer;
      const surviving = [];
      for (const buf of this._tempBuffers) {
        if (buf === keep || weightBuffers.has(buf)) {
          surviving.push(buf);
        } else {
          buf.destroy();
        }
      }
      this._tempBuffers = surviving;
    }
    return h;
  }

  /**
   * Run full-sequence forward and populate the KV cache (prefill).
   * This runs the standard forwardLayers but stores K,V at each layer.
   */
  async forwardLayersPrefill(hidden, layerStart, layerEnd, requestId) {
    const { hiddenSize, numHeads, headDim } = this.config;
    const seqLen = hidden.shape[0];
    const numLayers = layerEnd - layerStart + 1;
    const kvCache = this.getOrCreateKVCache(requestId, layerStart, numLayers);
    const weightBuffers = new Set(this.loader.buffers.values());

    let h = hidden;
    for (let l = layerStart; l <= layerEnd; l++) {
      const prefix = `transformer.h.${l}`;

      // Standard forward layer computation
      const ln1Out = await this._layerNorm(
        h.buffer, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.ln_1.weight`),
        this.loader.getBuffer(`${prefix}.ln_1.bias`)
      );

      const qkvOut = await this._matmul(
        ln1Out, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.attn.c_attn.weight`), hiddenSize, 3 * hiddenSize
      );
      await this._addBias(qkvOut, seqLen, 3 * hiddenSize,
        this.loader.getBuffer(`${prefix}.attn.c_attn.bias`)
      );

      // Extract full K and V for caching: [seqLen, hiddenSize]
      const kBuf = this._createBuffer(`prefill_k_l${l}`, seqLen * hiddenSize * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      const vBuf = this._createBuffer(`prefill_v_l${l}`, seqLen * hiddenSize * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

      // K is rows of qkvOut at columns [hiddenSize..2*hiddenSize]
      // V is rows of qkvOut at columns [2*hiddenSize..3*hiddenSize]
      // Since QKV is [seqLen, 3*hiddenSize] contiguous, we need per-row extraction
      // Use head_slice kernel to extract full K and V sections
      for (let h_idx = 0; h_idx < numHeads; h_idx++) {
        const kSlice = await this._extractHeadSlice(qkvOut, seqLen, 3 * hiddenSize, hiddenSize, h_idx, headDim, numHeads);
        const vSlice = await this._extractHeadSlice(qkvOut, seqLen, 3 * hiddenSize, 2 * hiddenSize, h_idx, headDim, numHeads);

        // Copy head slices into contiguous K and V buffers
        await this._copyHeadToOutput(kSlice, kBuf, seqLen, h_idx, headDim, numHeads);
        await this._copyHeadToOutput(vSlice, vBuf, seqLen, h_idx, headDim, numHeads);
      }

      // Store K,V in cache for all positions
      kvCache.appendBatch(l, kBuf, vBuf, 0, seqLen);

      // Standard attention (full sequence)
      const attnOut = await this._multiHeadAttention(qkvOut, seqLen, numHeads, headDim);

      const projOut = await this._matmul(
        attnOut, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.attn.c_proj.weight`), hiddenSize, hiddenSize
      );
      await this._addBias(projOut, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.attn.c_proj.bias`)
      );

      const residual1 = await this._residualAdd(h.buffer, projOut, seqLen * hiddenSize);

      const ln2Out = await this._layerNorm(
        residual1, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.ln_2.weight`),
        this.loader.getBuffer(`${prefix}.ln_2.bias`)
      );

      const ffnInnerDim = 4 * hiddenSize;
      const fcOut = await this._matmul(
        ln2Out, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.mlp.c_fc.weight`), hiddenSize, ffnInnerDim
      );
      await this._addBias(fcOut, seqLen, ffnInnerDim,
        this.loader.getBuffer(`${prefix}.mlp.c_fc.bias`)
      );

      const geluOut = await this._gelu(fcOut, seqLen * ffnInnerDim);

      const ffnOut = await this._matmul(
        geluOut, seqLen, ffnInnerDim,
        this.loader.getBuffer(`${prefix}.mlp.c_proj.weight`), ffnInnerDim, hiddenSize
      );
      await this._addBias(ffnOut, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.mlp.c_proj.bias`)
      );

      const residual2 = await this._residualAdd(residual1, ffnOut, seqLen * hiddenSize);
      h = { buffer: residual2, shape: [seqLen, hiddenSize] };

      // Cleanup temp buffers
      const keep = h.buffer;
      const surviving = [];
      for (const buf of this._tempBuffers) {
        if (buf === keep || weightBuffers.has(buf)) {
          surviving.push(buf);
        } else {
          buf.destroy();
        }
      }
      this._tempBuffers = surviving;
    }

    return h;
  }

  // ─── Cached Attention Kernel ────────────────────────────────

  /**
   * Multi-head attention using KV cache for single query token.
   * Q is from the new token's QKV, K and V come from the cache.
   */
  async _multiHeadAttentionCached(qkvBuf, numHeads, headDim, kvCache, layerIdx, cacheLen, headMask) {
    const hiddenSize = numHeads * headDim;
    const outputBuf = this._createBuffer("mha_cached_out", hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const { kBuffer: fullKBuf, vBuffer: fullVBuf } = kvCache.getKV(layerIdx);
    const scale = 1.0 / Math.sqrt(headDim);

    for (let h = 0; h < numHeads; h++) {
      // Head pruning: skip computation for low-importance heads
      // Output buffer is zero-initialized by WebGPU, so skipped heads contribute zeros
      if (headMask && !headMask[h]) continue;
      // Extract Q for this head from QKV [1, 3*hiddenSize]
      // Q is at columns [0..hiddenSize], head h at [h*headDim..(h+1)*headDim]
      const qBuf = await this._extractHeadSlice(qkvBuf, 1, 3 * hiddenSize, 0, h, headDim, numHeads);

      // Extract K_cache and V_cache for this head from the full cache buffer
      // Cache is [maxSeqLen, hiddenSize], head h at columns [h*headDim..(h+1)*headDim]
      // We need [cacheLen, headDim] slice
      const kCacheBuf = await this._extractHeadSlice(fullKBuf, cacheLen, hiddenSize, 0, h, headDim, numHeads);
      const vCacheBuf = await this._extractHeadSlice(fullVBuf, cacheLen, hiddenSize, 0, h, headDim, numHeads);

      // Pass 1: Compute scores = Q·K_cache^T / sqrt(d)
      const scoresBuf = this._createBuffer(`cached_scores_h${h}`, cacheLen * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

      const scoreParams = new ArrayBuffer(16);
      const sv = new DataView(scoreParams);
      sv.setUint32(0, cacheLen, true);
      sv.setUint32(4, headDim, true);
      sv.setFloat32(8, scale, true);
      sv.setUint32(12, 0, true);
      const scoreParamBuf = this._createBuffer(`cached_score_params_h${h}`, 16,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(scoreParamBuf, 0, new Uint8Array(scoreParams));

      const scoresPipeline = this._getOrCreatePipeline("attention_cached", "compute_scores_cached", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);

      let bg = this.device.createBindGroup({
        layout: scoresPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: scoreParamBuf } },
          { binding: 1, resource: { buffer: qBuf } },
          { binding: 2, resource: { buffer: kCacheBuf } },
          { binding: 3, resource: { buffer: scoresBuf } },
        ],
      });

      let encoder = this.device.createCommandEncoder();
      let pass = encoder.beginComputePass();
      pass.setPipeline(scoresPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(cacheLen / 256));
      pass.end();
      this.device.queue.submit([encoder.finish()]);

      // Pass 2: Softmax over single row
      const softmaxParams = new ArrayBuffer(16);
      const smv = new DataView(softmaxParams);
      smv.setUint32(0, cacheLen, true);
      smv.setUint32(4, headDim, true);
      smv.setFloat32(8, scale, true);
      smv.setUint32(12, 0, true);
      const softmaxParamBuf = this._createBuffer(`cached_sm_params_h${h}`, 16,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(softmaxParamBuf, 0, new Uint8Array(softmaxParams));

      const softmaxPipeline = this._getOrCreatePipeline("attention_cached_softmax", "softmax_cached", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);

      bg = this.device.createBindGroup({
        layout: softmaxPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: softmaxParamBuf } },
          { binding: 1, resource: { buffer: scoresBuf } },
        ],
      });

      encoder = this.device.createCommandEncoder();
      pass = encoder.beginComputePass();
      pass.setPipeline(softmaxPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1); // single row
      pass.end();
      this.device.queue.submit([encoder.finish()]);

      // Pass 3: Weighted sum = scores · V_cache -> [1, headDim]
      const headOutBuf = this._createBuffer(`cached_head_out_h${h}`, headDim * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

      const wsParams = new ArrayBuffer(16);
      const wsv = new DataView(wsParams);
      wsv.setUint32(0, cacheLen, true);
      wsv.setUint32(4, headDim, true);
      wsv.setFloat32(8, scale, true);
      wsv.setUint32(12, 0, true);
      const wsParamBuf = this._createBuffer(`cached_ws_params_h${h}`, 16,
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(wsParamBuf, 0, new Uint8Array(wsParams));

      const wsPipeline = this._getOrCreatePipeline("attention_cached_ws", "weighted_sum_cached", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);

      bg = this.device.createBindGroup({
        layout: wsPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: wsParamBuf } },
          { binding: 1, resource: { buffer: scoresBuf } },
          { binding: 2, resource: { buffer: vCacheBuf } },
          { binding: 3, resource: { buffer: headOutBuf } },
        ],
      });

      encoder = this.device.createCommandEncoder();
      pass = encoder.beginComputePass();
      pass.setPipeline(wsPipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(headDim / 256));
      pass.end();
      this.device.queue.submit([encoder.finish()]);

      // Copy head output to the concatenated output buffer
      await this._copyHeadToOutput(headOutBuf, outputBuf, 1, h, headDim, numHeads);
    }

    return outputBuf;
  }

  // ─── Internal Kernel Dispatchers ──────────────────────────

  async _layerNorm(inputBuf, seqLen, hiddenSize, gammaBuf, betaBuf) {
    const outputBuf = this._createBuffer("ln_out", seqLen * hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const params = new ArrayBuffer(16);
    const view = new DataView(params);
    view.setUint32(0, seqLen, true);
    view.setUint32(4, hiddenSize, true);
    view.setFloat32(8, 1e-5, true); // epsilon
    view.setUint32(12, 0, true);

    const paramBuf = this._createBuffer("ln_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, new Uint8Array(params));

    const pipeline = this._getOrCreatePipeline("layernorm", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: gammaBuf } },
        { binding: 3, resource: { buffer: betaBuf } },
        { binding: 4, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(seqLen); // one workgroup per row
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return outputBuf;
  }

  async _matmul(inputBuf, M, K, weightBuf, _K, N) {
    const outputBuf = this._createBuffer("mm_out", M * N * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const params = new Uint32Array([M, K, N, 0]);
    const paramBuf = this._createBuffer("mm_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this._getOrCreatePipeline("matmul", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: weightBuf } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return outputBuf;
  }

  /**
   * Matrix multiply with transposed B: C = A × Bᵀ
   * A is [M, K], B is [N, K] (stored row-major), C is [M, N]
   * Used for lm_head where weight is [vocab_size, hidden_size].
   */
  async _matmulTransB(inputBuf, M, K, weightBuf, N) {
    const outputBuf = this._createBuffer("mm_transB_out", M * N * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const params = new Uint32Array([M, K, N, 0]);
    const paramBuf = this._createBuffer("mm_transB_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this._getOrCreatePipeline("matmul_transB", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: weightBuf } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(M / 8), Math.ceil(N / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return outputBuf;
  }

  async _multiHeadAttention(qkvBuf, seqLen, numHeads, headDim, headMask) {
    const hiddenSize = numHeads * headDim;
    const outputBuf = this._createBuffer("mha_out", seqLen * hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // For each head, run the 3-pass attention kernel
    // QKV buffer layout: [seq, 3*hidden] → Q=[seq, hidden], K=[seq, hidden], V=[seq, hidden]
    // We process one head at a time, each head has contiguous Q[h], K[h], V[h] slices
    for (let h = 0; h < numHeads; h++) {
      // Head pruning: skip computation for low-importance heads
      if (headMask && !headMask[h]) continue;
      // Extract Q, K, V for this head from the combined QKV buffer
      // GPT-2 QKV layout: [seq_len, 3 * hidden_size] where Q = [:, 0:hidden], K = [:, hidden:2*hidden], V = [:, 2*hidden:3*hidden]
      // Within Q, head h occupies columns [h*head_dim : (h+1)*head_dim]
      const qBuf = await this._extractHeadSlice(qkvBuf, seqLen, 3 * hiddenSize, 0, h, headDim, numHeads);
      const kBuf = await this._extractHeadSlice(qkvBuf, seqLen, 3 * hiddenSize, hiddenSize, h, headDim, numHeads);
      const vBuf = await this._extractHeadSlice(qkvBuf, seqLen, 3 * hiddenSize, 2 * hiddenSize, h, headDim, numHeads);

      const scale = 1.0 / Math.sqrt(headDim);

      // Pass 1: Compute scores = Q·Kᵀ / sqrt(d) with causal mask
      const scoresBuf = this._createBuffer(`scores_h${h}`, seqLen * seqLen * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

      const scoreParams = new ArrayBuffer(16);
      const sv = new DataView(scoreParams);
      sv.setUint32(0, seqLen, true);
      sv.setUint32(4, headDim, true);
      sv.setFloat32(8, scale, true);
      sv.setUint32(12, 0, true);
      const scoreParamBuf = this._createBuffer(`score_params_h${h}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(scoreParamBuf, 0, new Uint8Array(scoreParams));

      const scoresPipeline = this._getOrCreatePipeline("attention", "compute_scores", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);

      let bg = this.device.createBindGroup({
        layout: scoresPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: scoreParamBuf } },
          { binding: 1, resource: { buffer: qBuf } },
          { binding: 2, resource: { buffer: kBuf } },
          { binding: 3, resource: { buffer: scoresBuf } },
        ],
      });

      let enc = this.device.createCommandEncoder();
      let p = enc.beginComputePass();
      p.setPipeline(scoresPipeline);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(seqLen / 8));
      p.end();
      this.device.queue.submit([enc.finish()]);

      // Pass 2: Softmax over scores (row-wise)
      const softmaxParams = new ArrayBuffer(16);
      const smv = new DataView(softmaxParams);
      smv.setUint32(0, seqLen, true);
      smv.setUint32(4, headDim, true);
      smv.setFloat32(8, scale, true);
      smv.setUint32(12, 0, true);
      const softmaxParamBuf = this._createBuffer(`sm_params_h${h}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(softmaxParamBuf, 0, new Uint8Array(softmaxParams));

      const softmaxPipeline = this._getOrCreatePipeline("attention_softmax", "softmax_rows", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);

      bg = this.device.createBindGroup({
        layout: softmaxPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: softmaxParamBuf } },
          { binding: 1, resource: { buffer: scoresBuf } },
        ],
      });

      enc = this.device.createCommandEncoder();
      p = enc.beginComputePass();
      p.setPipeline(softmaxPipeline);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(seqLen);
      p.end();
      this.device.queue.submit([enc.finish()]);

      // Pass 3: Weighted sum = softmax_scores · V
      const headOutBuf = this._createBuffer(`head_out_h${h}`, seqLen * headDim * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

      const wsParams = new ArrayBuffer(16);
      const wsv = new DataView(wsParams);
      wsv.setUint32(0, seqLen, true);
      wsv.setUint32(4, headDim, true);
      wsv.setFloat32(8, scale, true);
      wsv.setUint32(12, 0, true);
      const wsParamBuf = this._createBuffer(`ws_params_h${h}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(wsParamBuf, 0, new Uint8Array(wsParams));

      const wsPipeline = this._getOrCreatePipeline("attention_ws", "weighted_sum", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);

      bg = this.device.createBindGroup({
        layout: wsPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: wsParamBuf } },
          { binding: 1, resource: { buffer: scoresBuf } },
          { binding: 2, resource: { buffer: vBuf } },
          { binding: 3, resource: { buffer: headOutBuf } },
        ],
      });

      enc = this.device.createCommandEncoder();
      p = enc.beginComputePass();
      p.setPipeline(wsPipeline);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(Math.ceil(seqLen / 8), Math.ceil(headDim / 8));
      p.end();
      this.device.queue.submit([enc.finish()]);

      // Copy head output into the correct slice of the concatenated output
      await this._copyHeadToOutput(headOutBuf, outputBuf, seqLen, h, headDim, numHeads);
    }

    return outputBuf;
  }

  async _gelu(inputBuf, totalElements) {
    const outputBuf = this._createBuffer("gelu_out", totalElements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const params = new Uint32Array([totalElements, 0, 0, 0]);
    const paramBuf = this._createBuffer("gelu_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this._getOrCreatePipeline("gelu", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: inputBuf } },
        { binding: 2, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalElements / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return outputBuf;
  }

  async _residualAdd(aBuf, bBuf, totalElements) {
    const outputBuf = this._createBuffer("res_out", totalElements * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const params = new Uint32Array([totalElements, 0, 0, 0]);
    const paramBuf = this._createBuffer("res_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this._getOrCreatePipeline("residual_add", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: aBuf } },
        { binding: 2, resource: { buffer: bBuf } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(totalElements / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return outputBuf;
  }

  /**
   * Add bias to a [rows, cols] matrix in-place on GPU.
   * bias is [cols], broadcast across rows.
   */
  async _addBias(matBuf, rows, cols, biasBuf) {
    const total = rows * cols;
    const params = new Uint32Array([rows, cols, 0, 0]);
    const paramBuf = this._createBuffer("bias_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this._getOrCreatePipeline("bias_add", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: matBuf } },
        { binding: 2, resource: { buffer: biasBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Extract a single attention head's slice from the combined QKV buffer on GPU.
   */
  async _extractHeadSlice(qkvBuf, seqLen, qkvCols, sectionOffset, headIdx, headDim, numHeads) {
    const total = seqLen * headDim;
    const outputBuf = this._createBuffer(`head_slice_h${headIdx}`, total * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const params1 = new Uint32Array([seqLen, qkvCols, sectionOffset, headIdx * headDim]);
    const paramBuf1 = this._createBuffer(`hs_params1_h${headIdx}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf1, 0, params1);

    const params2 = new Uint32Array([headDim, 0, 0, 0]);
    const paramBuf2 = this._createBuffer(`hs_params2_h${headIdx}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf2, 0, params2);

    const pipeline = this._getOrCreatePipeline("head_slice", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf1 } },
        { binding: 1, resource: { buffer: paramBuf2 } },
        { binding: 2, resource: { buffer: qkvBuf } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return outputBuf;
  }

  /**
   * Copy a head's output [seq, headDim] into the concatenated output [seq, hidden] on GPU.
   */
  async _copyHeadToOutput(headBuf, outputBuf, seqLen, headIdx, headDim, numHeads) {
    const hiddenSize = numHeads * headDim;
    const total = seqLen * headDim;

    const params = new Uint32Array([seqLen, headDim, hiddenSize, headIdx * headDim]);
    const paramBuf = this._createBuffer(`hc_params_h${headIdx}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, params);

    const pipeline = this._getOrCreatePipeline("head_concat", "main", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramBuf } },
        { binding: 1, resource: { buffer: headBuf } },
        { binding: 2, resource: { buffer: outputBuf } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / 256));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  // ─── Buffer Utilities ─────────────────────────────────────

  _createBuffer(label, size, usage) {
    const buf = this.device.createBuffer({ label, size, usage });
    this._tempBuffers.push(buf);
    return buf;
  }

  /**
   * Destroy all temporary buffers created during inference.
   * Keeps weight buffers (owned by ShardLoader) intact.
   */
  _cleanupTempBuffers() {
    for (const buf of this._tempBuffers) {
      buf.destroy();
    }
    this._tempBuffers = [];
  }

  async _readBuffer(buffer, offset, size) {
    const stagingBuf = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offset, stagingBuf, 0, size);
    this.device.queue.submit([encoder.finish()]);

    await stagingBuf.mapAsync(GPUMapMode.READ);
    const data = stagingBuf.getMappedRange().slice(0);
    stagingBuf.unmap();
    stagingBuf.destroy();
    return data;
  }

  /**
   * Get or create a cached compute pipeline.
   */
  _getOrCreatePipeline(shaderName, entryPoint, layoutEntries) {
    const key = `${shaderName}:${entryPoint}`;
    if (this.pipelines[key]) return this.pipelines[key];

    // Map pipeline keys to their shader module names
    let moduleName = shaderName;
    if (shaderName === "attention_softmax" || shaderName === "attention_ws") {
      moduleName = "attention";
    } else if (shaderName === "attention_cached_softmax" || shaderName === "attention_cached_ws") {
      moduleName = "attention_cached";
    }

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: layoutEntries,
    });

    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    const pipeline = this.device.createComputePipeline({
      layout: pipelineLayout,
      compute: {
        module: this.shaderModules[moduleName],
        entryPoint,
      },
    });

    this.pipelines[key] = pipeline;
    return pipeline;
  }

  /**
   * Serialize a GPU tensor to a transferable object (base64 Float32Array).
   */
  async serializeTensor(tensor) {
    const byteSize = tensor.shape.reduce((a, b) => a * b, 1) * 4;
    const data = await this._readBuffer(tensor.buffer, 0, byteSize);
    const bytes = new Uint8Array(data);

    // Convert to base64
    let binary = "";
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return {
      shape: tensor.shape,
      dtype: "float32",
      data: btoa(binary),
    };
  }

  /**
   * Deserialize a received tensor and upload to a GPU buffer.
   */
  deserializeTensor(tensorMsg) {
    const { shape, data } = tensorMsg;
    const binary = atob(data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const floatData = new Float32Array(bytes.buffer);

    const buffer = this._createBuffer("deserialized", floatData.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(buffer, 0, floatData);

    return { buffer, shape };
  }

  // ─── Binary Protocol (v2) ───────────────────────────────────────

  /**
   * Serialize a GPU tensor to a raw ArrayBuffer (no base64, no JSON).
   * Used with the binary wire protocol.
   */
  async serializeTensorBinary(tensor) {
    const byteSize = tensor.shape.reduce((a, b) => a * b, 1) * 4;
    const data = await this._readBuffer(tensor.buffer, 0, byteSize);
    return {
      shape: tensor.shape,
      data, // raw ArrayBuffer — no base64 encoding
    };
  }

  /**
   * Deserialize a raw ArrayBuffer tensor and upload to GPU.
   * Used with the binary wire protocol.
   */
  deserializeTensorBinary(payload, shape) {
    // payload is a Uint8Array view — create Float32Array from its underlying buffer
    const floatData = new Float32Array(
      payload.buffer, payload.byteOffset, payload.byteLength / 4
    );

    const buffer = this._createBuffer("deserialized_bin", floatData.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(buffer, 0, floatData);

    return { buffer, shape };
  }

  // ─── Quantized Serialization (int8) ─────────────────────────────

  /**
   * Serialize a GPU tensor as int8-quantized for wire transfer.
   * Returns packed buffer: [int8_data...][float32_scale]
   */
  async serializeTensorQuantized(tensor) {
    const byteSize = tensor.shape.reduce((a, b) => a * b, 1) * 4;
    const data = await this._readBuffer(tensor.buffer, 0, byteSize);
    const float32 = new Float32Array(data);

    // Sanity check: if input has NaN, fall back to unquantized
    for (let i = 0; i < Math.min(float32.length, 64); i++) {
      if (!isFinite(float32[i])) {
        console.warn("[pipeline] NaN/Inf in activation — falling back to unquantized");
        return { ...await this.serializeTensorBinary(tensor), fallbackUnquantized: true };
      }
    }

    // Per-channel for multi-token (prefill), per-tensor for single token (cached step)
    // Single token has 1 row so per-channel adds overhead with no accuracy benefit
    const rows = tensor.shape[0] || 1;
    if (rows === 1) {
      const { data: int8Data, scale } = quantizeInt8(float32);
      const packed = packQuantized(int8Data, scale);
      return { shape: tensor.shape, data: packed };
    }

    const cols = tensor.shape[1] || float32.length;
    const { data: int8Data, scales } = quantizeInt8PerChannel(float32, cols);
    const packed = packQuantizedPerChannel(int8Data, scales);

    return {
      shape: tensor.shape,
      data: packed,
    };
  }

  /**
   * Deserialize a quantized tensor (int8) and upload to GPU as float32.
   * Auto-detects per-tensor vs per-channel format.
   */
  deserializeTensorQuantized(payload, shape) {
    const rows = shape[0] || 1;
    let floatData;

    if (rows === 1) {
      // Single token — per-tensor format: [int8_data][float32_scale]
      const { int8Data, scale } = unpackQuantized(payload);
      floatData = dequantizeInt8(int8Data, scale);
    } else {
      // Multi-token — per-channel format: [int8_data][scales][numRows]
      const { int8Data, scales } = unpackQuantizedPerChannel(payload);
      const cols = shape[1] || int8Data.length;
      floatData = dequantizeInt8PerChannel(int8Data, scales, cols);
    }

    const buffer = this._createBuffer("deserialized_quant", floatData.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(buffer, 0, floatData);

    return { buffer, shape };
  }

  // ─── Delta-Encoded Serialization ──────────────────────────────

  /**
   * Serialize a GPU tensor as delta-encoded int8 for wire transfer.
   * Computes delta = current - previous, quantizes the delta.
   * Returns { shape, data, isDelta, sparsity }.
   *
   * @param {object} tensor - { buffer, shape }
   * @param {Float32Array|null} previousFloat32 - Previous activation (null = send full)
   * @returns {Promise<{ shape, data, isDelta, sparsity, currentFloat32 }>}
   */
  async serializeTensorDelta(tensor, previousFloat32) {
    const byteSize = tensor.shape.reduce((a, b) => a * b, 1) * 4;
    const rawData = await this._readBuffer(tensor.buffer, 0, byteSize);
    const currentFloat32 = new Float32Array(rawData);

    // If no previous activation or shape mismatch, fall back to full send
    if (!previousFloat32 || previousFloat32.length !== currentFloat32.length) {
      const { data: int8Data, scale } = quantizeInt8(currentFloat32);
      const packed = packQuantized(int8Data, scale);
      return {
        shape: tensor.shape,
        data: packed,
        isDelta: false,
        sparsity: 0,
        currentFloat32,
      };
    }

    // Compute and quantize delta
    const delta = computeDelta(currentFloat32, previousFloat32);
    const sparsity = deltaSparsity(delta);
    const { data: int8Data, scale } = quantizeInt8(delta);
    const packed = packQuantized(int8Data, scale);

    return {
      shape: tensor.shape,
      data: packed,
      isDelta: true,
      sparsity,
      currentFloat32,
    };
  }

  /**
   * Deserialize a delta-encoded tensor: dequantize delta, add to previous.
   *
   * @param {Uint8Array} payload - Wire format (int8 + scale)
   * @param {number[]} shape
   * @param {Float32Array} previousFloat32 - Previous activation to add delta to
   * @returns {{ buffer, shape, currentFloat32 }}
   */
  deserializeTensorDeltaApply(payload, shape, previousFloat32) {
    const { int8Data, scale } = unpackQuantized(payload);
    const deltaFloat32 = dequantizeInt8(int8Data, scale);
    const currentFloat32 = applyDelta(deltaFloat32, previousFloat32);

    const buffer = this._createBuffer("deserialized_delta", currentFloat32.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(buffer, 0, currentFloat32);

    return { buffer, shape, currentFloat32 };
  }
}
