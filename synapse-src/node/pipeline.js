/**
 * Pipeline — Forward pass orchestration for GPT-2 transformer layers.
 *
 * Manages WebGPU compute shader dispatches for:
 *   embed → [layernorm → attention → residual → layernorm → FFN → residual] × N → final_ln → lm_head
 *
 * Each browser node runs a subset of layers (e.g., layers 0-5 or 6-11).
 */

export class Pipeline {
  constructor(device, shardLoader) {
    this.device = device;
    this.loader = shardLoader;
    this.config = shardLoader.getModelConfig();
    this.pipelines = {};   // cached compute pipelines
    this.shaderModules = {}; // cached shader modules
    this._initialized = false;
    this._tempBuffers = []; // track temporary buffers for cleanup
  }

  /**
   * Initialize compute pipelines by loading and compiling all WGSL shaders.
   */
  async init() {
    const shaderNames = [
      "matmul", "attention", "layernorm", "gelu", "residual_add", "embed",
      "bias_add", "head_slice", "head_concat",
    ];

    for (const name of shaderNames) {
      const code = await (await fetch(`/node/kernels/${name}.wgsl`)).text();
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
  async forwardLayer(hidden, layerIdx) {
    const { hiddenSize, numHeads, headDim } = this.config;
    const seqLen = hidden.shape[0];
    const prefix = `transformer.h.${layerIdx}`;

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
    const attnOut = await this._multiHeadAttention(qkvOut, seqLen, numHeads, headDim);

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

    let h = hidden;
    for (let l = layerStart; l <= layerEnd; l++) {
      h = await this.forwardLayer(h, l);
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

    // lm_head: [hidden_size, vocab_size] — project to logits
    const logits = await this._matmul(
      lnOut, seqLen, hiddenSize,
      this.loader.getBuffer("lm_head.weight"), hiddenSize, vocabSize
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

  async _multiHeadAttention(qkvBuf, seqLen, numHeads, headDim) {
    const hiddenSize = numHeads * headDim;
    const outputBuf = this._createBuffer("mha_out", seqLen * hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // For each head, run the 3-pass attention kernel
    // QKV buffer layout: [seq, 3*hidden] → Q=[seq, hidden], K=[seq, hidden], V=[seq, hidden]
    // We process one head at a time, each head has contiguous Q[h], K[h], V[h] slices
    for (let h = 0; h < numHeads; h++) {
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

    // For attention, the shaderModule name maps differently
    let moduleName = shaderName;
    if (shaderName === "attention_softmax" || shaderName === "attention_ws") {
      moduleName = "attention";
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
}
