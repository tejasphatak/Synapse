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
import { TensorSerializer } from "./tensor-serializer.js";

export class Pipeline {
  constructor(device, shardLoader) {
    this.device = device;
    this.loader = shardLoader;
    // shardLoader may be null for self-test / pre-assignment path. Use a
    // GPT-2 117M default config — kernels only need shape info, not weights.
    this.config = shardLoader
      ? shardLoader.getModelConfig()
      : { vocabSize: 50257, hiddenSize: 768, numHeads: 12, headDim: 64, maxSeqLen: 1024, numLayers: 12 };
    this.pipelines = {};   // cached compute pipelines
    this.shaderModules = {}; // cached shader modules
    this._initialized = false;
    this._tempBuffers = []; // track temporary buffers for cleanup
    this.kvCaches = new Map(); // requestId -> KVCache
    this.earlyExit = new EarlyExitDetector(); // disabled by default, tracks metrics
    this.modRouter = null; // MixtureOfDepths — initialized when layer range is known
    this.headPruner = null; // HeadPruner — initialized when layer range is known
    this.serializer = new TensorSerializer(
      device,
      (label, size, usage) => this._createBuffer(label, size, usage),
      (buffer, offset, size) => this._readBuffer(buffer, offset, size),
    );
    // Per-method cumulative timing (AOP — wraps every method at construction).
    // Caller must opt in via enableDebugProfile() before first forward —
    // otherwise methods run at native speed without per-call instrumentation.
    this._perfCounters = {};
  }

  enableDebugProfile() { this._installPerfAspect(); }

  /**
   * Run a closure with a shared command encoder. Every kernel dispatched
   * inside `fn` attaches its compute pass to the shared encoder instead
   * of submitting its own — collapsing N submit fences into 1. Caller
   * returns a value from fn (typically the final buffer).
   */
  async _withBatchedEncoder(fn) {
    if (this._currentEncoder) return fn(); // already batched by outer scope
    this._currentEncoder = this.device.createCommandEncoder();
    try {
      const r = await fn();
      this.device.queue.submit([this._currentEncoder.finish()]);
      return r;
    } finally {
      this._currentEncoder = null;
    }
  }

  /**
   * Add a compute pass to the shared encoder when batching is active; else
   * create a one-shot encoder + submit. All kernel JS wrappers route
   * through this instead of inlining encoder/pass boilerplate.
   */
  _dispatch(pipeline, bindGroup, wx, wy = 1, wz = 1) {
    const own = !this._currentEncoder;
    const enc = this._currentEncoder || this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wx, wy, wz);
    pass.end();
    if (own) this.device.queue.submit([enc.finish()]);
  }

  /**
   * Attach a buffer-to-buffer copy to the shared encoder or submit one-off.
   */
  _copyBuffer(src, srcOffset, dst, dstOffset, size) {
    const own = !this._currentEncoder;
    const enc = this._currentEncoder || this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
    if (own) this.device.queue.submit([enc.finish()]);
  }

  _perfReset() { this._perfCounters = {}; }
  _perfSnapshot() {
    const out = {};
    for (const [k, v] of Object.entries(this._perfCounters)) {
      out[k] = { ms: +v.ms.toFixed(2), calls: v.calls };
    }
    return out;
  }

  /**
   * Install timing aspect on every async method of Pipeline. Runs once at
   * construction. Skips the perf plumbing itself so we don't double-count.
   */
  _installPerfAspect() {
    const skip = new Set(["constructor", "_installPerfAspect", "_perfReset",
                          "_perfSnapshot"]);
    const proto = Object.getPrototypeOf(this);
    for (const name of Object.getOwnPropertyNames(proto)) {
      if (skip.has(name)) continue;
      const orig = proto[name];
      if (typeof orig !== "function") continue;
      const self = this;
      // Replace on THIS instance, not prototype, so other Pipeline instances
      // (if any, e.g., self-test) aren't affected and hot-reload's new class
      // isn't polluted.
      this[name] = function (...args) {
        const t0 = performance.now();
        let r;
        try { r = orig.apply(self, args); }
        catch (e) {
          const c = self._perfCounters[name] || (self._perfCounters[name] = { ms: 0, calls: 0 });
          c.ms += performance.now() - t0; c.calls += 1;
          throw e;
        }
        // Handle both sync + async returns.
        if (r && typeof r.then === "function") {
          return r.finally(() => {
            const c = self._perfCounters[name] || (self._perfCounters[name] = { ms: 0, calls: 0 });
            c.ms += performance.now() - t0; c.calls += 1;
          });
        }
        const c = self._perfCounters[name] || (self._perfCounters[name] = { ms: 0, calls: 0 });
        c.ms += performance.now() - t0; c.calls += 1;
        return r;
      };
    }
  }

  /**
   * Initialize compute pipelines by loading and compiling all WGSL shaders.
   */
  async init() {
    const shaderNames = [
      "matmul", "matmul_transB", "attention", "attention_cached", "layernorm", "gelu",
      "residual_add", "embed", "bias_add", "head_slice", "head_concat",
      // Gemma-family kernels (rmsnorm, rope, elementwise_mul, attention_gqa).
      // Safe to load unconditionally — each compiles once and incurs zero
      // runtime cost until its JS wrapper is called. Fetch errors are
      // swallowed so older deployments without the files don't break.
      "rmsnorm", "rope", "elementwise_mul", "attention_gqa", "gemma_embed",
    ];

    for (const name of shaderNames) {
      try {
        const resp = await fetch(`/node/kernels/${name}.wgsl?v=${Date.now()}`);
        // Treat missing `.ok` as truthy (test mocks don't set it); only skip
        // on explicit false. Real Response objects always set it.
        if (resp.ok === false) {
          console.warn(`[pipeline] skipping kernel ${name}: ${resp.status}`);
          continue;
        }
        const code = await resp.text();
        this.shaderModules[name] = this.device.createShaderModule({
          label: name,
          code,
        });
      } catch (e) {
        console.warn(`[pipeline] kernel ${name} compile failed: ${e.message}`);
        continue;
      }
    }

    this._initialized = true;
  }

  /**
   * Pre-flight self-test: runs synthetic inputs through each kernel and checks
   * for NaN/Inf/garbage. Devices that fail shouldn't join the inference pool.
   * Returns {pass: bool, failures: [{kernel, reason}]}.
   */
  async runSelfTest() {
    const failures = [];
    // Use the real model's hidden size so kernel workgroup assumptions hold.
    // layernorm.wgsl dispatches workgroup_size=256 expecting hiddenSize elements.
    const seqLen = 4;
    const hiddenSize = this.config.hiddenSize || 768;

    const mkBuf = (label, bytes, usage) => this._createBuffer(label, bytes,
      usage ?? (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC));
    const hasBadFloat = (arr) => Array.from(arr).some(v => Number.isNaN(v) || !Number.isFinite(v));

    // Test 1: LayerNorm with near-zero variance (all-ones input). Many GPU
    // drivers silently produce NaN from divide-by-zero if epsilon is too small
    // or handled wrong. This is the canonical failure mode.
    try {
      const input = new Float32Array(seqLen * hiddenSize).fill(1.0);
      const inBuf = mkBuf("st_ln_in", input.byteLength);
      this.device.queue.writeBuffer(inBuf, 0, input);
      const gamma = new Float32Array(hiddenSize).fill(1.0);
      const gammaBuf = mkBuf("st_ln_g", gamma.byteLength);
      this.device.queue.writeBuffer(gammaBuf, 0, gamma);
      const beta = new Float32Array(hiddenSize).fill(0.0);
      const betaBuf = mkBuf("st_ln_b", beta.byteLength);
      this.device.queue.writeBuffer(betaBuf, 0, beta);
      const outBuf = await this._layerNorm(inBuf, seqLen, hiddenSize, gammaBuf, betaBuf);
      const out = new Float32Array(await this._readBuffer(outBuf, 0, seqLen * hiddenSize * 4));
      if (hasBadFloat(out)) failures.push({ kernel: "layernorm_zerovar", sample: Array.from(out.slice(0, 4)) });
    } catch (e) {
      failures.push({ kernel: "layernorm", error: String(e.message || e) });
    }

    // Test 2: GELU at extreme values (tanh overflow, denormals, etc.).
    try {
      const input = new Float32Array([
        -100, -10, -1, -0.5, -0.001, 0, 0.001, 0.5,
        1, 10, 100, 1000, -1000, 1e20, -1e20, 1.5,
      ]);
      const inBuf = mkBuf("st_gelu", input.byteLength);
      this.device.queue.writeBuffer(inBuf, 0, input);
      await this._gelu(inBuf, input.length);
      const out = new Float32Array(await this._readBuffer(inBuf, 0, input.byteLength));
      if (hasBadFloat(out)) failures.push({ kernel: "gelu_extremes", sample: Array.from(out) });
    } catch (e) {
      failures.push({ kernel: "gelu", error: String(e.message || e) });
    }

    // Test 3: Matmul with known small inputs. Output should equal 16*1.0 = 16.0
    // for each of the 4 output positions.
    try {
      const a = new Float32Array(seqLen * hiddenSize).fill(1.0); // [4, 16]
      const aBuf = mkBuf("st_mm_a", a.byteLength);
      this.device.queue.writeBuffer(aBuf, 0, a);
      const b = new Float32Array(hiddenSize * hiddenSize).fill(1.0); // [16, 16]
      const bBuf = mkBuf("st_mm_b", b.byteLength);
      this.device.queue.writeBuffer(bBuf, 0, b);
      const outBuf = await this._matmul(aBuf, seqLen, hiddenSize, bBuf, hiddenSize, hiddenSize);
      const out = new Float32Array(await this._readBuffer(outBuf, 0, seqLen * hiddenSize * 4));
      if (hasBadFloat(out)) {
        failures.push({ kernel: "matmul_nan", sample: Array.from(out.slice(0, 4)) });
      } else {
        const expected = hiddenSize;
        const err = Math.max(...Array.from(out).map(v => Math.abs(v - expected)));
        if (err > 0.01) failures.push({ kernel: "matmul_wrong", expected, err: +err.toFixed(4), sample: Array.from(out.slice(0, 4)) });
      }
    } catch (e) {
      failures.push({ kernel: "matmul", error: String(e.message || e) });
    }

    // Test 4: Chained multi-kernel pipeline. Some device bugs only manifest
    // when kernels feed each other's intermediate state (e.g., Intel
    // 2026-04-15: individual kernels pass, but chaining layernorm →
    // matmul → residual → gelu produces NaN). Simulate the core of a
    // transformer layer without weights: three sequential layernorm +
    // matmul ops against identity-ish matrices, checking for NaN/Inf
    // and sane magnitude growth at each step.
    try {
      const input = new Float32Array(seqLen * hiddenSize);
      for (let i = 0; i < input.length; i++) input[i] = Math.sin(i * 0.01) * 0.5;
      let curBuf = mkBuf("st_chain_in", input.byteLength);
      this.device.queue.writeBuffer(curBuf, 0, input);

      const gamma = new Float32Array(hiddenSize).fill(1.0);
      const gammaBuf = mkBuf("st_chain_g", gamma.byteLength);
      this.device.queue.writeBuffer(gammaBuf, 0, gamma);
      const beta = new Float32Array(hiddenSize).fill(0.0);
      const betaBuf = mkBuf("st_chain_b", beta.byteLength);
      this.device.queue.writeBuffer(betaBuf, 0, beta);

      // Identity-like matrix (diagonal 1.0) lets us preserve signal through
      // matmul so we can detect NaN introduction without weight noise.
      const wId = new Float32Array(hiddenSize * hiddenSize);
      for (let i = 0; i < hiddenSize; i++) wId[i * hiddenSize + i] = 1.0;
      const wBuf = mkBuf("st_chain_w", wId.byteLength);
      this.device.queue.writeBuffer(wBuf, 0, wId);

      let chainFailed = false;
      for (let step = 0; step < 3 && !chainFailed; step++) {
        const lnOut = await this._layerNorm(curBuf, seqLen, hiddenSize, gammaBuf, betaBuf);
        const mmOut = await this._matmul(lnOut, seqLen, hiddenSize, wBuf, hiddenSize, hiddenSize);
        const out = new Float32Array(await this._readBuffer(mmOut, 0, seqLen * hiddenSize * 4));
        let nans = 0, sum2 = 0;
        for (let i = 0; i < out.length; i++) {
          const v = out[i];
          if (Number.isNaN(v) || !Number.isFinite(v)) nans++;
          else sum2 += v * v;
        }
        const rms = Math.sqrt(sum2 / Math.max(1, out.length - nans));
        if (nans > 0) {
          failures.push({ kernel: "chain_nan", step, nans, sampleIdx: Array.from(out.slice(0, 4)) });
          chainFailed = true;
        } else if (rms > 100 || rms < 0.001) {
          // Post-layernorm + identity-matmul rms should be ~1. Extreme values
          // indicate broken layernorm or matmul numerics.
          failures.push({ kernel: "chain_rms_out_of_range", step, rms: +rms.toFixed(3) });
          chainFailed = true;
        }
        curBuf = mmOut;
      }
    } catch (e) {
      failures.push({ kernel: "chain", error: String(e.message || e) });
    }

    return { pass: failures.length === 0, failures };
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
   * Gemma embedding: token IDs → hidden state [seq_len, hidden_size].
   * No positional embedding (RoPE handles that). Output is scaled by
   * sqrt(hidden_size) per Gemma 3 spec (embed_scale).
   *
   * Implementation: CPU-side gather from the loader's float16 tensor bytes
   * directly into a Float32Array, then upload. Simpler than a dedicated
   * WGSL kernel and runs once per prefill so perf cost is negligible.
   */
  async gemmaEmbed(tokenIds) {
    const { hiddenSize } = this.config;
    const seqLen = tokenIds.length;
    const embedBuf = this.loader.getBuffer("model.embed_tokens.weight");
    if (!embedBuf) throw new Error("Gemma embed: model.embed_tokens.weight not loaded");

    // Upload token IDs and dispatch gemma_embed.wgsl. Gather runs on-GPU
    // so we don't need to read back the (huge) embedding matrix.
    const tokBuf = this._createBuffer("gemma_embed_tokids", seqLen * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(tokBuf, 0, new Uint32Array(tokenIds));

    const outputBuf = this._createBuffer("gemma_embed_out", seqLen * hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    const params = new ArrayBuffer(16);
    const v = new DataView(params);
    v.setUint32(0, seqLen, true);
    v.setUint32(4, hiddenSize, true);
    v.setFloat32(8, Math.sqrt(hiddenSize), true);
    const pBuf = this._createBuffer("gemma_embed_params", 16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(pBuf, 0, new Uint8Array(params));

    const pipe = this._getOrCreatePipeline("gemma_embed", "gemma_embed", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ]);
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pBuf } },
        { binding: 1, resource: { buffer: tokBuf } },
        { binding: 2, resource: { buffer: embedBuf } },
        { binding: 3, resource: { buffer: outputBuf } },
      ],
    });
    const total = seqLen * hiddenSize;
    this._dispatch(pipe, bg, Math.ceil(total / 256));
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
  async sampleToken(logitsTensor, opts = 0.8) {
    const { vocabSize } = this.config;
    const seqLen = logitsTensor.shape[0];

    // Backward-compat: callers that pass a plain number get treated as
    // { temperature: number } with default top-k/top-p. Objects override.
    const sOpts = typeof opts === "number" ? { temperature: opts } : (opts || {});
    const temperature = sOpts.temperature ?? 0.8;
    const topK = sOpts.topK ?? 40;           // 0 disables; 40 is a common default
    const topP = sOpts.topP ?? 0.95;         // 0 disables; 0.95 is nucleus default

    // Lazy-load banned tokens from manifest (Gemma has 6242 <unused*>
    // tokens scattered across the vocab that shouldn't be sampled).
    if (!this._bannedTokensSet && this.loader?.manifest?.banned_sampling_tokens) {
      this._bannedTokensSet = new Set(this.loader.manifest.banned_sampling_tokens);
    }
    const banned = this._bannedTokensSet;

    // Read only the last position's logits from GPU
    const offset = (seqLen - 1) * vocabSize * 4;
    const logits = await this._readBuffer(logitsTensor.buffer, offset, vocabSize * 4);
    const logitsF32 = new Float32Array(logits);

    // Snapshot raw (pre-temperature) logits so the parity verifier can
    // compare against HF/numpy output without having to re-apply temperature.
    const rawLogits = Float32Array.from(logitsF32);

    // Temperature scaling
    if (temperature !== 1.0) {
      for (let i = 0; i < logitsF32.length; i++) logitsF32[i] /= temperature;
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
    for (let i = 0; i < probs.length; i++) probs[i] /= sumExp;

    // Ban unused/special tokens: zero their probs so they can't survive
    // top-k/top-p. Applied BEFORE the filter so they're guaranteed out.
    if (banned && banned.size) {
      for (const id of banned) {
        if (id < probs.length) probs[id] = 0;
      }
      // Renormalize after zeroing
      let renormSum = 0;
      for (let i = 0; i < probs.length; i++) renormSum += probs[i];
      if (renormSum > 0) {
        for (let i = 0; i < probs.length; i++) probs[i] /= renormSum;
      }
    }

    // Top-K + Top-P filter. For 262k-vocab Gemma, keeping only top-40
    // tokens concentrates mass and eliminates long-tail noise that
    // causes "Thank You. Thank You." style loops under pure multinomial.
    if ((topK > 0 && topK < probs.length) || (topP > 0 && topP < 1)) {
      const order = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
      // Top-K: zero everything past K
      const kCut = topK > 0 ? Math.min(topK, order.length) : order.length;
      // Top-P: accumulate prob mass, cut when cumulative > topP
      let cumulative = 0;
      let pCut = order.length;
      if (topP > 0 && topP < 1) {
        for (let i = 0; i < order.length; i++) {
          cumulative += probs[order[i]];
          if (cumulative >= topP) { pCut = i + 1; break; }
        }
      }
      const keep = Math.max(1, Math.min(kCut, pCut));
      const mask = new Uint8Array(probs.length);
      for (let i = 0; i < keep; i++) mask[order[i]] = 1;
      // Re-normalize survivors
      let survSum = 0;
      for (let i = 0; i < probs.length; i++) {
        if (!mask[i]) probs[i] = 0; else survSum += probs[i];
      }
      if (survSum > 0) {
        for (let i = 0; i < probs.length; i++) probs[i] /= survSum;
      }
    }

    // Instrumentation: log top-5 tokens with probabilities. Helps diagnose
    // drift (if EOT=50256 dominates the softmax, accumulated activation drift
    // has pushed the logits; if normal tokens dominate but sampler still
    // picks weirdly, sampler is at fault).
    if (this._sampleCallCount === undefined) this._sampleCallCount = 0;
    this._sampleCallCount++;
    if (this._sampleCallCount <= 8 || this._sampleCallCount % 10 === 0) {
      // Rank by RAW (pre-temperature) logits so ordering and values are
      // directly comparable to HF / numpy reference output.
      const order = Array.from(rawLogits.keys()).sort((a, b) => rawLogits[b] - rawLogits[a]);
      const top5 = order.slice(0, 5).map((i) => `${i}:${probs[i].toFixed(3)}`);
      const top20 = order.slice(0, 20).map((i) => [i, +rawLogits[i].toFixed(4)]);
      let maxL = -Infinity, minL = Infinity;
      for (let i = 0; i < rawLogits.length; i++) {
        const v = rawLogits[i];
        if (v > maxL) maxL = v;
        if (v < minL) minL = v;
      }
      this._lastSampleTop = { top: top5, top20, maxLogit: maxL, minLogit: minL };
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
  getOrCreateKVCache(requestId, layerStart, numLayers, kvHiddenSize = null) {
    if (!this.kvCaches.has(requestId)) {
      const { hiddenSize, maxSeqLen } = this.config;
      this.kvCaches.set(requestId, new KVCache(
        this.device, numLayers, layerStart, hiddenSize, maxSeqLen,
        kvHiddenSize ?? hiddenSize,
      ));
    }
    return this.kvCaches.get(requestId);
  }

  /**
   * Gemma-specific cached-decode multi-layer orchestrator. Single-token
   * step across the [layerStart, layerEnd] range, using a KVCache sized
   * for numKvHeads * headDim.
   */
  async forwardLayersGemmaCached(hidden, layerStart, layerEnd, cfg, ropeBufs, requestId, seqPos) {
    const numLayers = layerEnd - layerStart + 1;
    const kvHidden = cfg.numKvHeads * cfg.headDim;
    const kvCache = this.getOrCreateKVCache(requestId, layerStart, numLayers, kvHidden);
    let cur = hidden.buffer;
    for (let l = layerStart; l <= layerEnd; l++) {
      cur = await this._withBatchedEncoder(
        () => this.forwardLayerGemmaCached(cur, l, seqPos, cfg, ropeBufs, kvCache)
      );
    }
    return { buffer: cur, shape: [1, cfg.hiddenSize] };
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

        // Free temp buffers from skip path immediately — on mobile GPUs,
        // letting them accumulate until the next non-skipped layer is wasteful.
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

    // Sub-kernel diagnostic helper. Pushes into this._subKernelTrace if
    // it's set externally (typically only on layer 0 of shard 0 to limit
    // readback cost). Captures rms + nans after each kernel inside layer 0.
    const trace = async (label, buf, elements) => {
      if (!this._subKernelTrace || this._subKernelTrace._stop) return;
      const f = new Float32Array(await this._readBuffer(buf, 0, elements * 4));
      let nans = 0, sum2 = 0, mn = Infinity, mx = -Infinity;
      for (let i = 0; i < f.length; i++) {
        const v = f[i];
        if (Number.isNaN(v)) nans++;
        else { sum2 += v * v; if (v < mn) mn = v; if (v > mx) mx = v; }
      }
      this._subKernelTrace.push({
        label, nans,
        rms: +Math.sqrt(sum2 / Math.max(1, f.length - nans)).toFixed(3),
        min: isFinite(mn) ? +mn.toFixed(3) : null,
        max: isFinite(mx) ? +mx.toFixed(3) : null,
      });
      if (nans > 0) this._subKernelTrace._stop = true;
    };

    let h = hidden;
    for (let l = layerStart; l <= layerEnd; l++) {
      const prefix = `transformer.h.${l}`;

      // Standard forward layer computation
      const ln1Out = await this._layerNorm(
        h.buffer, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.ln_1.weight`),
        this.loader.getBuffer(`${prefix}.ln_1.bias`)
      );
      if (l === layerStart) await trace("ln1", ln1Out, seqLen * hiddenSize);

      const qkvOut = await this._matmul(
        ln1Out, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.attn.c_attn.weight`), hiddenSize, 3 * hiddenSize
      );
      if (l === layerStart) await trace("qkv_matmul", qkvOut, seqLen * 3 * hiddenSize);
      await this._addBias(qkvOut, seqLen, 3 * hiddenSize,
        this.loader.getBuffer(`${prefix}.attn.c_attn.bias`)
      );
      if (l === layerStart) await trace("qkv_bias", qkvOut, seqLen * 3 * hiddenSize);

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
      if (l === layerStart) await trace("attention", attnOut, seqLen * hiddenSize);

      const projOut = await this._matmul(
        attnOut, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.attn.c_proj.weight`), hiddenSize, hiddenSize
      );
      if (l === layerStart) await trace("attn_proj_matmul", projOut, seqLen * hiddenSize);
      await this._addBias(projOut, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.attn.c_proj.bias`)
      );
      if (l === layerStart) await trace("attn_proj_bias", projOut, seqLen * hiddenSize);

      const residual1 = await this._residualAdd(h.buffer, projOut, seqLen * hiddenSize);
      if (l === layerStart) await trace("residual1", residual1, seqLen * hiddenSize);

      const ln2Out = await this._layerNorm(
        residual1, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.ln_2.weight`),
        this.loader.getBuffer(`${prefix}.ln_2.bias`)
      );
      if (l === layerStart) await trace("ln2", ln2Out, seqLen * hiddenSize);

      const ffnInnerDim = 4 * hiddenSize;
      const fcOut = await this._matmul(
        ln2Out, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.mlp.c_fc.weight`), hiddenSize, ffnInnerDim
      );
      if (l === layerStart) await trace("fc_matmul", fcOut, seqLen * ffnInnerDim);
      await this._addBias(fcOut, seqLen, ffnInnerDim,
        this.loader.getBuffer(`${prefix}.mlp.c_fc.bias`)
      );
      if (l === layerStart) await trace("fc_bias", fcOut, seqLen * ffnInnerDim);

      const geluOut = await this._gelu(fcOut, seqLen * ffnInnerDim);
      if (l === layerStart) await trace("gelu", geluOut, seqLen * ffnInnerDim);

      const ffnOut = await this._matmul(
        geluOut, seqLen, ffnInnerDim,
        this.loader.getBuffer(`${prefix}.mlp.c_proj.weight`), ffnInnerDim, hiddenSize
      );
      if (l === layerStart) await trace("ffn_matmul", ffnOut, seqLen * hiddenSize);
      await this._addBias(ffnOut, seqLen, hiddenSize,
        this.loader.getBuffer(`${prefix}.mlp.c_proj.bias`)
      );
      if (l === layerStart) await trace("ffn_bias", ffnOut, seqLen * hiddenSize);

      const residual2 = await this._residualAdd(residual1, ffnOut, seqLen * hiddenSize);
      if (l === layerStart) await trace("residual2", residual2, seqLen * hiddenSize);
      h = { buffer: residual2, shape: [seqLen, hiddenSize] };

      // Per-layer NaN trace: expose for node-level logging so we can
      // pinpoint which layer first introduces NaN in a multi-shard pipeline.
      if (this._nanTrace) {
        const sz = seqLen * hiddenSize * 4;
        const bufData = await this._readBuffer(h.buffer, 0, sz);
        const f = new Float32Array(bufData);
        let nans = 0, sum2 = 0;
        for (let i = 0; i < f.length; i++) {
          const v = f[i];
          if (Number.isNaN(v)) nans++;
          else sum2 += v * v;
        }
        this._nanTrace.push({
          layer: l,
          nans,
          rms: +Math.sqrt(sum2 / Math.max(1, f.length - nans)).toFixed(3),
        });
      }

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

  // ─── Gemma-family kernels ───────────────────────────────────
  // Thin JS wrappers over rmsnorm.wgsl, rope.wgsl, elementwise_mul.wgsl,
  // attention_gqa.wgsl. Each is independently testable — the full Gemma
  // forward loop calls them in sequence in a later method.

  /**
   * RMSNorm: y[i] = x[i] / sqrt(mean(x[i]^2) + eps) * gamma[i].
   * Drop-in for Gemma / Llama / Qwen / Mistral (replaces LayerNorm).
   */
  async _rmsNorm(inputBuf, seqLen, hiddenSize, gammaBuf, eps = 1e-6, gammaBias = 0.0) {
    // gammaBias: 0.0 for Llama/GPT-style (y = x/rms * gamma),
    //            1.0 for Gemma 3 family (y = x/rms * (1 + gamma)).
    const outputBuf = this._createBuffer("rms_out", seqLen * hiddenSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const params = new ArrayBuffer(16);
    const view = new DataView(params);
    view.setUint32(0, seqLen, true);
    view.setUint32(4, hiddenSize, true);
    view.setFloat32(8, eps, true);
    view.setFloat32(12, gammaBias, true);
    const paramBuf = this._createBuffer("rms_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, new Uint8Array(params));

    const pipeline = this._getOrCreatePipeline("rmsnorm", "rmsnorm", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ]);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuf } },
        { binding: 1, resource: { buffer: gammaBuf } },
        { binding: 2, resource: { buffer: outputBuf } },
        { binding: 3, resource: { buffer: paramBuf } },
      ],
    });
    this._dispatch(pipeline, bindGroup, seqLen); // one workgroup per row
    return outputBuf;
  }

  /**
   * RoPE (in-place): rotates pairs of dimensions in Q or K by precomputed
   * cos/sin angles. xBuf shape [seq, num_heads, head_dim]. Interleaved pairs.
   * startPos = position offset for cached-step decode (0 for prefill).
   */
  async _rope(xBuf, seqLen, numHeads, headDim, cosCacheBuf, sinCacheBuf, startPos = 0) {
    if (headDim % 2 !== 0) throw new Error(`head_dim must be even, got ${headDim}`);
    const params = new ArrayBuffer(16);
    const v = new DataView(params);
    v.setUint32(0, seqLen, true);
    v.setUint32(4, numHeads, true);
    v.setUint32(8, headDim, true);
    v.setUint32(12, startPos, true);
    const paramBuf = this._createBuffer("rope_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, new Uint8Array(params));

    const pipeline = this._getOrCreatePipeline("rope", "rope", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ]);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: xBuf } },
        { binding: 1, resource: { buffer: cosCacheBuf } },
        { binding: 2, resource: { buffer: sinCacheBuf } },
        { binding: 3, resource: { buffer: paramBuf } },
      ],
    });
    const totalPairs = seqLen * numHeads * (headDim / 2);
    this._dispatch(pipeline, bindGroup, Math.ceil(totalPairs / 256));
    return xBuf; // in-place
  }

  /** c = a * b (elementwise). Same-length 1D buffers, optionally in-place (c===a). */
  async _elementwiseMul(aBuf, bBuf, length, outBuf = null) {
    const resultBuf = outBuf ?? this._createBuffer("emul_out", length * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const params = new ArrayBuffer(16);
    new DataView(params).setUint32(0, length, true);
    const paramBuf = this._createBuffer("emul_params", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(paramBuf, 0, new Uint8Array(params));

    const pipeline = this._getOrCreatePipeline("elementwise_mul", "elementwise_mul", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ]);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: aBuf } },
        { binding: 1, resource: { buffer: bBuf } },
        { binding: 2, resource: { buffer: resultBuf } },
        { binding: 3, resource: { buffer: paramBuf } },
      ],
    });
    this._dispatch(pipeline, bindGroup, Math.ceil(length / 256));
    return resultBuf;
  }

  /**
   * GQA / MQA attention with optional sliding-window causal mask.
   * Three-pass kernel: qk_scores → softmax_rows → attend.
   *
   * Q: [seq, num_q_heads, head_dim]
   * K: [seq, num_kv_heads, head_dim]
   * V: [seq, num_kv_heads, head_dim]
   * Returns: [seq, num_q_heads, head_dim] (same shape as Q).
   *
   * numKvHeads === numQHeads → standard MHA.
   * numKvHeads <  numQHeads  → GQA.
   * numKvHeads === 1          → MQA (Gemma 3 1B default).
   * windowSize === 0         → pure causal (no sliding window).
   * windowSize  > 0           → sliding-window causal (Gemma 3 uses 512 on some layers).
   */
  async _attentionGqa(qBuf, kBuf, vBuf, seqLen, numQHeads, numKvHeads, headDim, windowSize = 0, invSqrtScale = null) {
    // Scores buffer: [numQHeads, seqLen, seqLen]
    const scoresBuf = this._createBuffer("gqa_scores", numQHeads * seqLen * seqLen * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    const outputBuf = this._createBuffer("gqa_out", seqLen * numQHeads * headDim * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    // Pass 1: Q·Kᵀ/√d, causal + sliding mask. Params1: [S,HQ,HK,D,W,0,0,0] → 32 bytes.
    const p1 = new ArrayBuffer(32);
    const v1 = new DataView(p1);
    v1.setUint32(0, seqLen, true);
    v1.setUint32(4, numQHeads, true);
    v1.setUint32(8, numKvHeads, true);
    v1.setUint32(12, headDim, true);
    v1.setUint32(16, windowSize, true);
    v1.setFloat32(20, invSqrtScale ?? (1.0 / Math.sqrt(headDim)), true);
    const p1Buf = this._createBuffer("gqa_p1", 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(p1Buf, 0, new Uint8Array(p1));

    const pipe1 = this._getOrCreatePipeline("attention_gqa_p1", "qk_scores", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ], "attention_gqa");
    const bg1 = this.device.createBindGroup({
      layout: pipe1.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: qBuf } },
        { binding: 1, resource: { buffer: kBuf } },
        { binding: 2, resource: { buffer: scoresBuf } },
        { binding: 3, resource: { buffer: p1Buf } },
      ],
    });

    // Pass 2: softmax over rows. Params2: [S,0,0,0] → 16 bytes.
    const p2 = new ArrayBuffer(16);
    new DataView(p2).setUint32(0, seqLen, true);
    const p2Buf = this._createBuffer("gqa_p2", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(p2Buf, 0, new Uint8Array(p2));

    const pipe2 = this._getOrCreatePipeline("attention_gqa_p2", "softmax_rows", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ], "attention_gqa");
    const bg2 = this.device.createBindGroup({
      layout: pipe2.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: scoresBuf } },
        { binding: 1, resource: { buffer: p2Buf } },
      ],
    });

    // Pass 3: attend — weighted sum of V. Params3: [S,HQ,HK,D] → 16 bytes.
    const p3 = new ArrayBuffer(16);
    const v3v = new DataView(p3);
    v3v.setUint32(0, seqLen, true);
    v3v.setUint32(4, numQHeads, true);
    v3v.setUint32(8, numKvHeads, true);
    v3v.setUint32(12, headDim, true);
    const p3Buf = this._createBuffer("gqa_p3", 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(p3Buf, 0, new Uint8Array(p3));

    const pipe3 = this._getOrCreatePipeline("attention_gqa_p3", "attend", [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ], "attention_gqa");
    const bg3 = this.device.createBindGroup({
      layout: pipe3.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: scoresBuf } },
        { binding: 1, resource: { buffer: vBuf } },
        { binding: 2, resource: { buffer: outputBuf } },
        { binding: 3, resource: { buffer: p3Buf } },
      ],
    });

    // Three passes routed through _dispatch so they attach to the outer
    // shared encoder when the layer is batched.
    this._dispatch(pipe1, bg1, Math.ceil(seqLen / 8), Math.ceil(seqLen / 8), numQHeads);
    this._dispatch(pipe2, bg2, seqLen, numQHeads, 1);
    this._dispatch(pipe3, bg3, Math.ceil(seqLen / 8), Math.ceil(headDim / 8), numQHeads);
    return outputBuf;
  }

  /**
   * One transformer layer of Gemma 3 (prefill / no-KV-cache path).
   *
   * Composes the four Gemma-family kernels + matmul + residual_add in the
   * architecture's exact order:
   *
   *   residual = x
   *   h = rmsnorm(x, input_layernorm.weight)
   *   q = matmul(h, q_proj); k = matmul(h, k_proj); v = matmul(h, v_proj)
   *   q = rope(q, cos, sin);   k = rope(k, cos, sin)
   *   attn = attention_gqa(q, k, v, window_size)
   *   o = matmul(attn, o_proj)
   *   x = residual_add(residual, o)                              ← residual around attn
   *   residual = x
   *   h = rmsnorm(x, post_attention_layernorm.weight)
   *   gate = matmul(h, gate_proj); up = matmul(h, up_proj)
   *   gated = elementwise_mul(gelu(gate), up)
   *   down = matmul(gated, down_proj)
   *   x = residual_add(residual, down)                            ← residual around mlp
   *   return x
   *
   * Caller owns: reading layer weights off this.loader + uploading cos/sin
   * buffers once per forward (not per layer). See forwardLayersGemmaPrefill
   * for that orchestration (shipped separately).
   *
   * @param {GPUBuffer} xBuf — [seqLen, hiddenSize] input, fp32
   * @param {number} l — layer index (for weight lookup)
   * @param {object} cfg — model config with { hiddenSize, numQHeads, numKvHeads, headDim, intermediateSize, windowSize, rmsEps }
   * @param {GPUBuffer} cosBuf / sinBuf — precomputed RoPE caches
   * @returns {Promise<GPUBuffer>} — output buffer [seqLen, hiddenSize]
   */
  async forwardLayerGemmaPrefill(xBuf, l, seqLen, cfg, ropeBufs) {
    const { hiddenSize, numQHeads, numKvHeads, headDim, intermediateSize,
            windowSize, rmsEps, invSqrtScale, layerTypes } = cfg;
    const prefix = `model.layers.${l}`;
    const getW = (name) => {
      const b = this.loader.getBuffer(`${prefix}.${name}.weight`);
      if (!b) throw new Error(`missing weight: ${prefix}.${name}.weight`);
      return b;
    };

    // Layer 0 sub-kernel probe: when _subKernelTrace array is set by the
    // caller, stats every intermediate so we can diff live vs numpy.
    const probeSub = l === 0 && Array.isArray(this._subKernelTrace);
    const probe = async (buf, elems, name) => {
      if (!probeSub) return;
      const f = new Float32Array(await this._readBuffer(buf, 0, elems * 4));
      let mn = Infinity, mx = -Infinity, sum2 = 0;
      for (let i = 0; i < f.length; i++) {
        const v = f[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum2 += v * v;
      }
      this._subKernelTrace.push({
        name, rms: +Math.sqrt(sum2 / elems).toFixed(4),
        min: +mn.toFixed(3), max: +mx.toFixed(3), n: elems,
      });
    };

    // Per-layer attention type selects (1) sliding-window size and
    // (2) which RoPE cache to use. Gemma 3 interleaves sliding+full.
    const isSliding = layerTypes ? layerTypes[l] === "sliding_attention" : (windowSize > 0);
    const layerWindow = isSliding ? (windowSize || 0) : 0;
    const cosBuf = isSliding && ropeBufs.cosLocal ? ropeBufs.cosLocal : ropeBufs.cos;
    const sinBuf = isSliding && ropeBufs.sinLocal ? ropeBufs.sinLocal : ropeBufs.sin;

    // ── Attention block ──────────────────────────────────────────
    const residual1 = xBuf;

    await probe(xBuf, seqLen * hiddenSize, "xBuf (layer_in)");

    // Gemma 3: RMSNorm uses y = x/rms * (1 + gamma). gammaBias = 1.0.
    const ln1 = await this._rmsNorm(xBuf, seqLen, hiddenSize,
      getW("input_layernorm"), rmsEps, 1.0);
    await probe(ln1, seqLen * hiddenSize, "ln1");

    const qFull = numQHeads * headDim;
    const kvFull = numKvHeads * headDim;
    const q = await this._matmulTransB(ln1, seqLen, hiddenSize, getW("self_attn.q_proj"), qFull);
    const k = await this._matmulTransB(ln1, seqLen, hiddenSize, getW("self_attn.k_proj"), kvFull);
    const v = await this._matmulTransB(ln1, seqLen, hiddenSize, getW("self_attn.v_proj"), kvFull);
    await probe(q, seqLen * qFull, "q_proj");
    await probe(k, seqLen * kvFull, "k_proj");
    await probe(v, seqLen * kvFull, "v_proj");

    await this._rmsNormInPlace(q, seqLen * numQHeads, headDim, getW("self_attn.q_norm"), rmsEps, 1.0);
    await this._rmsNormInPlace(k, seqLen * numKvHeads, headDim, getW("self_attn.k_norm"), rmsEps, 1.0);
    await probe(q, seqLen * qFull, "q_norm");
    await probe(k, seqLen * kvFull, "k_norm");

    await this._rope(q, seqLen, numQHeads,  headDim, cosBuf, sinBuf, 0);
    await this._rope(k, seqLen, numKvHeads, headDim, cosBuf, sinBuf, 0);
    await probe(q, seqLen * qFull, "q_rope");
    await probe(k, seqLen * kvFull, "k_rope");

    const attnOut = await this._attentionGqa(q, k, v, seqLen,
      numQHeads, numKvHeads, headDim, layerWindow, invSqrtScale);
    await probe(attnOut, seqLen * qFull, "attn_gqa");

    const oProj = await this._matmulTransB(attnOut, seqLen, qFull, getW("self_attn.o_proj"), hiddenSize);
    await probe(oProj, seqLen * hiddenSize, "o_proj");

    const oNorm = await this._rmsNorm(oProj, seqLen, hiddenSize,
      getW("post_attention_layernorm"), rmsEps, 1.0);
    await probe(oNorm, seqLen * hiddenSize, "o_norm");
    const afterAttn = await this._residualAdd(residual1, oNorm, seqLen * hiddenSize);
    await probe(afterAttn, seqLen * hiddenSize, "after_attn_residual");

    const residual2 = afterAttn;
    const ln2 = await this._rmsNorm(afterAttn, seqLen, hiddenSize,
      getW("pre_feedforward_layernorm"), rmsEps, 1.0);
    await probe(ln2, seqLen * hiddenSize, "ln2");

    const gate = await this._matmulTransB(ln2, seqLen, hiddenSize, getW("mlp.gate_proj"), intermediateSize);
    const up   = await this._matmulTransB(ln2, seqLen, hiddenSize, getW("mlp.up_proj"),   intermediateSize);
    await probe(gate, seqLen * intermediateSize, "gate_pre_gelu");
    await probe(up,   seqLen * intermediateSize, "up");
    const total = seqLen * intermediateSize;
    // _gelu returns a NEW buffer (not in-place). Capture it.
    const gateAct = await this._gelu(gate, total);
    await probe(gateAct, seqLen * intermediateSize, "gate_gelu");
    const gated = await this._elementwiseMul(gateAct, up, total);
    await probe(gated, seqLen * intermediateSize, "gated");
    const down = await this._matmulTransB(gated, seqLen, intermediateSize, getW("mlp.down_proj"), hiddenSize);
    await probe(down, seqLen * hiddenSize, "down");

    const downNorm = await this._rmsNorm(down, seqLen, hiddenSize,
      getW("post_feedforward_layernorm"), rmsEps, 1.0);
    await probe(downNorm, seqLen * hiddenSize, "down_norm");
    const finalOut = await this._residualAdd(residual2, downNorm, seqLen * hiddenSize);
    await probe(finalOut, seqLen * hiddenSize, "layer_0_final");
    return finalOut;
  }

  /**
   * Single-token Gemma embed for cached decode. Mirrors gemmaEmbed but
   * handles one token_id → one row, with the same sqrt(hidden) scale.
   */
  async gemmaEmbedSingle(tokenId) {
    // Single-token case is just gemmaEmbed with a 1-element token list —
    // keeps one on-GPU code path.
    return this.gemmaEmbed([tokenId]);
  }

  /**
   * GQA/MQA cached attention for single-query decode.
   * q:        [numQHeads, headDim]  (flat buffer, the current token's rotated Q)
   * kvCache:  KVCache with kvHiddenSize == numKvHeads * headDim
   * Returns:  [numQHeads * headDim] buffer (concatenated head outputs).
   *
   * Per-query-head dispatch of attention_cached.wgsl. For MQA (numKvHeads=1)
   * all Q heads share the same K/V slice. For GQA, Q head h routes to
   * KV head (h / group) where group = numQHeads / numKvHeads.
   *
   * Note: sliding-window masking via `cacheStart` — caller bounds the
   * cache-length window so we don't re-implement per-position masking in
   * the 1×N softmax.
   */
  async _attentionGqaCached(qBuf, cfg, kvCache, layerIdx, seqPos, windowSize = 0) {
    const { numQHeads, numKvHeads, headDim, invSqrtScale } = cfg;
    const group = numQHeads / numKvHeads;
    const scale = invSqrtScale ?? (1.0 / Math.sqrt(headDim));
    const { kBuffer: fullK, vBuffer: fullV } = kvCache.getKV(layerIdx);
    const cacheLenFull = kvCache.seqLen; // includes the current position we just appended
    const cacheStart = windowSize > 0 ? Math.max(0, cacheLenFull - windowSize) : 0;
    const cacheLen = cacheLenFull - cacheStart;
    const kvHidden = numKvHeads * headDim;

    const outputBuf = this._createBuffer("gemma_cached_out", numQHeads * headDim * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    for (let h = 0; h < numQHeads; h++) {
      const kvH = Math.floor(h / group);

      // Extract Q head slice from the flat [numQHeads*headDim] buffer.
      const qHead = this._createBuffer(`gemma_qcached_h${h}`, headDim * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      this._copyBuffer(qBuf, h * headDim * 4, qHead, 0, headDim * 4);

      // Slice K and V cache rows [cacheStart, cacheLenFull) for kvH.
      const kSlice = await this._extractHeadSlice(fullK, cacheLen, kvHidden, cacheStart, kvH, headDim, numKvHeads);
      const vSlice = await this._extractHeadSlice(fullV, cacheLen, kvHidden, cacheStart, kvH, headDim, numKvHeads);

      // Run the 3-pass cached attention.
      const scores = this._createBuffer(`gemma_cs_h${h}`, cacheLen * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      const p = new ArrayBuffer(16);
      const pv = new DataView(p);
      pv.setUint32(0, cacheLen, true);
      pv.setUint32(4, headDim, true);
      pv.setFloat32(8, scale, true);
      const pBuf = this._createBuffer(`gemma_cp_h${h}`, 16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      this.device.queue.writeBuffer(pBuf, 0, new Uint8Array(p));

      const scorePipe = this._getOrCreatePipeline("attention_cached", "compute_scores_cached", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);
      const smPipe = this._getOrCreatePipeline("attention_cached_softmax", "softmax_cached", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);
      const wsPipe = this._getOrCreatePipeline("attention_cached_ws", "weighted_sum_cached", [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ]);

      const headOut = this._createBuffer(`gemma_hout_h${h}`, headDim * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      const bg1 = this.device.createBindGroup({ layout: scorePipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: pBuf } },
        { binding: 1, resource: { buffer: qHead } },
        { binding: 2, resource: { buffer: kSlice } },
        { binding: 3, resource: { buffer: scores } },
      ]});
      const bg2 = this.device.createBindGroup({ layout: smPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: pBuf } },
        { binding: 1, resource: { buffer: scores } },
      ]});
      const bg3 = this.device.createBindGroup({ layout: wsPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: pBuf } },
        { binding: 1, resource: { buffer: scores } },
        { binding: 2, resource: { buffer: vSlice } },
        { binding: 3, resource: { buffer: headOut } },
      ]});

      this._dispatch(scorePipe, bg1, Math.ceil(cacheLen / 256));
      this._dispatch(smPipe, bg2, 1);
      this._dispatch(wsPipe, bg3, Math.ceil(headDim / 256));
      this._copyBuffer(headOut, 0, outputBuf, h * headDim * 4, headDim * 4);
    }

    return outputBuf;
  }

  /**
   * One Gemma transformer layer for single-token cached decode.
   *
   * Same composition as forwardLayerGemmaPrefill (q_norm/k_norm, dual RoPE,
   * pre/post FFN norms) but seq=1 and attention reads from KVCache instead
   * of re-computing over full history.
   *
   * Appends the new K,V to the cache before attending.
   *
   * @param {GPUBuffer} xBuf — single-row [1, hidden] input
   * @param {number} l — absolute layer index
   * @param {number} seqPos — 0-indexed position of this token in the sequence
   * @param {object} cfg — same as forwardLayerGemmaPrefill + { invSqrtScale, layerTypes }
   * @param {{cos,sin,cosLocal?,sinLocal?}} ropeBufs
   * @param {KVCache} kvCache — must be constructed with kvHiddenSize == numKvHeads*headDim
   * @returns {Promise<GPUBuffer>} — [1, hidden] output
   */
  async forwardLayerGemmaCached(xBuf, l, seqPos, cfg, ropeBufs, kvCache) {
    const { hiddenSize, numQHeads, numKvHeads, headDim, intermediateSize,
            windowSize, rmsEps, layerTypes } = cfg;
    const prefix = `model.layers.${l}`;
    const getW = (name) => {
      const b = this.loader.getBuffer(`${prefix}.${name}.weight`);
      if (!b) throw new Error(`missing weight: ${prefix}.${name}.weight`);
      return b;
    };

    const isSliding = layerTypes ? layerTypes[l] === "sliding_attention" : (windowSize > 0);
    const layerWindow = isSliding ? (windowSize || 0) : 0;
    const cosBuf = isSliding && ropeBufs.cosLocal ? ropeBufs.cosLocal : ropeBufs.cos;
    const sinBuf = isSliding && ropeBufs.sinLocal ? ropeBufs.sinLocal : ropeBufs.sin;

    const residual1 = xBuf;
    const ln1 = await this._rmsNorm(xBuf, 1, hiddenSize, getW("input_layernorm"), rmsEps, 1.0);

    const qFull = numQHeads * headDim;
    const kvFull = numKvHeads * headDim;
    const q = await this._matmulTransB(ln1, 1, hiddenSize, getW("self_attn.q_proj"), qFull);
    const k = await this._matmulTransB(ln1, 1, hiddenSize, getW("self_attn.k_proj"), kvFull);
    const v = await this._matmulTransB(ln1, 1, hiddenSize, getW("self_attn.v_proj"), kvFull);

    await this._rmsNormInPlace(q, numQHeads,  headDim, getW("self_attn.q_norm"), rmsEps, 1.0);
    await this._rmsNormInPlace(k, numKvHeads, headDim, getW("self_attn.k_norm"), rmsEps, 1.0);

    // RoPE at the current absolute position.
    await this._rope(q, 1, numQHeads,  headDim, cosBuf, sinBuf, seqPos);
    await this._rope(k, 1, numKvHeads, headDim, cosBuf, sinBuf, seqPos);

    // Append to KV cache at seqPos, then attend against the (possibly
    // windowed) cache.
    kvCache.append(l, k, v, seqPos);
    const attnOut = await this._attentionGqaCached(q, cfg, kvCache, l, seqPos, layerWindow);

    const oProj = await this._matmulTransB(attnOut, 1, qFull, getW("self_attn.o_proj"), hiddenSize);
    const oNorm = await this._rmsNorm(oProj, 1, hiddenSize, getW("post_attention_layernorm"), rmsEps, 1.0);
    const afterAttn = await this._residualAdd(residual1, oNorm, hiddenSize);

    const residual2 = afterAttn;
    const ln2 = await this._rmsNorm(afterAttn, 1, hiddenSize, getW("pre_feedforward_layernorm"), rmsEps, 1.0);
    const gate = await this._matmulTransB(ln2, 1, hiddenSize, getW("mlp.gate_proj"), intermediateSize);
    const up   = await this._matmulTransB(ln2, 1, hiddenSize, getW("mlp.up_proj"),   intermediateSize);
    const gateAct = await this._gelu(gate, intermediateSize);
    const gated = await this._elementwiseMul(gateAct, up, intermediateSize);
    const down = await this._matmulTransB(gated, 1, intermediateSize, getW("mlp.down_proj"), hiddenSize);
    const downNorm = await this._rmsNorm(down, 1, hiddenSize, getW("post_feedforward_layernorm"), rmsEps, 1.0);
    return this._residualAdd(residual2, downNorm, hiddenSize);
  }

  /** RMSNorm where the "row" dimension isn't seq_len — e.g. q_norm over
   *  (seqLen*numHeads) rows of headDim. Thin wrapper over _rmsNorm that
   *  overwrites inputBuf via copy after norm. */
  async _rmsNormInPlace(buf, rows, cols, gammaBuf, eps, gammaBias = 0.0) {
    const out = await this._rmsNorm(buf, rows, cols, gammaBuf, eps, gammaBias);
    this._copyBuffer(out, 0, buf, 0, rows * cols * 4);
    return buf;
  }

  /**
   * Full Gemma prefill across a contiguous range of layers. Stops at the
   * end of the range — the caller (shard boundary) then sends the hidden
   * state to the next shard, exactly like forwardLayersPrefill does for
   * GPT-2. If this shard is the last one, caller follows with
   * gemmaFinalNormAndLmHead() to produce logits.
   *
   * @param {object} hidden — { buffer, shape: [seqLen, hiddenSize] }
   * @param {number} layerStart / layerEnd — inclusive range assigned to this shard
   * @param {object} cfg — as forwardLayerGemmaPrefill
   * @param {GPUBuffer} cosBuf / sinBuf — precomputed RoPE caches for the session
   * @returns {Promise<{buffer, shape}>}
   */
  async forwardLayersGemmaPrefill(hidden, layerStart, layerEnd, cfg, ropeBufs) {
    const seqLen = hidden.shape[0];
    let cur = hidden.buffer;
    const weightBuffers = new Set(this.loader.buffers.values());

    // Layer-level divergence probe: when _nanTrace is enabled by the caller,
    // also stats every layer's output so we can compare to numpy reference.
    // Cheap — one readback per layer, logged via _layerStats array.
    const wantStats = Array.isArray(this._nanTrace);
    if (wantStats && !this._layerStats) this._layerStats = [];

    for (let l = layerStart; l <= layerEnd; l++) {
      // Batch every kernel dispatch in this layer into ONE GPU submit —
      // collapses ~16 queue.submit fences into 1.
      cur = await this._withBatchedEncoder(
        () => this.forwardLayerGemmaPrefill(cur, l, seqLen, cfg, ropeBufs)
      );

      if (wantStats) {
        const sz = seqLen * cfg.hiddenSize * 4;
        const f = new Float32Array(await this._readBuffer(cur, 0, sz));
        let mn = Infinity, mx = -Infinity, sum2 = 0, nans = 0;
        // Sample the LAST row (final position) — matches what numpy trace logs.
        const start = (seqLen - 1) * cfg.hiddenSize;
        for (let i = start; i < f.length; i++) {
          const v = f[i];
          if (Number.isNaN(v)) { nans++; continue; }
          if (v < mn) mn = v;
          if (v > mx) mx = v;
          sum2 += v * v;
        }
        this._layerStats.push({
          layer: l,
          min: +mn.toFixed(3), max: +mx.toFixed(3),
          rms: +Math.sqrt(sum2 / cfg.hiddenSize).toFixed(4),
          nans,
        });
      }

      // Clean up temp buffers between layers to bound peak memory on
      // mobile (same pattern as forwardLayersPrefill for GPT-2).
      const keep = cur;
      const surviving = [];
      for (const buf of this._tempBuffers) {
        if (buf === keep || weightBuffers.has(buf)) surviving.push(buf);
        else buf.destroy();
      }
      this._tempBuffers = surviving;
    }

    return { buffer: cur, shape: [seqLen, cfg.hiddenSize] };
  }

  /**
   * Final norm + LM head projection, called only on the last shard after
   * forwardLayersGemmaPrefill. Gemma uses weight tying — lm_head weights
   * are the same as embed_tokens; loader exposes both names. If only
   * embed_tokens exists (strict tying), caller passes embedBuf as
   * headWeight.
   *
   * Returns a buffer of shape [seqLen, vocabSize] with raw logits.
   * Sampling (with optional softcap) happens on coord side.
   */
  async gemmaFinalNormAndLmHead(hidden, cfg, normGamma, headWeight) {
    const seqLen = hidden.shape[0];
    // Do NOT wrap in _withBatchedEncoder: the caller immediately reads the
    // logits back, so any deferred submit + mapAsync ordering is
    // brittle. Submit each kernel individually and let GPU scheduler
    // overlap them.
    const normed = await this._rmsNorm(hidden.buffer, seqLen, cfg.hiddenSize, normGamma, cfg.rmsEps, 1.0);
    const logits = await this._matmulTransB(normed, seqLen, cfg.hiddenSize, headWeight, cfg.vocabSize);
    return { buffer: logits, shape: [seqLen, cfg.vocabSize] };
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
    this._dispatch(pipeline, bindGroup, Math.ceil(M / 8), Math.ceil(N / 8));
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

    this._dispatch(pipeline, bindGroup, Math.ceil(totalElements / 256));
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
    this._dispatch(pipeline, bindGroup, Math.ceil(totalElements / 256));
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
    this._dispatch(pipeline, bindGroup, Math.ceil(total / 256));
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
    // Pool lookup: reuse an idle buffer with same (size, usage). Keyed on
    // both because GPUBufferUsage flags must match exactly for binding.
    if (!this._bufPool) this._bufPool = new Map();
    const key = `${size}|${usage}`;
    const bucket = this._bufPool.get(key);
    if (bucket && bucket.length) {
      const buf = bucket.pop();
      this._tempBuffers.push(buf);
      return buf;
    }
    const buf = this.device.createBuffer({ label, size, usage });
    buf._poolKey = key; // remember bucket for recycle
    this._tempBuffers.push(buf);
    return buf;
  }

  /**
   * Return temp buffers to the pool (not destroy). Weight buffers owned
   * by ShardLoader are never in _tempBuffers so they're safe.
   */
  _cleanupTempBuffers() {
    if (!this._bufPool) this._bufPool = new Map();
    for (const buf of this._tempBuffers) {
      const key = buf._poolKey;
      if (!key) { buf.destroy(); continue; }
      let bucket = this._bufPool.get(key);
      if (!bucket) { bucket = []; this._bufPool.set(key, bucket); }
      // Cap bucket size so we don't grow unbounded across many request types.
      if (bucket.length >= 32) { buf.destroy(); continue; }
      bucket.push(buf);
    }
    this._tempBuffers = [];
  }

  /**
   * Fully free the pool. Call on shutdown.
   */
  _drainBufferPool() {
    if (!this._bufPool) return;
    for (const bucket of this._bufPool.values()) for (const b of bucket) b.destroy();
    this._bufPool.clear();
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
  _getOrCreatePipeline(shaderName, entryPoint, layoutEntries, moduleNameOverride) {
    const key = `${shaderName}:${entryPoint}`;
    if (this.pipelines[key]) return this.pipelines[key];

    // Map pipeline keys to their shader module names
    let moduleName = moduleNameOverride || shaderName;
    if (!moduleNameOverride) {
      if (shaderName === "attention_softmax" || shaderName === "attention_ws") {
        moduleName = "attention";
      } else if (shaderName === "attention_cached_softmax" || shaderName === "attention_cached_ws") {
        moduleName = "attention_cached";
      }
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

  // ─── Tensor Serialization (delegated to TensorSerializer) ────

  async serializeTensor(tensor) { return this.serializer.serialize(tensor); }
  deserializeTensor(tensorMsg) { return this.serializer.deserialize(tensorMsg); }
  async serializeTensorBinary(tensor) { return this.serializer.serializeBinary(tensor); }
  deserializeTensorBinary(payload, shape) { return this.serializer.deserializeBinary(payload, shape); }
  async serializeTensorQuantized(tensor) { return this.serializer.serializeQuantized(tensor); }
  deserializeTensorQuantized(payload, shape) { return this.serializer.deserializeQuantized(payload, shape); }
  async serializeTensorInt4(tensor) { return this.serializer.serializeInt4(tensor); }
  deserializeTensorInt4(payload, shape) { return this.serializer.deserializeInt4(payload, shape); }
  async serializeTensorDelta(tensor, prev) { return this.serializer.serializeDelta(tensor, prev); }
  deserializeTensorDeltaApply(payload, shape, prev) { return this.serializer.deserializeDeltaApply(payload, shape, prev); }
}
