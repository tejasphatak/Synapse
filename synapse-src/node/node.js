/**
 * Synapse Node — Browser-side WebSocket client that loads a model shard
 * and participates in distributed inference.
 *
 * Lifecycle:
 * 1. Initialize WebGPU
 * 2. Connect to coordinator
 * 3. Send JOIN with capabilities
 * 4. Receive ASSIGN_SHARD → download weights → upload to GPU
 * 5. Mark as READY
 * 6. Process INFERENCE_REQUEST / ACTIVATION messages → run pipeline → send output
 */

import { ShardLoader } from "./shard-loader.js?v=20260415-gemma";
import { Pipeline } from "./pipeline.js?v=20260415-transB";
import {
  MessageType,
  PROTOCOL_V2,
  createJoinMessage,
  createPingMessage,
  createActivationMessage,
  createOutputMessage,
  createNodeReadyMessage,
} from "../protocol/messages.js";
import {
  isBinaryMessage,
  decodeBinaryMessage,
  encodeBinaryMessage,
  encodeBinaryOutput,
  decodeOutputTokens,
  BinaryMsgType,
  QuantMode,
  Flags,
  getQuantMode,
  setQuantFlags,
  requestIdToUint32,
  uint32ToRequestId,
  registerRequestId,
} from "../protocol/binary.js";
import { unpackQuantized, dequantizeInt8, unpackQuantizedPerChannel, dequantizeInt8PerChannel } from "../protocol/quantize.js";
import { compressPayload, decompressPayload } from "../protocol/entropy.js";
import { AdaptivePrecisionSelector } from "../protocol/adaptive-precision.js";
import { SpeculativeController } from "./speculative.js";
import { P2PChannel } from "./p2p.js";

// Polyfill: crypto.randomUUID is only available in secure contexts (HTTPS)
// and on Chrome 92+. Headless Chrome on HTTP / older builds lack it. Fall back
// to crypto.getRandomValues-based RFC-4122 v4 UUID so nodes can join from
// insecure contexts during development and from older substrates.
function _randomUUID() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const b = new Uint8Array(16);
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(b);
  } else {
    for (let i = 0; i < 16; i++) b[i] = Math.floor(Math.random() * 256);
  }
  b[6] = (b[6] & 0x0f) | 0x40; // version 4
  b[8] = (b[8] & 0x3f) | 0x80; // variant RFC 4122
  const h = Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

export class SynapseNode {
  constructor(statusCallback = null, inherit = null) {
    // If `inherit` is provided, adopt state from a previous SynapseNode
    // instance (hot-reload path). Keeps WebGPU device, shard buffers, and
    // KV caches alive so we skip 10-30s of tensor re-upload.
    this._inherited = inherit;
    this.nodeId = inherit?.nodeId ?? `node-${_randomUUID().slice(0, 8)}`;
    this.ws = null;
    this.device = null;
    this.loader = null;
    this.pipeline = null;
    this.shardId = null;
    this.layerStart = null;
    this.layerEnd = null;
    this.isFirstNode = false;
    this.isLastNode = false;
    this.status = "initializing";
    this.topology = null;
    this.onStatus = statusCallback || (() => {});
    this.onOutput = null;
    this.pingInterval = null;
    this.baseUrl = "";
    this.useBinaryProtocol = false;
    this.useQuantization = true;
    // Delta encoding disabled 2026-04-15 — root-cause bug: sender stores
    // unquantized lastSentActivation while receiver stores INT8-roundtripped
    // lastRecvActivation, so the "previous" state diverges from step 0 and
    // all subsequent delta-apply operations drift by the quantization error.
    // Real fix: sender must store the dequantized-after-quantize version
    // (match receiver's exact state). Until then, absolute INT8 on every hop.
    this.useDeltaEncoding = false;
    this._binaryRequestIds = new Map(); // string -> uint32
    this._lastSentActivation = new Map(); // requestId -> Float32Array
    this._lastRecvActivation = new Map(); // requestId -> Float32Array
    this.speculative = null; // initialized after pipeline is ready
    this.useSpeculation = true;
    this.p2p = null; // initialized when topology assigns a downstream peer
    this.useP2P = true;
    this.adaptivePrecision = null; // initialized when layer range is known
    this.useAdaptivePrecision = true;
    // Wire quant override: URL param ?quant=int4 forces INT4 on the wire
    // (bypassing adaptive selector). Default: INT8 + adaptive promotion.
    // Used for A/B bandwidth measurement. See 2026-04-15 int4 finding.
    this.forceQuantMode = null;
    if (typeof window !== "undefined" && window.location) {
      const params = new URLSearchParams(window.location.search);
      const q = params.get("quant");
      if (q === "int4") this.forceQuantMode = "int4";
      else if (q === "int8") this.forceQuantMode = "int8";
      else if (q === "none") this.forceQuantMode = "none";
    }
  }

  /**
   * Resume from inherited state (hot reload path). Adopts the previous
   * instance's WebGPU device, shard pipeline, and WebSocket. Re-binds
   * WS event handlers to THIS instance. Skips the 10-30s shard upload.
   */
  async _resumeFromInherited(coordinatorUrl) {
    const inh = this._inherited;
    this.device = inh.device;
    // Build a FRESH ShardLoader from the currently-imported class so any
    // bug-fixes/cache-key bumps in shard-loader.js take effect for future
    // loadShard() calls. The pipeline's GPU buffers are already populated and
    // unaffected by loader lifecycle.
    // Reuse the inherited loader — it holds the uploaded shard buffers +
    // (for Gemma) the RoPE cos/sin caches. Replacing it with a fresh
    // ShardLoader that only has manifest loses those and leaves
    // pipeline.loader references dangling.
    this.loader = inh.loader;
    if (!this.loader) {
      this.loader = new ShardLoader(this.device);
      try { await this.loader.loadManifest(coordinatorUrl.replace(/^ws/, "http")); } catch (_) {}
    }
    this.pipeline = inh.pipeline;
    this.shardId = inh.shardId;
    this.layerStart = inh.layerStart;
    this.layerEnd = inh.layerEnd;
    this.isFirstNode = inh.isFirstNode;
    this.isLastNode = inh.isLastNode;
    this.topology = inh.topology;
    this.adaptivePrecision = inh.adaptivePrecision;
    this.gpuInfo = {
      isMobile: this._isMobile(),
      vendor: this.device?.adapterInfo?.vendor || "unknown",
    };

    // Rebuild speculative controller on top of the kept pipeline so it picks
    // up any behavior changes in the new module.
    if (this.useSpeculation && this.pipeline) {
      this.speculative = new SpeculativeController(this.pipeline);
      this.speculative.enabled = false;
      this.speculative.warmupSteps = 5;
      this.speculative.enableThreshold = 0.99;
    }

    // Adopt existing WebSocket if still open — no reconnect needed.
    if (inh.ws && inh.ws.readyState === 1) {
      this.ws = inh.ws;
      this.ws.onmessage = (event) => this._handleMessage(event.data);
      this.ws.onclose = () => {
        this._setStatus("disconnected");
        clearInterval(this.pingInterval);
        setTimeout(() => {
          if (this.status === "disconnected") {
            this._setStatus("reconnecting");
            this._connect(coordinatorUrl);
          }
        }, 3000);
      };
      this.ws.onerror = () => this._setStatus("error", "WebSocket error");
      // Restart heartbeat on the adopted socket.
      this.pingInterval = setInterval(() => {
        if (this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify(createPingMessage()));
        }
      }, 15000);
      this._sendLog("info", "hot_reload_resumed", { shardId: this.shardId });

      // If the node was assigned a shard but the pipeline is empty (prior
      // load had failed), request coord to re-send ASSIGN_SHARD so we
      // retry from clean state with the fresh code. Proxy-check by looking
      // at the old status — ready means shard loaded; anything else didn't.
      const pipelineLoaded = inh.status === "ready" || inh.status === "computing";
      if (this.shardId !== null && !pipelineLoaded) {
        console.log(`[node] hot-reload: shard ${this.shardId} was unloaded, requesting reassignment`);
        this._sendLog("info", "hot_reload_reassign_requested", { shardId: this.shardId });
        // Ask coord to reassign by un-assigning then letting tryAssignShards refill
        this.shardId = null;
        this.layerStart = null;
        this.layerEnd = null;
        // Send a fresh JOIN to trigger reassignment
        this.ws.send(JSON.stringify(createJoinMessage(this.nodeId, {
          webgpu: true, maxLayers: 6,
          mobile: this.gpuInfo?.isMobile || false,
          gpuVendor: this.gpuInfo?.vendor || "unknown",
          maxBufferMB: Math.round((this.device?.limits?.maxBufferSize || 0) / 1024 / 1024),
          userAgent: navigator.userAgent,
          protocolV2: true,
        })));
        this._setStatus("connected", "awaiting reassignment post hot-reload");
      } else {
        this._setStatus("ready", "hot-reload resumed");
      }
      return true;
    }

    // WebSocket was closed during handoff — fall back to fresh connect but
    // keep device/pipeline so we still skip the heavy re-upload.
    return this._connect(coordinatorUrl);
  }

  /**
   * Extract state for hot reload. Returns a plain object with references to
   * the expensive-to-recreate pieces (device, pipeline, shard assignment).
   * The new instance will adopt these via the `inherit` constructor arg.
   */
  extractHotReloadState() {
    return {
      nodeId: this.nodeId,
      ws: this.ws,
      device: this.device,
      loader: this.loader,
      pipeline: this.pipeline,
      shardId: this.shardId,
      layerStart: this.layerStart,
      layerEnd: this.layerEnd,
      isFirstNode: this.isFirstNode,
      isLastNode: this.isLastNode,
      topology: this.topology,
      adaptivePrecision: this.adaptivePrecision,
      status: this.status,
    };
  }

  /**
   * Shutdown without tearing down WebGPU state. Called on the OLD instance
   * before the NEW instance adopts its state. Stops timers, releases event
   * handlers, but leaves device + buffers alive for the successor.
   */
  softShutdown() {
    if (this.pingInterval) { clearInterval(this.pingInterval); this.pingInterval = null; }
    if (this.ws) { this.ws.onmessage = null; this.ws.onclose = null; this.ws.onerror = null; }
    this.status = "handoff";
  }

  /**
   * Initialize WebGPU and connect to the coordinator.
   */
  async start(coordinatorUrl) {
    this.baseUrl = coordinatorUrl.replace(/^ws/, "http");

    // Hot-reload fast path: if `inherit` state was passed to the constructor,
    // skip WebGPU init + shard download and resume from the existing pipeline.
    if (this._inherited) {
      return await this._resumeFromInherited(coordinatorUrl);
    }

    // Step 1: Initialize WebGPU
    this._setStatus("checking_webgpu");
    if (!navigator.gpu) {
      this._setStatus("error", "WebGPU not supported in this browser");
      return false;
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: this._isMobile() ? "low-power" : "high-performance",
    });
    if (!adapter) {
      this._setStatus("error", "No WebGPU adapter found");
      return false;
    }

    // Detect GPU capabilities
    this.gpuInfo = {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      isMobile: this._isMobile(),
      vendor: adapter.info?.vendor || "unknown",
      architecture: adapter.info?.architecture || "unknown",
    };

    this._setStatus("checking_webgpu",
      `GPU: ${this.gpuInfo.vendor} | Max buffer: ${Math.round(this.gpuInfo.maxBufferSize / 1024 / 1024)}MB | Mobile: ${this.gpuInfo.isMobile}`
    );

    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });

    this.device.lost.then((info) => {
      this._setStatus("error", `WebGPU device lost: ${info.message}`);
    });

    // Mobile: keep screen awake and handle visibility changes
    if (this.gpuInfo.isMobile) {
      this._setupMobileHandlers();
    }

    // Step 2: Self-test — prove this device's WebGPU stack can run our
    // kernels without NaN/Inf before joining the inference pool. Found on
    // 2026-04-15 that some GPUs (initial offender: Intel UHD/Arc) produce
    // all-NaN activations, corrupting any pipeline they join. Run synthetic
    // inputs through layernorm/gelu/matmul to catch this proactively.
    try {
      const testPipeline = new Pipeline(this.device, null);
      await testPipeline.init();
      const result = await testPipeline.runSelfTest();
      this.selfTestResult = result;
      testPipeline._cleanupTempBuffers();
      if (!result.pass) {
        // Still connect + send JOIN so coord can log the failure and feed the
        // self-healing loop. Coord policy: failing nodes appear in topology
        // but don't get shards assigned. Clear UI status for the user.
        const kernels = result.failures.map(f => f.kernel).join(", ");
        this._setStatus("error", `Self-test failed: ${kernels} (reporting to coord)`);
        console.warn("[node] Self-test failed — will connect + report:", result.failures);
        // Fall through to _connect — coord gets the failure via JOIN.capabilities.selfTest.
      } else {
        console.log("[node] Self-test passed");
      }
    } catch (err) {
      this._setStatus("error", `Self-test crashed: ${err.message}`);
      return false;
    }

    // Step 3: Connect to coordinator
    this._setStatus("connecting");
    return this._connect(coordinatorUrl);
  }

  /**
   * Establish WebSocket connection to coordinator.
   */
  _connect(url) {
    return new Promise((resolve) => {
      this.ws = new WebSocket(`${url}?type=node`);

      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => {
        this._setStatus("connected");

        // Send JOIN with device capabilities (advertise binary protocol support)
        const msg = createJoinMessage(this.nodeId, {
          webgpu: true,
          maxLayers: 6,
          mobile: this.gpuInfo?.isMobile || false,
          gpuVendor: this.gpuInfo?.vendor || "unknown",
          maxBufferMB: Math.round((this.gpuInfo?.maxBufferSize || 0) / 1024 / 1024),
          userAgent: navigator.userAgent,
          protocolV2: true,
          selfTest: { pass: this.selfTestResult?.pass ?? null, failures: this.selfTestResult?.failures || [] },
        });
        this.ws.send(JSON.stringify(msg));

        // Start heartbeat
        this.pingInterval = setInterval(() => {
          if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(createPingMessage()));
          }
        }, 15000);

        // Browser-keepalive (wakeLock + silent audio) reverted 2026-04-15
        // pending coherence-regression bisect. Coord-initiated PING every 10s
        // + 60s stale threshold already handles most mobile bg-throttling.

        resolve(true);
      };

      this.ws.onmessage = (event) => this._handleMessage(event.data);

      this.ws.onclose = () => {
        this._setStatus("disconnected");
        clearInterval(this.pingInterval);
        // Attempt reconnect after 3 seconds
        setTimeout(() => {
          if (this.status === "disconnected") {
            this._setStatus("reconnecting");
            this._connect(url);
          }
        }, 3000);
      };

      this.ws.onerror = (err) => {
        this._setStatus("error", "WebSocket error");
        resolve(false);
      };
    });
  }

  /**
   * Handle raw binary data received via P2P (WebRTC data channel).
   * Same format as WebSocket binary messages — SYN1 encoded activations.
   */
  async _handleBinaryMessage(raw) {
    try {
      const decoded = decodeBinaryMessage(raw);
      if (decoded.type === BinaryMsgType.ACTIVATION) {
        const requestId = uint32ToRequestId(decoded.requestId);
        await this._handleActivationBinary(decoded, requestId);
      }
    } catch (err) {
      console.error("[node] P2P binary decode error:", err);
    }
  }

  /**
   * Handle incoming WebSocket messages (binary or JSON).
   */
  async _handleMessage(raw) {
    // Binary protocol path — activation/output messages
    if (isBinaryMessage(raw)) {
      try {
        const decoded = decodeBinaryMessage(raw);
        if (decoded.type === BinaryMsgType.ACTIVATION) {
          const requestId = uint32ToRequestId(decoded.requestId);
          await this._handleActivationBinary(decoded, requestId);
        } else if (decoded.type === BinaryMsgType.OUTPUT) {
          // Nodes don't typically receive OUTPUT, but handle for completeness
          console.log("[node] Received binary OUTPUT");
        }
      } catch (err) {
        console.error("[node] Binary decode error:", err);
      }
      return;
    }

    // JSON protocol path — control messages + legacy activations
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch {
      console.error("[node] Invalid JSON received");
      return;
    }

    switch (msg.type) {
      case MessageType.ASSIGN_SHARD:
        await this._handleAssignShard(msg);
        break;

      case MessageType.INFERENCE_REQUEST:
        // Register the binary request ID mapping if provided
        if (msg.binaryRequestId != null) {
          registerRequestId(msg.requestId, msg.binaryRequestId);
          this.useBinaryProtocol = true;
        }
        await this._handleInferenceRequest(msg);
        break;

      case MessageType.INFERENCE_STEP:
        // KV-cached single-token step
        if (msg.binaryRequestId != null) {
          registerRequestId(msg.requestId, msg.binaryRequestId);
        }
        await this._handleInferenceStep(msg);
        break;

      case MessageType.KV_RESET:
        this.pipeline?.clearCache(msg.requestId);
        this._lastSentActivation.delete(msg.requestId);
        this._lastRecvActivation.delete(msg.requestId);
        this.speculative?.clear(msg.requestId);
        break;

      case MessageType.ACTIVATION:
        await this._handleActivation(msg);
        break;

      case MessageType.TOPOLOGY_UPDATE:
        this._handleTopologyUpdate(msg);
        break;

      case MessageType.PONG:
        // Heartbeat acknowledged
        break;

      case MessageType.PING:
        // Coord-initiated keepalive — reply with PONG. Receiving the frame
        // keeps the WS warm even if our outbound timers are throttled.
        if (this.ws.readyState === 1) {
          this.ws.send(JSON.stringify({ type: MessageType.PONG }));
        }
        break;

      case "P2P_SIGNAL":
        // WebRTC signaling relayed through coordinator
        if (this.p2p) {
          this.p2p.handleSignal(msg).catch(err =>
            console.warn("[node] P2P signal error:", err.message));
        } else if (this.useP2P) {
          // Incoming offer — create P2P channel as responder
          this.p2p = new P2PChannel(this.nodeId, this.ws);
          this.p2p.onMessage = (data) => this._handleBinaryMessage(data);
          this.p2p.onConnected = () => {
            this._sendLog("info", "p2p_connected", { peer: msg.from });
          };
          this.p2p.onDisconnected = () => {
            this._sendLog("info", "p2p_disconnected", { peer: msg.from });
            this.p2p?.close();
            this.p2p = null;
          };
          this.p2p.handleSignal(msg).catch(err =>
            console.warn("[node] P2P signal error:", err.message));
        }
        break;

      case MessageType.ERROR:
        console.error(`[node] Error from coordinator: ${msg.message}`);
        break;

      case MessageType.CLIENT_RELOAD:
        // Coordinator requested a hot-reload (e.g. after a node-code deploy).
        // Browser context: reload the page so the new node.js / pipeline.js
        // code is picked up. Non-browser clients log + ignore.
        console.log(`[node] CLIENT_RELOAD received — reloading in ${msg.delayMs ?? 1000}ms`);
        this._sendLog("info", "client_reload", { reason: msg.reason ?? "admin-triggered" });
        if (typeof window !== "undefined" && typeof window.location?.reload === "function") {
          setTimeout(() => window.location.reload(), msg.delayMs ?? 1000);
        }
        break;

      case MessageType.HOT_RELOAD:
        // Transparent hot-reload: dynamically re-import node.js and instance-
        // swap while keeping the WebGPU device + shard buffers alive. The
        // index.html host listens for this event and drives the swap because
        // it owns the class binding — the module itself can't replace its
        // own instance reference.
        console.log(`[node] HOT_RELOAD received — requesting in-place swap (reason: ${msg.reason})`);
        this._sendLog("info", "hot_reload_requested", { reason: msg.reason ?? "admin-triggered" });
        if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
          window.dispatchEvent(new CustomEvent("synapse:hot-reload", {
            detail: { reason: msg.reason ?? "admin-triggered", ts: msg.ts },
          }));
        }
        break;

      default:
        console.log(`[node] Unhandled message type: ${msg.type}`);
    }
  }

  /**
   * Handle shard assignment from coordinator.
   */
  async _handleAssignShard(msg) {
    this.shardId = msg.shardId;
    this.layerStart = msg.layerStart;
    this.layerEnd = msg.layerEnd;
    this._setStatus("loading_shard", `Shard ${msg.shardId} (layers ${msg.layerStart}-${msg.layerEnd})`);

    try {
      // Load manifest + weights
      this.loader = new ShardLoader(this.device);
      await this.loader.loadManifest(this.baseUrl);

      const result = await this.loader.loadShard(
        msg.shardId, msg.shardUrl, msg.sharedUrl,
        (loaded, total, phase) => {
          this._setStatus("loading_shard", `${phase}: ${Math.round((loaded / total) * 100)}%`);
        }
      );

      // Initialize the compute pipeline
      this.pipeline = new Pipeline(this.device, this.loader);
      await this.pipeline.init();

      // Initialize speculative execution
      if (this.useSpeculation) {
        this.speculative = new SpeculativeController(this.pipeline);
        // Start in prediction-only mode for the first few steps.
        // Auto-enables after warmup if prediction accuracy is high enough.
        this.speculative.enabled = false;
        this.speculative.warmupSteps = 5;
        this.speculative.enableThreshold = 0.99; // require 99%+ hit rate to auto-enable
      }

      // Initialize adaptive precision (1 output per node — tracks wire quantization error)
      if (this.useAdaptivePrecision) {
        this.adaptivePrecision = new AdaptivePrecisionSelector(1);
      }

      this._setStatus("ready", `Loaded ${result.tensorCount} tensors`);
      this._sendLog("perf", "shard_loaded", {
        tensorCount: result.tensorCount,
        layerStart: msg.layerStart,
        layerEnd: msg.layerEnd,
        gpuVendor: this.gpuInfo?.vendor,
        maxBufferMB: Math.round((this.gpuInfo?.maxBufferSize || 0) / 1024 / 1024),
      });

      // Tell coordinator we're ready
      this.ws.send(JSON.stringify(createNodeReadyMessage(this.nodeId, this.shardId)));

    } catch (err) {
      this._setStatus("error", `Failed to load shard: ${err.message}`);
      this._sendLog("error", "shard_load_error", { error: err.message });
      console.error("[node] Shard load error:", err);

      // Auto-retry: truncation errors and transient network issues are
      // commonly self-healing on a clean fetch. Retry up to 3 times with
      // exponential backoff before giving up. The truncation guard in
      // shard-loader.js ensures bad bytes aren't cached, so retries are safe.
      this._shardLoadAttempts = (this._shardLoadAttempts || 0) + 1;
      if (this._shardLoadAttempts <= 3) {
        const delay = 2000 * Math.pow(2, this._shardLoadAttempts - 1); // 2s, 4s, 8s
        console.log(`[node] Retrying shard load in ${delay}ms (attempt ${this._shardLoadAttempts + 1}/4)`);
        setTimeout(() => this._handleAssignShard(msg), delay);
      } else {
        this._sendLog("error", "shard_load_abandoned", { shardId: msg.shardId, attempts: this._shardLoadAttempts });
      }
    }
  }

  /**
   * Handle an inference request — prefill (only received by the first node).
   * Processes the full token sequence and populates the KV cache.
   */
  /**
   * Assemble the Gemma cfg + ropeBufs once per call. Cheap object creation,
   * read-only views into loader buffers — safe to recompute every invocation.
   */
  _buildGemmaRuntime() {
    const m = this.loader.manifest;
    const qAttn = m.query_pre_attn_scalar || m.head_dim;
    const cfg = {
      hiddenSize:       m.hidden_size,
      numQHeads:        m.num_attention_heads,
      numKvHeads:       m.num_key_value_heads,
      headDim:          m.head_dim,
      intermediateSize: m.intermediate_size,
      windowSize:       m.sliding_window || 0,
      rmsEps:           m.rms_norm_eps || 1e-6,
      vocabSize:        m.vocab_size,
      invSqrtScale:     1.0 / Math.sqrt(qAttn),
      layerTypes:       m.layer_types || null,
    };
    const ropeBufs = {
      cos:      this.loader.ropeCosBuffer,
      sin:      this.loader.ropeSinBuffer,
      cosLocal: this.loader.ropeCosLocalBuffer || null,
      sinLocal: this.loader.ropeSinLocalBuffer || null,
    };
    if (!ropeBufs.cos || !ropeBufs.sin) throw new Error("Gemma: RoPE cos/sin buffers not loaded");
    return { cfg, ropeBufs };
  }

  async _handleInferenceRequest(msg) {
    if (!this.pipeline) return;

    this._setStatus("computing", `Prefill ${msg.requestId} (${msg.tokenIds.length} tokens)`);
    const startTime = performance.now();

    try {
      const tokenIds = msg.tokenIds;

      // Embed all tokens — arch-specific path.
      const arch0 = this.loader?.manifest?.arch;
      let hidden = arch0 === "gemma"
        ? await this.pipeline.gemmaEmbed(tokenIds)
        : await this.pipeline.embed(tokenIds);

      // Diagnostic: stats of embedding output. If NaN here, embed kernel
      // is broken. If clean here but NaN after forwardLayersPrefill, the
      // bug is in one of the per-layer kernels.
      try {
        const sz = hidden.shape.reduce((a,b)=>a*b,1) * 4;
        const f = new Float32Array(await this.pipeline._readBuffer(hidden.buffer, 0, sz));
        let nans=0, mn=Infinity, mx=-Infinity, sum2=0;
        for (let i=0;i<f.length;i++) { const v=f[i]; if (Number.isNaN(v)) nans++; else { if(v<mn)mn=v; if(v>mx)mx=v; sum2+=v*v; } }
        this._sendLog("perf","post_embed_stats",{
          requestId: msg.requestId, shape: hidden.shape,
          min: isFinite(mn)?+mn.toFixed(4):null, max: isFinite(mx)?+mx.toFixed(4):null,
          nans, rms: +Math.sqrt(sum2/Math.max(1,f.length-nans)).toFixed(4),
        });
      } catch(_){}

      // Enable per-layer trace BEFORE the prefill so we can identify which
      // exact layer first introduces NaN in shard 0.
      this.pipeline._nanTrace = [];
      // Sub-kernel trace ONLY for shard 0 (where the bug originates) to
      // avoid expensive readbacks on healthy shards.
      // Enable sub-kernel trace on EVERY shard's first layer — captures
      // rms/nans after each sub-kernel (ln1 / qkv / attention / etc.) so we
      // can find where Intel introduces NaN regardless of which shard it's on.
      this.pipeline._subKernelTrace = [];

      // Run assigned layers — branch on architecture. Gemma-family models
      // use a different layer composition (RMSNorm + RoPE + GQA + gated MLP)
      // so route to forwardLayersGemmaPrefill when manifest.arch === "gemma".
      // KV cache for Gemma's MQA/GQA is a separate feature to ship; for now
      // prefill-only on Gemma (no per-token decode yet).
      const arch = this.loader?.manifest?.arch;
      if (arch === "gemma") {
        const { cfg, ropeBufs } = this._buildGemmaRuntime();
        hidden = await this.pipeline.forwardLayersGemmaPrefill(
          hidden, this.layerStart, this.layerEnd, cfg, ropeBufs
        );
      } else {
        hidden = await this.pipeline.forwardLayersPrefill(
          hidden, this.layerStart, this.layerEnd, msg.requestId
        );
      }

      if (this.pipeline._nanTrace) {
        this._sendLog("perf", "per_layer_nan_trace", {
          requestId: msg.requestId,
          shardId: this.shardId,
          layerRange: [this.layerStart, this.layerEnd],
          trace: this.pipeline._nanTrace,
        });
        this.pipeline._nanTrace = null;
      }
      if (this.pipeline._layerStats) {
        this._sendLog("perf", "gemma_layer_stats", {
          requestId: msg.requestId,
          shardId: this.shardId,
          stats: this.pipeline._layerStats,
        });
        this.pipeline._layerStats = null;
      }
      if (this.pipeline._subKernelTrace) {
        this._sendLog("perf", "sub_kernel_trace", {
          requestId: msg.requestId,
          shardId: this.shardId,
          layer: this.layerStart,
          trace: this.pipeline._subKernelTrace,
        });
        this.pipeline._subKernelTrace = null;
      }

      if (this.isLastNode) {
        await this._produceOutput(hidden, msg.requestId, msg.temperature ?? 1.0);
      } else {
        // Include seqPos so downstream nodes know the sequence length
        await this._sendActivation(hidden, msg.requestId, tokenIds.length);
      }

      const elapsed = performance.now() - startTime;
      this.pipeline._cleanupTempBuffers();
      this._setStatus("ready", `Prefill: ${elapsed.toFixed(0)}ms`);
      this._sendLog("perf", "prefill", {
        requestId: msg.requestId,
        tokenCount: msg.tokenIds.length,
        durationMs: +elapsed.toFixed(2),
        isFirstNode: this.isFirstNode,
        isLastNode: this.isLastNode,
      });

    } catch (err) {
      console.error("[node] Inference error:", err);
      const detail = {
        requestId: msg.requestId,
        message: err?.message || String(err),
        name: err?.name || null,
        stack: (err?.stack || "").split("\n").slice(0, 6).join("\n"),
      };
      this._sendLog("error", "prefill_error", detail);
      this.pipeline?._cleanupTempBuffers();
      this._setStatus("error", `Inference failed: ${detail.message}`);
    }
  }

  /**
   * Handle a single-token inference step (KV-cached path).
   * Only processes one new token using cached K,V from previous tokens.
   */
  async _handleInferenceStep(msg) {
    if (!this.pipeline) return;

    this._setStatus("computing", `Step ${msg.seqPos} for ${msg.requestId}`);
    const startTime = performance.now();

    try {
      const archStep = this.loader?.manifest?.arch;
      let hidden;

      if (archStep === "gemma") {
        const { cfg, ropeBufs } = this._buildGemmaRuntime();
        hidden = await this.pipeline.gemmaEmbedSingle(msg.tokenId);
        hidden = await this.pipeline.forwardLayersGemmaCached(
          hidden, this.layerStart, this.layerEnd, cfg, ropeBufs, msg.requestId, msg.seqPos,
        );
      } else {
        // Embed single token at the given position
        hidden = await this.pipeline.embedSingle(msg.tokenId, msg.seqPos);
        // Run layers with KV cache
        hidden = await this.pipeline.forwardLayersCached(
          hidden, this.layerStart, this.layerEnd, msg.requestId, msg.seqPos
        );
      }

      if (this.isLastNode) {
        await this._produceOutput(hidden, msg.requestId, msg.temperature ?? 1.0);
      } else {
        await this._sendActivation(hidden, msg.requestId, msg.seqPos + 1);
      }

      const elapsed = performance.now() - startTime;
      this.pipeline._cleanupTempBuffers();
      this._setStatus("ready", `Step ${msg.seqPos}: ${elapsed.toFixed(0)}ms`);
      this._sendLog("perf", "cached_step", {
        requestId: msg.requestId,
        seqPos: msg.seqPos,
        durationMs: +elapsed.toFixed(2),
        isFirstNode: this.isFirstNode,
        isLastNode: this.isLastNode,
      });

    } catch (err) {
      console.error("[node] Inference step error:", err);
      this._sendLog("error", "cached_step_error", { requestId: msg.requestId, error: err.message });
      this.pipeline?._cleanupTempBuffers();
      this._setStatus("error", `Step failed: ${err.message}`);
    }
  }

  /**
   * Handle a binary-encoded activation from the previous node.
   * Uses KV cache: if shape is [1, hidden] it's a single-token step,
   * if shape is [N, hidden] it's a prefill.
   */
  async _handleActivationBinary(decoded, requestId) {
    if (!this.pipeline) return;

    const seqLen = decoded.seqPos; // total sequence length from upstream
    const isPrefill = decoded.shape[0] > 1;

    this._setStatus("computing", `Processing ${isPrefill ? "prefill" : "step"} activation for ${requestId}`);
    const startTime = performance.now();

    try {
      // Decompress RLE if compressed flag is set
      const isCompressed = !!(decoded.flags & Flags.COMPRESSED);
      const payload = isCompressed ? decompressPayload(decoded.payload) : decoded.payload;

      // Detect quantized/delta payload and decode accordingly
      const quantMode = getQuantMode(decoded.flags);
      const isDelta = !!(decoded.flags & Flags.DELTA);
      let hidden;

      if (isDelta && quantMode === QuantMode.INT8) {
        // Delta-encoded int8: dequantize delta, add to previous activation
        const prev = this._lastRecvActivation.get(requestId);
        if (prev) {
          const result = this.pipeline.deserializeTensorDeltaApply(payload, decoded.shape, prev);
          hidden = { buffer: result.buffer, shape: result.shape };
          this._lastRecvActivation.set(requestId, result.currentFloat32);
        } else {
          // No previous — treat as regular int8 (first token or cache miss)
          hidden = this.pipeline.deserializeTensorQuantized(payload, decoded.shape);
        }
      } else if (quantMode === QuantMode.INT4) {
        hidden = this.pipeline.deserializeTensorInt4(payload, decoded.shape);
      } else if (quantMode === QuantMode.INT8) {
        hidden = this.pipeline.deserializeTensorQuantized(payload, decoded.shape);
        // Cache float32 for future delta decoding (only for single-token)
        if (!isPrefill && this.useDeltaEncoding) {
          const unpacked = unpackQuantized(payload);
          this._lastRecvActivation.set(requestId, dequantizeInt8(unpacked.int8Data, unpacked.scale));
        }
      } else {
        hidden = this.pipeline.deserializeTensorBinary(payload, decoded.shape);
      }

      if (isPrefill) {
        // Per-layer NaN trace: ALWAYS on during this debug session so we
        // can pinpoint which layer corrupts activation on failing devices.
        // Re-rate-limit once root cause found.
        const traceEnabled = true;
        if (traceEnabled) {
          this.pipeline._nanTrace = [];
        }

        // Prefill: full sequence, populate KV cache — no speculation on prefill
        if (this.loader?.manifest?.arch === "gemma") {
          const { cfg, ropeBufs } = this._buildGemmaRuntime();
          hidden = await this.pipeline.forwardLayersGemmaPrefill(
            hidden, this.layerStart, this.layerEnd, cfg, ropeBufs
          );
        } else {
          hidden = await this.pipeline.forwardLayersPrefill(
            hidden, this.layerStart, this.layerEnd, requestId
          );
        }

        if (traceEnabled && this.pipeline._nanTrace) {
          this._sendLog("perf", "per_layer_nan_trace", {
            requestId,
            shardId: this.shardId,
            layerRange: [this.layerStart, this.layerEnd],
            trace: this.pipeline._nanTrace,
          });
          this.pipeline._nanTrace = null;
        }
        // Seed the predictor with the prefill output if speculation is active
        if (this.speculative) {
          const sz = hidden.shape.reduce((a, b) => a * b, 1) * 4;
          const f32 = new Float32Array(await this.pipeline._readBuffer(hidden.buffer, 0, sz));
          this.speculative.predictor.observe(requestId, f32);
        }
      } else {
        // Cached step: single token — speculation possible (single or batch)
        const seqPos = seqLen - 1;
        let usedSpeculative = false;

        if (this.speculative) {
          const hiddenSize = hidden.shape.reduce((a, b) => a * b, 1) * 4;
          const hiddenFloat32 = new Float32Array(
            await this.pipeline._readBuffer(hidden.buffer, 0, hiddenSize)
          );

          // Unified speculation: verify pending (batch or single), observe, speculate next
          const result = await this.speculative.onActivationReceived(
            requestId, hiddenFloat32, hidden, seqPos, this.layerStart, this.layerEnd
          );

          // Instrumentation: if shadow mode just verified a prediction, emit
          // the per-step cosine so we can build a predictor-quality histogram
          // across many generations without needing warmup to finish.
          if (this.speculative.lastShadowCosine != null) {
            this._sendLog("perf", "speculation_cosine_sample", {
              shardId: this.shardId,
              seqPos,
              cosine: +this.speculative.lastShadowCosine.toFixed(4),
            });
          }

          if (this.speculative.enabled && result.useSpeculative) {
            hidden = result.speculativeHidden;
            usedSpeculative = true;
            this._sendLog("perf", "speculation_accepted", {
              requestId, seqLen, acceptedSteps: result.acceptedSteps,
            });
          } else if (result.rollbackPos != null && this.speculative.enabled) {
            // Speculation rejected — roll back KV cache
            const kvCache = this.pipeline.kvCaches.get(requestId);
            if (kvCache) kvCache.rollback(result.rollbackPos);
            this._sendLog("perf", "speculation_rejected", { requestId, seqLen });
          }

          // Auto-enable speculation after warmup if accuracy is high enough.
          // Shadow-speculation (always-predict during warmup) populates stats
          // so this check can actually fire. See project_speculation_bootstrap_deadlock_2026-04-15.
          if (!this.speculative.enabled && this.speculative.warmupSteps > 0 && !this.speculative._warmupExhausted) {
            const stats = this.speculative.predictor.getStats();
            const verified = stats.hits + stats.misses;
            if (verified >= this.speculative.warmupSteps) {
              if (stats.hitRate >= this.speculative.enableThreshold) {
                this.speculative.enabled = true;
                this._sendLog("perf", "speculation_auto_enabled", {
                  hitRate: +stats.hitRate.toFixed(4),
                  avgCosine: +stats.avgCosine.toFixed(6),
                  afterSteps: verified,
                });
                console.log(`[node] Speculative execution auto-enabled (hitRate=${stats.hitRate.toFixed(3)}, avgCosine=${stats.avgCosine.toFixed(4)})`);
              } else if (verified >= this.speculative.warmupSteps * 4) {
                // Gave the predictor 4× the warmup budget; hit rate still below threshold.
                // Stop wasting GPU on speculative work that will never be accepted.
                this.speculative._warmupExhausted = true;
                this._sendLog("perf", "speculation_warmup_exhausted", {
                  hitRate: +stats.hitRate.toFixed(4),
                  avgCosine: +stats.avgCosine.toFixed(6),
                  afterSteps: verified,
                });
                console.log(`[node] Speculation warmup exhausted — hit rate ${stats.hitRate.toFixed(3)} below ${this.speculative.enableThreshold}. Shadow mode off.`);
              }
            }
          }
        }

        if (!usedSpeculative) {
          if (this.loader?.manifest?.arch === "gemma") {
            const { cfg, ropeBufs } = this._buildGemmaRuntime();
            hidden = await this.pipeline.forwardLayersGemmaCached(
              hidden, this.layerStart, this.layerEnd, cfg, ropeBufs, requestId, seqPos,
            );
          } else {
            hidden = await this.pipeline.forwardLayersCached(
              hidden, this.layerStart, this.layerEnd, requestId, seqPos
            );
          }
        }
      }

      if (this.isLastNode) {
        await this._produceOutput(hidden, requestId);
      } else {
        await this._sendActivation(hidden, requestId, seqLen);
      }

      const elapsed = performance.now() - startTime;
      this.pipeline._cleanupTempBuffers();
      this._setStatus("ready", `Processed in ${elapsed.toFixed(0)}ms`);
      this._sendLog("perf", isPrefill ? "binary_prefill" : "binary_step", {
        requestId,
        durationMs: +elapsed.toFixed(2),
        quantized: quantMode === QuantMode.INT8,
        payloadBytes: decoded.payload.byteLength,
        seqLen,
      });
    } catch (err) {
      console.error("[node] Binary activation processing error:", err);
      this._sendLog("error", "binary_activation_error", { requestId, error: err.message });
      this.pipeline?._cleanupTempBuffers();
      this._setStatus("error", `Processing failed: ${err.message}`);
    }
  }

  /**
   * Handle a JSON-encoded activation message from the previous node.
   */
  async _handleActivation(msg) {
    if (!this.pipeline) return;

    const seqLen = msg.seqLen || msg.tensor?.shape?.[0] || 1;
    const isPrefill = msg.tensor?.shape?.[0] > 1;

    this._setStatus("computing", `Processing activation for ${msg.requestId}`);
    const startTime = performance.now();

    try {
      let hidden = this.pipeline.deserializeTensor(msg.tensor);

      const archJson = this.loader?.manifest?.arch;
      if (isPrefill) {
        if (archJson === "gemma") {
          const { cfg, ropeBufs } = this._buildGemmaRuntime();
          hidden = await this.pipeline.forwardLayersGemmaPrefill(
            hidden, this.layerStart, this.layerEnd, cfg, ropeBufs
          );
        } else {
          hidden = await this.pipeline.forwardLayersPrefill(
            hidden, this.layerStart, this.layerEnd, msg.requestId
          );
        }
      } else {
        const seqPos = seqLen - 1;
        if (archJson === "gemma") {
          const { cfg, ropeBufs } = this._buildGemmaRuntime();
          hidden = await this.pipeline.forwardLayersGemmaCached(
            hidden, this.layerStart, this.layerEnd, cfg, ropeBufs, msg.requestId, seqPos,
          );
        } else {
          hidden = await this.pipeline.forwardLayersCached(
            hidden, this.layerStart, this.layerEnd, msg.requestId, seqPos
          );
        }
      }

      if (this.isLastNode) {
        await this._produceOutput(hidden, msg.requestId);
      } else {
        await this._sendActivation(hidden, msg.requestId, seqLen);
      }

      const elapsed = performance.now() - startTime;
      this.pipeline._cleanupTempBuffers();
      this._setStatus("ready", `Processed in ${elapsed.toFixed(0)}ms`);

    } catch (err) {
      console.error("[node] Activation processing error:", err);
      this.pipeline?._cleanupTempBuffers();
      this._setStatus("error", `Processing failed: ${err.message}`);
    }
  }

  /**
   * Produce final output: run output head, sample token, send OUTPUT.
   */
  async _produceOutput(hidden, requestId, temperature = 1.0) {
    // Diagnostic: measure hidden state stats before LM head. If these are
    // zeros/NaN, the upstream pipeline failed. If they look normal but
    // logits are zero, the LM head itself is broken.
    try {
      const sz = hidden.shape.reduce((a, b) => a * b, 1) * 4;
      const buf = await this.pipeline._readBuffer(hidden.buffer, 0, sz);
      const f = new Float32Array(buf);
      let mn = Infinity, mx = -Infinity, nz = 0, sum2 = 0, nans = 0;
      for (let i = 0; i < f.length; i++) {
        const v = f[i];
        if (Number.isNaN(v)) { nans++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        if (v !== 0) nz++;
        sum2 += v * v;
      }
      this._sendLog("perf", "pre_lmhead_hidden", {
        requestId,
        shape: hidden.shape,
        min: +mn.toFixed(4), max: +mx.toFixed(4),
        nonZero: nz, nans,
        rms: +Math.sqrt(sum2 / f.length).toFixed(4),
      });
    } catch (_) { /* ignore */ }

    const archOut = this.loader?.manifest?.arch;
    let logitsTensor;
    if (archOut === "gemma") {
      // Gemma: tied lm_head → weight is embed_tokens.weight.
      const { cfg } = this._buildGemmaRuntime();
      const normGamma = this.loader.getBuffer("model.norm.weight");
      const headWeight = this.loader.getBuffer("model.embed_tokens.weight");
      if (!normGamma || !headWeight) throw new Error("Gemma output: model.norm or embed_tokens not loaded");
      logitsTensor = await this.pipeline.gemmaFinalNormAndLmHead(hidden, cfg, normGamma, headWeight);
    } else {
      logitsTensor = await this.pipeline.outputHead(hidden);
    }
    const tokenId = await this.pipeline.sampleToken(logitsTensor, temperature);

    // Drift diagnostic: emit top-5 tokens + logit range so we can see if
    // EOT-spam is argmax-from-drift (one token dominates) or sampler-weird.
    if (this.pipeline._lastSampleTop) {
      this._sendLog("perf", "sample_top5", {
        requestId, tokenId,
        top: this.pipeline._lastSampleTop.top,
        top20: this.pipeline._lastSampleTop.top20, // [[tokenId, rawLogit], ...] — for parity cross-checks
        logitRange: [
          +this.pipeline._lastSampleTop.minLogit.toFixed(3),
          +this.pipeline._lastSampleTop.maxLogit.toFixed(3),
        ],
      });
      this.pipeline._lastSampleTop = null;
    }

    if (this.useBinaryProtocol) {
      const numericId = requestIdToUint32(requestId);
      const binaryMsg = encodeBinaryOutput(numericId, [tokenId]);
      this.ws.send(binaryMsg);
    } else {
      const outputMsg = createOutputMessage(requestId, [tokenId], "");
      this.ws.send(JSON.stringify(outputMsg));
    }

    if (this.onOutput) {
      this.onOutput(tokenId, requestId);
    }
  }

  /**
   * Send activation to the next node via coordinator.
   * @param {number} seqLen - Current sequence length (for KV cache coordination)
   */
  async _sendActivation(hidden, requestId, seqLen = 0) {
    // Diagnostic: measure outgoing activation so we can trace which shard
    // first introduces NaN in multi-hop pipelines.
    try {
      const sz = hidden.shape.reduce((a, b) => a * b, 1) * 4;
      const buf = await this.pipeline._readBuffer(hidden.buffer, 0, sz);
      const f = new Float32Array(buf);
      let mn = Infinity, mx = -Infinity, nans = 0, sum2 = 0;
      for (let i = 0; i < f.length; i++) {
        const v = f[i];
        if (Number.isNaN(v)) { nans++; continue; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum2 += v * v;
      }
      this._sendLog("perf", "shard_output_stats", {
        requestId,
        shardId: this.shardId,
        shape: hidden.shape,
        min: isFinite(mn) ? +mn.toFixed(4) : null,
        max: isFinite(mx) ? +mx.toFixed(4) : null,
        nans,
        rms: +Math.sqrt(sum2 / f.length).toFixed(4),
      });
    } catch (_) { /* ignore */ }

    if (this.useBinaryProtocol) {
      let flags = 0;
      let serialized;

      // Determine quantization mode. Priority:
      //   1. URL param ?quant=... (forces mode for A/B measurement)
      //   2. Adaptive precision selector (per-layer promotion)
      //   3. Default INT8 (or NONE if useQuantization=false)
      let quantMode = this.useQuantization ? QuantMode.INT8 : QuantMode.NONE;
      if (this.forceQuantMode === "int4") quantMode = QuantMode.INT4;
      else if (this.forceQuantMode === "int8") quantMode = QuantMode.INT8;
      else if (this.forceQuantMode === "none") quantMode = QuantMode.NONE;
      else if (this.adaptivePrecision) {
        quantMode = this.adaptivePrecision.getMode(0);
      }

      // Try delta encoding for cached steps (single-token, shape[0] === 1)
      const isSingleToken = hidden.shape[0] === 1;
      if (this.useDeltaEncoding && quantMode === QuantMode.INT8 && isSingleToken) {
        const prev = this._lastSentActivation.get(requestId) || null;
        const result = await this.pipeline.serializeTensorDelta(hidden, prev);
        serialized = result;
        flags = setQuantFlags(flags, QuantMode.INT8);
        if (result.isDelta) {
          flags |= Flags.DELTA;
          this._sendLog("perf", "delta_send", {
            requestId, sparsity: +(result.sparsity * 100).toFixed(1),
            payloadBytes: result.data.byteLength,
          });
        }
        // Cache the float32 for next delta
        this._lastSentActivation.set(requestId, result.currentFloat32);
      } else if (quantMode === QuantMode.INT4) {
        serialized = await this.pipeline.serializeTensorInt4(hidden);
        if (!serialized.fallbackUnquantized) {
          flags = setQuantFlags(flags, QuantMode.INT4);
        }
      } else if (quantMode === QuantMode.INT8) {
        serialized = await this.pipeline.serializeTensorQuantized(hidden);
        if (!serialized.fallbackUnquantized) {
          flags = setQuantFlags(flags, QuantMode.INT8);
        }
      } else {
        serialized = await this.pipeline.serializeTensorBinary(hidden);
      }

      // Feed the adaptive precision selector with this activation's float32 data
      if (this.adaptivePrecision) {
        // Use currentFloat32 from delta path, or read from GPU
        const f32 = serialized.currentFloat32
          || new Float32Array(await this.pipeline._readBuffer(
               hidden.buffer, 0, hidden.shape.reduce((a, b) => a * b, 1) * 4));
        this.adaptivePrecision.observe(0, f32);
      }

      // Apply RLE compression on quantized payloads (lossless)
      let payloadData = serialized.data;
      if (getQuantMode(flags) !== QuantMode.NONE) {
        const result = compressPayload(payloadData);
        if (result) {
          payloadData = result.compressed;
          flags |= Flags.COMPRESSED;
          this._sendLog("perf", "rle_compress", {
            requestId, ratio: +result.ratio.toFixed(2),
            originalBytes: serialized.data.byteLength,
            compressedBytes: payloadData.byteLength,
          });
        }
      }

      const numericId = requestIdToUint32(requestId);
      const binaryMsg = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION,
        flags,
        seqLen,          // seqPos: downstream nodes use this for KV cache
        numericId,
        serialized.shape,
        payloadData
      );
      // Try P2P direct transfer, fall back to coordinator relay
      const sentP2P = this.p2p?.send(binaryMsg);
      if (!sentP2P) {
        this.ws.send(binaryMsg);
      }
    } else {
      const serialized = await this.pipeline.serializeTensor(hidden);
      const msg = createActivationMessage(
        this.nodeId,
        null, // coordinator will fill in the destination
        this.layerEnd + 1,
        requestId,
        serialized.data,
        serialized.shape
      );
      msg.seqLen = seqLen; // include for KV cache coordination
      this.ws.send(JSON.stringify(msg));
    }
  }

  /**
   * Handle topology updates to know our position in the pipeline.
   */
  _handleTopologyUpdate(msg) {
    this.topology = msg;
    if (msg.pipeline && msg.pipeline.length > 0) {
      this.isFirstNode = msg.pipeline[0] === this.nodeId;
      this.isLastNode = msg.pipeline[msg.pipeline.length - 1] === this.nodeId;

      // Initiate P2P to downstream peer if not last node
      if (this.useP2P && !this.isLastNode && !this.p2p) {
        const myIdx = msg.pipeline.indexOf(this.nodeId);
        if (myIdx >= 0 && myIdx < msg.pipeline.length - 1) {
          const downstreamId = msg.pipeline[myIdx + 1];
          this.p2p = new P2PChannel(this.nodeId, this.ws);
          this.p2p.onMessage = (data) => this._handleBinaryMessage(data);
          this.p2p.onConnected = () => {
            this._sendLog("info", "p2p_connected", { peer: downstreamId });
          };
          this.p2p.onDisconnected = () => {
            this._sendLog("info", "p2p_disconnected", { peer: downstreamId });
            this.p2p?.close();
            this.p2p = null;
          };
          this.p2p.initiate(downstreamId).catch(err => {
            console.warn("[node] P2P initiation failed:", err.message);
            this.p2p = null;
          });
        }
      }
    }
  }

  _setStatus(status, detail = "") {
    this.status = status;
    this.onStatus(status, detail, this);
  }

  /**
   * Ship a log entry to the coordinator for centralized collection.
   * @param {"perf"|"info"|"warn"|"error"} level
   * @param {string} event - e.g. "prefill", "cached_step", "activation_send"
   * @param {object} data - arbitrary metrics
   */
  _sendLog(level, event, data = {}) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({
      type: "NODE_LOG",
      nodeId: this.nodeId,
      level,
      event,
      data: { shardId: this.shardId, ...data },
      timestamp: Date.now(),
    }));
  }

  /**
   * Detect if running on a mobile device.
   */
  _isMobile() {
    return /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent) ||
      (navigator.maxTouchPoints > 1 && /Mac/.test(navigator.userAgent));
  }

  /**
   * Setup mobile-specific handlers for visibility, wake lock, and reconnection.
   */
  _setupMobileHandlers() {
    // Request wake lock to prevent screen sleep during inference
    this._requestWakeLock();

    // Handle visibility changes — mobile browsers throttle/kill background tabs
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") {
        // Tab became visible again — re-acquire wake lock and check connection
        this._requestWakeLock();
        if (this.ws?.readyState !== WebSocket.OPEN && this.status !== "destroyed") {
          this._setStatus("reconnecting", "Tab resumed — reconnecting");
          this._connect(`${this.baseUrl.replace(/^http/, "ws")}`);
        }
      } else {
        // Tab hidden — release wake lock to save battery
        this._releaseWakeLock();
        this._setStatus("ready", "Tab hidden — paused (switch back to resume)");
      }
    });

    console.log("[node] Mobile mode enabled — wake lock + visibility handlers active");
  }

  async _requestWakeLock() {
    try {
      if ("wakeLock" in navigator && !this._wakeLock) {
        this._wakeLock = await navigator.wakeLock.request("screen");
        this._wakeLock.addEventListener("release", () => {
          this._wakeLock = null;
          console.log("[node] Wake lock released");
        });
        console.log("[node] Wake lock acquired — screen will stay on");
      }
    } catch (e) {
      // Wake lock may fail if tab is not visible
      console.log("[node] Wake lock unavailable:", e.message);
    }
  }

  _releaseWakeLock() {
    if (this._wakeLock) {
      this._wakeLock.release();
      this._wakeLock = null;
    }
  }

  /**
   * Disconnect and clean up.
   */
  destroy() {
    clearInterval(this.pingInterval);
    this._releaseWakeLock();
    if (this.ws) this.ws.close();
    if (this.loader) this.loader.destroy();
    this.status = "destroyed";
  }
}
