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

import { ShardLoader } from "./shard-loader.js";
import { Pipeline } from "./pipeline.js";
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

export class SynapseNode {
  constructor(statusCallback = null) {
    this.nodeId = `node-${crypto.randomUUID().slice(0, 8)}`;
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
    this.useQuantization = true; // int8 quantization for activation transfer
    this._binaryRequestIds = new Map(); // string -> uint32
  }

  /**
   * Initialize WebGPU and connect to the coordinator.
   */
  async start(coordinatorUrl) {
    this.baseUrl = coordinatorUrl.replace(/^ws/, "http");

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

    // Step 2: Connect to coordinator
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
        });
        this.ws.send(JSON.stringify(msg));

        // Start heartbeat
        this.pingInterval = setInterval(() => {
          if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(createPingMessage()));
          }
        }, 15000);

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

      case MessageType.ERROR:
        console.error(`[node] Error from coordinator: ${msg.message}`);
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

      this._setStatus("ready", `Loaded ${result.tensorCount} tensors`);

      // Tell coordinator we're ready
      this.ws.send(JSON.stringify(createNodeReadyMessage(this.nodeId, this.shardId)));

    } catch (err) {
      this._setStatus("error", `Failed to load shard: ${err.message}`);
      console.error("[node] Shard load error:", err);
    }
  }

  /**
   * Handle an inference request — prefill (only received by the first node).
   * Processes the full token sequence and populates the KV cache.
   */
  async _handleInferenceRequest(msg) {
    if (!this.pipeline) return;

    this._setStatus("computing", `Prefill ${msg.requestId} (${msg.tokenIds.length} tokens)`);
    const startTime = performance.now();

    try {
      const tokenIds = msg.tokenIds;

      // Embed all tokens
      let hidden = await this.pipeline.embed(tokenIds);

      // Run assigned layers with KV cache prefill
      hidden = await this.pipeline.forwardLayersPrefill(
        hidden, this.layerStart, this.layerEnd, msg.requestId
      );

      if (this.isLastNode) {
        await this._produceOutput(hidden, msg.requestId);
      } else {
        // Include seqPos so downstream nodes know the sequence length
        await this._sendActivation(hidden, msg.requestId, tokenIds.length);
      }

      const elapsed = performance.now() - startTime;
      this.pipeline._cleanupTempBuffers();
      this._setStatus("ready", `Prefill: ${elapsed.toFixed(0)}ms`);

    } catch (err) {
      console.error("[node] Inference error:", err);
      this.pipeline?._cleanupTempBuffers();
      this._setStatus("error", `Inference failed: ${err.message}`);
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
      // Embed single token at the given position
      let hidden = await this.pipeline.embedSingle(msg.tokenId, msg.seqPos);

      // Run layers with KV cache
      hidden = await this.pipeline.forwardLayersCached(
        hidden, this.layerStart, this.layerEnd, msg.requestId, msg.seqPos
      );

      if (this.isLastNode) {
        await this._produceOutput(hidden, msg.requestId);
      } else {
        await this._sendActivation(hidden, msg.requestId, msg.seqPos + 1);
      }

      const elapsed = performance.now() - startTime;
      this.pipeline._cleanupTempBuffers();
      this._setStatus("ready", `Step ${msg.seqPos}: ${elapsed.toFixed(0)}ms`);

    } catch (err) {
      console.error("[node] Inference step error:", err);
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
      // Detect quantized payload and dequantize if needed
      const quantMode = getQuantMode(decoded.flags);
      let hidden;
      if (quantMode === QuantMode.INT8) {
        hidden = this.pipeline.deserializeTensorQuantized(decoded.payload, decoded.shape);
      } else {
        hidden = this.pipeline.deserializeTensorBinary(decoded.payload, decoded.shape);
      }

      if (isPrefill) {
        // Prefill: full sequence, populate KV cache
        hidden = await this.pipeline.forwardLayersPrefill(
          hidden, this.layerStart, this.layerEnd, requestId
        );
      } else {
        // Cached step: single token
        const seqPos = seqLen - 1; // 0-indexed position of the new token
        hidden = await this.pipeline.forwardLayersCached(
          hidden, this.layerStart, this.layerEnd, requestId, seqPos
        );
      }

      if (this.isLastNode) {
        await this._produceOutput(hidden, requestId);
      } else {
        await this._sendActivation(hidden, requestId, seqLen);
      }

      const elapsed = performance.now() - startTime;
      this.pipeline._cleanupTempBuffers();
      this._setStatus("ready", `Processed in ${elapsed.toFixed(0)}ms`);
    } catch (err) {
      console.error("[node] Binary activation processing error:", err);
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

      if (isPrefill) {
        hidden = await this.pipeline.forwardLayersPrefill(
          hidden, this.layerStart, this.layerEnd, msg.requestId
        );
      } else {
        const seqPos = seqLen - 1;
        hidden = await this.pipeline.forwardLayersCached(
          hidden, this.layerStart, this.layerEnd, msg.requestId, seqPos
        );
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
  async _produceOutput(hidden, requestId) {
    const logitsTensor = await this.pipeline.outputHead(hidden);
    const tokenId = await this.pipeline.sampleToken(logitsTensor, 0.8);

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
    if (this.useBinaryProtocol) {
      let flags = 0;
      let serialized;

      if (this.useQuantization) {
        serialized = await this.pipeline.serializeTensorQuantized(hidden);
        flags = setQuantFlags(flags, QuantMode.INT8);
      } else {
        serialized = await this.pipeline.serializeTensorBinary(hidden);
      }

      const numericId = requestIdToUint32(requestId);
      const binaryMsg = encodeBinaryMessage(
        BinaryMsgType.ACTIVATION,
        flags,
        seqLen,          // seqPos: downstream nodes use this for KV cache
        numericId,
        serialized.shape,
        serialized.data
      );
      this.ws.send(binaryMsg);
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
    }
  }

  _setStatus(status, detail = "") {
    this.status = status;
    this.onStatus(status, detail, this);
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
