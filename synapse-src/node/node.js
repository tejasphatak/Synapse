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
  createJoinMessage,
  createPingMessage,
  createActivationMessage,
  createOutputMessage,
  createNodeReadyMessage,
} from "../protocol/messages.js";

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

      this.ws.onopen = () => {
        this._setStatus("connected");

        // Send JOIN with device capabilities
        const msg = createJoinMessage(this.nodeId, {
          webgpu: true,
          maxLayers: 6,
          mobile: this.gpuInfo?.isMobile || false,
          gpuVendor: this.gpuInfo?.vendor || "unknown",
          maxBufferMB: Math.round((this.gpuInfo?.maxBufferSize || 0) / 1024 / 1024),
          userAgent: navigator.userAgent,
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
   * Handle incoming WebSocket messages.
   */
  async _handleMessage(raw) {
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
        await this._handleInferenceRequest(msg);
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
   * Handle an inference request (only received by the first node).
   */
  async _handleInferenceRequest(msg) {
    if (!this.pipeline) return;

    this._setStatus("computing", `Inference ${msg.requestId}`);
    const startTime = performance.now();

    try {
      const tokenIds = msg.tokenIds;

      // Embed tokens
      let hidden = await this.pipeline.embed(tokenIds);

      // Run assigned layers
      hidden = await this.pipeline.forwardLayers(hidden, this.layerStart, this.layerEnd);

      if (this.isLastNode) {
        // This node holds the final layers — produce output
        await this._produceOutput(hidden, msg.requestId);
      } else {
        // Send activation to next node via coordinator
        await this._sendActivation(hidden, msg.requestId);
      }

      const elapsed = performance.now() - startTime;
      this._setStatus("ready", `Last inference: ${elapsed.toFixed(0)}ms`);

    } catch (err) {
      console.error("[node] Inference error:", err);
      this._setStatus("error", `Inference failed: ${err.message}`);
    }
  }

  /**
   * Handle an activation message from the previous node.
   */
  async _handleActivation(msg) {
    if (!this.pipeline) return;

    this._setStatus("computing", `Processing activation for ${msg.requestId}`);
    const startTime = performance.now();

    try {
      // Deserialize the incoming tensor
      let hidden = this.pipeline.deserializeTensor(msg.tensor);

      // Run assigned layers
      hidden = await this.pipeline.forwardLayers(hidden, this.layerStart, this.layerEnd);

      if (this.isLastNode) {
        await this._produceOutput(hidden, msg.requestId);
      } else {
        await this._sendActivation(hidden, msg.requestId);
      }

      const elapsed = performance.now() - startTime;
      this._setStatus("ready", `Processed in ${elapsed.toFixed(0)}ms`);

    } catch (err) {
      console.error("[node] Activation processing error:", err);
      this._setStatus("error", `Processing failed: ${err.message}`);
    }
  }

  /**
   * Produce final output: run output head, sample token, send OUTPUT.
   */
  async _produceOutput(hidden, requestId) {
    const logitsTensor = await this.pipeline.outputHead(hidden);
    const tokenId = await this.pipeline.sampleToken(logitsTensor, 0.8);

    const outputMsg = createOutputMessage(requestId, [tokenId], "");
    this.ws.send(JSON.stringify(outputMsg));

    if (this.onOutput) {
      this.onOutput(tokenId, requestId);
    }
  }

  /**
   * Send activation to the next node via coordinator.
   */
  async _sendActivation(hidden, requestId) {
    const serialized = await this.pipeline.serializeTensor(hidden);

    const msg = createActivationMessage(
      this.nodeId,
      null, // coordinator will fill in the destination
      this.layerEnd + 1,
      requestId,
      serialized.data,
      serialized.shape
    );

    this.ws.send(JSON.stringify(msg));
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
