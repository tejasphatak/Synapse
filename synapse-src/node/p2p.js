/**
 * Synapse P2P — WebRTC data channel for direct node-to-node activation transfer.
 *
 * Eliminates the coordinator relay hop. Activations go directly between
 * browser tabs/devices. Coordinator is still used for:
 *   - ICE signaling (SDP offer/answer exchange)
 *   - Topology management (who connects to whom)
 *   - Fallback relay if P2P fails
 *
 * Architecture:
 *   Node 0 ──WebRTC DataChannel──> Node 1
 *        \                          /
 *         └── WebSocket (signaling) ──┘
 *              via Coordinator
 */

export class P2PChannel {
  constructor(nodeId, ws) {
    this.nodeId = nodeId;
    this.ws = ws; // WebSocket to coordinator for signaling
    this.peerConnection = null;
    this.dataChannel = null;
    this.remoteNodeId = null;
    this.connected = false;
    this.onMessage = null; // callback for incoming binary messages
    this.onConnected = null;
    this.onDisconnected = null;

    // Stats
    this.stats = {
      messagesSent: 0,
      messagesReceived: 0,
      bytesSent: 0,
      bytesReceived: 0,
      connectTimeMs: 0,
    };
  }

  /**
   * Initiate a P2P connection to the next node in the pipeline.
   * Called by the upstream node (the one sending activations).
   *
   * @param {string} remoteNodeId - the downstream node to connect to
   */
  async initiate(remoteNodeId) {
    this.remoteNodeId = remoteNodeId;
    const startTime = performance.now();

    this.peerConnection = new RTCPeerConnection({
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
        { urls: "stun:stun1.l.google.com:19302" },
      ],
    });

    // Create data channel — ordered + reliable for activations
    this.dataChannel = this.peerConnection.createDataChannel("activations", {
      ordered: true,
      // No maxRetransmits — we need reliable delivery for correctness
    });
    this.dataChannel.binaryType = "arraybuffer";

    this.dataChannel.onopen = () => {
      this.connected = true;
      this.stats.connectTimeMs = performance.now() - startTime;
      console.log(`[p2p] Data channel open to ${remoteNodeId} (${this.stats.connectTimeMs.toFixed(0)}ms)`);
      this.onConnected?.();
    };

    this.dataChannel.onclose = () => {
      this.connected = false;
      console.log(`[p2p] Data channel closed to ${remoteNodeId}`);
      this.onDisconnected?.();
    };

    this.dataChannel.onmessage = (event) => {
      this.stats.messagesReceived++;
      this.stats.bytesReceived += event.data.byteLength;
      this.onMessage?.(event.data);
    };

    // ICE candidates → send to coordinator for relay to remote node
    this.peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        this._sendSignal({
          type: "ice-candidate",
          candidate: event.candidate,
          from: this.nodeId,
          to: remoteNodeId,
        });
      }
    };

    // Create and send offer
    const offer = await this.peerConnection.createOffer();
    await this.peerConnection.setLocalDescription(offer);
    this._sendSignal({
      type: "sdp-offer",
      sdp: offer,
      from: this.nodeId,
      to: remoteNodeId,
    });
  }

  /**
   * Handle an incoming P2P signal from the coordinator.
   * Called when coordinator relays SDP or ICE from the remote node.
   */
  async handleSignal(signal) {
    if (signal.type === "sdp-offer") {
      // We're the responder — create peer connection and answer
      this.remoteNodeId = signal.from;
      const startTime = performance.now();

      this.peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: "stun:stun.l.google.com:19302" },
          { urls: "stun:stun1.l.google.com:19302" },
        ],
      });

      this.peerConnection.ondatachannel = (event) => {
        this.dataChannel = event.channel;
        this.dataChannel.binaryType = "arraybuffer";

        this.dataChannel.onopen = () => {
          this.connected = true;
          this.stats.connectTimeMs = performance.now() - startTime;
          console.log(`[p2p] Data channel open from ${signal.from} (${this.stats.connectTimeMs.toFixed(0)}ms)`);
          this.onConnected?.();
        };

        this.dataChannel.onclose = () => {
          this.connected = false;
          this.onDisconnected?.();
        };

        this.dataChannel.onmessage = (event) => {
          this.stats.messagesReceived++;
          this.stats.bytesReceived += event.data.byteLength;
          this.onMessage?.(event.data);
        };
      };

      this.peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          this._sendSignal({
            type: "ice-candidate",
            candidate: event.candidate,
            from: this.nodeId,
            to: signal.from,
          });
        }
      };

      await this.peerConnection.setRemoteDescription(signal.sdp);
      const answer = await this.peerConnection.createAnswer();
      await this.peerConnection.setLocalDescription(answer);
      this._sendSignal({
        type: "sdp-answer",
        sdp: answer,
        from: this.nodeId,
        to: signal.from,
      });

    } else if (signal.type === "sdp-answer") {
      await this.peerConnection.setRemoteDescription(signal.sdp);

    } else if (signal.type === "ice-candidate") {
      await this.peerConnection.addIceCandidate(signal.candidate);
    }
  }

  /**
   * Send a binary message directly to the peer.
   * Returns false if P2P is not available (caller should fall back to WS relay).
   *
   * @param {ArrayBuffer} data - SYN1 binary message
   * @returns {boolean} - true if sent via P2P, false if unavailable
   */
  send(data) {
    if (!this.connected || !this.dataChannel || this.dataChannel.readyState !== "open") {
      return false; // caller should fall back to coordinator relay
    }

    try {
      this.dataChannel.send(data);
      this.stats.messagesSent++;
      this.stats.bytesSent += data.byteLength;
      return true;
    } catch (err) {
      console.warn(`[p2p] Send failed, falling back to relay:`, err.message);
      return false;
    }
  }

  /**
   * Close the P2P connection.
   */
  close() {
    this.dataChannel?.close();
    this.peerConnection?.close();
    this.connected = false;
    this.dataChannel = null;
    this.peerConnection = null;
  }

  getStats() {
    return { ...this.stats, connected: this.connected, remoteNodeId: this.remoteNodeId };
  }

  // Send signaling message through coordinator WebSocket
  _sendSignal(signal) {
    this.ws.send(JSON.stringify({ type: "P2P_SIGNAL", ...signal }));
  }
}
