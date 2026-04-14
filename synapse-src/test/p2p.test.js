/**
 * P2P Channel Tests — WebRTC data channel signaling, send/receive, fallback.
 *
 * Mocks RTCPeerConnection and DataChannel since Node.js has no WebRTC.
 * Tests the control flow: signaling, connection lifecycle, message passing, stats.
 */

import { describe, it, beforeEach } from "node:test";
import assert from "node:assert/strict";

// ─── Mock WebRTC globals ─────────────────────────────────────

class MockDataChannel {
  constructor(label, opts) {
    this.label = label;
    this.ordered = opts?.ordered ?? true;
    this.binaryType = "arraybuffer";
    this.readyState = "connecting";
    this.onopen = null;
    this.onclose = null;
    this.onmessage = null;
    this._sent = [];
  }
  send(data) {
    if (this.readyState !== "open") throw new Error("Channel not open");
    this._sent.push(data);
  }
  close() {
    this.readyState = "closed";
    this.onclose?.();
  }
  _simulateOpen() {
    this.readyState = "open";
    this.onopen?.();
  }
  _simulateMessage(data) {
    this.onmessage?.({ data });
  }
}

class MockRTCPeerConnection {
  constructor(config) {
    this.config = config;
    this.localDescription = null;
    this.remoteDescription = null;
    this.onicecandidate = null;
    this.ondatachannel = null;
    this._dataChannels = [];
    this._iceCandidates = [];
    this._closed = false;
  }
  createDataChannel(label, opts) {
    const dc = new MockDataChannel(label, opts);
    this._dataChannels.push(dc);
    return dc;
  }
  async createOffer() {
    return { type: "offer", sdp: "mock-offer-sdp" };
  }
  async createAnswer() {
    return { type: "answer", sdp: "mock-answer-sdp" };
  }
  async setLocalDescription(desc) {
    this.localDescription = desc;
  }
  async setRemoteDescription(desc) {
    this.remoteDescription = desc;
  }
  async addIceCandidate(candidate) {
    this._iceCandidates.push(candidate);
  }
  close() {
    this._closed = true;
  }
  _simulateRemoteDataChannel(dc) {
    this.ondatachannel?.({ channel: dc });
  }
}

// Install globals
globalThis.RTCPeerConnection = MockRTCPeerConnection;
globalThis.performance = globalThis.performance || { now: () => Date.now() };

import { P2PChannel } from "../node/p2p.js";

// ─── Mock WebSocket ──────────────────────────────────────────

function mockWs() {
  const sent = [];
  return {
    send(data) { sent.push(JSON.parse(data)); },
    _sent: sent,
  };
}

// ─── Tests ───────────────────────────────────────────────────

describe("P2PChannel", () => {
  let ws, channel;

  beforeEach(() => {
    ws = mockWs();
    channel = new P2PChannel("node-A", ws);
  });

  describe("constructor", () => {
    it("initializes with correct defaults", () => {
      assert.equal(channel.nodeId, "node-A");
      assert.equal(channel.connected, false);
      assert.equal(channel.peerConnection, null);
      assert.equal(channel.dataChannel, null);
      assert.equal(channel.remoteNodeId, null);
    });

    it("initializes stats to zero", () => {
      const stats = channel.getStats();
      assert.equal(stats.messagesSent, 0);
      assert.equal(stats.messagesReceived, 0);
      assert.equal(stats.bytesSent, 0);
      assert.equal(stats.bytesReceived, 0);
      assert.equal(stats.connected, false);
    });
  });

  describe("initiate()", () => {
    it("creates RTCPeerConnection with STUN servers", async () => {
      await channel.initiate("node-B");
      assert.ok(channel.peerConnection);
      assert.equal(channel.peerConnection.config.iceServers.length, 2);
    });

    it("creates data channel named 'activations'", async () => {
      await channel.initiate("node-B");
      assert.ok(channel.dataChannel);
      assert.equal(channel.dataChannel.label, "activations");
      assert.equal(channel.dataChannel.binaryType, "arraybuffer");
    });

    it("sends SDP offer via coordinator WebSocket", async () => {
      await channel.initiate("node-B");
      const offer = ws._sent.find(m => m.type === "P2P_SIGNAL" && m.signalType === "sdp-offer" && m.sdp);
      assert.ok(offer, "should send an offer signal");
      assert.equal(offer.from, "node-A");
      assert.equal(offer.to, "node-B");
    });

    it("sets remoteNodeId", async () => {
      await channel.initiate("node-B");
      assert.equal(channel.remoteNodeId, "node-B");
    });

    it("marks connected on dataChannel open", async () => {
      await channel.initiate("node-B");
      assert.equal(channel.connected, false);
      channel.dataChannel._simulateOpen();
      assert.equal(channel.connected, true);
    });

    it("fires onConnected callback", async () => {
      let called = false;
      channel.onConnected = () => { called = true; };
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();
      assert.ok(called);
    });

    it("tracks connect time", async () => {
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();
      assert.ok(channel.stats.connectTimeMs >= 0);
    });

    it("relays ICE candidates via signaling", async () => {
      await channel.initiate("node-B");
      channel.peerConnection.onicecandidate({ candidate: { sdpMid: "0" } });
      const ice = ws._sent.find(m => m.type === "P2P_SIGNAL" && m.signalType === "ice-candidate" && m.candidate);
      assert.ok(ice);
      assert.equal(ice.from, "node-A");
      assert.equal(ice.to, "node-B");
    });

    it("ignores null ICE candidates", async () => {
      await channel.initiate("node-B");
      const before = ws._sent.length;
      channel.peerConnection.onicecandidate({ candidate: null });
      assert.equal(ws._sent.length, before);
    });
  });

  describe("handleSignal() — responder", () => {
    it("handles SDP offer: creates connection and sends answer", async () => {
      await channel.handleSignal({
        type: "sdp-offer",
        sdp: { type: "offer", sdp: "remote-offer" },
        from: "node-B",
      });

      assert.ok(channel.peerConnection);
      assert.equal(channel.remoteNodeId, "node-B");
      assert.equal(channel.peerConnection.remoteDescription.sdp, "remote-offer");
      assert.equal(channel.peerConnection.localDescription.sdp, "mock-answer-sdp");

      const answer = ws._sent.find(m => m.sdp?.type === "answer");
      assert.ok(answer);
      assert.equal(answer.from, "node-A");
      assert.equal(answer.to, "node-B");
    });

    it("handles SDP answer: sets remote description", async () => {
      await channel.initiate("node-B");
      await channel.handleSignal({
        type: "sdp-answer",
        sdp: { type: "answer", sdp: "remote-answer" },
      });
      assert.equal(channel.peerConnection.remoteDescription.sdp, "remote-answer");
    });

    it("handles ICE candidate: adds to peer connection", async () => {
      await channel.initiate("node-B");
      await channel.handleSignal({
        type: "ice-candidate",
        candidate: { sdpMid: "0", candidate: "candidate:..." },
      });
      assert.equal(channel.peerConnection._iceCandidates.length, 1);
    });

    it("sets up data channel events on ondatachannel", async () => {
      await channel.handleSignal({
        type: "sdp-offer",
        sdp: { type: "offer", sdp: "remote-offer" },
        from: "node-B",
      });

      // Simulate remote data channel arriving
      const remoteDc = new MockDataChannel("activations", {});
      channel.peerConnection._simulateRemoteDataChannel(remoteDc);

      assert.equal(channel.dataChannel, remoteDc);
      assert.equal(channel.dataChannel.binaryType, "arraybuffer");

      // Simulate open
      remoteDc._simulateOpen();
      assert.equal(channel.connected, true);
    });
  });

  describe("send()", () => {
    it("returns false when not connected", () => {
      const result = channel.send(new ArrayBuffer(10));
      assert.equal(result, false);
    });

    it("returns false when dataChannel is null", () => {
      channel.connected = true;
      const result = channel.send(new ArrayBuffer(10));
      assert.equal(result, false);
    });

    it("sends binary data and returns true when connected", async () => {
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();

      const data = new ArrayBuffer(24);
      const result = channel.send(data);
      assert.equal(result, true);
      assert.equal(channel.dataChannel._sent.length, 1);
      assert.equal(channel.dataChannel._sent[0], data);
    });

    it("tracks send stats", async () => {
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();

      channel.send(new ArrayBuffer(100));
      channel.send(new ArrayBuffer(200));

      assert.equal(channel.stats.messagesSent, 2);
      assert.equal(channel.stats.bytesSent, 300);
    });

    it("returns false on send error (fallback to relay)", async () => {
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();
      channel.dataChannel.send = () => { throw new Error("buffer full"); };

      const result = channel.send(new ArrayBuffer(10));
      assert.equal(result, false);
    });
  });

  describe("receive", () => {
    it("tracks receive stats via onMessage", async () => {
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();

      let received = null;
      channel.onMessage = (data) => { received = data; };

      const payload = new ArrayBuffer(50);
      channel.dataChannel._simulateMessage(payload);

      assert.equal(received, payload);
      assert.equal(channel.stats.messagesReceived, 1);
      assert.equal(channel.stats.bytesReceived, 50);
    });

    it("handles multiple messages", async () => {
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();

      let count = 0;
      channel.onMessage = () => { count++; };

      channel.dataChannel._simulateMessage(new ArrayBuffer(10));
      channel.dataChannel._simulateMessage(new ArrayBuffer(20));
      channel.dataChannel._simulateMessage(new ArrayBuffer(30));

      assert.equal(count, 3);
      assert.equal(channel.stats.messagesReceived, 3);
      assert.equal(channel.stats.bytesReceived, 60);
    });
  });

  describe("close()", () => {
    it("closes data channel and peer connection", async () => {
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();

      channel.close();
      assert.equal(channel.connected, false);
      assert.equal(channel.dataChannel, null);
      assert.equal(channel.peerConnection, null);
    });

    it("handles close when already null", () => {
      channel.close(); // should not throw
      assert.equal(channel.connected, false);
    });

    it("fires onDisconnected on data channel close", async () => {
      let disconnected = false;
      channel.onDisconnected = () => { disconnected = true; };
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();
      channel.dataChannel.close();
      assert.ok(disconnected);
    });
  });

  describe("getStats()", () => {
    it("includes connection state", async () => {
      await channel.initiate("node-B");
      channel.dataChannel._simulateOpen();

      const stats = channel.getStats();
      assert.equal(stats.connected, true);
      assert.equal(stats.remoteNodeId, "node-B");
    });

    it("returns a copy (not a reference)", () => {
      const stats = channel.getStats();
      stats.messagesSent = 999;
      assert.equal(channel.stats.messagesSent, 0);
    });
  });

  describe("full signaling flow (initiator + responder)", () => {
    it("completes offer/answer/ICE exchange", async () => {
      const wsA = mockWs();
      const wsB = mockWs();
      const nodeA = new P2PChannel("node-A", wsA);
      const nodeB = new P2PChannel("node-B", wsB);

      // A initiates → sends offer
      await nodeA.initiate("node-B");
      const offerSignal = wsA._sent.find(m => m.sdp);

      // B receives offer → sends answer
      await nodeB.handleSignal({
        type: "sdp-offer",
        sdp: offerSignal.sdp,
        from: "node-A",
      });
      const answerSignal = wsB._sent.find(m => m.sdp?.type === "answer");

      // A receives answer
      await nodeA.handleSignal({
        type: "sdp-answer",
        sdp: answerSignal.sdp,
      });

      // Both sides have descriptions set
      assert.ok(nodeA.peerConnection.localDescription);
      assert.ok(nodeA.peerConnection.remoteDescription);
      assert.ok(nodeB.peerConnection.localDescription);
      assert.ok(nodeB.peerConnection.remoteDescription);
    });
  });
});
