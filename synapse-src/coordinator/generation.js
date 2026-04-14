/**
 * Generation — Autoregressive generation state machine.
 *
 * Encapsulates the state and logic for one autoregressive generation session:
 * token accumulation, EOS detection, timeout, and statistics.
 * Decoupled from WebSocket/IO so the core logic is testable.
 */

const EOS_TOKEN = 50256; // GPT-2 end-of-sequence

export class Generation {
  /**
   * @param {string} id - Unique generation identifier
   * @param {number[]} promptTokens - Initial prompt token IDs
   * @param {number} maxTokens - Maximum tokens to generate
   * @param {number} [timeoutMs=60000] - Timeout for stalled generations
   */
  constructor(id, promptTokens, maxTokens, timeoutMs = 60000) {
    this.id = id;
    this.tokenIds = [...promptTokens];
    this.promptLen = promptTokens.length;
    this.generatedTokens = [];
    this.maxTokens = maxTokens;
    this.timeoutMs = timeoutMs;
    this.startTime = Date.now();
    this._lastTokenTime = null;
    this.prefillDone = false;
    this._binaryReqId = null;
    this.promptWs = null; // WebSocket of the prompt client that initiated this generation
  }

  /**
   * Record a newly generated token. Returns the generation status.
   * @param {number} token - The generated token ID
   * @returns {{ done: boolean, reason: string|null, seqPos: number }}
   */
  addToken(token) {
    this.generatedTokens.push(token);
    this.tokenIds.push(token);
    this._lastTokenTime = Date.now();
    this.prefillDone = true;

    const isEOS = token === EOS_TOKEN;
    const hitMax = this.generatedTokens.length >= this.maxTokens;

    return {
      done: isEOS || hitMax,
      reason: isEOS ? "eos" : hitMax ? "max_tokens" : null,
      seqPos: this.tokenIds.length - 1,
    };
  }

  /**
   * Check if this generation has timed out (no token produced within timeoutMs).
   * @param {number} [now] - Current timestamp (default: Date.now())
   * @returns {boolean}
   */
  isTimedOut(now = Date.now()) {
    const lastActivity = this._lastTokenTime || this.startTime;
    return now - lastActivity > this.timeoutMs;
  }

  /**
   * Get generation statistics.
   * @param {number} [now] - Current timestamp
   * @returns {{ elapsedMs: number, tokensPerSecond: number, totalTokens: number, promptLen: number }}
   */
  getStats(now = Date.now()) {
    const elapsedMs = now - this.startTime;
    const totalTokens = this.generatedTokens.length;
    const tokensPerSecond = elapsedMs > 0
      ? parseFloat((totalTokens / (elapsedMs / 1000)).toFixed(1))
      : 0;

    return { elapsedMs, tokensPerSecond, totalTokens, promptLen: this.promptLen };
  }

  /**
   * Get the next sequence position for a continuation step.
   * @returns {number}
   */
  get nextSeqPos() {
    return this.tokenIds.length - 1;
  }

  /**
   * Get the last generated token (for continuation).
   * @returns {number|null}
   */
  get lastToken() {
    return this.generatedTokens.length > 0
      ? this.generatedTokens[this.generatedTokens.length - 1]
      : null;
  }
}

/**
 * GenerationManager — Manages multiple active generations.
 *
 * Provides create, lookup, cleanup, and timeout sweeping
 * without coupling to any I/O layer.
 */
export class GenerationManager {
  constructor(timeoutMs = 60000) {
    this.generations = new Map();
    this.timeoutMs = timeoutMs;
  }

  /**
   * Create and register a new generation.
   * @param {string} id
   * @param {number[]} promptTokens
   * @param {number} maxTokens
   * @returns {Generation}
   */
  create(id, promptTokens, maxTokens) {
    const gen = new Generation(id, promptTokens, maxTokens, this.timeoutMs);
    this.generations.set(id, gen);
    return gen;
  }

  /**
   * Get a generation by ID.
   * @param {string} id
   * @returns {Generation|undefined}
   */
  get(id) {
    return this.generations.get(id);
  }

  /**
   * Remove a generation.
   * @param {string} id
   */
  remove(id) {
    this.generations.delete(id);
  }

  /**
   * Find all timed-out generations.
   * @param {number} [now]
   * @returns {Generation[]}
   */
  getTimedOut(now = Date.now()) {
    const timedOut = [];
    for (const gen of this.generations.values()) {
      if (gen.isTimedOut(now)) {
        timedOut.push(gen);
      }
    }
    return timedOut;
  }

  /**
   * Remove all timed-out generations. Returns the removed ones.
   * @param {number} [now]
   * @returns {Generation[]}
   */
  sweepTimedOut(now = Date.now()) {
    const timedOut = this.getTimedOut(now);
    for (const gen of timedOut) {
      this.generations.delete(gen.id);
    }
    return timedOut;
  }

  /**
   * Number of active generations.
   */
  get size() {
    return this.generations.size;
  }
}
