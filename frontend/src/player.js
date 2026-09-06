/**
 * player.js
 * ──────────
 * Gapless TTS audio player for raw linear16 PCM chunks.
 *
 * Continuous-playback refactor (test-latency-tracking branch):
 *   This version decouples playback from the capture state machine by making
 *   every operation turn-aware. It supports:
 *     - pause(turnId)       → freeze playback, retain unplayed samples
 *     - resumeQueue()       → idempotent reschedule from retained entries
 *     - markAudioEnd(turnId)→ signal that no more chunks will arrive for a turn
 *     - enqueue(turnId, …)  → append PCM data tagged to a specific turn
 *     - onFinished(turnId)  → completion callback identifies which turn finished
 *
 * Key invariants:
 *   - A playbackGeneration token is captured by each source.onended callback;
 *     stale callbacks (from a cleared or paused player) are silently dropped.
 *   - The queue retains decoded AudioBuffer objects with precise offset tracking
 *     so that pause/resume never replays already-heard samples.
 *   - Maximum 2 concurrent turns, 90 seconds, ~8.5 MB decoded PCM.
 *   - sourceNode references are nulled after stop() to prevent double-scheduling.
 *   - Exposes per-turn sample counts via decodedSamplesByTurn for test hooks.
 */

/** Default sample rate when the server message has no sample_rate_hz field. */
const DEFAULT_SAMPLE_RATE = 24_000;

/** 24 kHz mono Float32 = 96 KB/s → 90 seconds ≈ 8.64 MB */
const MAX_QUEUED_SECONDS = 90;
const MAX_QUEUED_BYTES   = MAX_QUEUED_SECONDS * DEFAULT_SAMPLE_RATE * 4; // Float32
const MAX_TURNS          = 2;

export class AudioPlayer {
  constructor() {
    /** @type {AudioContext|null} */
    this._ctx = null;

    /** Monotonic integer — incremented on each clear() call. */
    this._generation = 0;

    /** Time (Web Audio clock) when the next chunk should start. */
    this._nextStartTime = 0;

    /** Resolved sample rate for the current AudioContext. */
    this._ctxSampleRate = DEFAULT_SAMPLE_RATE;

    /**
     * The ordered playback queue. Each entry represents one decoded chunk.
     * @type {Array<QueueEntry>}
     *
     * @typedef {Object} QueueEntry
     * @property {string}                  turnId      — owning turn
     * @property {AudioBuffer}             buffer      — decoded PCM
     * @property {AudioBufferSourceNode|null} sourceNode — null when paused/unscheduled
     * @property {number}                  startTime   — Web Audio clock start
     * @property {number}                  duration    — total buffer duration (seconds)
     * @property {number}                  startOffset — samples already played (seconds)
     * @property {boolean}                 played      — true when onended fired normally
     */
    this._queue = [];

    /** Whether the player is in a paused state. */
    this._paused = false;

    /**
     * Set of turnIds for which markAudioEnd has been called.
     * @type {Set<string>}
     */
    this._flushedTurns = new Set();

    /**
     * Set of turnIds that have already been finalized (onFinished fired).
     * Prevents double-finalization.
     * @type {Set<string>}
     */
    this._finishedTurns = new Set();

    /**
     * Ordered list of distinct turnIds in playback order.
     * @type {string[]}
     */
    this._turnOrder = [];

    // ── Observable counters ──────────────────────────────────────────────
    this.audio_started       = 0;   // first chunk played per turn
    this.audio_finished      = 0;   // last chunk ended naturally
    this.audio_cleared       = 0;   // clear() calls

    /** Per-turn decoded Int16 sample counts. @type {Map<string, number>} */
    this.decodedSamplesByTurn = new Map();

    /** Global decoded sample count (sum of all turns, for backward compat). */
    this.decodedSampleCount = 0;

    /**
     * Running byte total of queued Float32 data across all turns.
     * Used for enforcing MAX_QUEUED_BYTES.
     */
    this._queuedBytes = 0;

    /** Called when playback starts (first chunk of any turn). @type {function(): void} */
    this.onStarted  = null;

    /**
     * Called when all audio for a specific turn finishes naturally.
     * @type {function(string): void}
     */
    this.onFinished = null;

    /** @type {function(boolean): void} */
    this.onStateChange = null;

    /**
     * Called when the queue overflows.
     * @type {function(string): void}  — receives the turnId that caused overflow
     */
    this.onOverflow = null;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Initializes or resumes the AudioContext.
   * MUST be called from a trusted user interaction (click handler) to bypass autoplay policies.
   */
  initContext() {
    this._ensureCtx(DEFAULT_SAMPLE_RATE);
  }

  /**
   * Decode and schedule a PCM audio chunk for gapless playback.
   *
   * @param {string} turnId      — the turn this chunk belongs to
   * @param {string} base64data  — base64-encoded raw linear16 PCM bytes
   * @param {Object} [meta]      — metadata from the audio_chunk server message
   * @param {number} [meta.sample_rate_hz=24000]
   * @param {number} [meta.channels=1]
   * @param {string} [meta.format='pcm_s16le']
   */
  enqueue(turnId, base64data, meta = {}) {
    const sampleRate = meta.sample_rate_hz || DEFAULT_SAMPLE_RATE;
    const channels   = meta.channels       || 1;

    // ── Bounds check: turn count ──────────────────────────────────────────
    const activeTurns = new Set(this._queue.filter(e => !e.played).map(e => e.turnId));
    if (!activeTurns.has(turnId) && activeTurns.size >= MAX_TURNS) {
      console.error(`[Player] Queue overflow: MAX_TURNS=${MAX_TURNS} reached, rejecting turnId=${turnId}`);
      if (this.onOverflow) this.onOverflow(turnId);
      return;
    }

    if (meta.sample_rate_hz && meta.sample_rate_hz !== DEFAULT_SAMPLE_RATE) {
      console.warn(`[Player] Non-default sample rate ${meta.sample_rate_hz}Hz from server`);
    }

    try {
      const buffer = this._decodeChunk(base64data, sampleRate, channels);

      // ── Bounds check: bytes ──────────────────────────────────────────────
      const chunkBytes = buffer.length * 4; // Float32
      if (this._queuedBytes + chunkBytes > MAX_QUEUED_BYTES) {
        console.error(`[Player] Queue overflow: ${this._queuedBytes + chunkBytes} > ${MAX_QUEUED_BYTES} bytes, rejecting turnId=${turnId}`);
        if (this.onOverflow) this.onOverflow(turnId);
        return;
      }

      // ── Per-turn sample accounting ───────────────────────────────────────
      const prevCount = this.decodedSamplesByTurn.get(turnId) || 0;
      this.decodedSamplesByTurn.set(turnId, prevCount + buffer.length);
      this.decodedSampleCount += buffer.length;
      this._queuedBytes += chunkBytes;

      // Track turn order
      if (!this._turnOrder.includes(turnId)) {
        this._turnOrder.push(turnId);
      }

      if (this._paused) {
        // When paused, enqueue without scheduling
        this._queue.push({
          turnId,
          buffer,
          sourceNode:  null,
          startTime:   0,
          duration:    buffer.duration,
          startOffset: 0,
          played:      false,
        });
      } else {
        this._scheduleBuffer(turnId, buffer, this._generation);
      }
    } catch (e) {
      console.error('[Player] Failed to decode chunk:', e);
    }
  }

  /**
   * Pause playback for a specific turn. All active AudioBufferSourceNodes are
   * stopped, their callbacks detached, and unplayed portions retained in the
   * queue with precise offset calculations.
   *
   * Calling pause() while already paused is a no-op.
   *
   * @param {string} _turnId — the turn being paused (for logging; all playback stops)
   */
  pause(_turnId) {
    if (this._paused) return;
    if (!this._ctx) return;

    this._paused = true;
    const now = this._ctx.currentTime;

    for (const entry of this._queue) {
      if (entry.played || !entry.sourceNode) continue;

      const effectiveStart = entry.startTime;
      const effectiveEnd   = effectiveStart + (entry.duration - entry.startOffset);

      if (effectiveEnd <= now) {
        // Already finished playing — mark completed
        entry.played = true;
        this._releaseEntryBytes(entry);
      } else if (effectiveStart <= now) {
        // Currently playing — compute how much has been heard
        const elapsed = now - effectiveStart;
        entry.startOffset += elapsed;
      }
      // else: scheduled for the future — startOffset stays unchanged

      // Detach callback and stop the node
      if (entry.sourceNode) {
        entry.sourceNode.onended = null;
        try { entry.sourceNode.stop(); } catch (_) {}
        entry.sourceNode = null;
      }
    }

    // Remove fully-played entries
    this._queue = this._queue.filter(e => !e.played);

    if (this.onStateChange) this.onStateChange(false);
    console.log(`[Player] Paused for turn=${_turnId} remaining_entries=${this._queue.length}`);
  }

  /**
   * Resume playback from the current queue position. Entries are rescheduled
   * exactly once from their stored offsets.
   *
   * Calling resumeQueue() while not paused is a no-op.
   */
  resumeQueue() {
    if (!this._paused) return;
    this._paused = false;

    if (this._queue.length === 0) {
      // Nothing to resume — check if any flushed turns should finalize
      this._checkFlushedTurnsCompletion();
      return;
    }

    const ctx = this._ensureCtx(this._ctxSampleRate);
    this._nextStartTime = ctx.currentTime;
    const gen = this._generation;

    let isFirstChunk = true;

    for (const entry of this._queue) {
      if (entry.played || entry.sourceNode) continue; // skip completed or already-scheduled

      const remainingDuration = entry.duration - entry.startOffset;
      if (remainingDuration <= 0) {
        entry.played = true;
        this._releaseEntryBytes(entry);
        continue;
      }

      const source = ctx.createBufferSource();
      source.buffer = entry.buffer;
      source.connect(ctx.destination);

      const startTime = Math.max(this._nextStartTime, ctx.currentTime);
      source.start(startTime, entry.startOffset);

      entry.sourceNode = source;
      entry.startTime  = startTime;

      this._nextStartTime = startTime + remainingDuration;

      if (isFirstChunk) {
        if (this.onStateChange) this.onStateChange(true);
        isFirstChunk = false;
      }

      const capturedTurnId = entry.turnId;
      source.onended = () => {
        if (this._generation !== gen) return;
        entry.played     = true;
        entry.sourceNode = null;
        this._releaseEntryBytes(entry);
        this._onChunkEnded(capturedTurnId, gen);
      };
    }

    // Clean up entries that became zero-duration during offset update
    this._queue = this._queue.filter(e => !e.played);
    this._checkFlushedTurnsCompletion();
  }

  /**
   * Signal that all server audio chunks have been delivered for a turn.
   * The actual completion fires via onFinished when the last source ends.
   *
   * Calling markAudioEnd with the same turnId twice is a no-op.
   *
   * @param {string} turnId
   */
  markAudioEnd(turnId) {
    if (this._flushedTurns.has(turnId)) return;
    this._flushedTurns.add(turnId);

    // If there are no queued chunks for this turn, it's an empty turn — finalize immediately
    const hasChunks = this._queue.some(e => e.turnId === turnId && !e.played);
    if (!hasChunks) {
      this._finalizeTurn(turnId);
    }
  }

  /**
   * Cancel all scheduled and playing audio immediately.
   * Detaches callbacks BEFORE closing context to prevent stale onFinished calls.
   *
   * @param {number} [expectedGeneration] — if provided, only clears if generation matches
   */
  clear(expectedGeneration) {
    if (expectedGeneration !== undefined && expectedGeneration !== this._generation) {
      console.warn(`[Player] clear() generation mismatch: expected=${expectedGeneration} current=${this._generation}`);
      return;
    }

    // Increment generation first — all pending callbacks will now mismatch and drop
    this._generation++;

    // Stop all active sources
    for (const entry of this._queue) {
      if (entry.sourceNode) {
        entry.sourceNode.onended = null;
        try { entry.sourceNode.stop(); } catch (_) {}
        entry.sourceNode = null;
      }
    }

    this._queue         = [];
    this._queuedBytes   = 0;
    this._paused        = false;
    this._flushedTurns.clear();
    this._finishedTurns.clear();
    this._turnOrder     = [];

    // Close context (frees audio hardware)
    if (this._ctx) {
      this._ctx.close().catch(() => {});
      this._ctx = null;
    }

    this._nextStartTime = 0;
    this.audio_cleared++;

    if (this.onStateChange) this.onStateChange(false);
    console.log(`[Player] Cleared gen=${this._generation} cleared_count=${this.audio_cleared}`);
  }

  /**
   * Flush was the old API — redirect to avoid breaking callers during migration.
   * @deprecated Use markAudioEnd(turnId) instead.
   */
  flush() {
    // Legacy no-op: callers should migrate to markAudioEnd(turnId).
    console.warn('[Player] flush() is deprecated — use markAudioEnd(turnId)');
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  /**
   * Subtract an entry's decoded bytes from the running total.
   * @private
   * @param {QueueEntry} entry
   */
  _releaseEntryBytes(entry) {
    const bytes = entry.buffer.length * 4;
    this._queuedBytes = Math.max(0, this._queuedBytes - bytes);
  }

  /**
   * Called when any chunk's onended fires. Checks whether the turn is complete.
   * @private
   * @param {string} turnId
   * @param {number} capturedGen
   */
  _onChunkEnded(turnId, capturedGen) {
    if (this._generation !== capturedGen) return;

    // Remove played entries for this turn
    this._queue = this._queue.filter(e => !(e.turnId === turnId && e.played));

    // Check if any entries for this turn remain
    const remaining = this._queue.filter(e => e.turnId === turnId && !e.played);

    if (remaining.length === 0 && this._flushedTurns.has(turnId)) {
      this._finalizeTurn(turnId);
    }

    // Check if the entire queue is empty
    const anyActive = this._queue.some(e => !e.played);
    if (!anyActive && !this._paused) {
      if (this.onStateChange) this.onStateChange(false);
    }
  }

  /**
   * Check if any flushed turns have no remaining chunks and should finalize.
   * @private
   */
  _checkFlushedTurnsCompletion() {
    for (const turnId of this._flushedTurns) {
      if (this._finishedTurns.has(turnId)) continue;
      const remaining = this._queue.filter(e => e.turnId === turnId && !e.played);
      if (remaining.length === 0) {
        this._finalizeTurn(turnId);
      }
    }
  }

  /**
   * Fire the onFinished callback for a turn, exactly once.
   * @private
   * @param {string} turnId
   */
  _finalizeTurn(turnId) {
    if (this._finishedTurns.has(turnId)) return;
    this._finishedTurns.add(turnId);

    this.audio_finished++;
    console.log(`[Player] Turn finished: turnId=${turnId} finished_count=${this.audio_finished}`);

    if (this.onFinished) this.onFinished(turnId);
  }

  /**
   * @private
   * @returns {AudioContext}
   */
  _ensureCtx(sampleRate) {
    if (!this._ctx || this._ctx.state === 'closed' || this._ctxSampleRate !== sampleRate) {
      if (this._ctx && this._ctx.state !== 'closed') {
        this._ctx.close().catch(() => {});
      }
      this._ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
      this._ctxSampleRate  = sampleRate;
      this._nextStartTime  = 0;
    }
    if (this._ctx.state === 'suspended') {
      this._ctx.resume();
    }
    return this._ctx;
  }

  /**
   * @private
   * @param {string} base64data
   * @param {number} sampleRate
   * @param {number} channels
   * @returns {AudioBuffer}
   */
  _decodeChunk(base64data, sampleRate, channels) {
    const ctx = this._ensureCtx(sampleRate);

    // base64 → Uint8Array
    const binary = atob(base64data);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    // Int16 → Float32
    const int16     = new Int16Array(bytes.buffer);
    const numFrames = Math.floor(int16.length / channels);
    const float32   = new Float32Array(numFrames);
    for (let i = 0; i < numFrames; i++) {
      float32[i] = int16[i * channels] / 32768.0;
    }

    const audioBuffer = ctx.createBuffer(1, numFrames, sampleRate);
    audioBuffer.copyToChannel(float32, 0);
    return audioBuffer;
  }

  /**
   * Schedule a buffer for gapless playback. Captures the generation token
   * so that the onended callback is dropped if clear() was called.
   * @private
   * @param {string} turnId
   * @param {AudioBuffer} buffer
   * @param {number} capturedGen
   */
  _scheduleBuffer(turnId, buffer, capturedGen) {
    const ctx = this._ensureCtx(this._ctxSampleRate);

    // First chunk of the entire queue — notify listeners
    const isFirstGlobal = this._queue.filter(e => !e.played).length === 0;
    if (isFirstGlobal) {
      this.audio_started++;
      if (this.onStarted)     this.onStarted();
      if (this.onStateChange) this.onStateChange(true);
      this._nextStartTime = ctx.currentTime;
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const startTime     = Math.max(this._nextStartTime, ctx.currentTime);
    source.start(startTime);
    this._nextStartTime = startTime + buffer.duration;

    const entry = {
      turnId,
      buffer,
      sourceNode:  source,
      startTime,
      duration:    buffer.duration,
      startOffset: 0,
      played:      false,
    };
    this._queue.push(entry);

    source.onended = () => {
      // Generation check — if clear() was called, this fires with wrong gen
      if (this._generation !== capturedGen) return;

      entry.played     = true;
      entry.sourceNode = null;
      this._releaseEntryBytes(entry);
      this._onChunkEnded(turnId, capturedGen);
    };
  }
}
