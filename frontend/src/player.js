/**
 * player.js
 * ──────────
 * Gapless TTS audio player for raw linear16 PCM chunks.
 *
 * Key invariants after refactor:
 *   - enqueue(base64data, meta) reads sample_rate_hz, channels, and format
 *     from the message metadata instead of hard-coding 24 kHz.
 *   - A playbackGeneration token is captured by each source.onended callback;
 *     stale callbacks (from a cleared player) are silently dropped.
 *   - clear(generation?) cancels all scheduled sources via source.stop(),
 *     detaches their onended callbacks, and only THEN closes/resets the context.
 *   - onFinished fires when the last chunk completes AND the generation still matches.
 *   - Exposes audio_started, audio_finished, audio_cleared counters and
 *     decodedSampleCount for test hooks.
 */

/** Default sample rate when the server message has no sample_rate_hz field. */
const DEFAULT_SAMPLE_RATE = 24_000;

export class AudioPlayer {
  constructor() {
    /** @type {AudioContext|null} */
    this._ctx = null;

    /** Number of chunks currently scheduled or playing. */
    this._scheduledCount = 0;

    /** Monotonic integer — incremented on each clear() call. */
    this._generation = 0;

    /** Time (Web Audio clock) when the next chunk should start. */
    this._nextStartTime = 0;

    /** All active AudioBufferSourceNode instances (for cancellation). */
    this._activeSources = new Set();

    /** Resolved sample rate for the current AudioContext. */
    this._ctxSampleRate = DEFAULT_SAMPLE_RATE;

    // ── Observable counters ──────────────────────────────────────────────
    this.audio_started       = 0;   // first chunk played per session
    this.audio_finished      = 0;   // last chunk ended naturally
    this.audio_cleared       = 0;   // clear() calls
    this.decodedSampleCount  = 0;   // total Int16 samples decoded

    /** Called when playback starts (first chunk). @type {function(): void} */
    this.onStarted  = null;
    /** Called when all audio finishes naturally. @type {function(): void} */
    this.onFinished = null;
    /** @type {function(boolean): void} */
    this.onStateChange = null;
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
   * @param {string} base64data — base64-encoded raw linear16 PCM bytes
   * @param {Object} [meta]     — metadata from the audio_chunk server message
   * @param {number} [meta.sample_rate_hz=24000]
   * @param {number} [meta.channels=1]
   * @param {string} [meta.format='pcm_s16le']
   */
  enqueue(base64data, meta = {}) {
    const sampleRate = meta.sample_rate_hz || DEFAULT_SAMPLE_RATE;
    const channels   = meta.channels       || 1;

    if (meta.sample_rate_hz && meta.sample_rate_hz !== DEFAULT_SAMPLE_RATE) {
      console.warn(`[Player] Non-default sample rate ${meta.sample_rate_hz}Hz from server`);
    }

    try {
      const buffer = this._decodeChunk(base64data, sampleRate, channels);
      this.decodedSampleCount += buffer.length;
      this._scheduleBuffer(buffer, this._generation);
    } catch (e) {
      console.error('[Player] Failed to decode chunk:', e);
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

    // Stop all active sources (silences them immediately)
    for (const src of this._activeSources) {
      try {
        src.onended = null;   // detach before stop() to prevent spurious callbacks
        src.stop();
      } catch (_) {}
    }
    this._activeSources.clear();
    this._scheduledCount = 0;

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
   * Called on audio_end — all server chunks have been delivered.
   * Actual completion fires via onFinished when the last source ends.
   */
  flush() {
    // All chunks are already scheduled in the Web Audio timeline.
    // Nothing to do here; onended callbacks handle completion.
  }

  // ── Private helpers ───────────────────────────────────────────────────────

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
   * @param {AudioBuffer} buffer
   * @param {number} capturedGen
   */
  _scheduleBuffer(buffer, capturedGen) {
    const ctx = this._ensureCtx(this._ctxSampleRate);

    // First chunk — notify listeners
    if (this._scheduledCount === 0) {
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
    this._scheduledCount++;
    this._activeSources.add(source);

    source.onended = () => {
      // Generation check — if clear() was called, this fires with wrong gen
      if (this._generation !== capturedGen) return;

      this._activeSources.delete(source);
      this._scheduledCount = Math.max(0, this._scheduledCount - 1);

      if (this._scheduledCount <= 0) {
        this.audio_finished++;
        if (this.onStateChange) this.onStateChange(false);
        if (this.onFinished)    this.onFinished();
        console.log(`[Player] Playback finished gen=${capturedGen} finished_count=${this.audio_finished}`);
      }
    };
  }
}
