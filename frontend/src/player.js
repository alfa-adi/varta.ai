/**
 * player.js
 * ──────────
 * Gapless TTS audio player for raw linear16 PCM chunks.
 *
 * The server sends base64-encoded raw PCM (16-bit signed, mono).
 * The label says "mp3" but the actual bytes from Sarvam's Bulbul v3 TTS
 * are raw linear16 PCM — NOT a valid MP3 container.
 *
 * We decode the base64 → Int16Array → Float32Array, then schedule
 * playback via Web Audio API AudioBufferSourceNode for gapless output.
 */

export class AudioPlayer {
  constructor() {
    /** @type {AudioContext|null} */
    this._ctx = null;

    /** @type {{ buffer: AudioBuffer, duration: number }[]} */
    this._queue = [];

    /** @type {boolean} */
    this._isPlaying = false;

    /** @type {number} Next scheduled start time (Web Audio clock) */
    this._nextStartTime = 0;

    /** @type {AudioBufferSourceNode|null} */
    this._currentSource = null;

    /** @type {number} Track how many chunks are currently scheduled/playing */
    this._scheduledCount = 0;

    /** @type {function(): void} Called when all audio finishes */
    this.onFinished = null;

    /** @type {function(boolean): void} Called when playing state changes */
    this.onStateChange = null;

    // TTS sample rate from Sarvam Bulbul v3 (configured as speech_sample_rate=24000)
    this._sampleRate = 24000;
  }

  /** @private */
  _ensureCtx() {
    if (!this._ctx) {
      this._ctx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: this._sampleRate,
      });
    }
    if (this._ctx.state === 'suspended') {
      this._ctx.resume();
    }
    return this._ctx;
  }

  /**
   * Decode a base64 PCM chunk into an AudioBuffer.
   * @private
   * @param {string} base64data - base64-encoded raw linear16 PCM bytes
   * @returns {AudioBuffer}
   */
  _decodeChunk(base64data) {
    const ctx = this._ensureCtx();

    // base64 → raw bytes
    const binaryStr = atob(base64data);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }

    // Raw bytes → Int16 samples (little-endian)
    const int16 = new Int16Array(bytes.buffer);
    const numSamples = int16.length;

    // Int16 → Float32 (Web Audio API format)
    const float32 = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
      float32[i] = int16[i] / 32768.0;
    }

    // Create AudioBuffer (mono, at TTS sample rate)
    const audioBuffer = ctx.createBuffer(1, numSamples, this._sampleRate);
    audioBuffer.copyToChannel(float32, 0);

    return audioBuffer;
  }

  /**
   * Add a base64-encoded PCM chunk to the queue and schedule playback.
   * @param {string} base64data - base64-encoded raw linear16 PCM bytes
   */
  enqueue(base64data) {
    try {
      const buffer = this._decodeChunk(base64data);
      this._scheduleBuffer(buffer);
    } catch (e) {
      console.error('[Player] Failed to decode chunk:', e);
    }
  }

  /**
   * Schedule an AudioBuffer for gapless playback.
   * @private
   * @param {AudioBuffer} buffer
   */
  _scheduleBuffer(buffer) {
    const ctx = this._ensureCtx();

    if (!this._isPlaying) {
      this._isPlaying = true;
      this._nextStartTime = ctx.currentTime;
      if (this.onStateChange) this.onStateChange(true);
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    // Schedule exactly after the previous chunk ends → gapless
    const startTime = Math.max(this._nextStartTime, ctx.currentTime);
    source.start(startTime);

    this._nextStartTime = startTime + buffer.duration;
    this._scheduledCount++;

    source.onended = () => {
      this._scheduledCount--;
      if (this._scheduledCount <= 0) {
        this._isPlaying = false;
        this._scheduledCount = 0;
        if (this.onStateChange) this.onStateChange(false);
        if (this.onFinished) this.onFinished();
      }
    };
  }

  /**
   * Flush — ensure any remaining scheduled audio plays out.
   * Called on audio_end.
   */
  flush() {
    // Nothing to do — all chunks are already scheduled via Web Audio API.
    // The onended callbacks will fire when playback completes.
  }

  /**
   * Stop all playback and clear the queue.
   */
  clear() {
    if (this._ctx) {
      // Closing the context stops all scheduled sources immediately
      this._ctx.close().catch(() => {});
      this._ctx = null;
    }
    this._queue = [];
    this._isPlaying = false;
    this._scheduledCount = 0;
    this._nextStartTime = 0;
    if (this.onStateChange) this.onStateChange(false);
  }
}
