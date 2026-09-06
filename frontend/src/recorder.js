/**
 * recorder.js
 * ────────────
 * Manages the browser microphone, AudioContext, and AudioWorklet pipeline.
 *
 * Audio spec: 16kHz, mono, 16-bit signed little-endian (pcm_s16le).
 * The worklet resamples from whatever rate the browser provides.
 *
 * Invariants after refactor:
 *   - start() and stop() are guarded: overlapping calls return without action.
 *   - Both methods return Promises that always settle.
 *   - The bounded send queue (maxDepth chunks) emits AUDIO_BACKPRESSURE if
 *     the queue stays above the high-water mark for more than 1 second.
 *   - Worklet rate errors (AUDIO_SAMPLE_RATE_UNSUPPORTED) are surfaced as
 *     rejected start() promises.
 *   - Mic track ending unexpectedly calls onError.
 */

/** Max queued (unsent) chunks before backpressure kicks in (≈ 800 ms). */
const SEND_QUEUE_MAX       = 40;
/** If queue stays > 50% full for this long (ms), emit backpressure error. */
const BACKPRESSURE_TIME_MS = 1_000;

export class Recorder {
  constructor() {
    /** @type {MediaStream|null} Reused across turns to avoid mobile mic resets */
    this._stream         = null;
    /** @type {AudioContext|null} Reused across turns */
    this._audioCtx       = null;
    /** @type {AudioWorkletNode|null} */
    this._worklet        = null;
    /** @type {MediaStreamAudioSourceNode|null} */
    this._source         = null;
    /** @type {boolean} */
    this._workletLoaded  = false;

    // ── State flags ──────────────────────────────────────────────────────
    /** True while recording is active. */
    this.isRecording     = false;
    /** True while start() is in progress (mic acquisition, worklet load). */
    this._starting       = false;
    /** True while stop() is in progress. */
    this._stopping       = false;

    // ── Bounded send queue metrics ────────────────────────────────────────
    /** Current number of pending (unsent) chunks. */
    this._queueDepth     = 0;
    /** Peak queue depth seen in this turn. */
    this._queueMax       = 0;
    /** Number of chunks dropped due to queue overflow. */
    this._droppedChunks  = 0;
    /** Timestamp (performance.now) when queue first exceeded HWM; null if below. */
    this._hwmSince       = null;

    /**
     * Called when a non-fatal recorder error occurs.
     * @type {function(string, string): void}  (code, message)
     */
    this.onError = null;

    /**
     * Called when the worklet reports its actual input sample rate.
     * @type {function(number): void}
     */
    this.onRateDetected = null;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Start recording. Sends PCM chunks via liveWS.sendChunk().
   * @param {import('./wsClient.js').LiveWS} liveWS
   * @returns {Promise<void>} Resolves when recording starts; rejects on mic/worklet error.
   */
  async start(liveWS) {
    if (this.isRecording || this._starting) return;
    this._starting = true;

    // Reset queue metrics for this turn
    this._queueDepth    = 0;
    this._queueMax      = 0;
    this._droppedChunks = 0;
    this._hwmSince      = null;

    try {
      // ── Acquire mic (once, reused across turns) ─────────────────────────
      if (!this._stream || !this._stream.active) {
        this._stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount:     1,
            sampleRate:       16_000,  // hint; browser may not honor
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl:  true,
          },
        });

        // Surface mic track ending unexpectedly (e.g. device unplugged)
        this._stream.getAudioTracks()[0].addEventListener('ended', () => {
          if (this.isRecording) {
            const msg = 'Microphone track ended unexpectedly';
            console.error('[Recorder]', msg);
            this._teardownWorklet();
            this.isRecording = false;
            if (this.onError) this.onError('MIC_TRACK_ENDED', msg);
          }
        });
      }

      // ── Create AudioContext (once, reused) ──────────────────────────────
      if (!this._audioCtx) {
        this._audioCtx = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: 16_000,  // hint; browser may run at a different rate
        });
      }

      if (this._audioCtx.state === 'suspended') {
        await this._audioCtx.resume();
      }

      // ── Load worklet (once) ─────────────────────────────────────────────
      if (!this._workletLoaded) {
        await this._audioCtx.audioWorklet.addModule('/static/worklet/pcm-processor.js');
        this._workletLoaded = true;
      }

      // ── Build audio graph for this turn ────────────────────────────────
      this._source  = this._audioCtx.createMediaStreamSource(this._stream);
      this._worklet = new AudioWorkletNode(this._audioCtx, 'pcm-processor');

      // ── Handle messages from worklet ────────────────────────────────────
      this._worklet.port.onmessage = (e) => {
        const data = e.data;

        // Rate report: informational + pass to app layer
        if (data && data.type === 'rate') {
          console.log(`[Recorder] Worklet rate: input=${data.sampleRate} target=${data.targetRate}`);
          if (this.onRateDetected) this.onRateDetected(data.sampleRate);
          return;
        }

        // Worklet error (e.g. AUDIO_SAMPLE_RATE_UNSUPPORTED)
        if (data && data.type === 'error') {
          const code = data.code || 'WORKLET_ERROR';
          const msg  = `Worklet error: ${code} (rate=${data.rate ?? 'unknown'})`;
          console.error('[Recorder]', msg);
          if (this.onError) this.onError(code, msg);
          return;
        }

        // PCM chunk (ArrayBuffer) — apply queue backpressure
        if (data instanceof ArrayBuffer) {
          if (this._queueDepth >= SEND_QUEUE_MAX) {
            this._droppedChunks++;
            this._hwmSince = this._hwmSince ?? performance.now();
            const overMs = performance.now() - this._hwmSince;
            if (overMs >= BACKPRESSURE_TIME_MS) {
              const msg = `Audio backpressure: queue full for ${overMs.toFixed(0)}ms, dropped=${this._droppedChunks}`;
              console.error('[Recorder]', msg);
              if (this.onError) this.onError('AUDIO_BACKPRESSURE', msg);
            }
            return;
          }

          this._hwmSince = null;
          this._queueDepth++;
          if (this._queueDepth > this._queueMax) this._queueMax = this._queueDepth;

          liveWS.sendChunk(data);

          // Decrement queue depth after send (synchronous — WebSocket.send() is non-blocking)
          this._queueDepth = Math.max(0, this._queueDepth - 1);
        }
      };

      this._source.connect(this._worklet);
      // Connect to destination to keep the audio graph alive
      this._worklet.connect(this._audioCtx.destination);

      this.isRecording = true;
    } finally {
      this._starting = false;
    }
  }

  /**
   * Stop recording. Disconnects the worklet; no more PCM frames after this.
   * @param {import('./wsClient.js').LiveWS|null} liveWS — if provided, sends stop_recording
   * @param {string|null} turnId — current turn ID, sent in stop_recording
   * @returns {Promise<void>}
   */
  async stop(liveWS = null, turnId = null) {
    if (!this.isRecording || this._stopping) return;
    this._stopping = true;

    try {
      this.isRecording = false;
      this._teardownWorklet();

      if (liveWS && turnId) {
        liveWS.stop(turnId);
      }

      console.log(`[Recorder] Stopped. queue_max=${this._queueMax} dropped=${this._droppedChunks}`);
    } finally {
      this._stopping = false;
    }
  }

  /**
   * Full cleanup — release mic hardware. Call on page unload.
   */
  destroy() {
    this.isRecording = false;
    this._teardownWorklet();

    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
    if (this._audioCtx) {
      this._audioCtx.close().catch(() => {});
      this._audioCtx      = null;
      this._workletLoaded = false;
    }
  }

  // ── Queue metrics (read-only) ─────────────────────────────────────────────

  get queueDepth()   { return this._queueDepth;   }
  get queueMax()     { return this._queueMax;      }
  get droppedChunks(){ return this._droppedChunks; }

  // ── Private helpers ───────────────────────────────────────────────────────

  _teardownWorklet() {
    if (this._worklet) {
      this._worklet.port.onmessage = null;
      try { this._worklet.disconnect(); } catch (_) {}
      this._worklet = null;
    }
    if (this._source) {
      try { this._source.disconnect(); } catch (_) {}
      this._source = null;
    }
  }
}
