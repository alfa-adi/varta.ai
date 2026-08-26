/**
 * recorder.js
 * ────────────
 * Manages the browser microphone, AudioContext, and AudioWorklet pipeline.
 *
 * Audio spec: 16kHz, mono, 16-bit signed little-endian (pcm_s16le)
 * Worklet:    /worklet/pcm-processor.js (sends 20ms chunks = 640 bytes)
 *
 * Key design decisions:
 *   - MediaStream is acquired once and reused across turns (prevents mobile
 *     browsers from sending blank audio when getUserMedia is called repeatedly).
 *   - AudioContext is created once (reused, resumed if suspended).
 *   - Only the WorkletNode is connected/disconnected per turn.
 */

export class Recorder {
  constructor() {
    /** @type {MediaStream|null} */
    this._stream    = null;
    /** @type {AudioContext|null} */
    this._audioCtx  = null;
    /** @type {AudioWorkletNode|null} */
    this._worklet   = null;
    /** @type {MediaStreamAudioSourceNode|null} */
    this._source    = null;
    /** @type {boolean} */
    this.isRecording = false;
    /** @type {boolean} */
    this._workletLoaded = false;
  }

  /**
   * Start recording and streaming PCM chunks to the given LiveWS.
   * @param {import('./wsClient.js').LiveWS} liveWS
   */
  async start(liveWS) {
    if (this.isRecording) return;

    // ── Acquire mic (once, reused) ────────────────────────────────────────
    if (!this._stream) {
      this._stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount:          1,
          sampleRate:            16000,
          echoCancellation:      true,
          noiseSuppression:      true,
          autoGainControl:       true,
        },
      });
    }

    // ── Create AudioContext (once, reused) ─────────────────────────────────
    if (!this._audioCtx) {
      this._audioCtx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
      });
    }

    // Resume if suspended (browsers suspend AudioContext until user gesture)
    if (this._audioCtx.state === 'suspended') {
      await this._audioCtx.resume();
    }

    // ── Load worklet module (once) ────────────────────────────────────────
    if (!this._workletLoaded) {
      await this._audioCtx.audioWorklet.addModule('/static/worklet/pcm-processor.js');
      this._workletLoaded = true;
    }

    // ── Build audio graph for this turn ───────────────────────────────────
    this._source  = this._audioCtx.createMediaStreamSource(this._stream);
    this._worklet = new AudioWorkletNode(this._audioCtx, 'pcm-processor');

    // Each message from the worklet is an ArrayBuffer of Int16 PCM samples.
    // Send it directly as a binary WebSocket frame — the server expects raw bytes.
    this._worklet.port.onmessage = (e) => {
      liveWS.sendChunk(e.data);
    };

    this._source.connect(this._worklet);
    // Connect to destination to keep the audio graph alive (worklet won't
    // process if not connected to an output — browser optimisation).
    this._worklet.connect(this._audioCtx.destination);

    this.isRecording = true;
  }

  /**
   * Stop recording. Disconnects the worklet to prevent stale PCM frames.
   * Optionally sends the stop_recording signal to the server.
   *
   * @param {import('./wsClient.js').LiveWS|null} liveWS - if provided, calls liveWS.stop()
   * @param {boolean} sendStop - whether to send stop_recording to server (default true)
   */
  stop(liveWS = null, sendStop = true) {
    if (!this.isRecording) return;
    this.isRecording = false;

    // Cut the worklet pipeline — no more PCM frames after this
    if (this._worklet) {
      this._worklet.port.onmessage = null;
      this._worklet.disconnect();
      this._worklet = null;
    }
    if (this._source) {
      this._source.disconnect();
      this._source = null;
    }

    // Signal server to flush utterance and run NMT+TTS
    if (sendStop && liveWS) {
      liveWS.stop();
    }
  }

  /**
   * Full cleanup — called on page unload. Releases mic hardware.
   */
  destroy() {
    this.stop(null, false);

    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
    if (this._audioCtx) {
      this._audioCtx.close().catch(() => {});
      this._audioCtx  = null;
      this._workletLoaded = false;
    }
  }
}
