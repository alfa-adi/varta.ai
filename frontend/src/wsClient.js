/**
 * wsClient.js
 * ────────────
 * Manages a persistent WebSocket connection for one speaker.
 *
 * Connection state machine (per instance):
 *   IDLE → CONNECTING → OPEN → CLOSING → CLOSED
 *
 * Key invariants:
 *   - open() stores the socket BEFORE awaiting, so a second concurrent call
 *     coalesces on the same openPromise rather than creating a new socket.
 *   - Every event handler captures a generationId at registration time and
 *     exits silently if the current generation has changed.
 *   - close() is idempotent and will not null a newer socket.
 *   - stop() sends { type: "stop_recording", turn_id } — carries the active turn ID.
 *   - sendTurnStart() sends { type: "turn_start", turn_id, ... }.
 *
 * Wire protocol additions (v1):
 *   Browser → Server:
 *     { type: "turn_start",      turn_id, input_speaker, output_speaker, client_started_at }
 *     { type: "stop_recording",  turn_id, client_stopped_at }
 *   Server → Browser:
 *     { type: "server_ready",    protocol_version, session_id, ... }
 *     { type: "transcript_partial", turn_id, text, language_code }
 *     { type: "transcript_final",   turn_id, text, language_code, language_confidence }
 *     { type: "language_detected",  turn_id, language_code }
 *     { type: "audio_chunk",        turn_id, format, sample_rate_hz, channels, data }
 *     { type: "audio_end",          turn_id, reason, server_completed_at }
 *     { type: "turn_error",         turn_id, code, message, retryable }
 *     { type: "turn_cancelled",     turn_id, reason }
 */

/** @readonly */
export const ConnState = Object.freeze({
  IDLE:       'IDLE',
  CONNECTING: 'CONNECTING',
  OPEN:       'OPEN',
  CLOSING:    'CLOSING',
  CLOSED:     'CLOSED',
});

/** How long (ms) to wait for a socket to open before failing. */
const OPEN_TIMEOUT_MS = 10_000;

export class LiveWS {
  /**
   * @param {string} sessionId
   * @param {string} speaker  — "a" or "b"
   */
  constructor(sessionId, speaker) {
    this.sessionId = sessionId;
    this.speaker   = speaker;

    /** @type {WebSocket|null} */
    this._ws = null;

    /** Current connection state. */
    this._state = ConnState.IDLE;

    /**
     * Monotonically increasing integer — incremented every time a new native
     * WebSocket is constructed. Event handlers capture their generation and
     * silently exit if it no longer matches the instance's current generation.
     * @type {number}
     */
    this._generation = 0;

    /**
     * Shared promise returned to all callers while state === CONNECTING.
     * Resolved when the socket opens, rejected on timeout or error.
     * @type {Promise<void>|null}
     */
    this._openPromise = null;

    /** @type {number|null} setTimeout handle for the open timeout */
    this._openTimeoutId = null;

    /** @type {function(Object): void} */
    this.onMessage = null;
    /** @type {function(Event): void} */
    this.onError   = null;
    /** @type {function(number, string): void} close(code, reason) */
    this.onClose   = null;
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /** Read-only current connection state. */
  get state() { return this._state; }

  /** True only when state === OPEN. */
  get isOpen() { return this._state === ConnState.OPEN; }

  /**
   * Open the WebSocket. Idempotent while CONNECTING or OPEN.
   * Returns the shared openPromise so concurrent calls coalesce.
   * @returns {Promise<void>}
   */
  async open() {
    if (this._state === ConnState.OPEN)       return;
    if (this._state === ConnState.CONNECTING) return this._openPromise;

    this._state = ConnState.CONNECTING;

    // Increment generation BEFORE constructing the socket so handlers
    // captured in _attachHandlers close over the correct generation.
    const gen = ++this._generation;

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url   = `${proto}://${location.host}/ws/asr/${this.sessionId}/${this.speaker}`;

    let resolveOpen, rejectOpen;
    this._openPromise = new Promise((res, rej) => {
      resolveOpen = res;
      rejectOpen  = rej;
    });

    // ── Open timeout ──────────────────────────────────────────────────────
    this._openTimeoutId = setTimeout(() => {
      if (this._generation !== gen) return;
      rejectOpen(new Error(`[WS/${this.speaker}] Open timed out after ${OPEN_TIMEOUT_MS}ms`));
      this._state = ConnState.CLOSED;
      this._openPromise = null;
      try { this._ws?.close(); } catch (_) {}
      this._ws = null;
    }, OPEN_TIMEOUT_MS);

    // ── Construct socket ──────────────────────────────────────────────────
    this._ws = new WebSocket(url);
    this._attachHandlers(gen, resolveOpen, rejectOpen);

    return this._openPromise;
  }

  /**
   * Send a raw PCM audio chunk as a binary WebSocket frame.
   * No-op unless state === OPEN.
   * @param {ArrayBuffer} int16ArrayBuffer
   */
  sendChunk(int16ArrayBuffer) {
    if (this._state === ConnState.OPEN && this._ws) {
      this._ws.send(int16ArrayBuffer);
    }
  }

  /**
   * Send turn_start — must be called before the first audio chunk.
   * @param {string} turnId
   * @param {string} outputSpeaker — the speaker who will HEAR the TTS reply
   */
  sendTurnStart(turnId, outputSpeaker) {
    if (this._state === ConnState.OPEN && this._ws) {
      this._ws.send(JSON.stringify({
        type:              'turn_start',
        turn_id:           turnId,
        input_speaker:     this.speaker,
        output_speaker:    outputSpeaker,
        client_started_at: Date.now(),
      }));
    }
  }

  /**
   * Signal end of recording — triggers ASR flush → NMT → TTS on the server.
   * @param {string} turnId
   */
  stop(turnId) {
    if (this._state === ConnState.OPEN && this._ws) {
      this._ws.send(JSON.stringify({
        type:             'stop_recording',
        turn_id:          turnId,
        client_stopped_at: Date.now(),
      }));
    }
  }

  /**
   * Close the WebSocket. Idempotent.
   * @param {number} [code=1000]
   * @param {string} [reason='']
   */
  close(code = 1000, reason = '') {
    if (this._state === ConnState.CLOSING || this._state === ConnState.CLOSED) return;
    if (this._state === ConnState.IDLE) {
      this._state = ConnState.CLOSED;
      return;
    }

    const gen = this._generation;
    this._state = ConnState.CLOSING;
    clearTimeout(this._openTimeoutId);
    this._openTimeoutId = null;

    if (this._ws) {
      try { this._ws.close(code, reason); } catch (_) {}
      // onclose handler will transition to CLOSED when it fires.
    }
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  _attachHandlers(gen, resolveOpen, rejectOpen) {
    const ws = this._ws;

    ws.onopen = () => {
      if (this._generation !== gen) { ws.close(); return; }
      clearTimeout(this._openTimeoutId);
      this._openTimeoutId = null;
      this._state = ConnState.OPEN;
      resolveOpen();
      this._openPromise = null;
      console.log(`[WS/${this.speaker}] Connected (gen=${gen})`);
    };

    ws.onmessage = (evt) => {
      if (this._generation !== gen) return;
      try {
        const msg = JSON.parse(evt.data);
        if (this.onMessage) this.onMessage(msg);
      } catch (e) {
        console.error(`[WS/${this.speaker}] Parse error:`, e);
      }
    };

    ws.onerror = (evt) => {
      if (this._generation !== gen) return;
      console.error(`[WS/${this.speaker}] Error (gen=${gen}):`, evt);
      rejectOpen(new Error(`WebSocket error for speaker ${this.speaker}`));
      this._openPromise = null;
      if (this.onError) this.onError(evt);
    };

    ws.onclose = (evt) => {
      if (this._generation !== gen) return;
      // Only null the socket if it's still ours
      if (this._ws === ws) this._ws = null;
      this._state = ConnState.CLOSED;
      this._openPromise = null;
      clearTimeout(this._openTimeoutId);
      this._openTimeoutId = null;
      console.log(`[WS/${this.speaker}] Closed code=${evt.code} reason="${evt.reason}" gen=${gen}`);
      if (this.onClose) this.onClose(evt.code, evt.reason);
    };
  }
}
