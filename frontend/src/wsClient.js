/**
 * wsClient.js
 * ────────────
 * Manages a persistent WebSocket connection for one speaker.
 *
 * Protocol (client → server):
 *   Binary frames  — raw PCM (Int16, 16kHz, mono) audio chunks
 *   Text JSON      — { "type": "stop_recording" }
 *
 * Protocol (server → client):
 *   { "type": "transcript_partial", "transcript": str }
 *   { "type": "transcript_final",   "transcript": str }
 *   { "type": "language_detected",  "language": str, "speaker": str }
 *   { "type": "audio_chunk",  "data": base64, "format": "mp3" }
 *   { "type": "audio_end" }
 *   { "type": "error", "message": str }
 */

export class LiveWS {
  /**
   * @param {string} sessionId - UUID from /session/create
   * @param {string} speaker   - "a" or "b"
   */
  constructor(sessionId, speaker) {
    this.sessionId = sessionId;
    this.speaker   = speaker;
    this.ws        = null;
    this.isOpen    = false;

    /** @type {function(Object): void} */
    this.onMessage = null;

    /** @type {function(Event): void} */
    this.onError   = null;

    /** @type {function(): void} */
    this.onClose   = null;
  }

  /**
   * Open the WebSocket connection. Resolves when the connection is ready.
   * @returns {Promise<void>}
   */
  async open() {
    if (this.isOpen) return;

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url   = `${proto}://${location.host}/ws/asr/${this.sessionId}/${this.speaker}`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.isOpen = true;
      console.log(`[WS/${this.speaker}] Connected`);
    };

    this.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (this.onMessage) this.onMessage(msg);
      } catch (e) {
        console.error(`[WS/${this.speaker}] Failed to parse message:`, e);
      }
    };

    this.ws.onerror = (evt) => {
      console.error(`[WS/${this.speaker}] Error:`, evt);
      if (this.onError) this.onError(evt);
    };

    this.ws.onclose = () => {
      console.log(`[WS/${this.speaker}] Closed`);
      this.isOpen = false;
      this.ws     = null;
      if (this.onClose) this.onClose();
    };

    // Wait for the connection to actually open before resolving
    await new Promise((resolve, reject) => {
      if (this.ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      this.ws.addEventListener('open',  resolve, { once: true });
      this.ws.addEventListener('error', reject,  { once: true });
    });
  }

  /**
   * Send a raw PCM audio chunk as a binary WebSocket frame.
   * ⚠️ Must be raw ArrayBuffer — NOT base64, NOT JSON-wrapped.
   *    The server reads `msg["bytes"]` directly.
   * @param {ArrayBuffer} int16ArrayBuffer - Int16 PCM samples
   */
  sendChunk(int16ArrayBuffer) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(int16ArrayBuffer);
    }
  }

  /**
   * Signal end of recording — triggers ASR flush → NMT → TTS on the server.
   */
  stop() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'stop_recording' }));
    }
  }

  /**
   * Close the WebSocket connection entirely.
   */
  close() {
    if (this.ws) {
      this.ws.close();
      this.ws     = null;
      this.isOpen = false;
    }
  }
}
