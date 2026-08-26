/**
 * analytics.js
 * ─────────────
 * Reports client-side timing data to the server for latency dashboards.
 *
 * Endpoint: POST /metrics/browser
 * All fields are integers (milliseconds).
 */

/**
 * Report a single turn's timing to the server.
 * Fire-and-forget — failures are silently ignored.
 *
 * @param {Object} data
 * @param {string} data.sessionId
 * @param {number} data.uploadMs       - time spent sending audio
 * @param {number} data.serverWaitMs   - time waiting for server response
 * @param {number} data.parseMs        - time to parse the response
 * @param {number} data.audioDecodeMs  - time to decode + start playing TTS
 * @param {number} data.totalMs        - total round-trip (button press → audio plays)
 */
export async function reportTurn(data) {
  try {
    const form = new FormData();
    form.append('session_id',      data.sessionId);
    form.append('upload_ms',       String(data.uploadMs       || 0));
    form.append('server_wait_ms',  String(data.serverWaitMs   || 0));
    form.append('parse_ms',        String(data.parseMs         || 0));
    form.append('audio_decode_ms', String(data.audioDecodeMs  || 0));
    form.append('total_ms',        String(data.totalMs         || 0));

    await fetch('/metrics/browser', { method: 'POST', body: form });
  } catch {
    // fire-and-forget — never block the UI
  }
}
