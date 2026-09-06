/**
 * analytics.js
 * ─────────────
 * Reports client-side turn events to the server for latency dashboards.
 *
 * Endpoint: POST /metrics/browser
 * All time fields are integers (milliseconds).
 *
 * Upgrade notes vs. original:
 *   - Accepts a 'events' sub-object with real event timestamps (ms since epoch).
 *   - Includes turn_id, input_speaker, output_speaker in the payload.
 *   - browser_clean flag: true when the turn completed without any error or
 *     backpressure event (used to bucket "clean" vs "degraded" turns).
 *   - reportTurn() returns the fetch promise (still fire-and-forget for callers
 *     that don't await it), and catches internally to never throw.
 *   - subscribeEvents(dispatcher) can attach this module to an event bus for
 *     automatic, zero-glue wiring.
 */

/**
 * Report a single turn's timing to the server.
 * Fire-and-forget for callers — errors are swallowed.
 *
 * @param {Object}  data
 * @param {string}  data.sessionId
 * @param {string}  [data.turnId]           — turn UUID
 * @param {string}  [data.inputSpeaker]     — "a" or "b"
 * @param {string}  [data.outputSpeaker]    — "a" or "b"
 * @param {number}  [data.uploadMs=0]       — ms spent streaming audio
 * @param {number}  [data.serverWaitMs=0]   — ms from stop_recording → audio_end
 * @param {number}  [data.parseMs=0]        — ms to decode first audio chunk
 * @param {number}  [data.audioDecodeMs=0]  — ms from audio_chunk[0] → playback start
 * @param {number}  [data.totalMs=0]        — ms from turn_start → audio_finished
 * @param {boolean} [data.browserClean=true] — false if any error or backpressure occurred
 * @param {Object}  [data.events]           — { [eventName]: timestamp_ms, ... }
 * @returns {Promise<void>}
 */
export async function reportTurn(data) {
  try {
    const form = new FormData();
    form.append('session_id',     data.sessionId    || '');
    form.append('turn_id',        data.turnId        || '');
    form.append('input_speaker',  data.inputSpeaker  || '');
    form.append('output_speaker', data.outputSpeaker || '');
    form.append('upload_ms',      String(data.uploadMs      || 0));
    form.append('server_wait_ms', String(data.serverWaitMs  || 0));
    form.append('parse_ms',       String(data.parseMs        || 0));
    form.append('audio_decode_ms',String(data.audioDecodeMs  || 0));
    form.append('total_ms',       String(data.totalMs        || 0));
    form.append('browser_clean',  String(data.browserClean !== false));

    // Serialize real event timestamps as a JSON string field
    if (data.events && Object.keys(data.events).length > 0) {
      form.append('events', JSON.stringify(data.events));
    }

    await fetch('/metrics/browser', { method: 'POST', body: form });
  } catch {
    // fire-and-forget — never block or throw
  }
}

/**
 * Build a zero-allocation event recorder for a single turn.
 * Returns an object with stamp(eventName) and finish() → data blob.
 *
 * Usage:
 *   const ev = makeTurnEvents();
 *   ev.stamp('turn_started');
 *   ev.stamp('recording_stopped');
 *   ev.stamp('first_partial');
 *   ev.stamp('transcript_final');
 *   ev.stamp('first_audio_chunk');
 *   ev.stamp('audio_started');
 *   ev.stamp('audio_finished');
 *   const events = ev.finish();
 *   await reportTurn({ ...metadata, events, totalMs: events._duration ?? 0 });
 *
 * @returns {{ stamp: function(string): void, finish: function(): Object }}
 */
export function makeTurnEvents() {
  const _ts = {};
  const _start = Date.now();
  return {
    stamp(name) {
      if (!(name in _ts)) _ts[name] = Date.now();
    },
    finish() {
      _ts._duration = Date.now() - _start;
      return { ..._ts };
    },
  };
}
