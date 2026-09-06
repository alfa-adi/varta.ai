/**
 * app.js
 * ───────
 * Main application entry point — wires together all modules.
 *
 * Connection model (per speaker):
 *   One LiveWS per active session.speaker pair.
 *   Stored in connRecords[speaker]: { ws, turnId, turnState, events, t0, hadError }.
 *
 * Turn state machine (per speaker):
 *   IDLE → RECORDING → WAITING → PLAYING → IDLE
 *
 *   IDLE:      Button shows "Press to record". WS may or may not be open.
 *   RECORDING: Mic is live, PCM chunks flowing, turn_start sent.
 *   WAITING:   stop_recording sent; waiting for server transcript + audio.
 *   PLAYING:   audio_chunk frames arriving; TTS is playing.
 *
 * Key invariants:
 *   - audio_chunk is only forwarded to the player if the turn_id matches the
 *     current active turn. Stale chunks from a previous turn are dropped.
 *   - audio_end resets the state to IDLE and fires analytics.
 *   - turn_error / turn_cancelled reset state and show the error UI.
 *   - toggleRecord() guard: a second press while WAITING is silently ignored
 *     (prevents double-submit). A press while PLAYING clears the player.
 *   - __vartaTestHooks is exposed on window for browser automation tests.
 */

import './main.css';
import { createSession }       from './session.js';
import { LiveWS, ConnState }   from './wsClient.js';
import { Recorder }            from './recorder.js';
import { AudioPlayer }         from './player.js';
import { reportTurn, makeTurnEvents } from './analytics.js';
import * as UI                 from './ui.js';

// ── Turn states ───────────────────────────────────────────────────────────────

const TurnState = Object.freeze({
  IDLE:      'IDLE',
  STARTING:  'STARTING',
  RECORDING: 'RECORDING',
  WAITING:   'WAITING',
  PLAYING:   'PLAYING',
});

// ── Global session state ──────────────────────────────────────────────────────

let sessionId = null;
let sessionPromise = null;

// Conversation-level gate: only one speaker may own a turn at a time.
// This is deliberately separate from the per-speaker turn state because the
// two WebSockets can otherwise start independently and interleave audio.
let activeSpeaker = null;

/**
 * Per-speaker connection record.
 * @typedef {{ ws: LiveWS|null, turnId: string|null, turnState: string,
 *             events: ReturnType<typeof makeTurnEvents>|null,
 *             t0: number, hadError: boolean }} ConnRecord
 * @type {{ a: ConnRecord, b: ConnRecord }}
 */
const connRecords = {
  a: { ws: null, turnId: null, turnState: TurnState.IDLE, events: null, t0: 0, hadError: false },
  b: { ws: null, turnId: null, turnState: TurnState.IDLE, events: null, t0: 0, hadError: false },
};

/** @type {{ a: Recorder, b: Recorder }} */
const recorders = { a: new Recorder(), b: new Recorder() };

/** @type {{ a: AudioPlayer, b: AudioPlayer }} */
const players = { a: new AudioPlayer(), b: new AudioPlayer() };

// ── Player wiring ─────────────────────────────────────────────────────────────

for (const sp of ['a', 'b']) {
  const inputSp = sp === 'a' ? 'b' : 'a';
  players[sp].onStateChange = (playing) => UI.setAudioPlaying(sp, playing);
  players[sp].onStarted     = ()        => {
    connRecords[inputSp].events?.stamp('audio_started');
  };
  players[sp].onFinished    = ()        => {
    connRecords[inputSp].events?.stamp('audio_finished');
    _finishTurn(inputSp, 'player_finished');
  };
}

// ── Recorder wiring ───────────────────────────────────────────────────────────

for (const sp of ['a', 'b']) {
  recorders[sp].onError = (code, message) => {
    console.error(`[App] Recorder error (${sp}): ${code} — ${message}`);
    connRecords[sp].hadError = true;
    _resetToIdle(sp);
    UI.showError(message || 'Recording error. Please try again.');
  };
  recorders[sp].onRateDetected = (rate) => {
    console.log(`[App] Mic rate (${sp}): ${rate} Hz`);
  };
}

// ── Session ───────────────────────────────────────────────────────────────────

async function ensureSession() {
  if (sessionId) return;
  if (sessionPromise) return sessionPromise;

  sessionPromise = (async () => {
    try {
      const data = await createSession('', '');
      sessionId = data.session_id;
      UI.showSessionBadge(sessionId);
      console.log('[App] Session created:', sessionId);
    } catch (e) {
      UI.showError('Failed to create session: ' + e.message);
      throw e;
    } finally {
      sessionPromise = null;
    }
  })();

  return sessionPromise;
}

function setActiveSpeaker(speaker) {
  activeSpeaker = speaker;
  UI.setSpeakerLock(speaker);
}

function releaseActiveSpeaker(speaker) {
  if (activeSpeaker !== speaker) return;
  activeSpeaker = null;
  UI.setSpeakerLock(null);
}

// ── WebSocket management ──────────────────────────────────────────────────────

async function ensureLiveWS(speaker) {
  const rec = connRecords[speaker];

  // Already open — nothing to do
  if (rec.ws && rec.ws.state === ConnState.OPEN) return;

  // Coalesce concurrent callers while this speaker's socket is connecting.
  if (rec.ws && rec.ws.state === ConnState.CONNECTING) {
    await rec.ws.open();
    return;
  }

  await ensureSession();

  // If stale, close before recreating
  if (rec.ws) {
    rec.ws.onMessage = null;
    rec.ws.onError   = null;
    rec.ws.onClose   = null;
    rec.ws.close();
    rec.ws = null;
  }

  const ws = new LiveWS(sessionId, speaker);

  ws.onMessage = (msg) => _handleServerMsg(speaker, msg);

  ws.onError = () => {
    connRecords[speaker].hadError = true;
    UI.showError('Connection error. Please try again.');
  };

  ws.onClose = (code, reason) => {
    console.log(`[App] WS closed (${speaker}): code=${code} reason="${reason}"`);

    // An old connection must not release the record or turn owned by a newer
    // connection for the same speaker.
    if (connRecords[speaker].ws !== ws) return;
    connRecords[speaker].ws = null;

    // If we were mid-turn, clean up state
    const state = connRecords[speaker].turnState;
    if (state !== TurnState.IDLE) {
      connRecords[speaker].hadError = true;
      _resetToIdle(speaker);
      if (code !== 1000 && code !== 1001) {
        UI.showError(code === 4409
          ? 'Duplicate connection. Only one tab can record per session.'
          : 'Connection lost. Please try again.'
        );
      }
    }
  };

  await ws.open();
  rec.ws = ws;
}

// ── Server message handler ────────────────────────────────────────────────────

function _handleServerMsg(speaker, msg) {
  const rec = connRecords[speaker];

  switch (msg.type) {

    case 'server_ready':
      console.log(`[App] server_ready (${speaker}):`, msg.protocol_version ?? 'v?');
      break;

    case 'transcript_partial': {
      // Accept even without turn_id (legacy grace period)
      const turnMatch = !msg.turn_id || msg.turn_id === rec.turnId;
      if (!turnMatch) break;
      UI.setLiveTranscript(speaker, msg.text ?? msg.transcript ?? '');
      rec.events?.stamp('first_partial');
      break;
    }

    case 'transcript_final': {
      const turnMatch = !msg.turn_id || msg.turn_id === rec.turnId;
      if (!turnMatch) break;
      const text = msg.text ?? msg.transcript ?? '';
      UI.clearLiveTranscript(speaker);
      UI.addBubble(speaker, text, msg.language_code || '');
      rec.events?.stamp('transcript_final');
      // Transition to WAITING if still RECORDING (server beat the stop signal)
      if (rec.turnState === TurnState.RECORDING) {
        rec.turnState = TurnState.WAITING;
      }
      break;
    }

    case 'language_detected': {
      const turnMatch = !msg.turn_id || msg.turn_id === rec.turnId;
      if (!turnMatch) break;
      UI.setLanguageLabel(speaker, msg.language_code || msg.language || '');
      break;
    }

    case 'audio_chunk': {
      // Strict turn_id guard — drop stale chunks
      if (msg.turn_id && msg.turn_id !== rec.turnId) {
        console.warn(`[App] Dropped stale audio_chunk turn=${msg.turn_id} current=${rec.turnId}`);
        break;
      }

      // TTS audio plays on the OUTPUT speaker's panel
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';

      if (rec.turnState === TurnState.WAITING) {
        rec.turnState = TurnState.PLAYING;
        UI.setSpinner(speaker, false);
        UI.setLabel(speaker, 'Speaking…');
        rec.events?.stamp('first_audio_chunk');
      }

      players[outputSpeaker].enqueue(msg.data, {
        sample_rate_hz: msg.sample_rate_hz,
        channels:       msg.channels,
        format:         msg.format,
      });
      break;
    }

    case 'audio_end': {
      if (msg.turn_id && msg.turn_id !== rec.turnId) break;
      rec.events?.stamp('audio_end_received');
      // audio_end means all server chunks are in the pipe.
      // Player's onFinished will fire when the last one plays out → _finishTurn.
      players[speaker === 'a' ? 'b' : 'a'].flush();
      // If no audio was sent (empty TTS), finishTurn immediately.
      if (rec.turnState !== TurnState.PLAYING) {
        _finishTurn(speaker, 'audio_end_no_audio');
      }
      break;
    }

    case 'turn_error': {
      if (msg.turn_id && msg.turn_id !== rec.turnId) break;
      console.error(`[App] turn_error (${speaker}): ${msg.code} — ${msg.message}`);
      rec.hadError = true;
      _resetToIdle(speaker);
      UI.showError(msg.message || 'A server error occurred. Please try again.');
      break;
    }

    case 'turn_cancelled': {
      if (msg.turn_id && msg.turn_id !== rec.turnId) break;
      console.warn(`[App] turn_cancelled (${speaker}): ${msg.reason}`);
      _resetToIdle(speaker);
      break;
    }

    // ── Legacy fallbacks (old server) ────────────────────────────────────
    case 'error':
      console.error(`[App] legacy error (${speaker}):`, msg.message);
      rec.hadError = true;
      _resetToIdle(speaker);
      UI.showError(msg.message || 'An error occurred.');
      break;

    default:
      console.log(`[App] Unknown message: ${msg.type}`, msg);
  }
}

// ── Turn lifecycle helpers ────────────────────────────────────────────────────

function _finishTurn(speaker, reason) {
  const rec = connRecords[speaker];

  // Idempotent — may be called from audio_end OR player.onFinished
  if (rec.turnState === TurnState.IDLE) return;

  rec.events?.stamp('turn_finished');
  const events = rec.events?.finish() ?? {};

  // Report analytics
  reportTurn({
    sessionId,
    turnId:        rec.turnId,
    inputSpeaker:  speaker,
    outputSpeaker: speaker === 'a' ? 'b' : 'a',
    uploadMs:      events.recording_stopped && events.turn_started
                     ? events.recording_stopped - events.turn_started : 0,
    serverWaitMs:  events.first_audio_chunk && events.recording_stopped
                     ? events.first_audio_chunk - events.recording_stopped : 0,
    parseMs:       0,
    audioDecodeMs: events.audio_started && events.first_audio_chunk
                     ? events.audio_started - events.first_audio_chunk : 0,
    totalMs:       events._duration ?? 0,
    browserClean:  !rec.hadError,
    events,
  });

  console.log(`[App] Turn finished (${speaker}): reason=${reason} dur=${events._duration ?? '?'}ms clean=${!rec.hadError}`);

  recorders[speaker].isRecording && recorders[speaker].stop(null, null);
  rec.turnId    = null;
  rec.turnState = TurnState.IDLE;
  rec.events    = null;
  releaseActiveSpeaker(speaker);

  UI.setRecordButton(speaker, false);
  UI.setLabel(speaker, 'Press to record');
  UI.setSpinner(speaker, false);
}

function _resetToIdle(speaker) {
  const rec = connRecords[speaker];
  if (recorders[speaker].isRecording) {
    recorders[speaker].stop(null, null);
  }
  const outputSpeaker = speaker === 'a' ? 'b' : 'a';
  players[outputSpeaker].clear();

  rec.turnId    = null;
  rec.turnState = TurnState.IDLE;
  rec.events    = null;
  releaseActiveSpeaker(speaker);

  UI.setRecordButton(speaker, false);
  UI.setLabel(speaker, 'Press to record');
  UI.setSpinner(speaker, false);
}

// ── Toggle record ─────────────────────────────────────────────────────────────

async function toggleRecord(speaker) {
  const rec = connRecords[speaker];

  // Initialize both AudioContexts now that we have a trusted user gesture.
  players.a.initContext();
  players.b.initContext();

  try {
    if (activeSpeaker && activeSpeaker !== speaker) {
      const activeLabel = activeSpeaker.toUpperCase();
      UI.showError(`Speaker ${activeLabel} is currently using the microphone.`);
      return;
    }

    if (rec.turnState === TurnState.WAITING) {
      // Mid-turn: ignore double-press (server is processing)
      console.log(`[App] Ignored toggleRecord in WAITING state (${speaker})`);
      return;
    }

    if (rec.turnState === TurnState.STARTING) {
      // Prevent a second click while the socket/microphone is being prepared.
      return;
    }

    if (rec.turnState === TurnState.PLAYING) {
      // User pressed while other speaker is playing — interrupt playback
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';
      players[outputSpeaker].clear();
      _resetToIdle(speaker);
      return;
    }

    if (rec.turnState === TurnState.RECORDING) {
      // ── STOP ────────────────────────────────────────────────────────────
      rec.events?.stamp('recording_stopped');

      // Stop mic (send no stop_recording yet — we send it via ws.stop below)
      await recorders[speaker].stop(null, null);

      // Send stop_recording with turn_id
      if (rec.ws && rec.turnId) {
        rec.ws.stop(rec.turnId);
      }

      rec.turnState = TurnState.WAITING;
      UI.setRecordButton(speaker, false);
      UI.setLabel(speaker, 'Processing…');
      UI.setSpinner(speaker, true);

    } else {
      // ── START ────────────────────────────────────────────────────────────
      setActiveSpeaker(speaker);
      rec.turnState = TurnState.STARTING;
      await ensureLiveWS(speaker);

      // Clear the other speaker's player before we begin
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';
      players[outputSpeaker].clear();

      // Assign a new turn ID
      const turnId = crypto.randomUUID ? crypto.randomUUID() : _uuid4();
      rec.turnId    = turnId;
      rec.hadError  = false;
      rec.events    = makeTurnEvents();
      rec.events.stamp('turn_started');

      // Send turn_start before the first audio chunk
      rec.ws.sendTurnStart(turnId, outputSpeaker);

      // Start microphone + worklet
      await recorders[speaker].start(rec.ws);

      rec.turnState = TurnState.RECORDING;
      rec.t0        = Date.now();

      UI.setRecordButton(speaker, true);
      UI.setLabel(speaker, 'Recording…');
      UI.setSpinner(speaker, false);
      UI.clearLiveTranscript(speaker);
    }

  } catch (e) {
    console.error('[App] toggleRecord error:', e);
    rec.hadError = true;
    _resetToIdle(speaker);
    UI.showError('Recording failed: ' + (e.message || 'Unknown error'));
  }
}

// ── Expose to HTML ────────────────────────────────────────────────────────────
window.toggleRecord = toggleRecord;

// ── Test hooks (for browser automation) ──────────────────────────────────────
/**
 * Exposed for Playwright / browser test suites.
 * Do not use in production application code.
 */
window.__vartaTestHooks = {
  /** Returns the current TurnState for the given speaker. */
  getTurnState: (sp) => connRecords[sp]?.turnState,
  /** Returns the active turn ID for the given speaker. */
  getTurnId:    (sp) => connRecords[sp]?.turnId,
  /** Returns the WebSocket state string for the given speaker. */
  getWSState:   (sp) => connRecords[sp]?.ws?.state ?? 'NONE',
  /** Returns player counters for the given speaker. */
  getPlayerCounters: (sp) => ({
    started:    players[sp]?.audio_started,
    finished:   players[sp]?.audio_finished,
    cleared:    players[sp]?.audio_cleared,
    decoded:    players[sp]?.decodedSampleCount,
  }),
  /** Returns recorder queue stats for the given speaker. */
  getRecorderStats: (sp) => ({
    depth:   recorders[sp]?.queueDepth,
    max:     recorders[sp]?.queueMax,
    dropped: recorders[sp]?.droppedChunks,
  }),
  /** Returns the current session ID. */
  getSessionId: () => sessionId,
  /** Returns the speaker currently holding the conversation microphone lock. */
  getActiveSpeaker: () => activeSpeaker,
  /** TurnState enum (for assertions in tests). */
  TurnState,
};

// ── Cleanup on unload ─────────────────────────────────────────────────────────
window.addEventListener('beforeunload', () => {
  recorders.a.destroy();
  recorders.b.destroy();
  connRecords.a.ws?.close();
  connRecords.b.ws?.close();
});

// ── Boot ──────────────────────────────────────────────────────────────────────
console.log('[Varta] App loaded');

// ── Internal utils ────────────────────────────────────────────────────────────

/** Fallback UUID v4 generator for browsers without crypto.randomUUID. */
function _uuid4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}
