/**
 * app.js
 * ───────
 * Main application entry point — wires together all modules.
 *
 * Connection model (per speaker):
 *   One LiveWS per active session.speaker pair.
 *   Stored in connRecords[speaker]: { ws, captureState, activeCaptureTurnId, turns }.
 *
 * State model (continuous-playback refactor):
 *   captureState tracks microphone recording only:
 *     IDLE → STARTING → RECORDING → WAITING → IDLE
 *
 *   Playback state is tracked independently per turn in the turns Map:
 *     turns.get(turnId) → TurnRecord { serverAudioEnded, playbackStarted,
 *                                       playbackPaused, playbackFinished, terminalEvent }
 *
 *   This separation allows a speaker to begin Turn 2 recording while Turn 1
 *   audio is still playing (paused during capture, resumed after).
 *
 * Key invariants:
 *   - audio_chunk is only forwarded to the player if the turn_id exists in
 *     the turns Map. Stale chunks for unknown turns are dropped.
 *   - audio_end records serverAudioEnded on the TurnRecord. The turn is NOT
 *     finalized until playback completes or fails.
 *   - captureState reaches IDLE when audio_end arrives (server pipeline done).
 *     Playback may still be active for that turn.
 *   - A maximum of 2 TurnRecords may exist per speaker at any time.
 *   - activeSpeaker lock is held for the entire chained sequence and released
 *     only when captureState === IDLE AND turns.size === 0.
 *   - __vartaTestHooks is exposed on window for browser automation tests.
 */

import './main.css';
import { createSession }       from './session.js';
import { LiveWS, ConnState }   from './wsClient.js';
import { Recorder }            from './recorder.js';
import { AudioPlayer }         from './player.js';
import { reportTurn, makeTurnEvents } from './analytics.js';
import * as UI                 from './ui.js';

// ── Turn states (capture only) ────────────────────────────────────────────────

const CaptureState = Object.freeze({
  IDLE:      'IDLE',
  STARTING:  'STARTING',
  RECORDING: 'RECORDING',
  WAITING:   'WAITING',
});

// ── Terminal event values ─────────────────────────────────────────────────────

const TerminalEvent = Object.freeze({
  AUDIO_END:               'audio_end',
  NO_SPEECH:               'no_speech',
  NMT_ERROR:               'nmt_error',
  TTS_ERROR:               'tts_error',
  CANCELLED:               'cancelled',
  WEBSOCKET_CLOSED:        'websocket_closed',
  PLAYBACK_ERROR:          'playback_error',
  PLAYBACK_QUEUE_OVERFLOW: 'playback_queue_overflow',
});

// ── Global session state ──────────────────────────────────────────────────────

let sessionId = null;
let sessionPromise = null;

// Conversation-level gate: only one speaker may own a turn at a time.
// This is deliberately separate from the per-speaker turn state because the
// two WebSockets can otherwise start independently and interleave audio.
let activeSpeaker = null;

/**
 * Per-turn record tracking playback and analytics independently.
 * @typedef {Object} TurnRecord
 * @property {string}  turnId
 * @property {ReturnType<typeof makeTurnEvents>} events
 * @property {boolean} serverAudioEnded
 * @property {boolean} playbackStarted
 * @property {boolean} playbackPaused
 * @property {boolean} playbackFinished
 * @property {string|null} terminalEvent
 */

/**
 * Per-speaker connection record.
 * captureState tracks microphone only; turns Map tracks playback per turn.
 * @typedef {{ ws: LiveWS|null, captureState: string,
 *             activeCaptureTurnId: string|null,
 *             turns: Map<string, TurnRecord>,
 *             hadError: boolean }} ConnRecord
 * @type {{ a: ConnRecord, b: ConnRecord }}
 */
const connRecords = {
  a: { ws: null, captureState: CaptureState.IDLE, activeCaptureTurnId: null, turns: new Map(), hadError: false },
  b: { ws: null, captureState: CaptureState.IDLE, activeCaptureTurnId: null, turns: new Map(), hadError: false },
};

/** @type {{ a: Recorder, b: Recorder }} */
const recorders = { a: new Recorder(), b: new Recorder() };

/** @type {{ a: AudioPlayer, b: AudioPlayer }} */
const players = { a: new AudioPlayer(), b: new AudioPlayer() };

// ── Player wiring ─────────────────────────────────────────────────────────────

for (const sp of ['a', 'b']) {
  const inputSp = sp === 'a' ? 'b' : 'a';

  players[sp].onStateChange = (playing) => UI.setAudioPlaying(sp, playing);

  players[sp].onStarted = () => {
    // Find the earliest non-finished turn for this input speaker
    for (const [, turn] of connRecords[inputSp].turns) {
      if (!turn.playbackStarted) {
        turn.playbackStarted = true;
        turn.events?.stamp('audio_started');
        break;
      }
    }
  };

  players[sp].onFinished = (turnId) => {
    const turn = connRecords[inputSp].turns.get(turnId);
    if (turn) {
      turn.playbackFinished = true;
      turn.events?.stamp('audio_finished');
      _finishTurn(inputSp, turnId, 'player_finished');
    }

    // After this turn finishes, check if there's a paused queue to resume
    _maybeResumePlayback(inputSp);
  };

  players[sp].onOverflow = (turnId) => {
    const turn = connRecords[inputSp].turns.get(turnId);
    if (turn) {
      turn.terminalEvent = TerminalEvent.PLAYBACK_QUEUE_OVERFLOW;
      turn.events?.stamp('playback_queue_overflow');
    }
    console.error(`[App] Playback queue overflow for turn=${turnId}`);
    // Resume existing paused Turn 1 audio if possible
    const outputSpeaker = inputSp === 'a' ? 'b' : 'a';
    players[outputSpeaker].resumeQueue();
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

/**
 * Check whether the speaker lock can be safely released.
 * The lock is only released when captureState is IDLE AND no turns remain.
 */
function _maybeReleaseLock(speaker) {
  const rec = connRecords[speaker];
  if (rec.captureState === CaptureState.IDLE && rec.turns.size === 0) {
    releaseActiveSpeaker(speaker);
    UI.setRecordButton(speaker, false);
    UI.setLabel(speaker, 'Press to record');
    UI.setSpinner(speaker, false);
  }
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

    // If we were mid-capture, clean up capture state
    const captureState = connRecords[speaker].captureState;
    if (captureState !== CaptureState.IDLE) {
      connRecords[speaker].hadError = true;

      // Mark active capture turn as websocket_closed
      const activeTurnId = connRecords[speaker].activeCaptureTurnId;
      if (activeTurnId) {
        const turn = connRecords[speaker].turns.get(activeTurnId);
        if (turn && !turn.terminalEvent) {
          turn.terminalEvent = TerminalEvent.WEBSOCKET_CLOSED;
          turn.events?.stamp('websocket_closed');
        }
      }

      // Reset capture state but let buffered playback finish
      _resetCaptureToIdle(speaker);

      // Resume any paused playback so Turn 1 audio plays out locally
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';
      players[outputSpeaker].resumeQueue();

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
      const turnMatch = !msg.turn_id || msg.turn_id === rec.activeCaptureTurnId;
      if (!turnMatch) break;
      UI.setLiveTranscript(speaker, msg.text ?? msg.transcript ?? '');
      const turn = rec.turns.get(msg.turn_id || rec.activeCaptureTurnId);
      turn?.events?.stamp('first_partial');
      break;
    }

    case 'transcript_final': {
      const turnId = msg.turn_id || rec.activeCaptureTurnId;
      const turn = rec.turns.get(turnId);
      if (!turn) break;
      const text = msg.text ?? msg.transcript ?? '';
      UI.clearLiveTranscript(speaker);
      UI.addBubble(speaker, text, msg.language_code || '');
      turn.events?.stamp('transcript_final');
      // Transition to WAITING if still RECORDING (server beat the stop signal)
      if (rec.captureState === CaptureState.RECORDING) {
        rec.captureState = CaptureState.WAITING;
      }
      break;
    }

    case 'language_detected': {
      const turnId = msg.turn_id || rec.activeCaptureTurnId;
      const turn = rec.turns.get(turnId);
      if (!turn) break;
      UI.setLanguageLabel(speaker, msg.language_code || msg.language || '');
      break;
    }

    case 'audio_chunk': {
      const turnId = msg.turn_id;
      // Strict turn_id guard — drop chunks for unknown turns
      if (!turnId || !rec.turns.has(turnId)) {
        console.warn(`[App] Dropped stale audio_chunk turn=${turnId} (not in turns map)`);
        break;
      }

      const turn = rec.turns.get(turnId);

      // TTS audio plays on the OUTPUT speaker's panel
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';

      if (rec.captureState === CaptureState.WAITING && turnId === rec.activeCaptureTurnId) {
        UI.setSpinner(speaker, false);
        UI.setLabel(speaker, 'Speaking…');
        turn.events?.stamp('first_audio_chunk');
      }

      players[outputSpeaker].enqueue(turnId, msg.data, {
        sample_rate_hz: msg.sample_rate_hz,
        channels:       msg.channels,
        format:         msg.format,
      });
      break;
    }

    case 'audio_end': {
      const turnId = msg.turn_id;
      const turn = rec.turns.get(turnId);
      if (!turn) break;

      turn.serverAudioEnded = true;
      turn.terminalEvent = TerminalEvent.AUDIO_END;
      turn.events?.stamp('audio_end_received');

      // Signal the player that no more chunks will arrive for this turn
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';
      players[outputSpeaker].markAudioEnd(turnId);

      // Release capture state — the server pipeline is done.
      // This allows a new turn to start even if playback is still active.
      if (rec.activeCaptureTurnId === turnId) {
        rec.captureState = CaptureState.IDLE;
        rec.activeCaptureTurnId = null;
        UI.setSpinner(speaker, false);
      }

      // If no audio was sent (empty TTS), finishTurn immediately.
      if (!turn.playbackStarted) {
        _finishTurn(speaker, turnId, 'audio_end_no_audio');
      }
      break;
    }

    case 'turn_error': {
      const turnId = msg.turn_id || rec.activeCaptureTurnId;
      const turn = rec.turns.get(turnId);
      if (!turn) break;

      console.error(`[App] turn_error (${speaker}): ${msg.code} — ${msg.message}`);

      // Classify the terminal event
      const code = msg.code || '';
      if (code.includes('NMT'))       turn.terminalEvent = TerminalEvent.NMT_ERROR;
      else if (code.includes('TTS'))  turn.terminalEvent = TerminalEvent.TTS_ERROR;
      else if (code === 'FINAL_TRANSCRIPT_TIMEOUT') turn.terminalEvent = TerminalEvent.NO_SPEECH;
      else                            turn.terminalEvent = TerminalEvent.CANCELLED;

      turn.events?.stamp('turn_error');
      rec.hadError = true;

      // If this was the active capture turn, reset capture
      if (rec.activeCaptureTurnId === turnId) {
        _resetCaptureToIdle(speaker);
      }

      // Resume any paused playback for the other turn
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';
      players[outputSpeaker].resumeQueue();

      // Finalize this specific turn
      _finishTurn(speaker, turnId, 'turn_error');

      UI.showError(msg.message || 'A server error occurred. Please try again.');
      break;
    }

    case 'turn_cancelled': {
      const turnId = msg.turn_id || rec.activeCaptureTurnId;
      const turn = rec.turns.get(turnId);
      if (!turn) break;

      console.warn(`[App] turn_cancelled (${speaker}): ${msg.reason}`);
      turn.terminalEvent = TerminalEvent.CANCELLED;
      turn.events?.stamp('turn_cancelled');

      if (rec.activeCaptureTurnId === turnId) {
        _resetCaptureToIdle(speaker);
      }

      // Resume paused playback
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';
      players[outputSpeaker].resumeQueue();

      _finishTurn(speaker, turnId, 'turn_cancelled');
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

/**
 * Create a new TurnRecord for the given speaker and turn ID.
 * @param {string} speaker
 * @param {string} turnId
 * @returns {TurnRecord}
 */
function _createTurnRecord(speaker, turnId) {
  const turn = {
    turnId,
    events:           makeTurnEvents(),
    serverAudioEnded: false,
    playbackStarted:  false,
    playbackPaused:   false,
    playbackFinished: false,
    terminalEvent:    null,
  };
  connRecords[speaker].turns.set(turnId, turn);
  return turn;
}

/**
 * Finalize a specific turn: emit analytics, remove the TurnRecord,
 * and check whether the speaker lock can be released.
 */
function _finishTurn(speaker, turnId, reason) {
  const rec = connRecords[speaker];
  const turn = rec.turns.get(turnId);

  if (!turn) return; // already cleaned up

  turn.events?.stamp('turn_finished');
  const events = turn.events?.finish() ?? {};

  // Report analytics
  reportTurn({
    sessionId,
    turnId:        turn.turnId,
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

  console.log(`[App] Turn finished (${speaker}): turnId=${turnId} reason=${reason} dur=${events._duration ?? '?'}ms terminal=${turn.terminalEvent}`);

  // Remove the turn record
  rec.turns.delete(turnId);

  // Check if the speaker lock can be released
  _maybeReleaseLock(speaker);
}

/**
 * Reset only the capture state to IDLE. Does NOT clear playback or turns Map.
 * Used when the server pipeline finishes or errors, but playback may continue.
 */
function _resetCaptureToIdle(speaker) {
  const rec = connRecords[speaker];
  if (recorders[speaker].isRecording) {
    recorders[speaker].stop(null, null);
  }
  rec.captureState = CaptureState.IDLE;
  rec.activeCaptureTurnId = null;

  UI.setRecordButton(speaker, false);
  UI.setLabel(speaker, rec.turns.size > 0 ? 'Playing…' : 'Press to record');
  UI.setSpinner(speaker, false);
}

/**
 * Full nuclear reset: stops recording, clears playback, removes all turns,
 * releases the speaker lock. Used only for catastrophic failures and legacy errors.
 */
function _resetToIdle(speaker) {
  const rec = connRecords[speaker];
  if (recorders[speaker].isRecording) {
    recorders[speaker].stop(null, null);
  }
  const outputSpeaker = speaker === 'a' ? 'b' : 'a';
  players[outputSpeaker].clear();

  rec.captureState = CaptureState.IDLE;
  rec.activeCaptureTurnId = null;
  rec.turns.clear();
  releaseActiveSpeaker(speaker);

  UI.setRecordButton(speaker, false);
  UI.setLabel(speaker, 'Press to record');
  UI.setSpinner(speaker, false);
}

/**
 * After a turn finishes playback, check if the next turn's audio should resume.
 */
function _maybeResumePlayback(speaker) {
  const rec = connRecords[speaker];
  const outputSpeaker = speaker === 'a' ? 'b' : 'a';

  // If capture is IDLE and there's still paused audio, resume it
  if (rec.captureState === CaptureState.IDLE) {
    players[outputSpeaker].resumeQueue();
  }
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

    if (rec.captureState === CaptureState.WAITING) {
      // Mid-turn: ignore double-press (server is processing)
      console.log(`[App] Ignored toggleRecord in WAITING state (${speaker})`);
      return;
    }

    if (rec.captureState === CaptureState.STARTING) {
      // Prevent a second click while the socket/microphone is being prepared.
      return;
    }

    if (rec.captureState === CaptureState.RECORDING) {
      // ── STOP ────────────────────────────────────────────────────────────
      const activeTurnId = rec.activeCaptureTurnId;
      const turn = rec.turns.get(activeTurnId);
      turn?.events?.stamp('recording_stopped');

      // Stop mic
      await recorders[speaker].stop(null, null);

      // Send stop_recording with turn_id
      if (rec.ws && activeTurnId) {
        rec.ws.stop(activeTurnId);
      }

      rec.captureState = CaptureState.WAITING;
      UI.setRecordButton(speaker, false);
      UI.setLabel(speaker, 'Processing…');
      UI.setSpinner(speaker, true);

    } else {
      // ── START ────────────────────────────────────────────────────────────
      // captureState must be IDLE to start a new turn
      if (rec.captureState !== CaptureState.IDLE) return;

      setActiveSpeaker(speaker);
      rec.captureState = CaptureState.STARTING;
      await ensureLiveWS(speaker);

      // Pause the other speaker's player if audio is currently playing
      // (Turn 1 audio pauses so Turn 2 can record)
      const outputSpeaker = speaker === 'a' ? 'b' : 'a';

      // Check if there's active playback for a previous turn
      const hasPreviousTurn = rec.turns.size > 0;
      if (hasPreviousTurn) {
        // Pause playback — do NOT clear it
        const previousTurnId = [...rec.turns.keys()][0];
        const previousTurn = rec.turns.get(previousTurnId);
        if (previousTurn && !previousTurn.playbackFinished) {
          previousTurn.playbackPaused = true;
          previousTurn.events?.stamp('playback_paused');
          players[outputSpeaker].pause(previousTurnId);
        }
      } else {
        // No previous turn — clear any stale audio
        players[outputSpeaker].clear();
      }

      // Assign a new turn ID
      const turnId = crypto.randomUUID ? crypto.randomUUID() : _uuid4();
      rec.activeCaptureTurnId = turnId;
      rec.hadError = false;

      const turn = _createTurnRecord(speaker, turnId);
      turn.events.stamp('turn_started');

      // Send turn_start before the first audio chunk
      rec.ws.sendTurnStart(turnId, outputSpeaker);

      // Start microphone + worklet
      await recorders[speaker].start(rec.ws);

      rec.captureState = CaptureState.RECORDING;

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
  /** Returns the current CaptureState for the given speaker. */
  getCaptureState: (sp) => connRecords[sp]?.captureState,
  /** Returns the active capture turn ID for the given speaker. */
  getActiveTurnId: (sp) => connRecords[sp]?.activeCaptureTurnId,
  /** Returns the entire turns Map for the given speaker (for inspection). */
  getTurns:        (sp) => connRecords[sp]?.turns,
  /** Returns the WebSocket state string for the given speaker. */
  getWSState:      (sp) => connRecords[sp]?.ws?.state ?? 'NONE',
  /** Returns player counters for the given speaker. */
  getPlayerCounters: (sp) => ({
    started:    players[sp]?.audio_started,
    finished:   players[sp]?.audio_finished,
    cleared:    players[sp]?.audio_cleared,
    decoded:    players[sp]?.decodedSampleCount,
    decodedByTurn: Object.fromEntries(players[sp]?.decodedSamplesByTurn ?? new Map()),
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
  /** CaptureState enum (for assertions in tests). */
  CaptureState,
  /** TerminalEvent enum (for assertions in tests). */
  TerminalEvent,
  // Legacy aliases for backward compat with existing tests
  /** @deprecated Use getCaptureState instead */
  getTurnState: (sp) => connRecords[sp]?.captureState,
  /** @deprecated Use getActiveTurnId instead */
  getTurnId:    (sp) => connRecords[sp]?.activeCaptureTurnId,
  /** @deprecated Use CaptureState instead */
  TurnState: CaptureState,
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
