/**
 * app.js
 * ───────
 * Main application entry point — wires together all modules.
 *
 * Lifecycle:
 *   1. On load → createSession() → store sessionId
 *   2. toggleRecord(speaker):
 *      a. ensureLiveWS(speaker) — creates LiveWS, registers handleServerMsg
 *      b. if recording → stop + send stop_recording
 *      c. if not → start recording + update UI
 *   3. handleServerMsg(speaker, msg):
 *      Dispatches server events to the correct UI + player actions
 */

import './main.css';
import { createSession }     from './session.js';
import { LiveWS }            from './wsClient.js';
import { Recorder }          from './recorder.js';
import { AudioPlayer }       from './player.js';
import { reportTurn }        from './analytics.js';
import * as UI               from './ui.js';

// ── State ────────────────────────────────────────────────────────────────────

let sessionId = null;

/** @type {{ a: LiveWS|null, b: LiveWS|null }} */
const wsClients = { a: null, b: null };

/** @type {{ a: Recorder, b: Recorder }} */
const recorders = { a: new Recorder(), b: new Recorder() };

/** @type {{ a: AudioPlayer, b: AudioPlayer }} */
const players = { a: new AudioPlayer(), b: new AudioPlayer() };

/** Per-turn timing for analytics */
const turnTimers = { a: 0, b: 0 };

// Wire up player state change → UI indicator
players.a.onStateChange = (playing) => UI.setAudioPlaying('a', playing);
players.b.onStateChange = (playing) => UI.setAudioPlaying('b', playing);

// ── Session ──────────────────────────────────────────────────────────────────

async function ensureSession() {
  if (sessionId) return;
  try {
    const data = await createSession('', '');
    sessionId = data.session_id;
    UI.showSessionBadge(sessionId);
    console.log('[App] Session created:', sessionId);
  } catch (e) {
    UI.showError('Failed to create session: ' + e.message);
    throw e;
  }
}

// ── WebSocket ────────────────────────────────────────────────────────────────

async function ensureLiveWS(speaker) {
  if (wsClients[speaker] && wsClients[speaker].isOpen) return;

  await ensureSession();

  const ws = new LiveWS(sessionId, speaker);

  ws.onMessage = (msg) => handleServerMsg(speaker, msg);

  ws.onError = () => {
    UI.showError('Connection error. Please try again.');
  };

  ws.onClose = () => {
    wsClients[speaker] = null;

    // If we were recording when the WS died, clean up
    if (recorders[speaker].isRecording) {
      recorders[speaker].stop(null, false);
      UI.setRecordButton(speaker, false);
      UI.setLabel(speaker, 'Press to record');
      UI.setSpinner(speaker, false);
      UI.showError('Connection lost. Please try again.');
    }
  };

  await ws.open();
  wsClients[speaker] = ws;
}

// ── Server message handler ───────────────────────────────────────────────────

function handleServerMsg(speaker, msg) {
  switch (msg.type) {

    case 'transcript_partial':
      UI.setLiveTranscript(speaker, msg.transcript);
      break;

    case 'transcript_final':
      UI.clearLiveTranscript(speaker);
      UI.addBubble(speaker, msg.transcript, msg.language || '');
      break;

    case 'language_detected':
      UI.setLanguageLabel(msg.speaker || speaker, msg.language);
      break;

    case 'audio_chunk': {
      // TTS audio plays in the OTHER speaker's panel
      const otherSpeaker = speaker === 'a' ? 'b' : 'a';
      players[otherSpeaker].enqueue(msg.data);
      break;
    }

    case 'audio_end':
      // ⚠️ Critical: force-stop the microphone to prevent hallucination loop
      recorders[speaker].stop(null, false);
      UI.setRecordButton(speaker, false);
      UI.setLabel(speaker, 'Press to record');
      UI.setSpinner(speaker, false);

      // Flush any remaining TTS audio
      players[speaker === 'a' ? 'b' : 'a'].flush();

      // Report analytics
      if (turnTimers[speaker]) {
        reportTurn({
          sessionId,
          uploadMs:      0,
          serverWaitMs:  Date.now() - turnTimers[speaker],
          parseMs:       0,
          audioDecodeMs: 0,
          totalMs:       Date.now() - turnTimers[speaker],
        });
        turnTimers[speaker] = 0;
      }
      break;

    case 'error':
      UI.showError(msg.message || 'An error occurred.');
      recorders[speaker].stop(null, false);
      UI.setRecordButton(speaker, false);
      UI.setLabel(speaker, 'Press to record');
      UI.setSpinner(speaker, false);
      break;

    default:
      console.log(`[App] Unknown message type: ${msg.type}`, msg);
  }
}

// ── Toggle Record ────────────────────────────────────────────────────────────

async function toggleRecord(speaker) {
  try {
    if (recorders[speaker].isRecording) {
      // ── STOP ─────────────────────────────────────────────────────────────
      recorders[speaker].stop(wsClients[speaker], true);
      UI.setRecordButton(speaker, false);
      UI.setLabel(speaker, 'Processing…');
      UI.setSpinner(speaker, true);
    } else {
      // ── START ────────────────────────────────────────────────────────────
      await ensureLiveWS(speaker);

      // Stop the other speaker if they're playing audio
      const otherSpeaker = speaker === 'a' ? 'b' : 'a';
      players[otherSpeaker].clear();

      await recorders[speaker].start(wsClients[speaker]);

      UI.setRecordButton(speaker, true);
      UI.setLabel(speaker, 'Recording…');
      UI.setSpinner(speaker, false);
      UI.clearLiveTranscript(speaker);

      // Start timer for analytics
      turnTimers[speaker] = Date.now();
    }
  } catch (e) {
    console.error('[App] toggleRecord error:', e);
    recorders[speaker].stop(null, false);
    UI.setRecordButton(speaker, false);
    UI.setLabel(speaker, 'Press to record');
    UI.setSpinner(speaker, false);
    UI.showError('Recording failed: ' + e.message);
  }
}

// ── Expose to HTML ───────────────────────────────────────────────────────────
window.toggleRecord = toggleRecord;

// ── Cleanup on page unload ───────────────────────────────────────────────────
window.addEventListener('beforeunload', () => {
  recorders.a.destroy();
  recorders.b.destroy();
  if (wsClients.a) wsClients.a.close();
  if (wsClients.b) wsClients.b.close();
});

// ── Boot ─────────────────────────────────────────────────────────────────────
console.log('[Varta] App loaded');
