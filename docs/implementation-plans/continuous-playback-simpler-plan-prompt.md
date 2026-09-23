# Prompt: Create the Continuous Playback Implementation Plan

You are reviewing and planning changes for the Varta.ai realtime translation application.

**Repository:** `alfa-adi/varta.ai`  
**Branch:** `test-latency-tracking`  
**Task type:** Planning only. Do not modify source code, generated assets, tests, or deployment files.

Create a detailed implementation-plan Markdown file for the following feature.

## Feature requirement

When the same speaker begins Turn 2 while the translated audio from Turn 1 is still playing:

1. Pause Turn 1 audio immediately.
2. Record and process Turn 2.
3. Do not play translated audio while Turn 2 is being recorded or processed.
4. After Turn 2 reaches its server terminal state, resume only the unplayed remainder of Turn 1 audio.
5. After the Turn 1 remainder finishes, play all translated audio for Turn 2.
6. Never replay audio from Turn 1 that has already been heard.
7. Keep the other speaker locked out for the entire chained sequence.
8. Release the speaker lock only after Turn 2 and all queued playback have completed or failed safely.

The first version must remain deliberately conservative:

- Turn 2 may start only after Turn 1 has received the server-side `audio_end` event.
- Do not design concurrent ASR turns on the same upstream session.
- Do not allow Turn 1 and Turn 2 audio to interleave.
- Do not clear the player or destroy Turn 1’s remaining audio.
- Do not replace Vite, vanilla JavaScript, FastAPI, Render, or Sarvam.

## Required repository review

Before writing the plan, inspect the current branch and trace the complete lifecycle through these files:

- `frontend/src/app.js`
- `frontend/src/player.js`
- `frontend/src/recorder.js`
- `frontend/src/wsClient.js`
- `frontend/src/ui.js`
- `frontend/src/analytics.js`
- `frontend/public/worklet/pcm-processor.js`
- `web/server.py`
- `web/connection_manager.py` if present
- `web/protocol.py` if present
- `frontend/tests/`
- `tests/e2e/` if present
- `frontend/playwright.config.js` if present
- `docs/implementation-plans/01-realtime-stack-stabilization-implementation-plan.md`
- `docs/implementation-plans/02-browser-streaming-validation-implementation-plan.md`

Also inspect the current generated browser bundle and determine whether it matches the source files. Report any discrepancy.

Do not assume that an attached plan is correct. Treat attached documents as reference material and verify every claim against the current branch.

## Required plan structure

The generated implementation-plan Markdown file must contain the following sections.

### 1. Scope and safety boundary

State clearly:

- what this feature changes;
- what it intentionally does not change;
- why it is safer than fully concurrent ASR/NMT/TTS turns;
- the exact point at which Turn 2 is allowed to begin;
- whether the plan changes backend code, frontend code, tests, or generated assets.

If the current server cannot safely accept Turn 2 immediately after `audio_end`, identify the race and include the smallest required backend change.

### 2. Current-code understanding

Describe the current state machine and identify where capture state, server processing state, playback state, and speaker-lock state are currently combined.

Explicitly inspect and report:

- how `turn_id` is stored and validated;
- how `audio_end` is handled;
- how the output speaker is mapped back to the input turn owner;
- how `AudioPlayer` schedules PCM buffers;
- whether `AudioBufferSourceNode` objects can actually pause and resume;
- what `clear()` currently destroys;
- how player completion callbacks identify their owning turn;
- how the active-speaker lock is acquired and released;
- whether server turn ownership is released before or after `audio_end` is delivered.

Call out any existing playback-owner/turn-owner bug before proposing the new feature.

### 3. Proposed state model

Do not use one shared state variable to represent both capture and playback.

Propose an explicit model containing at least:

- `captureState`;
- `activeCaptureTurnId`;
- a per-turn record keyed by `turn_id`;
- `serverAudioEnded`;
- `playbackStarted`;
- `playbackPaused`;
- `playbackFinished`;
- `terminalEvent`;
- speaker-lock ownership;
- bounded queued audio state.

Explain which state transitions are legal and which are rejected.

The plan must preserve this invariant:

```text
At most one speaker captures or owns an active server turn,
but that speaker may have one previous turn draining in playback.
```

### 4. Player pause/resume design

Design a real pause/resume mechanism for PCM playback.

Do not claim that `AudioBufferSourceNode.stop()` can resume itself.

Explain how the implementation will:

- identify the currently playing buffer;
- calculate or retain the unplayed offset;
- stop future scheduled sources without invoking stale completion callbacks;
- preserve the remaining PCM for Turn 1;
- enqueue Turn 2 audio separately;
- resume Turn 1’s remainder before Turn 2 audio;
- prevent duplicate playback;
- handle audio chunks that arrive just before or after pause;
- handle an empty audio turn;
- handle malformed or unsupported audio metadata.

If the safest design is to retain decoded buffers and resume from an offset, state that explicitly. If the design instead uses smaller chunks and requeues unplayed chunks, document the precision tradeoff.

Define the proposed player API, including turn-aware methods such as:

```js
pause(turnId)
enqueue(turnId, base64data, metadata)
markAudioEnd(turnId)
resumeQueue()
onFinished(turnId)
```

The exact API may differ, but the plan must preserve turn identity and ordering.

### 5. Backend compatibility check

Verify whether backend changes are required.

The plan must address:

- `active_turn_id` ownership;
- server turn lease release timing;
- the race between sending `audio_end` and accepting the next `turn_start`;
- whether Turn 2 is rejected before Turn 1’s server pipeline finishes;
- one ASR reader and one upstream session per owner;
- terminal-event uniqueness;
- cleanup if Turn 1 or Turn 2 fails;
- single-speaker enforcement across tabs and workers.

Do not introduce concurrent ASR turns unless the plan explicitly proves that the provider session and adapter support them.

### 6. Analytics and traceability

Define separate timestamps and records for both turns.

At minimum, preserve:

- `turn_started`;
- `recording_stopped`;
- `server_audio_end_received`;
- `playback_paused`;
- `playback_resumed`;
- `audio_started`;
- `audio_finished`;
- `turn_finished`;
- error or cancellation events.

Explain how Turn 1 playback events cannot be attributed to Turn 2.

Do not report Turn 1 as complete merely because Turn 2 started recording. Turn completion must be defined explicitly, and the plan must distinguish server completion from browser playback completion.

### 7. Test plan

Create a browser-driven test matrix. Do not use Python endpoint tests as the browser release gate.

Include at least:

#### B27 — Pause Turn 1 for Turn 2 recording

- Start Turn 1.
- Stop Turn 1.
- Wait for `audio_end` while playback remains active.
- Start Turn 2.
- Assert playback pauses without clearing Turn 1.
- Assert no audio is audible during Turn 2 capture/processing.
- Assert Turn 1’s already-played audio is not replayed.

#### B28 — Ordered resume

- Complete Turn 2.
- Assert Turn 1’s unplayed remainder finishes first.
- Assert Turn 2 audio plays second.
- Assert both turns have separate IDs and completion records.

#### B29 — Playback completion during Turn 2

- Allow Turn 1’s remainder to finish while Turn 2 is recording or processing.
- Assert Turn 2 remains active and is not reset by Turn 1’s callback.

#### B30 — Failure and cleanup

- Inject a Turn 1 error, Turn 2 error, browser disconnect, and page unload.
- Assert the correct turn is finalized.
- Assert the player, recorder, WebSocket, active-speaker lock, and queued audio are cleaned up.

Also include tests for:

- immediate click before `audio_end`;
- repeated Turn 2 starts;
- delayed Turn 1 chunks;
- duplicate `audio_end`;
- stale Turn 1 errors;
- malformed audio metadata;
- long Turn 1 queue;
- Speaker B attempting to speak during the chain;
- reconnect after the chain completes.

### 8. Latency and memory analysis

Explain the expected tradeoffs numerically, without inventing measurements.

Estimate:

- perceived latency saved by allowing Turn 2 to start early;
- additional delay before Turn 2 audio is heard;
- maximum queued playback duration;
- memory limit for retained PCM;
- behavior when the queue limit is exceeded;
- whether the feature changes provider latency or only conversational idle time.

Clearly separate measured values from estimates.

### 9. Rollout and rollback

Define:

- a feature flag or safe activation boundary if appropriate;
- browser metrics to monitor;
- failure thresholds;
- how to disable the behavior without losing normal sequential playback;
- how to verify that the generated `web/static` bundle matches source;
- the exact test and build commands.

### 10. Acceptance criteria

The plan is complete only when it specifies criteria for:

- no Turn 1 audio truncation;
- no duplicate Turn 1 playback;
- ordered Turn 1 remainder then Turn 2 audio;
- no stale event mutation;
- no incorrect analytics attribution;
- no active-speaker lock leak;
- no unbounded PCM queue;
- no backend turn-lease race;
- passing browser tests;
- passing existing unit/protocol tests;
- reproducible production bundle generation.

## Important design constraints

- Do not clear the output player when Turn 2 starts.
- Do not reuse a single `events` object for multiple turns.
- Do not use one global `playingTurnId` as a substitute for per-turn records.
- Do not release the active-speaker lock when Turn 1 merely becomes paused.
- Do not accept Turn 2 before the server has safely completed Turn 1’s pipeline unless backend concurrency is explicitly designed and tested.
- Do not claim that “audio_end” means browser playback has finished.
- Do not claim that a passing fake stub test validates Sarvam provider behavior.
- Preserve the existing application stack.

## Required final output

Write one Markdown implementation plan with:

- current-code findings;
- files and symbols to modify;
- state-transition diagrams;
- protocol and ownership changes;
- player pause/resume design;
- browser test matrix B27–B30 and supporting cases;
- latency/memory tradeoffs;
- rollout/rollback plan;
- unresolved questions only where a user decision is genuinely required.

Do not write implementation code. Do not modify the repository while producing the plan.

