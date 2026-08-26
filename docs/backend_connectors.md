# Varta.AI — Backend Connectors Reference

> **Audience:** Front-end developers building or rebuilding the Varta.AI UI.  
> **Back-end file:** `web/server.py` (FastAPI, ~1138 lines).  
> **No back-end changes are needed** when rebuilding the front-end — all connectors below are stable.

---

## Table of Contents
1. [Session — Create](#1-session--create)
2. [WebSocket — Live ASR + NMT + TTS](#2-websocket--live-asr--nmt--tts)
   - [Client → Server Messages](#21-client--server-messages)
   - [Server → Client Messages](#22-server--client-messages)
3. [REST — Translate Speaker A](#3-rest--translate-speaker-a)
4. [REST — Translate Speaker B](#4-rest--translate-speaker-b)
5. [REST — Translate Single (Stateless)](#5-rest--translate-single-stateless)
6. [REST — Translate Dual (Both Speakers)](#6-rest--translate-dual-both-speakers)
7. [REST — Browser Metrics](#7-rest--browser-metrics)
8. [REST — Health Check](#8-rest--health-check)
9. [Static Assets](#9-static-assets)
10. [Rate Limiting](#10-rate-limiting)
11. [Error Handling](#11-error-handling)
12. [Full JS Usage Example](#12-full-js-usage-example)

---

## 1. Session — Create

Creates a new bilingual conversation session. A session stores the detected
language for each speaker so it persists across multiple recording turns.

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `/session/create` |
| **Content-Type** | `multipart/form-data` |
| **Auth** | None |
| **Rate limit** | 15 req/min per IP |

### Request Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `lang_a` | `string` | No | BCP-47 language code for Speaker A (e.g. `"hi-IN"`). Leave empty for auto-detect. |
| `lang_b` | `string` | No | BCP-47 language code for Speaker B. Leave empty for auto-detect. |

### Response
```json
{
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "lang_a": null,
  "lang_b": null
}
```

### JS Example
```js
async function createSession(langA = '', langB = '') {
  const form = new FormData();
  form.append('lang_a', langA);
  form.append('lang_b', langB);
  const res = await fetch('/session/create', { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Session creation failed: ${res.status}`);
  const { session_id } = await res.json();
  return session_id;
}
```

### Notes
- Empty string `""` is treated the same as `null` — the ASR model will auto-detect the language from the first audio clip.
- Once a language is detected by the ASR model it is saved back into the session automatically by the WebSocket handler.
- Sessions are stored in-memory (or Redis if `REDIS_URL` is set) with a 2-hour TTL.

---

## 2. WebSocket — Live ASR + NMT + TTS

The **core real-time path**. One persistent WebSocket connection per speaker
per conversation. Handles microphone audio streaming, transcription, translation,
and TTS audio streaming — all on the same socket.

| Field | Value |
|-------|-------|
| **Protocol** | `ws://` (plain) or `wss://` (TLS, required on Render/production) |
| **URL** | `ws[s]://<host>/ws/asr/<session_id>/<speaker>` |
| **`speaker`** | Must be `"a"` or `"b"`. Any other value → server closes with code 1003. |
| **Persistent?** | Yes. The Sarvam ASR connection (Saaras v3) is kept alive between turns. The browser WebSocket should stay open for the entire conversation. |
| **Re-use across turns** | Yes. After each turn ends (`audio_end`), the same WebSocket is reused for the next turn. Do NOT close and reopen it per turn. |

### 2.1 Client → Server Messages

#### Binary frame — PCM Audio Chunk
Send the raw microphone audio as **binary WebSocket frames**.

| Field | Value |
|-------|-------|
| Frame type | Binary (`ArrayBuffer`) |
| Encoding | **Raw `Int16` PCM** — NOT base64, NOT JSON-wrapped |
| Sample rate | **16 000 Hz** |
| Channels | **Mono (1 channel)** |
| Bit depth | **16-bit signed little-endian** |
| Chunk size | 20 ms → **320 samples = 640 bytes** per frame |

> ⚠️ **Critical protocol note:** The server reads binary frames with
> `msg["bytes"]` and passes them directly to Sarvam's ASR WebSocket.
> If you wrap the audio in JSON (e.g. `{ type: 'audio_chunk', data: base64 }`),
> the server will **ignore it** — it only reads raw binary frames.
> This was the primary bug in the previous UI.

**How to send correctly:**
```js
// workletNode.port.onmessage gives you e.data = Int16Array ArrayBuffer
workletNode.port.onmessage = (e) => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(e.data);        // ✅ raw ArrayBuffer — correct
    // ws.send(JSON.stringify({type:'audio_chunk', data: btoa(...)}))  ❌ wrong
  }
};
```

#### Text frame — Stop Recording
```json
{ "type": "stop_recording" }
```
Signals the server to flush the current utterance to the ASR model (sends
`speech_end` to Sarvam) which triggers the final transcript → NMT → TTS pipeline.

**When to send:** When the user releases the record button or when a timeout occurs.

---

### 2.2 Server → Client Messages

All server messages are JSON text frames.

#### `transcript_partial`
Live partial transcription — displayed in the preview bar while the user is speaking.
```json
{
  "type": "transcript_partial",
  "transcript": "नमस्ते मेरा"
}
```
Update the live preview UI element. Do NOT add to the conversation bubble yet.

#### `transcript_final`
The confirmed, complete utterance text.
```json
{
  "type": "transcript_final",
  "transcript": "नमस्ते मेरा नाम रोहन है"
}
```
Clear the live preview bar. Add a conversation bubble.

#### `language_detected`
Fired once when the ASR model identifies the speaker's language.
```json
{
  "type": "language_detected",
  "language": "hi-IN",
  "speaker": "a"
}
```
Update the language label in the speaker's panel. Save locally to pre-fill the next session if desired.

#### `audio_chunk`
One chunk of the TTS (text-to-speech) audio response. Multiple chunks arrive
sequentially — they must be **queued and played gaplessly**.
```json
{
  "type": "audio_chunk",
  "data": "<base64-encoded MP3 bytes>",
  "format": "mp3"
}
```
- Decode `data` from base64 to binary.
- Feed to a gapless audio player (see [§12 Full Example](#12-full-js-usage-example)).
- Play in the **other** speaker's panel (if Speaker A speaks → audio plays in Speaker B's panel).

#### `audio_end`
Signals that all TTS chunks have been sent for this turn.
```json
{ "type": "audio_end" }
```
**On receiving this:**
1. Stop the microphone recording (disconnect the AudioWorklet node so no stale PCM frames leak).
2. Hide the spinner.
3. Reset the label to "Press to record".
4. Flush any remaining audio queue.

> ⚠️ This is the most important message to handle correctly. Failing to stop the
> microphone here causes Sarvam to receive silent PCM after the turn ends,
> which triggers its hallucination loop (auto-generated garbage transcripts).

#### `error`
A pipeline error occurred (empty transcript, ASR timeout, NMT failure, etc.).
```json
{
  "type": "error",
  "message": "No speech detected. Please try again."
}
```
**On receiving this:**
1. Same as `audio_end` — stop the mic, hide the spinner, reset label.
2. Show the error message to the user.

---

## 3. REST — Translate Speaker A

Turn-by-turn fallback path. Used by Playwright tests and when WebSocket is unavailable.

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `/translate/speaker_a` |
| **Content-Type** | `multipart/form-data` |
| **Rate limit** | 15 req/min per IP |

### Request Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | File | Yes | Audio file (WAV, WebM, MP3, etc.) |
| `session_id` | `string` | Yes | Session ID from `/session/create` |

### Response
```json
{
  "transcript":       "नमस्ते मेरा नाम रोहन है",
  "translation":      "Hello, my name is Rohan",
  "src_language":     "hi-IN",
  "tgt_language":     "en-IN",
  "audio_b64":        "<base64 MP3>",
  "audio_format":     "mp3",
  "total_latency_ms": 1423,
  "timing": {
    "server": { "total_ms": 145, "session_load_ms": 2, ... },
    "asr":    { "total_ms": 830, "api_ms": 810, ... },
    "nmt":    { "total_ms": 210, "api_ms": 195, ... },
    "tts":    { "total_ms": 338, "api_ms": 320, ... },
    "browser": {}
  }
}
```

If Speaker B's language is not yet known, returns:
```json
{ "status": "buffered", "message": "Waiting for Speaker B language detection" }
```

---

## 4. REST — Translate Speaker B

Identical to Speaker A but processes Speaker B's audio.

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `/translate/speaker_b` |
| **Content-Type** | `multipart/form-data` |

Request and response fields are identical to `/translate/speaker_a`.

---

## 5. REST — Translate Single (Stateless)

One-way translation with explicit language codes. No session required.

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `/translate/single` |
| **Content-Type** | `multipart/form-data` |

### Request Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | File | Yes | Audio file |
| `src_language` | `string` | Yes | BCP-47 source language (e.g. `"hi-IN"`) |
| `tgt_language` | `string` | Yes | BCP-47 target language (e.g. `"en-IN"`) |

Response format is the same as `/translate/speaker_a` (without `timing.browser`).

---

## 6. REST — Translate Dual (Both Speakers)

Both speakers' audio in one request. Both translations returned together. All ASR/NMT/TTS calls run in parallel internally.

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `/translate/dual` |
| **Content-Type** | `multipart/form-data` |

### Request Fields
| Field | Type | Required |
|-------|------|----------|
| `audio_a` | File | Yes |
| `audio_b` | File | Yes |
| `session_id` | `string` | Yes |

### Response
```json
{
  "for_speaker_a": { "transcript": "...", "translation": "...", "audio_b64": "...", ... },
  "for_speaker_b": { "transcript": "...", "translation": "...", "audio_b64": "...", ... }
}
```

---

## 7. REST — Browser Metrics

Reports client-side timing data. Merged into the most recent `request_latency` MongoDB document for this session.

| Field | Value |
|-------|-------|
| **Method** | `POST` |
| **URL** | `/metrics/browser` |
| **Content-Type** | `multipart/form-data` |
| **Rate limit** | None |

### Request Fields
| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `string` | Session ID |
| `upload_ms` | `int` | Time to upload audio |
| `server_wait_ms` | `int` | Time waiting for server response |
| `parse_ms` | `int` | Time to parse JSON response |
| `audio_decode_ms` | `int` | Time to decode and start playing audio |
| `total_ms` | `int` | Total round-trip time (button press → audio plays) |

### Response
```json
{ "status": "ok" }
```

### JS Example
```js
async function reportMetrics(sessionId, timings) {
  const form = new FormData();
  form.append('session_id', sessionId);
  Object.entries(timings).forEach(([k, v]) => form.append(k, v));
  await fetch('/metrics/browser', { method: 'POST', body: form });
}
```

---

## 8. REST — Health Check

| Field | Value |
|-------|-------|
| **Method** | `GET` |
| **URL** | `/health` |
| **Auth** | None |
| **Rate limit** | None |

### Response
```json
{ "status": "ok", "version": "0.1.0" }
```

Use this to check if the server is up before attempting a session create.

---

## 9. Static Assets

| Path | Description |
|------|-------------|
| `GET /` | Serves `web/static/index.html` directly (not through StaticFiles — server reads and returns it) |
| `GET /static/*` | Everything under `web/static/` served as-is |
| `GET /static/worklet/pcm-processor.js` | AudioWorklet processor — must be served from this path. The JS loads it with `audioCtx.audioWorklet.addModule('/static/worklet/pcm-processor.js')` |

> **Build note:** After `npm run build`, Vite outputs to `web/static/`. The
> worklet file must be in `web/static/worklet/pcm-processor.js` (not in
> `web/static/assets/`) so the absolute URL stays predictable.

---

## 10. Rate Limiting

All endpoints except `/health`, `/metrics/browser`, and `/static/*` are rate-limited at **15 requests per minute per IP address** using SlowAPI.

When exceeded, the server returns:
```http
HTTP 429 Too Many Requests
```

The front-end should catch 429 responses and show a user-friendly message:
> "Too many requests. Please wait a moment and try again."

---

## 11. Error Handling

### HTTP Errors
| Code | Meaning | Front-end Action |
|------|---------|-----------------|
| `400` | Bad request (missing fields) | Show form validation error |
| `404` | Session not found | Re-create session and retry |
| `429` | Rate limited | Show "please wait" message |
| `500` | Pipeline error (ASR/NMT/TTS failure) | Show generic error, allow retry |

### WebSocket Close Codes
| Code | Meaning |
|------|---------|
| `1000` | Normal closure |
| `1003` | Invalid speaker value (not `"a"` or `"b"`) |
| `1006` | Abnormal closure (network drop) — reconnect |

### WebSocket `error` Messages
The server sends `{ "type": "error", "message": "..." }` for:
- Empty transcript (silence recorded)
- ASR model error
- NMT translation error
- TTS synthesis error
- Session not found mid-stream

---

## 12. Full JS Usage Example

A minimal but complete implementation of a single-speaker recording turn:

```js
// ── 1. Create session ────────────────────────────────────────────────────────
const { session_id } = await fetch('/session/create', {
  method: 'POST',
  body: Object.assign(new FormData(), { append: (k,v) => (new FormData()).append(k,v) })
}).then(r => r.json());

// ── 2. Open WebSocket ────────────────────────────────────────────────────────
const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws/asr/${session_id}/a`);
ws.binaryType = 'arraybuffer'; // not strictly needed — send side only

await new Promise((resolve, reject) => {
  ws.addEventListener('open',  resolve, { once: true });
  ws.addEventListener('error', reject,  { once: true });
});

// ── 3. Set up AudioContext + WorkletNode ─────────────────────────────────────
const audioCtx = new AudioContext({ sampleRate: 16000 });
await audioCtx.audioWorklet.addModule('/static/worklet/pcm-processor.js');
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const source = audioCtx.createMediaStreamSource(stream);
const worklet = new AudioWorkletNode(audioCtx, 'pcm-processor');

// ── 4. Stream audio ── send as raw binary, NOT base64 JSON ───────────────────
worklet.port.onmessage = (e) => {
  if (ws.readyState === WebSocket.OPEN) ws.send(e.data); // raw Int16 ArrayBuffer
};
source.connect(worklet).connect(audioCtx.destination);

// ── 5. Handle server messages ────────────────────────────────────────────────
ws.onmessage = async (evt) => {
  const msg = JSON.parse(evt.data);

  if (msg.type === 'transcript_partial') {
    document.getElementById('preview').textContent = msg.transcript;

  } else if (msg.type === 'transcript_final') {
    document.getElementById('preview').textContent = '';
    addBubble('a', msg.transcript);

  } else if (msg.type === 'language_detected') {
    document.getElementById('lang-a').textContent = msg.language;

  } else if (msg.type === 'audio_chunk') {
    // Decode and play MP3 chunk in Speaker B's panel
    const binary = Uint8Array.from(atob(msg.data), c => c.charCodeAt(0));
    const buffer = await audioCtx.decodeAudioData(binary.buffer);
    const src = audioCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(audioCtx.destination);
    src.start();

  } else if (msg.type === 'audio_end') {
    // ⚠️ Always stop the mic here — prevents silent PCM hallucination loop
    worklet.port.onmessage = null;
    worklet.disconnect();
    document.getElementById('spinner').hidden = true;
    document.getElementById('btn').textContent = 'Press to record';

  } else if (msg.type === 'error') {
    worklet.port.onmessage = null;
    worklet.disconnect();
    alert(msg.message);
  }
};

// ── 6. Stop recording ────────────────────────────────────────────────────────
function stopRecording() {
  worklet.port.onmessage = null;     // cut the mic immediately
  worklet.disconnect();
  // Do NOT stop MediaStream tracks — keep mic alive for next turn
  ws.send(JSON.stringify({ type: 'stop_recording' }));
}
```

---

## Supported Languages (BCP-47 Codes)
| Language | Code |
|----------|------|
| Hindi | `hi-IN` |
| Tamil | `ta-IN` |
| Telugu | `te-IN` |
| Bengali | `bn-IN` |
| Marathi | `mr-IN` |
| Gujarati | `gu-IN` |
| Kannada | `kn-IN` |
| Malayalam | `ml-IN` |
| Punjabi | `pa-IN` |
| Odia | `od-IN` |
| English (Indian) | `en-IN` |

---

*Last updated: 2026-08-24. Back-end version: 0.1.0.*
