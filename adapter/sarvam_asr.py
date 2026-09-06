"""
adapter/sarvam_asr.py
─────────────────────
Sarvam Saaras ASR adapters — legacy file-upload and live streaming.

Two adapters:

1. SarvamASRAdapter  — legacy REST/WS file-upload path used by /translate/* endpoints.
   Opens a new WS per request, converts audio to WAV, returns a full transcript.
   Unchanged in this refactor.

2. SarvamLiveASRAdapter  — live microphone streaming (realtime path).
   Maintains a persistent WS to saaras:v3-realtime per browser connection.
   Browser streams raw PCM (pcm_s16le, 16 kHz, mono) → server wraps as
   JSON audio_input messages → Saaras returns partial + final transcripts.

   Key invariants after refactor:
   - All lifecycle methods (start_session, stream_chunk, signal_speech_end,
     ping, reconnect, close) are serialized by a single asyncio.Lock.
   - start_session() waits for session.begin before accepting the first audio frame.
   - Each utterance: speech_start → N × audio_input → speech_end.
   - flush is reserved for watchdog forced-finalization only.
   - graceful close sends "end" and awaits session.end.
   - Reconnect cancels and awaits the old reader before replacing _ws.
   - Fatal / auth / quota errors are NOT retried.
   - Transcript queue is bounded (TRANSCRIPT_QUEUE_MAX items).
   - Odia is normalized: Sarvam realtime "or-IN" → Varta "od-IN".
"""

import asyncio
import base64
import io
import json
import time
import urllib.parse
from collections.abc import AsyncIterator

import websockets
import websockets.exceptions
from websockets.connection import State as _WSState

from pipeline.types import ASRInput, ASROutput
from web.protocol import TRANSCRIPT_QUEUE_MAX

from .base import BaseASRAdapter
from .sarvam_protocol import (
    ASR_ENCODING,
    ASR_ENDPOINTING,
    ASR_LANGUAGE_AUTO,
    ASR_MODEL,
    ASR_SAMPLE_RATE,
    ASR_STREAM_TYPE,
    REALTIME_ASR_URL,
    RECONNECT_BASE_DELAY_SEC,
    RECONNECT_MAX_ATTEMPTS,
    RECONNECT_MAX_DELAY_SEC,
    SarvamASRCommand,
    SarvamASREvent,
    sarvam_to_varta_lang,
    varta_to_sarvam_lang,
)

# ── Legacy file-upload endpoint ───────────────────────────────────────────────
_WS_URL            = "wss://api.sarvam.ai/speech-to-text/ws"
_AUDIO_CHUNK_BYTES = 65536
_RECV_TIMEOUT_SEC  = 3.0
_WAV_SAMPLE_RATE   = 16_000


def _to_wav_bytes(audio_bytes: bytes, audio_format: str) -> bytes:
    """Convert audio bytes to 16 kHz mono WAV (legacy path only)."""
    if audio_format == "wav":
        return audio_bytes
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)
    seg = seg.set_frame_rate(_WAV_SAMPLE_RATE).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


# ── Legacy adapter (file-upload REST path) ────────────────────────────────────

class SarvamASRAdapter(BaseASRAdapter):
    """Saaras v3 ASR via WebSocket — used by /translate/* REST endpoints only.

    For live microphone streaming, use SarvamLiveASRAdapter instead.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def transcribe(self, asr_input: ASRInput) -> ASROutput:
        t0 = time.perf_counter()
        lang_code = getattr(asr_input, "language_hint", None) or ""

        query_dict = {
            "model":       "saaras:v3",
            "mode":        getattr(asr_input, "mode", "transcribe"),
            "sample_rate": str(_WAV_SAMPLE_RATE),
        }
        if lang_code:
            query_dict["language-code"] = lang_code
        ws_url  = f"{_WS_URL}?{urllib.parse.urlencode(query_dict)}"
        headers = {"Api-Subscription-Key": self._api_key}

        transcripts: list[str] = []
        detected_language = lang_code
        tcp_ms = api_ms = parse_ms = 0
        audio  = b""

        try:
            tcp_start = time.perf_counter()
            async with websockets.connect(
                ws_url, additional_headers=headers,
                ping_interval=20, ping_timeout=10, open_timeout=15,
            ) as ws:
                tcp_ms = int((time.perf_counter() - tcp_start) * 1000)

                audio_format = getattr(asr_input, "audio_format", "wav")
                audio = _to_wav_bytes(asr_input.audio_bytes, audio_format)

                for offset in range(0, len(audio), _AUDIO_CHUNK_BYTES):
                    await ws.send(json.dumps({
                        "audio": {
                            "data":        base64.b64encode(audio[offset:offset + _AUDIO_CHUNK_BYTES]).decode(),
                            "sample_rate": str(_WAV_SAMPLE_RATE),
                            "encoding":    "audio/wav",
                        }
                    }))
                await ws.send(json.dumps({"type": "flush"}))

                api_start = time.perf_counter()
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=_RECV_TIMEOUT_SEC)
                    except asyncio.TimeoutError:
                        break
                    api_ms += int((time.perf_counter() - api_start) * 1000)
                    parse_start = time.perf_counter()
                    try:
                        msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
                        if msg.get("type") == "data":
                            data = msg.get("data", {})
                            t    = data.get("transcript", "")
                            if t:
                                transcripts.append(t)
                            lang_from_api = data.get("language_code", "")
                            if lang_from_api:
                                detected_language = lang_from_api
                            if t:
                                parse_ms += int((time.perf_counter() - parse_start) * 1000)
                                break
                    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
                        pass
                    parse_ms  += int((time.perf_counter() - parse_start) * 1000)
                    api_start  = time.perf_counter()

        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError as exc:
            raise RuntimeError(f"Saaras v3 WS closed with error: {exc}") from exc
        except websockets.exceptions.WebSocketException as exc:
            raise RuntimeError(f"Saaras v3 WS error: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Saaras v3 WS network error: {exc}") from exc

        latency_ms      = int((time.perf_counter() - t0) * 1000)
        final_language  = detected_language or lang_code or "unknown"
        transcript_text = (" ".join(transcripts)).strip()
        print(f"[ASR] audio_bytes={len(asr_input.audio_bytes)} wav={len(audio)} "
              f"text='{transcript_text[:40]}' lang={final_language} latency={latency_ms}ms")

        return ASROutput(
            transcript=transcript_text,
            detected_language=final_language,
            confidence=1.0,
            latency_ms=latency_ms,
            model_id="sarvam/saaras-v3",
            tcp_ms=tcp_ms,
            api_ms=api_ms,
            parse_ms=parse_ms,
        )


# ── Live streaming adapter (persistent WS, realtime protocol) ─────────────────

class SarvamLiveASRAdapter:
    """
    Persistent WebSocket connection to Saaras v3-realtime streaming ASR.

    Protocol (this adapter → Sarvam):
        {"event": "speech_start"}
        {"event": "audio_input", "audio": "<base64 mono 16kHz s16le>"}  × N
        {"event": "speech_end"}
        {"event": "flush"}     ← watchdog / forced-finalization only
        {"event": "ping"}
        {"event": "end"}       ← graceful adapter shutdown

    Protocol (Sarvam → this adapter):
        {"event": "session.begin",       "request_id": str, ...}
        {"event": "transcript.partial",  "text": str, "language": str}
        {"event": "transcript.final",    "text": str, "language": str, "language_confidence": float}
        {"event": "error",               "code": str, "is_fatal": bool, "message": str}
        {"event": "pong"}
        {"event": "session.end",         "audio_duration_s": float}

    Lifecycle invariants:
    - All public methods are serialized by _lifecycle_lock.
    - start_session() waits for session.begin before setting _ready = True.
    - Only one reader task (_background_reader) is active at a time.
    - Reconnect cancels and awaits the old reader before replacing _ws.
    - close() is idempotent; may be called from any context.
    """

    def __init__(self, api_key: str):
        self._api_key            = api_key
        self._ws                 = None
        self._recv_queue: asyncio.Queue | None = None
        self._reader_task: asyncio.Task | None = None
        self._lifecycle_lock     = asyncio.Lock()
        self._ready              = False   # True after session.begin received
        self._in_utterance       = False
        self._detected_language  = ""

        # ── Counters exposed for observability ───────────────────────────────
        self.upstream_connects:           int   = 0
        self.upstream_reconnects:         int   = 0
        self.reader_starts:               int   = 0
        self.reader_cancels:              int   = 0
        self.queue_overflows:             int   = 0
        self.session_begin_request_id:    str   = ""
        self.session_end_audio_duration_s: float = 0.0
        self.provider_error_code:         str   = ""
        self.provider_error_fatal:        bool  = False
        self.upstream_close_reason:       str   = ""

    # ── Public interface ──────────────────────────────────────────────────────

    async def start_session(self, language_hint: str = "") -> None:
        """
        Open a persistent WS to saaras:v3-realtime and start the background reader.
        Waits for session.begin before returning.

        language_hint: BCP-47 code (e.g. "hi-IN") or "" / "auto" for auto-detect.
                       Never pass "unknown" — use "auto" instead.
        """
        async with self._lifecycle_lock:
            await self._connect(language_hint)

    async def stream_chunk(self, pcm_bytes: bytes) -> None:
        """
        Forward a 16 kHz mono s16le PCM chunk to Saaras.
        Reconnects (serialized) if the connection is dead.
        """
        async with self._lifecycle_lock:
            if not self._is_open():
                print("[LiveASR] Connection dead — reconnecting...")
                await self._reconnect_under_lock()
                self._in_utterance = False

            if not self._ready:
                print("[LiveASR] Waiting for session.begin before sending audio")
                await self._wait_for_ready()

            if not self._in_utterance:
                await self._ws.send(json.dumps({"event": SarvamASRCommand.SPEECH_START}))
                self._in_utterance = True

            payload = json.dumps({
                "event": SarvamASRCommand.AUDIO_INPUT,
                "audio": base64.b64encode(pcm_bytes).decode(),
            })
            await self._ws.send(payload)

    async def signal_speech_end(self) -> None:
        """Signal manual end of utterance (triggers transcript.final from Saaras)."""
        async with self._lifecycle_lock:
            if not self._is_open():
                print("[LiveASR] Connection dead during signal_speech_end — reconnecting...")
                await self._reconnect_under_lock()
                self._in_utterance = False
                return   # no audio was sent on this reconnect; no speech to end
            await self._ws.send(json.dumps({"event": SarvamASRCommand.SPEECH_END}))
            self._in_utterance = False
            print("[LiveASR] speech_end sent")

    async def force_flush(self) -> None:
        """
        Send a Sarvam flush — watchdog/forced-finalization only.
        Do NOT call this as a normal stop; use signal_speech_end() instead.
        """
        async with self._lifecycle_lock:
            if self._is_open():
                await self._ws.send(json.dumps({"event": SarvamASRCommand.FLUSH}))
                print("[LiveASR] flush sent (watchdog)")

    async def send_ping(self) -> None:
        """Send a Sarvam-protocol ping (not a WebSocket control ping)."""
        async with self._lifecycle_lock:
            if self._is_open():
                await self._ws.send(json.dumps({"event": SarvamASRCommand.PING}))

    async def close(self) -> None:
        """
        Gracefully close the Saaras WS and stop the background reader.
        Idempotent — safe to call multiple times.
        """
        async with self._lifecycle_lock:
            await self._close_under_lock()

    async def listen_transcripts(self) -> AsyncIterator[dict]:
        """
        Continuously yields transcript dicts as they arrive from Saaras.
        Must be consumed by exactly ONE owner — not broadcast to multiple consumers.

        Yields:
            {
                "transcript":   str,
                "is_partial":   bool,
                "language":     str,   # Varta BCP-47 (od-IN normalized)
                "language_confidence": float | None,
            }
        """
        if self._recv_queue is None:
            raise RuntimeError("start_session() must be called before listen_transcripts()")

        while True:
            frame = await self._recv_queue.get()
            event_type = frame.get("event", "")

            if event_type == SarvamASREvent.TRANSCRIPT_PARTIAL:
                lang = sarvam_to_varta_lang(frame.get("language", "") or "")
                if lang:
                    self._detected_language = lang
                yield {
                    "transcript":         frame.get("text", ""),
                    "is_partial":         True,
                    "language":           lang or self._detected_language,
                    "language_confidence": None,
                }

            elif event_type == SarvamASREvent.TRANSCRIPT_FINAL:
                lang       = sarvam_to_varta_lang(frame.get("language", "") or "")
                confidence = frame.get("language_confidence")
                if lang:
                    self._detected_language = lang
                yield {
                    "transcript":         frame.get("text", ""),
                    "is_partial":         False,
                    "language":           lang or self._detected_language,
                    "language_confidence": confidence,
                }

            elif event_type == SarvamASREvent.ERROR:
                # Surface fatal errors to the consumer; non-fatal are logged.
                is_fatal = frame.get("is_fatal", False)
                self.provider_error_code  = frame.get("code", "")
                self.provider_error_fatal = is_fatal
                print(f"[LiveASR] Saaras error: code={self.provider_error_code} "
                      f"fatal={is_fatal} msg={frame.get('message', '')}")
                if is_fatal:
                    # Yield a sentinel that the consumer can turn into a turn_error
                    yield {
                        "transcript":          "",
                        "is_partial":          False,
                        "language":            self._detected_language,
                        "language_confidence": None,
                        "_provider_error":     frame,
                    }
                    return

            elif event_type == SarvamASREvent.SESSION_END:
                self.session_end_audio_duration_s = frame.get("audio_duration_s", 0.0)
                print(f"[LiveASR] session.end — audio_duration_s={self.session_end_audio_duration_s}")
                return  # generator exhausted; owner must handle

            # pong and unknown events are silently dropped

    @property
    def detected_language(self) -> str:
        """Returns the Varta-normalized language Saaras detected, or '' if unknown."""
        return self._detected_language

    @property
    def queue_depth(self) -> int:
        return self._recv_queue.qsize() if self._recv_queue else 0

    # ── Private helpers ───────────────────────────────────────────────────────

    def _is_open(self) -> bool:
        if self._ws is None:
            return False
        try:
            if hasattr(self._ws, "open"):
                return self._ws.open
            return self._ws.state == _WSState.OPEN
        except Exception:
            return False

    def _build_url(self, language_hint: str) -> str:
        # Use a valid BCP-47 or "auto"; never "unknown"
        lang = language_hint.strip() if language_hint else ""
        if not lang or lang.lower() == "unknown":
            lang = ASR_LANGUAGE_AUTO
        # Normalize od-IN → or-IN for the Sarvam realtime endpoint
        lang = varta_to_sarvam_lang(lang)

        params = urllib.parse.urlencode({
            "model":         ASR_MODEL,
            "language_code": lang,
            "stream_type":   ASR_STREAM_TYPE,
            "endpointing":   ASR_ENDPOINTING,
            "encoding":      ASR_ENCODING,
            "sample_rate":   str(ASR_SAMPLE_RATE),
        })
        return f"{REALTIME_ASR_URL}?{params}"

    async def _connect(self, language_hint: str) -> None:
        """Open WS, start reader, wait for session.begin. Called under _lifecycle_lock."""
        url     = self._build_url(language_hint)
        headers = {"Api-Subscription-Key": self._api_key}

        if self._recv_queue is None:
            self._recv_queue = asyncio.Queue(maxsize=TRANSCRIPT_QUEUE_MAX)

        try:
            self._ws = await websockets.connect(
                url, extra_headers=headers,
                ping_interval=20, ping_timeout=10, open_timeout=15,
            )
        except TypeError:
            # Older websockets versions use additional_headers
            self._ws = await websockets.connect(
                url, additional_headers=headers,
                ping_interval=20, ping_timeout=10, open_timeout=15,
            )

        self.upstream_connects += 1
        self._ready = False

        self._reader_task = asyncio.create_task(
            self._background_reader(), name="saaras-bg-reader"
        )
        self.reader_starts += 1
        print(f"[LiveASR] Connected → {url}")

        await self._wait_for_ready()

    async def _wait_for_ready(self) -> None:
        """Block until session.begin arrives (up to 8 s). Called under _lifecycle_lock."""
        from web.protocol import Timeout
        deadline = time.monotonic() + Timeout.ASR_SESSION_BEGIN
        while not self._ready:
            if time.monotonic() > deadline:
                raise RuntimeError("Timed out waiting for session.begin from Saaras")
            await asyncio.sleep(0.05)

    async def _reconnect_under_lock(self) -> None:
        """
        Serialized reconnect. Cancels old reader, closes old socket, opens new one.
        Retries up to RECONNECT_MAX_ATTEMPTS with capped exponential backoff.
        Must be called under _lifecycle_lock.
        """
        # 1. Cancel and await old reader
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self.reader_cancels += 1

        # 2. Close old socket
        if self._is_open():
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._ready = False

        # 3. Retry with backoff
        delay  = RECONNECT_BASE_DELAY_SEC
        for attempt in range(1, RECONNECT_MAX_ATTEMPTS + 1):
            try:
                await self._connect(self._detected_language or "")
                self.upstream_reconnects += 1
                print(f"[LiveASR] Reconnected (attempt {attempt})")
                return
            except Exception as exc:
                print(f"[LiveASR] Reconnect attempt {attempt} failed: {exc}")
                if attempt < RECONNECT_MAX_ATTEMPTS:
                    await asyncio.sleep(min(delay, RECONNECT_MAX_DELAY_SEC))
                    delay *= 2

        raise RuntimeError(
            f"Failed to reconnect to Saaras after {RECONNECT_MAX_ATTEMPTS} attempts"
        )

    async def _close_under_lock(self) -> None:
        """Close reader and WS. Called under _lifecycle_lock."""
        # Cancel reader
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self.reader_cancels += 1

        # Send graceful end if socket is open
        if self._is_open():
            try:
                await self._ws.send(json.dumps({"event": SarvamASRCommand.END}))
            except Exception:
                pass
            try:
                await self._ws.close()
            except Exception:
                pass
            print("[LiveASR] Session closed")

        self._ws    = None
        self._ready = False

    async def _background_reader(self) -> None:
        """
        Continuously reads all incoming frames from Saaras into _recv_queue.
        Runs as a background asyncio task. Handles session.begin specially
        (sets _ready flag); all other frames go to the queue.
        """
        try:
            async for raw in self._ws:
                try:
                    frame = (
                        json.loads(raw)
                        if isinstance(raw, str)
                        else json.loads(raw.decode())
                    )
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"[LiveASR-reader] Parse error: {e} | raw={str(raw)[:80]!r}")
                    continue

                event_type = frame.get("event", "?")
                print(f"[LiveASR-reader] event={event_type!r}")

                if event_type == SarvamASREvent.SESSION_BEGIN:
                    self.session_begin_request_id = frame.get("request_id", "")
                    self._ready = True
                    print(f"[LiveASR-reader] session.begin — request_id={self.session_begin_request_id}")
                    continue  # do not enqueue; handled inline

                if event_type == SarvamASREvent.ERROR:
                    self.provider_error_code  = frame.get("code", "")
                    self.provider_error_fatal = frame.get("is_fatal", False)
                    # Fall through to queue so listen_transcripts() sees it

                # Bounded enqueue
                try:
                    self._recv_queue.put_nowait(frame)
                except asyncio.QueueFull:
                    self.queue_overflows += 1
                    print(f"[LiveASR-reader] Queue full — dropped event={event_type!r}")

        except asyncio.CancelledError:
            pass  # normal cancellation during close/reconnect
        except websockets.exceptions.ConnectionClosedOK:
            self.upstream_close_reason = "closed_ok"
            print("[LiveASR-reader] WS closed normally")
        except websockets.exceptions.ConnectionClosedError as exc:
            self.upstream_close_reason = f"closed_error:{exc.code}"
            print(f"[LiveASR-reader] WS closed with error: {exc}")
        except websockets.exceptions.WebSocketException as exc:
            self.upstream_close_reason = f"ws_error:{type(exc).__name__}"
            print(f"[LiveASR-reader] WS error: {exc}")
        except Exception as exc:
            self.upstream_close_reason = f"unexpected:{type(exc).__name__}"
            print(f"[LiveASR-reader] Unexpected error: {exc}")
