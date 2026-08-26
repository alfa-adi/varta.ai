"""
Sarvam Saaras v3 ASR Adapter — WebSocket implementation.

Two adapters in this module:

1. SarvamASRAdapter  — legacy file-upload path used by REST endpoints.
   Opens a new WS per request, converts audio to WAV, returns full transcript.

2. SarvamLiveASRAdapter  — live microphone streaming path (NEW).
   Maintains a persistent WS to saaras:v3-realtime per session/speaker.
   Browser streams raw PCM (pcm_s16le) → server relays binary frames →
   Saaras returns partial + final transcripts in real time.
   No file conversion. No reconnect between turns.

API references:
  Legacy:  wss://api.sarvam.ai/speech-to-text/ws  (saaras:v3)
  Realtime: wss://api.sarvam.ai/speech-to-text-realtime/ws (saaras:v3-realtime)
"""

import asyncio
import base64
import io
import json
import time
import urllib.parse
from typing import AsyncIterator

import websockets
import websockets.exceptions
from websockets.connection import State as _WSState

from .base import BaseASRAdapter
from pipeline.types import ASRInput, ASROutput

# ── Legacy file-upload endpoint ───────────────────────────────────────────────
_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"

# ── Realtime live streaming endpoint ─────────────────────────────────────────
_REALTIME_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

# Send audio in 64 KB chunks to avoid a single oversized WS frame.
_AUDIO_CHUNK_BYTES = 65536

# How long to wait for each transcript message after sending flush.
_RECV_TIMEOUT_SECONDS = 3.0

# Sample rate for both legacy and realtime paths.
_WAV_SAMPLE_RATE = 16000


def _to_wav_bytes(audio_bytes: bytes, audio_format: str) -> bytes:
    """
    Convert audio bytes to 16kHz mono WAV in-memory.
    Used only by the legacy SarvamASRAdapter (file-upload path).
    The live SarvamLiveASRAdapter uses raw PCM and never calls this.
    """
    if audio_format == "wav":
        return audio_bytes

    from pydub import AudioSegment
    fmt = audio_format if audio_format != "webm" else "webm"
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    seg = seg.set_frame_rate(_WAV_SAMPLE_RATE).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


# ── Legacy adapter (file-upload REST path) ────────────────────────────────────

class SarvamASRAdapter(BaseASRAdapter):
    """Saaras v3 ASR via WebSocket — used by /translate/* REST endpoints.

    For live microphone streaming, use SarvamLiveASRAdapter instead.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def transcribe(self, asr_input: ASRInput) -> ASROutput:
        t0 = time.perf_counter()

        lang_code = getattr(asr_input, "language_hint", None) or ""

        query_dict = {
            "model": "saaras:v3",
            "mode": getattr(asr_input, "mode", "transcribe"),
            "sample_rate": str(_WAV_SAMPLE_RATE),
        }
        if lang_code:
            query_dict["language-code"] = lang_code
        params = urllib.parse.urlencode(query_dict)
        ws_url = f"{_WS_URL}?{params}"

        headers = {"Api-Subscription-Key": self._api_key}
        transcripts: list[str] = []
        detected_language: str = lang_code

        tcp_ms = 0
        api_ms = 0
        parse_ms = 0
        audio = b""

        try:
            tcp_start = time.perf_counter()
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=15,
            ) as ws:
                tcp_ms = int((time.perf_counter() - tcp_start) * 1000)

                audio_format = getattr(asr_input, "audio_format", "wav")
                audio = _to_wav_bytes(asr_input.audio_bytes, audio_format)

                for offset in range(0, len(audio), _AUDIO_CHUNK_BYTES):
                    chunk = audio[offset : offset + _AUDIO_CHUNK_BYTES]
                    await ws.send(json.dumps({
                        "audio": {
                            "data": base64.b64encode(chunk).decode("utf-8"),
                            "sample_rate": str(_WAV_SAMPLE_RATE),
                            "encoding": "audio/wav",
                        }
                    }))

                await ws.send(json.dumps({"type": "flush"}))

                api_start = time.perf_counter()

                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=_RECV_TIMEOUT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        break

                    api_ms += int((time.perf_counter() - api_start) * 1000)
                    parse_start = time.perf_counter()
                    try:
                        msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                        if msg.get("type") == "data":
                            data = msg.get("data", {})
                            t = data.get("transcript", "")
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
                    parse_ms += int((time.perf_counter() - parse_start) * 1000)
                    api_start = time.perf_counter()

        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosedError as exc:
            raise RuntimeError(f"Saaras v3 WS closed with error: {exc}") from exc
        except websockets.exceptions.WebSocketException as exc:
            raise RuntimeError(f"Saaras v3 WS error: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Saaras v3 WS network error: {exc}") from exc

        latency_ms = int((time.perf_counter() - t0) * 1000)
        final_language = detected_language or lang_code or "unknown"
        transcript_text = (" ".join(transcripts)).strip()

        print(f"[ASR] audio_bytes: {len(asr_input.audio_bytes)} | wav: {len(audio)} | text: '{transcript_text[:40]}' | lang: {final_language} | latency: {latency_ms}ms")

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


# ── Live streaming adapter (persistent WS, SDK-confirmed protocol) ─────────────

class SarvamLiveASRAdapter:
    """
    Persistent WebSocket connection to Saaras v3 streaming ASR.

    Protocol confirmed directly from sarvam API documentation:

    Send audio:
        {"event": "audio_input", "audio": "<base64-pcm>"}

    Send flush:
        {"event": "speech_end"}

    Receive (per frame):
        {"event": "transcript.partial", "text": "...", "language": "..."}
        {"event": "transcript.final", "text": "...", "language": "...", "language_confidence": ...}

    URL: wss://api.sarvam.ai/speech-to-text-realtime/ws
    Query params: model=saaras:v3-realtime, encoding=linear16, sample_rate=16000, endpointing=manual

    One instance per speaker per session, stored in server._live_asr_sessions.
    The WS stays open between turns — only closed when browser session ends.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._ws = None                        # websockets connection handle
        self._recv_queue: asyncio.Queue = None # decoded JSON frames from Saaras
        self._reader_task: asyncio.Task = None # background frame reader
        self._detected_language: str = ""
        self._in_utterance: bool = False

    def _is_open(self) -> bool:
        """Check if WS is connected and usable."""
        if self._ws is None:
            return False
        try:
            # Works for both legacy (has .open) and new (has .state) websockets
            if hasattr(self._ws, 'open'):
                return self._ws.open
            return self._ws.state == _WSState.OPEN
        except Exception:
            return False

    async def start_session(self, language_hint: str = "unknown") -> None:
        """
        Open a persistent WS to saaras:v3 streaming and start bg reader.
        Call once per session/speaker.

        language_hint: BCP-47 code (e.g. "hi-IN") or "unknown" for auto-detect.
        """
        query_dict = {
            "model":         "saaras:v3-realtime",
            "encoding":      "linear16",
            "sample_rate":   str(_WAV_SAMPLE_RATE),
            "language_code": language_hint if language_hint else "auto",
            "endpointing":   "manual",
        }
        params = urllib.parse.urlencode(query_dict)
        url = f"{_REALTIME_WS_URL}?{params}"
        headers = {"Api-Subscription-Key": self._api_key}

        if self._recv_queue is None:
            self._recv_queue = asyncio.Queue()

        # Use legacy websockets connect — same as SDK (extra_headers for v14+,
        # additional_headers for v12/v13). Try both.
        try:
            self._ws = await websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=15,
            )
        except TypeError:
            # Older websockets versions use additional_headers
            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=15,
            )

        self._reader_task = asyncio.create_task(
            self._background_reader(), name="saaras-bg-reader"
        )
        print(f"[LiveASR] Session opened -> {url}")

    async def stream_chunk(self, pcm_bytes: bytes) -> None:
        """
        Forward a PCM chunk to Saaras.
        Wraps raw PCM bytes as base64 inside the SDK's AudioMessage format.
        Called ~50 times/sec (20ms chunks) while user is recording.
        """
        if not self._is_open():
            print("[LiveASR] Connection dead, auto-reconnecting...")
            await self.start_session(language_hint=self._detected_language or "unknown")
            self._in_utterance = False

        if not self._in_utterance:
            await self._ws.send(json.dumps({"event": "speech_start"}))
            self._in_utterance = True

        payload = json.dumps({
            "event": "audio_input",
            "audio": base64.b64encode(pcm_bytes).decode()
        })
        await self._ws.send(payload)

    async def signal_speech_end(self) -> None:
        """Signal end of utterance for manual endpointing."""
        if not self._is_open():
            print("[LiveASR] Connection dead during signal_speech_end, auto-reconnecting...")
            await self.start_session(language_hint=self._detected_language or "unknown")
            self._in_utterance = False
        await self._ws.send(json.dumps({"event": "speech_end"}))
        self._in_utterance = False
        print("[LiveASR] speech_end sent")

    async def listen_transcripts(self) -> AsyncIterator[dict]:
        """
        Continuously yields transcript dicts as they arrive from Saaras.
        Yields: { "transcript": str, "is_partial": bool, "language": str }
        """
        while True:
            frame = await self._recv_queue.get()
            event_type = frame.get("event", "")

            if event_type in ("transcript.partial", "transcript.final"):
                transcript = frame.get("text", "")
                language = frame.get("language", "")

                if language:
                    self._detected_language = language

                yield {
                    "transcript": transcript,
                    "is_partial": event_type == "transcript.partial",
                    "language":   language or self._detected_language,
                }
            elif event_type == "error":
                print(f"[LiveASR] Saaras error frame: {frame}")
                # We do not break here; just log and continue listening.
            else:
                pass # ignore non-transcript frames

    @property
    def detected_language(self) -> str:
        """Returns the language Saaras auto-detected, or empty string if unknown."""
        return self._detected_language

    async def _background_reader(self) -> None:
        """
        Continuously reads all incoming frames from Saaras into self._recv_queue.
        Runs as a background asyncio task so frames are never lost between flushes.
        """
        try:
            async for raw in self._ws:
                try:
                    frame = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
                    event_type = frame.get("event", "?")
                    print(f"[LiveASR-reader] Received event={event_type!r}")
                    await self._recv_queue.put(frame)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    raw_preview = str(raw)[:100]
                    print(f"[LiveASR-reader] Parse error: {e} | raw={raw_preview!r}")
        except websockets.exceptions.ConnectionClosedOK:
            print("[LiveASR-reader] WS closed normally")
        except websockets.exceptions.WebSocketException as exc:
            print(f"[LiveASR-reader] WS error: {exc}")
        except Exception as exc:
            print(f"[LiveASR-reader] Unexpected error: {exc}")

    async def close(self) -> None:
        """Gracefully close the Saaras WS and stop the background reader."""
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._is_open():
            await self._ws.close()
            print("[LiveASR] Session closed")
