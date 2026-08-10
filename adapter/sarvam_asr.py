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


# ── Live streaming adapter (persistent WS, realtime endpoint) ─────────────────

class SarvamLiveASRAdapter:
    """
    Persistent WebSocket connection to Saaras v3 realtime ASR.

    Lifecycle:
      1. Call start_session() once — opens WS, sends config, starts bg reader.
      2. While user is recording: call stream_chunk(pcm_bytes) for each 20ms packet.
      3. On stop: call flush_utterance() — async-iterates partial/final transcripts.
         The WS stays open after flush — ready for the next utterance immediately.
      4. Call close() when the session ends (browser disconnects permanently).

    Audio spec: pcm_s16le, 16kHz, mono. Browser AudioWorklet produces this directly.
    No file conversion at any point.

    One instance per speaker per session, stored in server._live_asr_sessions.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._ws = None                   # websockets connection handle
        self._recv_queue: asyncio.Queue = None  # decoded JSON frames from Saaras
        self._reader_task: asyncio.Task = None  # background frame reader
        self._detected_language: str = ""

    async def start_session(self, language_hint: str = "auto") -> None:
        """
        Open a persistent WS to saaras:v3-realtime and configure it.
        Call once per session/speaker. Blocks until the WS handshake completes.
        """
        query_dict = {
            "model": "saaras:v3-realtime",
            "input_audio_codec": "pcm_s16le",
            "sample_rate": str(_WAV_SAMPLE_RATE),
        }
        if language_hint and language_hint != "auto":
            query_dict["language-code"] = language_hint
            
        params = urllib.parse.urlencode(query_dict)
        url = f"{_REALTIME_WS_URL}?{params}"
        headers = {"Api-Subscription-Key": self._api_key}

        self._recv_queue = asyncio.Queue()
        self._ws = await websockets.connect(
            url,
            additional_headers=headers,
            ping_interval=20,
            ping_timeout=10,
            open_timeout=15,
        )
        # Background task drains all incoming Saaras frames into the queue.
        # This prevents the WS receive buffer from filling up between flushes.
        self._reader_task = asyncio.create_task(
            self._background_reader(), name="saaras-reader"
        )
        print(f"[LiveASR] Session opened → {url}")

    async def stream_chunk(self, pcm_bytes: bytes) -> None:
        """
        Forward a raw PCM binary frame to Saaras.
        Called ~50 times/sec (20ms chunks) while the user is recording.
        Fire-and-forget — no response expected per chunk.
        """
        if self._ws and not self._ws.closed:
            await self._ws.send(pcm_bytes)  # binary frame, no base64 encoding

    async def flush_utterance(self) -> AsyncIterator[dict]:
        """
        Signal end of utterance. Yields transcript dicts until the final
        transcript arrives from Saaras.

        Each yielded dict:
          { "transcript": str, "is_partial": bool, "language": str }

        The WS stays open after the final frame — ready for next utterance.
        """
        if not self._ws or self._ws.closed:
            return

        # Send flush signal — Saaras will finalize the current utterance
        await self._ws.send(json.dumps({"type": "flush"}))

        # Drain frames until we get a final (non-partial) transcript
        while True:
            try:
                frame = await asyncio.wait_for(
                    self._recv_queue.get(), timeout=10.0
                )
            except asyncio.TimeoutError:
                print("[LiveASR] Flush timeout — no final transcript within 10s")
                break

            msg_type = frame.get("type", "")
            data = frame.get("data", {})
            transcript = data.get("transcript", "")
            language = data.get("language_code", "")

            # Update detected language whenever Saaras tells us
            if language:
                self._detected_language = language

            # Saaras realtime sends "partial" frames while speaking,
            # "final" (or "data") after flush
            is_partial = msg_type in ("partial", "interim")
            is_final = msg_type in ("final", "data")

            if transcript or is_final:
                yield {
                    "transcript": transcript,
                    "is_partial": not is_final,
                    "language": language or self._detected_language,
                }

            if is_final:
                break  # Done — WS remains open for next utterance

    @property
    def detected_language(self) -> str:
        """Returns the language Saaras auto-detected, or empty string if unknown."""
        return self._detected_language

    async def _background_reader(self) -> None:
        """
        Continuously reads all incoming frames from Saaras and puts them
        into self._recv_queue. Runs as a background asyncio task.
        This ensures incoming frames don't pile up in the WS buffer while
        the server is waiting between flushes.
        """
        try:
            async for raw in self._ws:
                try:
                    frame = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
                    await self._recv_queue.put(frame)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # Ignore malformed frames silently
        except websockets.exceptions.ConnectionClosedOK:
            pass  # Normal close — session ended
        except websockets.exceptions.WebSocketException as exc:
            print(f"[LiveASR] WS reader error: {exc}")

    async def close(self) -> None:
        """
        Gracefully close the Saaras WS and stop the background reader.
        Call when the browser session ends permanently.
        """
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._ws and not self._ws.closed:
            await self._ws.close()
            print("[LiveASR] Session closed")