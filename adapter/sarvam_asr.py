"""
Sarvam Saaras v3 ASR Adapter — WebSocket implementation.

Replaces the REST multipart POST with a WebSocket connection to
wss://api.sarvam.ai/speech-to-text/ws, eliminating per-request TCP
reconnect overhead and REST serialization roundtrip.

Expected latency improvement: ~1307ms REST → ~600–900ms WS.

API reference:
  https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe/ws
"""

import asyncio
import base64
import io
import json
import time
import urllib.parse

import websockets
import websockets.exceptions

from .base import BaseASRAdapter
from pipeline.types import ASRInput, ASROutput

_WS_URL = "wss://api.sarvam.ai/speech-to-text/ws"

# Send audio in 64 KB chunks to avoid a single oversized WS frame.
_AUDIO_CHUNK_BYTES = 65536

# How long to wait for each transcript message after sending flush.
# Saaras v3 is typically sub-2s; 10s is a safe timeout for slow clips.
_RECV_TIMEOUT_SECONDS = 10.0

# Saaras WS only accepts audio/wav. Convert everything else using pydub.
_WAV_SAMPLE_RATE = 16000


def _to_wav_bytes(audio_bytes: bytes, audio_format: str) -> bytes:
    """
    Convert audio bytes to 16kHz mono WAV in-memory.
    Saaras v3 WS API only accepts audio/wav — webm/opus from the browser
    must be transcoded before sending.
    Uses pydub which in turn uses ffmpeg under the hood.
    """
    if audio_format == "wav":
        return audio_bytes  # Already WAV — pass through without touching

    from pydub import AudioSegment
    fmt = audio_format if audio_format != "webm" else "webm"
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    seg = seg.set_frame_rate(_WAV_SAMPLE_RATE).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()



class SarvamASRAdapter(BaseASRAdapter):
    """Saaras v3 ASR via WebSocket — replaces REST implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def transcribe(self, asr_input: ASRInput) -> ASROutput:
        t0 = time.perf_counter()

        # Adapt to existing ASRInput properties (which uses language_hint)
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

        # Adjust header based on websockets library version compatibility
        headers = {"Api-Subscription-Key": self._api_key}
        transcripts: list[str] = []
        detected_language: str = lang_code  # overwritten from WS response if available

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=15,
            ) as ws:
                # Convert to 16kHz mono WAV — Saaras WS only accepts audio/wav.
                # For WAV input this is a no-op; for webm/mp3/ogg it transcodes.
                audio_format = getattr(asr_input, "audio_format", "wav")
                audio = _to_wav_bytes(asr_input.audio_bytes, audio_format)

                # Send audio in chunks to stay within WS frame limits
                for offset in range(0, len(audio), _AUDIO_CHUNK_BYTES):
                    chunk = audio[offset : offset + _AUDIO_CHUNK_BYTES]
                    await ws.send(json.dumps({
                        "audio": {
                            "data": base64.b64encode(chunk).decode("utf-8"),
                            "sample_rate": str(_WAV_SAMPLE_RATE),
                            "encoding": "audio/wav",   # WS API only accepts wav
                        }
                    }))

                # Signal end of audio — server will finalize and close connection
                await ws.send(json.dumps({"type": "flush"}))

                # Drain transcript messages until server closes the connection
                while True:
                    try:
                        raw = await asyncio.wait_for(
                            ws.recv(), timeout=_RECV_TIMEOUT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        # No more messages within timeout — treat as complete
                        break
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "data":
                            data = msg.get("data", {})
                            t = data.get("transcript", "")
                            if t:
                                transcripts.append(t)
                            # Extract the language Saaras detected — this is the source
                            # of truth used downstream by TTS. Overwrite our initial guess.
                            lang_from_api = data.get("language_code", "")
                            if lang_from_api:
                                detected_language = lang_from_api
                    except (json.JSONDecodeError, KeyError, AttributeError):
                        pass  # Ignore malformed frames

        except websockets.exceptions.ConnectionClosedOK:
            pass  # Server closed normally after flush — expected
        except websockets.exceptions.ConnectionClosedError as exc:
            raise RuntimeError(f"Saaras v3 WS closed with error: {exc}") from exc
        except websockets.exceptions.WebSocketException as exc:
            raise RuntimeError(f"Saaras v3 WS error: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Saaras v3 WS network error: {exc}") from exc

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # If Saaras WS frames didn't include language_code (common when a hint is provided),
        # fall back to the input hint — matching the behaviour of the old REST adapter which did:
        #   body.get("language_code", input.language_hint or "unknown")
        final_language = detected_language or lang_code or "unknown"

        # Adapt to existing ASROutput which requires model_id
        return ASROutput(
            transcript=(" ".join(transcripts)).strip(),
            detected_language=final_language,
            confidence=1.0,
            latency_ms=latency_ms,
            model_id="sarvam/saaras-v3",
        )