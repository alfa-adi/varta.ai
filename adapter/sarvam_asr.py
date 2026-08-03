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


class SarvamASRAdapter(BaseASRAdapter):
    """Saaras v3 ASR via WebSocket — replaces REST implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def transcribe(self, asr_input: ASRInput) -> ASROutput:
        t0 = time.perf_counter()

        # Adapt to existing ASRInput properties (which uses language_hint)
        lang_code = getattr(asr_input, "language_hint", None) or ""

        params = urllib.parse.urlencode({
            "language-code": lang_code,
            "model": "saaras:v3",
            "mode": getattr(asr_input, "mode", "transcribe"),
            "sample_rate": str(getattr(asr_input, "sample_rate", 16000)),
        })
        ws_url = f"{_WS_URL}?{params}"

        # Adjust header based on websockets library version compatibility
        headers = {"Api-Subscription-Key": self._api_key}
        transcripts: list[str] = []
        detected_language: str = lang_code  # overwritten from WS response if available

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers, # Use additional_headers to be safe with newer websockets version
                ping_interval=20,
                ping_timeout=10,
                open_timeout=15,
            ) as ws:
                # Send audio in chunks to stay within WS frame limits
                audio = asr_input.audio_bytes
                for offset in range(0, len(audio), _AUDIO_CHUNK_BYTES):
                    chunk = audio[offset : offset + _AUDIO_CHUNK_BYTES]
                    await ws.send(json.dumps({
                        "audio": {
                            "data": base64.b64encode(chunk).decode("utf-8"),
                            "sample_rate": str(getattr(asr_input, "sample_rate", 16000)),
                            "encoding": f"audio/{getattr(asr_input, 'audio_format', 'wav')}",
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