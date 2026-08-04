"""
Sarvam Bulbul v3 TTS Adapter — WebSocket streaming implementation.

Replaces the REST JSON POST (benchmarked at ~3380ms avg, the dominant bottleneck)
with WebSocket streaming via the sarvamai Python SDK. Audio chunks stream back
as they are generated, rather than waiting for full synthesis before returning.

Also applies Fix 2: Opus codec replaces MP3, saving ~150–200ms browser decode time.

Expected latency improvement: ~3380ms REST → ~800–1200ms WS (to last chunk).

API reference:
  https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/streaming-api/web-socket
"""

import asyncio
import base64
import time

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse

from .base import BaseTTSAdapter
from pipeline.types import TTSInput, TTSOutput


# Voice map: (language_bcp47, gender) → Bulbul v3 speaker ID
# Source: https://docs.sarvam.ai/api-reference-docs/models/bulbul
_VOICE_MAP: dict[tuple[str, str], str] = {
    ("hi-IN", "female"): "priya",
    ("hi-IN", "male"):   "shubh",
    ("ta-IN", "female"): "kavya",
    ("ta-IN", "male"):   "gokul",
    ("te-IN", "female"): "shruti",
    ("te-IN", "male"):   "vijay",
    ("kn-IN", "female"): "roopa",
    ("kn-IN", "male"):   "kabir",
    ("ml-IN", "female"): "suhani",
    ("ml-IN", "male"):   "mani",
    ("bn-IN", "female"): "ishita",
    ("bn-IN", "male"):   "rohan",
    ("mr-IN", "female"): "pooja",
    ("mr-IN", "male"):   "rahul",
    ("gu-IN", "female"): "neha",
    ("gu-IN", "male"):   "amit",
    ("pa-IN", "female"): "simran",
    ("pa-IN", "male"):   "dev",
    ("od-IN", "female"): "rupali",
    ("od-IN", "male"):   "sumit",
    ("en-IN", "female"): "ritu",
    ("en-IN", "male"):   "aditya",
}
_DEFAULT_SPEAKER = "shubh"

# Languages supported by Bulbul v3 (11 languages: 10 Indian + English)
_BULBUL_V3_LANGS: frozenset[str] = frozenset({
    "hi-IN", "bn-IN", "ta-IN", "te-IN", "gu-IN",
    "kn-IN", "ml-IN", "mr-IN", "pa-IN", "od-IN", "en-IN",
})


class SarvamTTSAdapter(BaseTTSAdapter):
    """Bulbul v3 TTS via WebSocket streaming — replaces REST implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        # AsyncSarvamAI is safe to instantiate once per adapter instance;
        # each synthesise() call opens its own WS connection.
        self._client = AsyncSarvamAI(api_subscription_key=api_key)

    def supports_language(self, language: str) -> bool:
        return language in _BULBUL_V3_LANGS

    def _get_speaker(self, language: str, gender: str) -> str:
        return _VOICE_MAP.get((language, gender.lower()), _DEFAULT_SPEAKER)

    async def synthesise(self, tts_input: TTSInput) -> TTSOutput:
        if tts_input.language not in _BULBUL_V3_LANGS:
            raise ValueError(
                f"Bulbul v3 does not support '{tts_input.language}'. "
                f"Supported languages: {sorted(_BULBUL_V3_LANGS)}. "
                "Use IIT Madras TTS (BhashiniTTSAdapter) for remaining 11 languages."
            )

        t0 = time.perf_counter()
        gender = getattr(tts_input, "voice_gender", "female")
        speaker = self._get_speaker(tts_input.language, gender)

        audio_chunks: list[bytes] = []

        async with self._client.text_to_speech_streaming.connect(
            model="bulbul:v3",
            send_completion_event=True,  # Server sends "final" event when done
        ) as ws:
            # Config must be the first message after connect
            await ws.configure(
                target_language_code=tts_input.language,
                speaker=speaker,
                output_audio_codec="opus",   # Fix 2: Opus saves ~150-200ms vs mp3
                speech_sample_rate=24000,    # OPUS codec requires 24000 (or 16k etc)
                pace=getattr(tts_input, "pace", 1.0),                    # Normal speed; adjust if needed
            )

            await ws.convert(tts_input.text)
            await ws.flush()  # Signal end of text; server finalizes synthesis

            async for message in ws:
                if isinstance(message, AudioOutput):
                    # Each AudioOutput carries a base64-encoded audio chunk
                    audio_chunks.append(base64.b64decode(message.data.audio))
                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final":
                        break  # All chunks received; clean exit

        audio_bytes = b"".join(audio_chunks)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        print(f"[TTS] text len: {len(tts_input.text)} | audio_bytes: {len(audio_bytes)} | lang: {tts_input.language} | latency: {latency_ms}ms")

        return TTSOutput(
            audio_bytes=audio_bytes,
            audio_format="opus",  # Updated from "mp3" — browser must handle Opus
            language=tts_input.language,
            latency_ms=latency_ms,
            model_id="sarvam/bulbul-v3",
        )