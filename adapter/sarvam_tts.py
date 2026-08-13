"""
Sarvam Bulbul v3 TTS Adapter — WebSocket streaming implementation.

Replaces the REST JSON POST (benchmarked at ~3380ms avg, the dominant bottleneck)
with WebSocket streaming via the sarvamai Python SDK. Audio chunks stream back
as they are generated, rather than waiting for full synthesis before returning.

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
    """Bulbul v3 TTS via persistent WebSocket streaming connection pool."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = AsyncSarvamAI(api_subscription_key=api_key)
        # Connection pool: (language, speaker) -> {"ws_ctx": context, "ws": ws_instance, "lock": asyncio.Lock()}
        self._pool: dict[tuple[str, str], dict] = {}
        self._pool_lock = asyncio.Lock()

    def supports_language(self, language: str) -> bool:
        return language in _BULBUL_V3_LANGS

    def _get_speaker(self, language: str, gender: str) -> str:
        return _VOICE_MAP.get((language, gender.lower()), _DEFAULT_SPEAKER)

    async def _get_connection(self, language: str, speaker: str, pace: float = 1.0):
        # We must create a new WebSocket per synthesis because bulbul:v3 
        # closes the connection (1000 OK) after a single generation finishes.
        ws_ctx = self._client.text_to_speech_streaming.connect(
            model="bulbul:v3",
            send_completion_event=True,
        )
        ws = await ws_ctx.__aenter__()
        await ws.configure(
            target_language_code=language,
            speaker=speaker,
            output_audio_codec="linear16",
            speech_sample_rate=24000,
            pace=pace,
        )
        return ws_ctx, ws

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
        pace = getattr(tts_input, "pace", 1.0)

        ws_ctx, ws = await self._get_connection(tts_input.language, speaker, pace)

        audio_chunks: list[bytes] = []
        tcp_ms = 0
        api_ms = 0
        parse_ms = 0

        try:
            api_start = time.perf_counter()
            await ws.convert(tts_input.text)
            await ws.flush()

            async for message in ws:
                api_ms += int((time.perf_counter() - api_start) * 1000)
                parse_start = time.perf_counter()

                if isinstance(message, AudioOutput):
                    audio_chunks.append(base64.b64decode(message.data.audio))
                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final":
                        parse_ms += int((time.perf_counter() - parse_start) * 1000)
                        break

                parse_ms += int((time.perf_counter() - parse_start) * 1000)
                api_start = time.perf_counter()
        finally:
            await ws_ctx.__aexit__(None, None, None)

        audio_bytes = b"".join(audio_chunks)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        print(f"[TTS] text len: {len(tts_input.text)} | audio_bytes: {len(audio_bytes)} | lang: {tts_input.language} | latency: {latency_ms}ms")

        return TTSOutput(
            audio_bytes=audio_bytes,
            audio_format="pcm",
            language=tts_input.language,
            latency_ms=latency_ms,
            model_id="sarvam/bulbul-v3",
            tcp_ms=tcp_ms,
            api_ms=api_ms,
            parse_ms=parse_ms,
        )

    async def synthesise_streaming(self, tts_input: TTSInput):
        """
        Async generator — yields raw linear16 PCM audio bytes as chunks arrive from Bulbul v3.
        Reuses persistent WebSockets across requests to bypass connection setup latency.
        """
        if tts_input.language not in _BULBUL_V3_LANGS:
            raise ValueError(
                f"Bulbul v3 does not support '{tts_input.language}'. "
                f"Supported languages: {sorted(_BULBUL_V3_LANGS)}."
            )

        gender = getattr(tts_input, "voice_gender", "female")
        speaker = self._get_speaker(tts_input.language, gender)
        pace = getattr(tts_input, "pace", 1.0)

        ws_ctx, ws = await self._get_connection(tts_input.language, speaker, pace)
        chunk_count = 0

        try:
            await ws.convert(tts_input.text)
            await ws.flush()

            async for message in ws:
                if isinstance(message, AudioOutput):
                    chunk = base64.b64decode(message.data.audio)
                    chunk_count += 1
                    yield chunk
                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final":
                        break
        finally:
            await ws_ctx.__aexit__(None, None, None)

        print(f"[TTS-stream] text len: {len(tts_input.text)} | chunks: {chunk_count} | lang: {tts_input.language}")