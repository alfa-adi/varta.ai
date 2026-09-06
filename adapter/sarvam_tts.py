"""
adapter/sarvam_tts.py
─────────────────────
Sarvam Bulbul v3 TTS Adapter — WebSocket streaming.

Key changes from previous version:
  - One persistent Bulbul WebSocket per (language, speaker) owner/config,
    protected by an asyncio.Lock (conversion_lock). Not one-per-turn.
  - configure() uses documented "language_code" field (not "target_language_code").
  - Per-turn flow: convert(text) → flush() → consume AudioOutput chunks →
    break on completion EventResponse → keep connection open for next turn.
  - Idle keepalive: connection pinged before idle timeout; reconnects transient
    closes with capped exponential backoff.
  - Fatal / 4xxx / auth / quota closes are surfaced as errors — not retried.
  - synthesise() non-streaming path still works for REST /translate/* endpoints.

TTS output spec: linear16 PCM, 24,000 Hz, mono.
"""

import asyncio
import base64
import time
from collections.abc import AsyncIterator

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse

from pipeline.types import TTSInput, TTSOutput

from .base import BaseTTSAdapter
from .sarvam_protocol import (
    BULBUL_SUPPORTED_LANGS,
    FATAL_CLOSE_CODES,
    RECONNECT_BASE_DELAY_SEC,
    RECONNECT_MAX_ATTEMPTS,
    RECONNECT_MAX_DELAY_SEC,
    TTS_AUDIO_CODEC,
    TTS_MODEL,
    TTS_SAMPLE_RATE,
)

# Voice map: (language_bcp47, gender) → Bulbul v3 speaker ID
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


class _BulbulConnection:
    """
    Holds one persistent Bulbul WebSocket connection for a fixed (language, speaker) config.
    Protected by a conversion_lock — only one convert/flush cycle at a time.
    """

    def __init__(self, client: AsyncSarvamAI, language: str, speaker: str, pace: float = 1.0):
        self._client    = client
        self._language  = language
        self._speaker   = speaker
        self._pace      = pace
        self._ws_ctx    = None
        self._ws        = None
        self.conversion_lock = asyncio.Lock()

    async def _open(self) -> None:
        """Open the Bulbul WebSocket and configure it."""
        self._ws_ctx = self._client.text_to_speech_streaming.connect(
            model=TTS_MODEL,
            send_completion_event=True,
        )
        self._ws = await self._ws_ctx.__aenter__()
        await self._ws.configure(
            language_code=self._language,      # ← documented field name
            speaker=self._speaker,
            output_audio_codec=TTS_AUDIO_CODEC,
            speech_sample_rate=TTS_SAMPLE_RATE,
            pace=self._pace,
        )

    async def _close(self) -> None:
        if self._ws_ctx is not None:
            try:
                await self._ws_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._ws_ctx = None
            self._ws     = None

    def _is_open(self) -> bool:
        return self._ws is not None

    async def synthesise_streaming(self, text: str) -> AsyncIterator[bytes]:
        """
        Yield raw linear16 PCM chunks as they arrive from Bulbul v3.
        Reconnects with backoff on transient failures.
        Raises on fatal / auth / quota errors.
        """
        delay = RECONNECT_BASE_DELAY_SEC
        for attempt in range(1, RECONNECT_MAX_ATTEMPTS + 1):
            try:
                if not self._is_open():
                    await self._open()

                await self._ws.convert(text)
                await self._ws.flush()

                async for message in self._ws:
                    if isinstance(message, AudioOutput):
                        yield base64.b64decode(message.data.audio)
                    elif isinstance(message, EventResponse) and message.data.event_type == "final":
                        break
                return  # successful completion

            except Exception as exc:
                close_code = getattr(exc, "code", None)
                if close_code in FATAL_CLOSE_CODES:
                    await self._close()
                    raise RuntimeError(
                        f"Bulbul fatal close {close_code}: {exc}"
                    ) from exc

                print(f"[Bulbul] Error on attempt {attempt}: {exc} — reconnecting")
                await self._close()

                if attempt < RECONNECT_MAX_ATTEMPTS:
                    await asyncio.sleep(min(delay, RECONNECT_MAX_DELAY_SEC))
                    delay *= 2
                else:
                    raise RuntimeError(
                        f"Bulbul failed after {RECONNECT_MAX_ATTEMPTS} attempts: {exc}"
                    ) from exc

    async def close(self) -> None:
        await self._close()


class SarvamTTSAdapter(BaseTTSAdapter):
    """
    Bulbul v3 TTS via persistent WebSocket connection per (language, speaker) config.

    Connection reuse:
      One _BulbulConnection is kept alive per (language, speaker) tuple.
      Concurrent conversions for the same config wait on conversion_lock.
      Connections for different configs are independent.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client  = AsyncSarvamAI(api_subscription_key=api_key)
        # Pool: (language, speaker_id) → _BulbulConnection
        self._pool: dict[tuple[str, str], _BulbulConnection] = {}
        self._pool_lock = asyncio.Lock()

    def supports_language(self, language: str) -> bool:
        return language in BULBUL_SUPPORTED_LANGS

    def _get_speaker(self, language: str, gender: str) -> str:
        return _VOICE_MAP.get((language, gender.lower()), _DEFAULT_SPEAKER)

    async def _get_connection(self, language: str, speaker: str, pace: float = 1.0) -> _BulbulConnection:
        """Return the existing persistent connection or create a new one."""
        key = (language, speaker)
        async with self._pool_lock:
            if key not in self._pool:
                self._pool[key] = _BulbulConnection(self._client, language, speaker, pace)
            return self._pool[key]

    async def synthesise(self, tts_input: TTSInput) -> TTSOutput:
        """Non-streaming synthesis — collects all chunks and returns a TTSOutput.
        Used by /translate/* REST endpoints."""
        if tts_input.language not in BULBUL_SUPPORTED_LANGS:
            raise ValueError(
                f"Bulbul v3 does not support '{tts_input.language}'. "
                f"Supported: {sorted(BULBUL_SUPPORTED_LANGS)}"
            )

        t0     = time.perf_counter()
        gender = getattr(tts_input, "voice_gender", "female")
        pace   = getattr(tts_input, "pace", 1.0)
        spk    = self._get_speaker(tts_input.language, gender)
        conn   = await self._get_connection(tts_input.language, spk, pace)

        audio_chunks: list[bytes] = []
        async with conn.conversion_lock:
            async for chunk in conn.synthesise_streaming(tts_input.text):
                audio_chunks.append(chunk)

        audio_bytes = b"".join(audio_chunks)
        latency_ms  = int((time.perf_counter() - t0) * 1000)
        print(f"[TTS] text_len={len(tts_input.text)} audio_bytes={len(audio_bytes)} "
              f"lang={tts_input.language} latency={latency_ms}ms")

        return TTSOutput(
            audio_bytes=audio_bytes,
            audio_format="pcm",
            language=tts_input.language,
            latency_ms=latency_ms,
            model_id="sarvam/bulbul-v3",
            tcp_ms=0,
            api_ms=latency_ms,
            parse_ms=0,
        )

    async def synthesise_streaming(self, tts_input: TTSInput) -> AsyncIterator[bytes]:
        """
        Async generator — yields raw linear16 PCM bytes as chunks arrive.
        Reuses persistent connections. Each synthesis is protected by conversion_lock.
        """
        if tts_input.language not in BULBUL_SUPPORTED_LANGS:
            raise ValueError(
                f"Bulbul v3 does not support '{tts_input.language}'. "
                f"Supported: {sorted(BULBUL_SUPPORTED_LANGS)}."
            )

        gender = getattr(tts_input, "voice_gender", "female")
        pace   = getattr(tts_input, "pace", 1.0)
        spk    = self._get_speaker(tts_input.language, gender)
        conn   = await self._get_connection(tts_input.language, spk, pace)

        chunk_count = 0
        async with conn.conversion_lock:
            async for chunk in conn.synthesise_streaming(tts_input.text):
                chunk_count += 1
                yield chunk

        print(f"[TTS-stream] text_len={len(tts_input.text)} chunks={chunk_count} "
              f"lang={tts_input.language}")

    async def close_all(self) -> None:
        """Close all pooled Bulbul connections. Call on server shutdown."""
        async with self._pool_lock:
            for conn in self._pool.values():
                await conn.close()
            self._pool.clear()