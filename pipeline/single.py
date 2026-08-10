"""
pipeline/single.py
──────────────────
One-way translation pipeline: audio_in → transcript → translation → audio_out

Two pipeline entry points:

run()               — legacy file-based path: takes audio bytes, runs ASR+NMT+TTS,
                      returns a PipelineResult. Used by /translate/* REST endpoints.

run_from_transcript() — live streaming path (NEW): ASR is already done via the
                        persistent live WebSocket. Takes a ready transcript string,
                        runs NMT, then streams TTS chunks as an async generator.
                        First audio byte arrives ~100–200ms after call.

Usage:
    # Legacy (REST path)
    pipeline = SinglePipeline(...)
    result = await pipeline.run(audio_bytes, audio_format="wav")

    # Live streaming path
    async for chunk in pipeline.run_from_transcript(transcript, src_language):
        await ws.send_json({"type": "audio_chunk", "data": b64encode(chunk)})
"""

import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from adapter.base import BaseASRAdapter, BaseNMTAdapter, BaseTTSAdapter
from pipeline.types import (
    ASRInput, NMTInput, TTSInput, PipelineResult
)


@dataclass
class SinglePipeline:
    """
    Runs ASR → NMT → TTS for a single direction.
    Each adapter is injected at construction — no import of concrete classes here.
    This keeps the pipeline decoupled from specific providers.
    """
    asr_adapter: BaseASRAdapter
    nmt_adapter: BaseNMTAdapter
    tts_adapter: BaseTTSAdapter

    # Hardcoded for PoC — router will set these dynamically in v2
    src_language: str    # e.g. "hi-IN"
    tgt_language: str    # e.g. "ta-IN"

    async def run(
        self,
        audio_bytes:  bytes,
        audio_format: str = "wav",
        asr_mode:     str = "transcribe",
        voice_gender: str = "female",
    ) -> PipelineResult:
        """
        Full pipeline run: audio → translated audio.

        Step 1 — ASR: audio bytes become text in src_language
        Step 2 — NMT: src_language text becomes tgt_language text
        Step 3 — TTS: tgt_language text becomes audio bytes
        """
        wall_start = int(time.time() * 1000)

        # ── Step 1: ASR ──────────────────────────────────────────────
        # We pass src_language as a hint to improve accuracy.
        # The adapter still returns detected_language from the API response.
        asr_output = await self.asr_adapter.transcribe(
            ASRInput(
                audio_bytes   = audio_bytes,
                audio_format  = audio_format,
                language_hint = self.src_language,
                mode          = asr_mode,
            )
        )

        # ── Step 2: NMT ──────────────────────────────────────────────
        # Use the ASR-detected language as the true source.
        # This is more accurate than blindly using self.src_language,
        # especially once we add auto-detection.
        nmt_output = await self.nmt_adapter.translate(
            NMTInput(
                text         = asr_output.transcript,
                src_language = asr_output.detected_language,
                tgt_language = self.tgt_language,
            )
        )

        # ── Step 3: TTS ──────────────────────────────────────────────
        tts_output = await self.tts_adapter.synthesise(
            TTSInput(
                text         = nmt_output.translated_text,
                language     = self.tgt_language,
                voice_gender = voice_gender,
                audio_format = "mp3",   # mp3 is smallest — good for web
            )
        )

        total_ms = int(time.time() * 1000) - wall_start

        # ── Collect timing from all adapters ─────────────────────────
        timing = {
            "asr_total_ms": asr_output.latency_ms,
            "asr_tcp_ms":   asr_output.tcp_ms,
            "asr_api_ms":   asr_output.api_ms,
            "asr_parse_ms": asr_output.parse_ms,
            "nmt_total_ms": nmt_output.latency_ms,
            "nmt_tcp_ms":   nmt_output.tcp_ms,
            "nmt_api_ms":   nmt_output.api_ms,
            "nmt_parse_ms": nmt_output.parse_ms,
            "tts_total_ms": tts_output.latency_ms,
            "tts_tcp_ms":   tts_output.tcp_ms,
            "tts_api_ms":   tts_output.api_ms,
            "tts_parse_ms": tts_output.parse_ms,
        }

        return PipelineResult(
            source_transcript = asr_output.transcript,
            translated_text   = nmt_output.translated_text,
            audio_bytes       = tts_output.audio_bytes,
            audio_format      = tts_output.audio_format,
            src_language      = asr_output.detected_language,
            tgt_language      = self.tgt_language,
            total_latency_ms  = total_ms,
            timing            = timing,
        )

    async def run_from_transcript(
        self,
        transcript:   str,
        src_language: str,
        voice_gender: str = "female",
    ) -> AsyncIterator[bytes]:
        """
        Live streaming path — ASR is already done via the persistent WS.
        Takes a ready final transcript, runs NMT, then streams TTS opus chunks.

        Yields raw opus bytes as each chunk arrives from Bulbul v3.
        First byte arrives in ~100–200ms (NMT) + ~100ms (TTS first chunk).
        Caller should forward each chunk to the browser immediately over WS.

        Args:
            transcript:   Final transcript from SarvamLiveASRAdapter.flush_utterance()
            src_language: Detected source language (BCP-47 e.g. "hi-IN")
            voice_gender: "female" or "male" (default "female")

        Yields:
            bytes — decoded opus audio chunks, each 20–80ms of audio
        """
        # ── Step 1: NMT — translate the final transcript ──────────────
        # This is a single REST call (~200-400ms). No ASR step here.
        nmt_output = await self.nmt_adapter.translate(
            NMTInput(
                text         = transcript,
                src_language = src_language,
                tgt_language = self.tgt_language,
            )
        )

        # ── Step 2: TTS streaming — yield chunks as they arrive ───────
        # synthesise_streaming() is an async generator that yields each
        # opus chunk immediately as it decodes from the Bulbul v3 WS.
        async for chunk in self.tts_adapter.synthesise_streaming(
            TTSInput(
                text         = nmt_output.translated_text,
                language     = self.tgt_language,
                voice_gender = voice_gender,
            )
        ):
            yield chunk