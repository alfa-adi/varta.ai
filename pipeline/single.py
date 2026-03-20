"""
pipeline/single.py
──────────────────
One-way translation pipeline: audio_in → transcript → translation → audio_out

This is the simplest pipeline. It runs three adapters in sequence:
  ASR → NMT → TTS

Languages are HARDCODED at construction time for the PoC.
The router + resolver (not yet built) will replace the hardcoded values later.

Usage:
    pipeline = SinglePipeline(
        asr_adapter=SarvamASRAdapter(api_key),
        nmt_adapter=SarvamNMTAdapter(api_key),
        tts_adapter=SarvamTTSAdapter(api_key),
        src_language="hi-IN",
        tgt_language="ta-IN",
    )
    result = await pipeline.run(audio_bytes, audio_format="wav")
"""

import time
from dataclasses import dataclass

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

        return PipelineResult(
            source_transcript = asr_output.transcript,
            translated_text   = nmt_output.translated_text,
            audio_bytes       = tts_output.audio_bytes,
            audio_format      = tts_output.audio_format,
            src_language      = asr_output.detected_language,
            tgt_language      = self.tgt_language,
            total_latency_ms  = total_ms,
        )