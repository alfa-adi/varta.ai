"""
pipeline/dual.py
────────────────
Bidirectional simultaneous translation pipeline.

This implements the cross-language sourcing architecture:

  Speaker A audio ──→ ASR_A ──→ (transcript_A, detected lang_A)
                                        │
                        ┌───────────────┘ lang_A becomes target for Pipeline B
                        ▼
  NMT_A: translate(transcript_A, src=lang_A, tgt=lang_B) ──→ TTS_A(lang_B) ──→ Speaker B hears
                                        ▲
                        lang_B ─────────┘ (lang_B comes from ASR_B)

  Speaker B audio ──→ ASR_B ──→ (transcript_B, detected lang_B)
                                        │
                        ┌───────────────┘ lang_B becomes target for Pipeline A
                        ▼
  NMT_B: translate(transcript_B, src=lang_B, tgt=lang_A) ──→ TTS_B(lang_A) ──→ Speaker A hears
                                        ▲
                        lang_A ─────────┘ (lang_A comes from ASR_A)

KEY INSIGHT: ASR layer detects the language. That detected language becomes the
TARGET of the OTHER speaker's NMT layer. No manual language declaration needed
once both speakers have spoken once.

For the PoC: speakers declare their language upfront via the web UI.
The detected language from ASR is used to verify and override if needed.

PARALLELISM: Both ASR calls run simultaneously (asyncio.gather).
             Both NMT calls run simultaneously after ASR completes.
             Both TTS calls run simultaneously after NMT completes.
             Total wall time ≈ max(each layer) not sum(each layer).
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

from adapter.base import BaseASRAdapter, BaseNMTAdapter, BaseTTSAdapter
from pipeline.types import (
    ASRInput, NMTInput, TTSInput,
    ASROutput, NMTOutput, TTSOutput,
    PipelineResult
)


@dataclass
class DualPipelineResult:
    """Result from both directions in a single round."""
    for_speaker_b: PipelineResult    # What Speaker B should hear
    for_speaker_a: PipelineResult    # What Speaker A should hear


@dataclass
class SessionState:
    """
    Tracks the detected languages across turns.
    On first turn, languages come from the UI declaration.
    After that, the ASR overrides them with detected values.
    """
    lang_a: Optional[str] = None    # Speaker A's detected language (BCP-47)
    lang_b: Optional[str] = None    # Speaker B's detected language (BCP-47)

    def both_known(self) -> bool:
        return self.lang_a is not None and self.lang_b is not None


class DualPipeline:
    """
    Runs two translation pipelines in parallel — one per direction.

    Construction: inject shared adapter instances (both pipelines share
    the same adapter objects, saving memory and connection pool resources).

    Session state: tracks detected languages across turns so the cross-
    wiring (lang_a → NMT_B target, lang_b → NMT_A target) works correctly.
    """

    def __init__(
        self,
        asr: BaseASRAdapter,
        nmt: BaseNMTAdapter,
        tts: BaseTTSAdapter,
        initial_lang_a: Optional[str] = None,   # From UI declaration
        initial_lang_b: Optional[str] = None,   # From UI declaration
    ):
        self.asr = asr
        self.nmt = nmt
        self.tts = tts

        # Session state — updated by ASR detections each turn
        self.state = SessionState(
            lang_a=initial_lang_a,
            lang_b=initial_lang_b,
        )

    # ── Single-direction pipeline helper ─────────────────────────────────────

    async def _run_one_direction(
        self,
        audio_bytes:   bytes,
        audio_format:  str,
        language_hint: Optional[str],   # Hint for ASR
        tgt_language:  str,             # Where to translate TO
        voice_gender:  str,
    ) -> Tuple[PipelineResult, str]:
        """
        Run ASR → NMT → TTS for one speaker.
        Returns (PipelineResult, detected_source_language).
        The detected language is returned separately so the caller can
        update the session state and cross-wire the other direction.
        """
        wall_start = int(time.time() * 1000)

        # ASR: audio → text
        asr_out: ASROutput = await self.asr.transcribe(
            ASRInput(
                audio_bytes   = audio_bytes,
                audio_format  = audio_format,
                language_hint = language_hint,
                mode          = "transcribe",
            )
        )

        detected_src = asr_out.detected_language

        # NMT: text in detected_src → text in tgt_language
        nmt_out: NMTOutput = await self.nmt.translate(
            NMTInput(
                text         = asr_out.transcript,
                src_language = detected_src,
                tgt_language = tgt_language,
            )
        )

        # TTS: translated text → audio in tgt_language
        tts_out: TTSOutput = await self.tts.synthesise(
            TTSInput(
                text         = nmt_out.translated_text,
                language     = tgt_language,
                voice_gender = voice_gender,
                audio_format = "mp3",
            )
        )

        total_ms = int(time.time() * 1000) - wall_start

        result = PipelineResult(
            source_transcript = asr_out.transcript,
            translated_text   = nmt_out.translated_text,
            audio_bytes       = tts_out.audio_bytes,
            audio_format      = tts_out.audio_format,
            src_language      = detected_src,
            tgt_language      = tgt_language,
            total_latency_ms  = total_ms,
        )

        return result, detected_src

    # ── Both speakers send audio at the same time ─────────────────────────────

    async def process_both(
        self,
        audio_a:      bytes,
        audio_b:      bytes,
        audio_format: str = "wav",
        gender_a:     str = "female",
        gender_b:     str = "female",
    ) -> DualPipelineResult:
        """
        Process audio from BOTH speakers simultaneously.

        PHASE 1 — Parallel ASR:
          Both audio clips are transcribed at the same time.
          We get back (transcript_A, lang_A) and (transcript_B, lang_B).

        PHASE 2 — Cross-wire languages:
          lang_A from ASR_A → becomes target for Pipeline B's NMT
          lang_B from ASR_B → becomes target for Pipeline A's NMT
          This is the core of the cross-language sourcing architecture.

        PHASE 3 — Parallel NMT + TTS (chained):
          Both translation+synthesis pipelines run simultaneously.
        """

        # ── PHASE 1: Parallel ASR ────────────────────────────────────────────
        # asyncio.gather runs both coroutines concurrently.
        # Wall time = max(ASR_A latency, ASR_B latency) — not their sum.
        asr_a_task = self.asr.transcribe(
            ASRInput(
                audio_bytes   = audio_a,
                audio_format  = audio_format,
                language_hint = self.state.lang_a,  # Hint from session state
                mode          = "transcribe",
            )
        )
        asr_b_task = self.asr.transcribe(
            ASRInput(
                audio_bytes   = audio_b,
                audio_format  = audio_format,
                language_hint = self.state.lang_b,
                mode          = "transcribe",
            )
        )

        asr_a_out, asr_b_out = await asyncio.gather(asr_a_task, asr_b_task)

        # ── PHASE 2: Cross-wire ──────────────────────────────────────────────
        # The detected language from each ASR becomes the TARGET of the other.
        lang_a = asr_a_out.detected_language   # e.g. "hi-IN"
        lang_b = asr_b_out.detected_language   # e.g. "ta-IN"

        # Update session state so future turns use the corrected languages
        self.state.lang_a = lang_a
        self.state.lang_b = lang_b

        # ── PHASE 3: Parallel NMT → TTS ─────────────────────────────────────
        # NMT_A translates A's transcript INTO lang_B (Speaker B's language)
        # NMT_B translates B's transcript INTO lang_A (Speaker A's language)
        nmt_a_task = self.nmt.translate(
            NMTInput(text=asr_a_out.transcript, src_language=lang_a, tgt_language=lang_b)
        )
        nmt_b_task = self.nmt.translate(
            NMTInput(text=asr_b_out.transcript, src_language=lang_b, tgt_language=lang_a)
        )

        nmt_a_out, nmt_b_out = await asyncio.gather(nmt_a_task, nmt_b_task)

        # TTS_A synthesises in lang_B — Speaker B will hear this
        # TTS_B synthesises in lang_A — Speaker A will hear this
        tts_a_task = self.tts.synthesise(
            TTSInput(text=nmt_a_out.translated_text, language=lang_b,
                     voice_gender=gender_a, audio_format="mp3")
        )
        tts_b_task = self.tts.synthesise(
            TTSInput(text=nmt_b_out.translated_text, language=lang_a,
                     voice_gender=gender_b, audio_format="mp3")
        )

        tts_a_out, tts_b_out = await asyncio.gather(tts_a_task, tts_b_task)

        # ── Assemble results ─────────────────────────────────────────────────
        for_b = PipelineResult(
            source_transcript = asr_a_out.transcript,
            translated_text   = nmt_a_out.translated_text,
            audio_bytes       = tts_a_out.audio_bytes,
            audio_format      = "mp3",
            src_language      = lang_a,
            tgt_language      = lang_b,
            total_latency_ms  = tts_a_out.latency_ms + nmt_a_out.latency_ms + asr_a_out.latency_ms,
        )

        for_a = PipelineResult(
            source_transcript = asr_b_out.transcript,
            translated_text   = nmt_b_out.translated_text,
            audio_bytes       = tts_b_out.audio_bytes,
            audio_format      = "mp3",
            src_language      = lang_b,
            tgt_language      = lang_a,
            total_latency_ms  = tts_b_out.latency_ms + nmt_b_out.latency_ms + asr_b_out.latency_ms,
        )

        return DualPipelineResult(for_speaker_b=for_b, for_speaker_a=for_a)

    # ── Single speaker sends audio (turn-by-turn mode) ────────────────────────

    async def process_speaker_a(
        self, audio_bytes: bytes, audio_format: str = "wav", gender: str = "female"
    ) -> Optional[PipelineResult]:
        """
        Process only Speaker A's audio.
        Returns None if Speaker B's language hasn't been detected yet
        (can't cross-wire without it). The result is buffered server-side.
        """
        asr_out = await self.asr.transcribe(
            ASRInput(audio_bytes=audio_bytes, audio_format=audio_format,
                     language_hint=self.state.lang_a)
        )
        self.state.lang_a = asr_out.detected_language

        if self.state.lang_b is None:
            # Can't translate yet — return None and store transcript
            return None

        nmt_out = await self.nmt.translate(
            NMTInput(text=asr_out.transcript,
                     src_language=self.state.lang_a,
                     tgt_language=self.state.lang_b)
        )
        tts_out = await self.tts.synthesise(
            TTSInput(text=nmt_out.translated_text, language=self.state.lang_b,
                     voice_gender=gender, audio_format="mp3")
        )

        return PipelineResult(
            source_transcript = asr_out.transcript,
            translated_text   = nmt_out.translated_text,
            audio_bytes       = tts_out.audio_bytes,
            audio_format      = "mp3",
            src_language      = self.state.lang_a,
            tgt_language      = self.state.lang_b,
            total_latency_ms  = asr_out.latency_ms + nmt_out.latency_ms + tts_out.latency_ms,
        )

    async def process_speaker_b(
        self, audio_bytes: bytes, audio_format: str = "wav", gender: str = "female"
    ) -> Optional[PipelineResult]:
        """Mirror of process_speaker_a for Speaker B."""
        asr_out = await self.asr.transcribe(
            ASRInput(audio_bytes=audio_bytes, audio_format=audio_format,
                     language_hint=self.state.lang_b)
        )
        self.state.lang_b = asr_out.detected_language

        if self.state.lang_a is None:
            return None

        nmt_out = await self.nmt.translate(
            NMTInput(text=asr_out.transcript,
                     src_language=self.state.lang_b,
                     tgt_language=self.state.lang_a)
        )
        tts_out = await self.tts.synthesise(
            TTSInput(text=nmt_out.translated_text, language=self.state.lang_a,
                     voice_gender=gender, audio_format="mp3")
        )

        return PipelineResult(
            source_transcript = asr_out.transcript,
            translated_text   = nmt_out.translated_text,
            audio_bytes       = tts_out.audio_bytes,
            audio_format      = "mp3",
            src_language      = self.state.lang_b,
            tgt_language      = self.state.lang_a,
            total_latency_ms  = asr_out.latency_ms + nmt_out.latency_ms + tts_out.latency_ms,
        )