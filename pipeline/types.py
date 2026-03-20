"""
pipeline/types.py
─────────────────
Universal data contracts for all three adapter layers.
Every adapter accepts one of these as input and returns one as output.
The rest of the system ONLY speaks these types — never raw API shapes.
"""

from dataclasses import dataclass, field
from typing import Optional


# ── ASR ──────────────────────────────────────────────────────────────────────

@dataclass
class ASRInput:
    """What you hand to any ASR adapter."""
    audio_bytes: bytes          # Raw audio content
    audio_format: str = "wav"   # wav | mp3 | ogg | webm etc.
    language_hint: Optional[str] = None  # BCP-47 e.g. "hi-IN". None = auto-detect.
    mode: str = "transcribe"    # transcribe | codemix | verbatim | translit


@dataclass
class ASROutput:
    """What every ASR adapter returns — regardless of which API is behind it."""
    transcript: str
    detected_language: str      # BCP-47 code the ASR identified (e.g. "hi-IN")
    confidence: float           # 0.0–1.0 language detection confidence
    latency_ms: int
    model_id: str               # Which registry entry produced this


# ── NMT ──────────────────────────────────────────────────────────────────────

@dataclass
class NMTInput:
    """What you hand to any NMT adapter."""
    text: str
    src_language: str           # BCP-47 source (e.g. "hi-IN")
    tgt_language: str           # BCP-47 target (e.g. "ta-IN")


@dataclass
class NMTOutput:
    """What every NMT adapter returns."""
    translated_text: str
    src_language: str
    tgt_language: str
    latency_ms: int
    model_id: str


# ── TTS ──────────────────────────────────────────────────────────────────────

@dataclass
class TTSInput:
    """What you hand to any TTS adapter."""
    text: str
    language: str               # BCP-47 (e.g. "ta-IN")
    voice_gender: str = "female"
    audio_format: str = "mp3"
    sample_rate: int = 22050
    pace: float = 1.0           # 0.5–2.0


@dataclass
class TTSOutput:
    """What every TTS adapter returns."""
    audio_bytes: bytes          # Decoded audio (NOT base64)
    audio_format: str
    language: str
    latency_ms: int
    model_id: str


# ── Pipeline result (used by dual.py) ────────────────────────────────────────

@dataclass
class PipelineResult:
    """Full result from one direction of a translation pipeline."""
    source_transcript: str
    translated_text: str
    audio_bytes: bytes
    audio_format: str
    src_language: str
    tgt_language: str
    total_latency_ms: int