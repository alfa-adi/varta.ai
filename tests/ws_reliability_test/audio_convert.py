"""
test/ws_reliability_test/audio_convert.py
──────────────────────────────────────────
Converts FLEURS WebM audio fixtures to the exact PCM format that the
V2 WebSocket protocol expects.

V2 audio protocol (confirmed from pcm-processor.js + sarvam_asr.py):
  Format:      pcm_s16le (signed 16-bit little-endian)
  Sample rate: 16,000 Hz
  Channels:    1 (mono)
  Chunk size:  320 samples = 640 bytes = 20ms per WebSocket binary frame

The production browser pipeline:
  getUserMedia() → AudioContext(16kHz) → AudioWorklet → Int16Array(320) → WS binary frame

This module reproduces that conversion for test fixtures:
  WebM file → pydub decode → 16kHz mono pcm_s16le → 640-byte chunks

Note: pydub is already in requirements.txt. ffmpeg or libavcodec must be
installed on the system for WebM/Opus decoding (pydub calls ffmpeg under the hood).
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Optional


# ── Audio constants (must match pcm-processor.js and sarvam_asr.py) ──────────

SAMPLE_RATE   = 16_000   # Hz — Saaras v3-realtime requirement
CHANNELS      = 1        # mono
SAMPLE_WIDTH  = 2        # bytes (16-bit signed integer)
CHUNK_SAMPLES = 320      # 320 Int16 samples = 20ms at 16kHz
CHUNK_BYTES   = CHUNK_SAMPLES * SAMPLE_WIDTH   # 640 bytes per WebSocket frame


# ── Conversion ────────────────────────────────────────────────────────────────

def webm_to_pcm_chunks(webm_path: Path) -> tuple[list[bytes], float]:
    """
    Decode a WebM audio file and split into 640-byte PCM chunks.

    Returns:
        chunks (list[bytes]):  Each chunk is exactly CHUNK_BYTES (640) bytes.
                               The last chunk is zero-padded if needed.
        duration_sec (float): Estimated audio duration from PCM length.

    Raises:
        FileNotFoundError:    If webm_path does not exist.
        RuntimeError:         If pydub or ffmpeg fails to decode the file.
    """
    if not webm_path.exists():
        raise FileNotFoundError(f"Fixture not found: {webm_path}")

    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "pydub is required for WebM decoding. "
            "It is listed in requirements.txt — run: pip install pydub"
        )

    try:
        # pydub calls ffmpeg internally for WebM/Opus
        seg = AudioSegment.from_file(str(webm_path), format="webm")
    except Exception as exc:
        # Some systems may need explicit format hint
        try:
            seg = AudioSegment.from_file(str(webm_path), format="ogg")
        except Exception:
            raise RuntimeError(
                f"Failed to decode WebM fixture '{webm_path}': {exc}\n"
                "Ensure ffmpeg is installed: https://ffmpeg.org/download.html"
            ) from exc

    # Resample to match Saaras v3-realtime requirements
    seg = (
        seg
        .set_frame_rate(SAMPLE_RATE)    # 16kHz
        .set_channels(CHANNELS)          # mono
        .set_sample_width(SAMPLE_WIDTH)  # 16-bit
    )

    raw_pcm: bytes = seg.raw_data   # already pcm_s16le after set_sample_width(2)

    duration_sec = len(raw_pcm) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)

    # Split into CHUNK_BYTES-sized frames
    chunks: list[bytes] = []
    for offset in range(0, len(raw_pcm), CHUNK_BYTES):
        chunk = raw_pcm[offset : offset + CHUNK_BYTES]
        if len(chunk) < CHUNK_BYTES:
            chunk = chunk + b"\x00" * (CHUNK_BYTES - len(chunk))  # zero-pad last chunk
        chunks.append(chunk)

    return chunks, duration_sec


# ── Validation ────────────────────────────────────────────────────────────────

def validate_pcm_audio(pcm_bytes: bytes) -> tuple[bool, Optional[str]]:
    """
    Validate raw pcm_s16le bytes.

    Returns:
        (True, None)           — valid audio
        (False, error_message) — invalid

    Checks:
      1. Non-empty
      2. Even byte count (Int16 alignment)
      3. Non-silent (at least one sample with |amplitude| > 100)
    """
    if not pcm_bytes:
        return False, "PCM bytes are empty"

    if len(pcm_bytes) % 2 != 0:
        return False, f"PCM byte count {len(pcm_bytes)} is not even (not Int16-aligned)"

    sample_count = len(pcm_bytes) // 2
    samples = struct.unpack_from(f"<{sample_count}h", pcm_bytes)

    max_amp = max(abs(s) for s in samples)
    if max_amp < 100:
        return False, f"PCM appears silent (max amplitude: {max_amp})"

    return True, None


def decode_received_audio(audio_bytes: bytes) -> tuple[bool, Optional[str], int, float]:
    """
    Validate and decode received TTS audio (raw linear16 PCM, 24kHz mono).

    The server sends Bulbul v3 audio as raw pcm_s16le at 24kHz.
    The browser interprets it as Int16Array → Float32Array → AudioContext(24kHz).

    Returns:
        success (bool)
        error (Optional[str])
        sample_count (int)
        duration_sec (float)
    """
    TTS_SAMPLE_RATE = 24_000

    if not audio_bytes:
        return False, "No audio bytes received", 0, 0.0

    if len(audio_bytes) % 2 != 0:
        return False, f"Audio byte count {len(audio_bytes)} is not Int16-aligned", 0, 0.0

    sample_count = len(audio_bytes) // 2
    duration_sec = sample_count / TTS_SAMPLE_RATE

    if sample_count == 0:
        return False, "Zero audio samples after decode", 0, 0.0

    # Verify we can parse the bytes as Int16 without exception
    try:
        samples = struct.unpack_from(f"<{sample_count}h", audio_bytes)
        max_amp = max(abs(s) for s in samples)
        if max_amp == 0:
            return False, "TTS audio appears completely silent", sample_count, duration_sec
    except struct.error as exc:
        return False, f"Struct unpack failed: {exc}", 0, 0.0

    return True, None, sample_count, duration_sec


# ── Fixture cache (avoid re-decoding same file multiple times) ────────────────

_pcm_cache: dict[str, tuple[list[bytes], float]] = {}


def get_pcm_chunks_cached(webm_path: Path) -> tuple[list[bytes], float]:
    """
    Return cached PCM chunks for a fixture, decoding only on first call.
    This avoids re-running pydub/ffmpeg for every turn using the same file.
    """
    key = str(webm_path.resolve())
    if key not in _pcm_cache:
        _pcm_cache[key] = webm_to_pcm_chunks(webm_path)
    return _pcm_cache[key]


def clear_pcm_cache():
    """Clear the fixture PCM cache (call between test phases if needed)."""
    _pcm_cache.clear()
