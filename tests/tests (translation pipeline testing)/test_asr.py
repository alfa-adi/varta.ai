"""
tests/test_asr.py
─────────────────
Smoke test for the Sarvam ASR adapter.
Run with: python -m pytest tests/test_asr.py -v
Or just:  python tests/test_asr.py

Requires: SARVAM_API_KEY in .env
Requires: a real audio file at tests/fixtures/sample_hi.wav
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from adapter.sarvam_asr import SarvamASRAdapter
from pipeline.types import ASRInput

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "output.txt")
_log_file = None

def log(msg=""):
    print(msg, file=_log_file)


async def test_asr_with_file():
    api_key = os.getenv("SARVAM_API_KEY")
    assert api_key, "Set SARVAM_API_KEY in .env"

    adapter = SarvamASRAdapter(api_key)

    # Use any small Hindi audio file you have
    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "TestingAudio_AamirKhan_6.wav")
    if not os.path.exists(fixture):
        log("[!] No fixture file found at tests/fixtures/TestingAudio_AamirKhan_6.wav")
        log("   Create this file with a short Hindi audio clip to test.")
        return

    with open(fixture, "rb") as f:
        audio_bytes = f.read()

    log("Calling Sarvam ASR (Saaras v3)...")
    result = await adapter.transcribe(
        ASRInput(audio_bytes=audio_bytes, audio_format="wav", language_hint="hi-IN")
    )

    log(f"✅  Transcript:         {result.transcript}")
    log(f"    Detected language:  {result.detected_language}")
    log(f"    Confidence:         {result.confidence:.2f}")
    log(f"    Latency:            {result.latency_ms}ms")
    log(f"    Model:              {result.model_id}")

    assert result.transcript, "Transcript should not be empty"
    assert result.detected_language, "Detected language should not be empty"


async def test_asr_auto_detect():
    """Test that omitting language_hint triggers auto-detection."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key: return

    adapter = SarvamASRAdapter(api_key)

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_hi.wav")
    if not os.path.exists(fixture): return

    with open(fixture, "rb") as f:
        audio_bytes = f.read()

    log("\nCalling Sarvam ASR with NO language hint (auto-detect)...")
    result = await adapter.transcribe(
        ASRInput(audio_bytes=audio_bytes, audio_format="wav", language_hint=None)
    )

    log(f"✅  Auto-detected language: {result.detected_language}")
    log(f"    Transcript: {result.transcript}")


if __name__ == "__main__":
    _log_file = open(OUTPUT_FILE, "w", encoding="utf-8")
    try:
        asyncio.run(test_asr_with_file())
        asyncio.run(test_asr_auto_detect())
    finally:
        _log_file.close()
    print(f"Done. Results written to {OUTPUT_FILE}")