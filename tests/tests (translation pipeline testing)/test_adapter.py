"""
tests/test_nmt.py  +  tests/test_tts.py
Run: python tests/test_nmt.py   or  python tests/test_tts.py
"""

# ── test_nmt.py + test_tts.py ────────────────────────────────────────────────
import asyncio, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv()

from adapter.sarvam_nmt import SarvamNMTAdapter
from adapter.sarvam_tts import SarvamTTSAdapter
from pipeline.types import NMTInput, TTSInput

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "output.txt")
_log_file = None

def log(msg=""):
    print(msg, file=_log_file)


async def test_nmt():
    key = os.getenv("SARVAM_API_KEY")
    assert key, "Set SARVAM_API_KEY"

    adapter = SarvamNMTAdapter(key)

    test_cases = [
        ("नमस्ते, आप कैसे हैं?",   "hi-IN", "ta-IN"),
        ("नमस्ते, आप कैसे हैं?",   "hi-IN", "te-IN"),
        ("நான் நலமாக இருக்கிறேன்", "ta-IN", "hi-IN"),
    ]

    for text, src, tgt in test_cases:
        log(f"\nTranslating [{src}→{tgt}]: {text}")
        result = await adapter.translate(NMTInput(text=text, src_language=src, tgt_language=tgt))
        log(f"  ✅ {result.translated_text}  ({result.latency_ms}ms)")
        assert result.translated_text

    # Test short-circuit (same language)
    result = await adapter.translate(NMTInput(text="Hello", src_language="hi-IN", tgt_language="hi-IN"))
    assert result.translated_text == "Hello", "Same-language should short-circuit"
    assert result.latency_ms == 0, "Short-circuit should have 0 latency"
    log("\n✅ Same-language short-circuit works correctly")


async def test_tts():
    key = os.getenv("SARVAM_API_KEY")
    assert key, "Set SARVAM_API_KEY"

    adapter = SarvamTTSAdapter(key)

    test_cases = [
        ("नमस्ते, मैं अच्छा हूँ", "hi-IN", "female"),
        ("வணக்கம், நான் நலமாக இருக்கிறேன்", "ta-IN", "female"),
    ]

    os.makedirs("tests/output", exist_ok=True)

    for text, lang, gender in test_cases:
        log(f"\nSynthesising [{lang}/{gender}]: {text[:30]}...")
        result = await adapter.synthesise(
            TTSInput(text=text, language=lang, voice_gender=gender, audio_format="mp3")
        )
        out_path = f"tests/output/{lang}_{gender}.mp3"
        with open(out_path, "wb") as f:
            f.write(result.audio_bytes)
        log(f"  ✅ Audio saved to {out_path}  ({result.latency_ms}ms, {len(result.audio_bytes)} bytes)")
        assert len(result.audio_bytes) > 0

    # Test unsupported language
    try:
        await adapter.synthesise(TTSInput(text="test", language="doi-IN"))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        log(f"\n✅ Unsupported language correctly raises ValueError: {e}")


if __name__ == "__main__":
    _log_file = open(OUTPUT_FILE, "a", encoding="utf-8")  # append after ASR results
    try:
        asyncio.run(test_nmt())
        asyncio.run(test_tts())
    finally:
        _log_file.close()
    print(f"Done. Results appended to {OUTPUT_FILE}")