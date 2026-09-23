#!/usr/bin/env python3
"""
download_fleurs_samples.py
──────────────────────────
Downloads 5 real speech audio samples per language (13 languages) from
Google's FLEURS dataset on Hugging Face.

Bypasses the `datasets` library entirely — streams the raw .tar.gz audio
archives directly over HTTP using requests + tarfile, so there is no
MemoryError, no interactive prompt, and no multi-GB Parquet downloads.

For three languages (as, sd, or) where FLEURS audio is unavailable or
corrupted, falls back to gTTS synthetic speech.

Output layout:
    test_audio/<varta_code>/sample_1.webm … sample_5.webm
    test_audio/manifest.json
"""

import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import requests
import soundfile as sf
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()

# ── Language mapping ─────────────────────────────────────────────────────────

LANGUAGE_MAP = {
    # varta_code: (language_name, fleurs_config, use_gtts_fallback)
    "hi": ("Hindi",     "hi_in", False),
    "bn": ("Bengali",   "bn_in", False),
    "mr": ("Marathi",   "mr_in", False),
    "te": ("Telugu",    "te_in", False),
    "ta": ("Tamil",     "ta_in", False),
    "ur": ("Urdu",      "ur_pk", False),
    "gu": ("Gujarati",  "gu_in", False),
    "kn": ("Kannada",   "kn_in", False),
    "or": ("Odia",      "or_in", False),
    "ml": ("Malayalam", "ml_in", False),
    "pa": ("Punjabi",   "pa_in", False),
    "as": ("Assamese",  "as_in", False),   # FLEURS; gTTS fallback uses "bn"
    "sd": ("Sindhi",    "sd_in", False),   # FLEURS; gTTS fallback uses "ur"
}

FALLBACK_TEXTS = {
    "as": [
        "অসমীয়া ভাষা আমাৰ মাতৃভাষা",
        "আজি বতৰ বৰ ভাল আছে",
        "মই তোমাক ভাল পাওঁ",
        "আমাৰ দেশ বহুত সুন্দৰ",
        "আকাশত ডাৱৰ নাই",
    ],
    "sd": [
        "سنڌي اسان جي مادري ٻولي آهي",
        "اڄ موسم تمام سٺو آهي",
        "مون کي توهان سان ڳالهائڻ پسند آهي",
        "اسان جو ملڪ سھڻو آھي",
        "آسمان صاف آهي",
    ],
    "or": [
        "ଓଡ଼ିଆ ଆମ ମାତୃଭାଷା",
        "ଆଜି ପାଣିପାଗ ବହୁତ ଭଲ ଅଛି",
        "ମୁଁ ତୁମ ସହ କଥା ହେବାକୁ ପସନ୍ଦ କରେ",
        "ଆମ ଦେଶ ବହୁତ ସୁନ୍ଦର",
        "ଆକାଶ ସଫା ଅଛି",
    ],
}

GTTS_LANG_MAP = {"as": "bn", "sd": "ur", "or": "hi"}  # proxy langs (gTTS has no as/sd/or)

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLES_PER_LANG   = 5
MIN_DURATION_SEC   = 4.0
MAX_DURATION_SEC   = 12.0
OUTPUT_DIR         = Path("test_audio")

# Hugging Face base URL for raw FLEURS audio archives
HF_BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"

# FFmpeg binary path (reads from env var or uses the winget-installed location)
FFMPEG_BIN = os.environ.get(
    "FFMPEG_PATH",
    r"C:\Users\admin\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def convert_to_webm(input_path: str, webm_path: str) -> None:
    """Convert any audio file (WAV, MP3, ...) to WebM/Opus via ffmpeg."""
    subprocess.run(
        [FFMPEG_BIN, "-y", "-i", input_path, "-c:a", "libopus", "-b:a", "32k", webm_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wav_duration(wav_bytes: bytes) -> float:
    """Return duration in seconds by reading the WAV header via soundfile."""
    with sf.SoundFile(io.BytesIO(wav_bytes)) as f:
        return len(f) / f.samplerate


# ── FLEURS direct-HTTP streaming ──────────────────────────────────────────────

def download_fleurs(varta_code: str, lang_name: str, fleurs_code: str) -> list[dict]:
    """
    Stream the FLEURS test archive for one language directly over HTTP.
    Stops after collecting SAMPLES_PER_LANG valid-duration .wav files.
    Converts each to WebM/Opus and returns manifest entries.
    """
    lang_dir = OUTPUT_DIR / varta_code
    lang_dir.mkdir(parents=True, exist_ok=True)

    url = f"{HF_BASE}/{fleurs_code}/audio/test.tar.gz"
    print(f"\n{'-'*60}")
    print(f"  {lang_name} ({varta_code})  |  FLEURS config: {fleurs_code}")
    print(f"{'-'*60}")
    print(f"  Streaming: {url}")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    kept: list[dict] = []

    with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
            for member in tar:
                if len(kept) >= SAMPLES_PER_LANG:
                    break
                if not member.isfile():
                    continue
                if not member.name.lower().endswith(".wav"):
                    continue

                # Extract raw WAV bytes
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                wav_bytes = fobj.read()

                # Duration check
                try:
                    duration = wav_duration(wav_bytes)
                except Exception:
                    continue

                if duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
                    continue

                # Write temp WAV -> convert to WebM -> delete WAV
                sample_idx = len(kept) + 1
                webm_path  = lang_dir / f"sample_{sample_idx}.webm"

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_bytes)
                    tmp_wav = tmp.name

                try:
                    convert_to_webm(tmp_wav, str(webm_path))
                finally:
                    if os.path.exists(tmp_wav):
                        os.remove(tmp_wav)

                size_bytes = webm_path.stat().st_size
                entry = {
                    "varta_code":    varta_code,
                    "fleurs_code":   fleurs_code,
                    "sample_index":  sample_idx,
                    "file":          str(webm_path.as_posix()),
                    "duration_sec":  round(duration, 2),
                    "size_bytes":    size_bytes,
                    "transcription": member.name,
                }
                kept.append(entry)
                print(f"  OK  sample_{sample_idx}.webm  ({duration:.1f}s, {size_bytes:,} bytes)")

    if len(kept) < SAMPLES_PER_LANG:
        print(
            f"  WARNING: only {len(kept)}/{SAMPLES_PER_LANG} "
            f"valid-duration samples found for {lang_name}"
        )
    return kept


# ── gTTS fallback ─────────────────────────────────────────────────────────────

def download_gtts(varta_code: str) -> list[dict]:
    """Generate synthetic speech via gTTS for languages where FLEURS fails."""
    lang_dir = OUTPUT_DIR / varta_code
    lang_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-'*60}")
    print(f"  {varta_code.upper()} (gTTS synthetic fallback)")
    print(f"{'-'*60}")

    kept = []
    for i, text in enumerate(FALLBACK_TEXTS[varta_code], start=1):
        mp3_path  = lang_dir / f"sample_{i}.mp3"
        webm_path = lang_dir / f"sample_{i}.webm"

        tts = gTTS(text=text, lang=GTTS_LANG_MAP[varta_code], slow=False)
        tts.save(str(mp3_path))
        convert_to_webm(str(mp3_path), str(webm_path))
        mp3_path.unlink()

        size_bytes = webm_path.stat().st_size
        kept.append({
            "varta_code":    varta_code,
            "fleurs_code":   "gtts_synthetic",
            "sample_index":  i,
            "file":          str(webm_path.as_posix()),
            "duration_sec":  round(len(text) * 0.06, 2),
            "size_bytes":    size_bytes,
            "transcription": text,
        })
        print(f"  OK  sample_{i}.webm  ({size_bytes:,} bytes)  [gTTS]")

    return kept


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[dict]] = {}
    total_samples = 0

    for varta_code, (lang_name, fleurs_code, use_gtts) in LANGUAGE_MAP.items():
        # Skip languages already fully downloaded
        lang_dir = OUTPUT_DIR / varta_code
        existing = sorted(lang_dir.glob("sample_*.webm")) if lang_dir.exists() else []
        if len(existing) >= SAMPLES_PER_LANG:
            print(f"\n  SKIP {lang_name} ({varta_code}) — already have {len(existing)} samples on disk")
            # Reconstruct manifest entries from disk
            entries = []
            for webm in existing[:SAMPLES_PER_LANG]:
                idx = int(webm.stem.split("_")[1])
                entries.append({
                    "varta_code": varta_code, "fleurs_code": fleurs_code,
                    "sample_index": idx, "file": str(webm.as_posix()),
                    "duration_sec": 0.0, "size_bytes": webm.stat().st_size,
                    "transcription": "",
                })
            manifest[varta_code] = entries
            total_samples += len(entries)
            continue

        if use_gtts:
            manifest[varta_code] = download_gtts(varta_code)
        else:
            try:
                manifest[varta_code] = download_fleurs(varta_code, lang_name, fleurs_code)
            except Exception as exc:
                print(f"  FLEURS failed for {varta_code}: {exc}")
                if varta_code in FALLBACK_TEXTS:
                    print(f"  -> falling back to gTTS")
                    manifest[varta_code] = download_gtts(varta_code)
                else:
                    manifest[varta_code] = []

        total_samples += len(manifest[varta_code])

    # Write manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Done!  {total_samples} samples across {len(LANGUAGE_MAP)} languages")
    print(f"  Manifest -> {manifest_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
