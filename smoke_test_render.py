"""
smoke_test_render.py
--------------------
Quick terminal smoke test against the deployed varta.ai instance.
Sends test audio to both Speaker A and Speaker B endpoints (the server
uses a dual pipeline that buffers until both speakers have spoken).

Usage:
    python smoke_test_render.py
    python smoke_test_render.py --url http://localhost:8000
"""

import io
import json
import os
import sys
import time
import uuid

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Test audio files
SAMPLE_HI  = os.path.join(PROJECT_ROOT, "test", "datasets", "hi", "sample_1.webm")
SAMPLE_TA  = os.path.join(PROJECT_ROOT, "test", "datasets", "ta", "sample_1.webm")
DEFAULT_URL = "https://varta-ai-1-7wwd.onrender.com"


def send_audio(session, url, endpoint, audio_path, src_lang, tgt_lang, session_id):
    """Send audio to one speaker endpoint and return (status_code, body, elapsed_ms)."""
    import requests

    ext = audio_path.rsplit(".", 1)[-1].lower()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    files = {"audio": (f"recording.{ext}", io.BytesIO(audio_bytes), f"audio/{ext}")}
    data = {
        "session_id":   session_id,
        "src_language": src_lang,
        "tgt_language": tgt_lang,
    }

    t0 = time.perf_counter()
    resp = requests.post(f"{url}{endpoint}", files=files, data=data, timeout=60)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text[:500]}

    return resp.status_code, body, elapsed_ms, len(audio_bytes)


def print_result(label, status, body, elapsed_ms, audio_size):
    """Pretty-print one endpoint's result."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Status     : {status}")
    print(f"  Audio sent : {audio_size:,} bytes")
    print(f"  Round trip : {elapsed_ms}ms")

    if body.get("status") == "buffered":
        print(f"  Result     : BUFFERED (waiting for other speaker)")
        return

    print(f"  Transcript : {body.get('transcript', '(none)')}")
    print(f"  Translation: {body.get('translation', '(none)')}")

    timing = body.get("timing", {})
    asr = timing.get("asr", {})
    nmt = timing.get("nmt", {})
    tts = timing.get("tts", {})
    server = timing.get("server", {})

    if asr:
        print(f"\n  Server-side timing:")
        print(f"    ASR   : {asr.get('total_ms', '?')}ms")
        print(f"    NMT   : {nmt.get('total_ms', '?')}ms")
        print(f"    TTS   : {tts.get('total_ms', '?')}ms")
        print(f"    Total : {server.get('total_ms', '?')}ms")

    import base64
    audio_b64 = body.get("audio_base64", "")
    if audio_b64:
        audio_out = base64.b64decode(audio_b64)
        print(f"    Audio out: {len(audio_out):,} bytes ({body.get('audio_format', '?')})")

    # Check for deferred result (other speaker's buffered transcript)
    deferred = body.get("deferred")
    if deferred:
        print(f"\n  --- Deferred (buffered speaker's result) ---")
        print(f"  Transcript : {deferred.get('transcript', '(none)')}")
        print(f"  Translation: {deferred.get('translation', '(none)')}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test varta.ai")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    url = args.url
    session_id = str(uuid.uuid4())

    print(f"varta.ai Smoke Test")
    print(f"URL       : {url}")
    print(f"Session   : {session_id}")

    # Check audio files exist
    for f in [SAMPLE_HI, SAMPLE_TA]:
        if not os.path.exists(f):
            print(f"ERROR: Missing audio file: {f}")
            sys.exit(1)

    # Step 1: Speaker A (Hindi)
    print(f"\n>> Sending Speaker A (Hindi)...")
    s1_status, s1_body, s1_ms, s1_size = send_audio(
        None, url, "/translate/speaker_a", SAMPLE_HI, "hi-IN", "ta-IN", session_id
    )
    print_result("Speaker A (hi-IN -> ta-IN)", s1_status, s1_body, s1_ms, s1_size)

    # Step 2: Speaker B (Tamil) — this should trigger both translations
    print(f"\n>> Sending Speaker B (Tamil)...")
    s2_status, s2_body, s2_ms, s2_size = send_audio(
        None, url, "/translate/speaker_b", SAMPLE_TA, "ta-IN", "hi-IN", session_id
    )
    print_result("Speaker B (ta-IN -> hi-IN)", s2_status, s2_body, s2_ms, s2_size)

    # Summary
    print(f"\n{'='*55}")
    total_ms = s1_ms + s2_ms
    has_transcript = bool(s2_body.get("transcript"))
    print(f"  Total time : {total_ms}ms ({s1_ms}ms + {s2_ms}ms)")
    print(f"  Result     : {'PASS' if has_transcript else 'FAIL'}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
