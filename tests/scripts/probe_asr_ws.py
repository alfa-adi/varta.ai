"""
probe_asr_ws.py
───────────────
Smoke probe for the Sarvam Saaras v3 WebSocket ASR endpoint.

Loads a real Hindi webm sample, transcodes it to 16kHz mono WAV (as
the WS API requires), streams it to wss://api.sarvam.ai/speech-to-text/ws
in JSON+base64 frames, and prints every raw response frame.

Usage:
    python test/scripts/probe_asr_ws.py

Verified response format (2026-08-04):
    {
      "type": "data",
      "data": {
        "request_id": "...",
        "transcript": "...",
        "language_code": "hi-IN",
        "language_probability": null,
        "metrics": {
          "audio_duration": 8.48,
          "processing_latency": 0.252
        }
      }
    }
"""

import asyncio
import base64
import io
import json
import os
import sys
import time

# Allow running from any directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

import websockets

API_KEY = os.environ["SARVAM_API_KEY"]
SAMPLE  = os.path.join(PROJECT_ROOT, "test", "datasets", "hi", "sample_1.webm")
WS_URL  = "wss://api.sarvam.ai/speech-to-text/ws"


def _to_wav(raw: bytes, fmt: str = "webm") -> bytes:
    """Convert audio bytes to 16kHz mono WAV using pydub."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(raw), format=fmt)
    seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    return buf.getvalue()


async def probe():
    if not os.path.exists(SAMPLE):
        print(f"❌ Sample not found: {SAMPLE}")
        return

    with open(SAMPLE, "rb") as f:
        raw = f.read()

    print(f"Converting {len(raw):,} bytes webm → WAV ...")
    wav = _to_wav(raw)
    print(f"WAV size: {len(wav):,} bytes\n")

    url = f"{WS_URL}?model=saaras:v3&mode=transcribe&sample_rate=16000"
    headers = {"Api-Subscription-Key": API_KEY}
    print(f"Connecting to: {url}")

    t0 = time.perf_counter()
    async with websockets.connect(url, additional_headers=headers, open_timeout=15) as ws:
        print("Connected. Sending JSON+base64 audio frames ...\n")

        chunk_size = 65536
        for i in range(0, len(wav), chunk_size):
            chunk = wav[i : i + chunk_size]
            await ws.send(json.dumps({
                "audio": {
                    "data": base64.b64encode(chunk).decode("utf-8"),
                    "sample_rate": "16000",
                    "encoding": "audio/wav",
                }
            }))

        await ws.send(json.dumps({"type": "flush"}))
        print("→ flush sent. Waiting for response frames ...\n")

        while True:
            try:
                raw_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw_msg)
                print(f"FRAME: {json.dumps(msg, indent=2, ensure_ascii=False)}")
                print("---")
                if msg.get("type") == "data" and msg.get("data", {}).get("transcript"):
                    break  # Got transcript — done
            except asyncio.TimeoutError:
                print("(timeout — no more frames)")
                break
            except websockets.exceptions.ConnectionClosedOK:
                print("(server closed connection)")
                break

    latency_ms = int((time.perf_counter() - t0) * 1000)
    print(f"\nTotal round-trip: {latency_ms}ms")


if __name__ == "__main__":
    asyncio.run(probe())
