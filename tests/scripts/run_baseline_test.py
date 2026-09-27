#!/usr/bin/env python3
"""
run_baseline_test.py
────────────────────
Automates a 12-language-to-Hindi latency test against a deployed varta.ai
instance using Playwright browser automation.

For each of 12 spoke languages, runs one "conversation" of 5 turns
(alternating speakers), captures browser-side and server-side timing,
and writes every measurement into MongoDB.

Prerequisites:
    pip install playwright pymongo python-dotenv
    python -m playwright install chromium

Environment variables (read from .env):
    MONGO_URL          — MongoDB connection string (required)
    MONGO_TEST_DB      — Database name (default: varta_test_data)
    TARGET_URL         — Deployed varta.ai URL (default: http://localhost:8000)
"""

import asyncio
import io
import sys

# Force UTF-8 output on Windows (cp1252 can't handle arrows/emojis)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright, Response
from pymongo import MongoClient

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

MONGO_URL     = os.getenv("MONGO_URL")
MONGO_TEST_DB = os.getenv("MONGO_TEST_DB", "varta_test_data")
TARGET_URL    = os.getenv("TARGET_URL") or os.getenv("RENDER_URL", "http://localhost:8000")

if not MONGO_URL:
    print("ERROR: MONGO_URL environment variable is not set.")
    sys.exit(1)

# Cost tracking — updated based on real Sarvam API billing data (July 10, 2026)
# Total cost: ₹74.14 for 394 requests (~131 full pipeline turns)
COST_PER_ACTIVATION_INR = 0.56

# Rate-limit pacing: backend enforces 10 req/min per IP → 6s minimum gap.
# Using 7s for safety margin. Do NOT reduce this value.
INTER_TURN_DELAY_SEC = 7

TURNS_PER_CONVERSATION = 10  # 5 spoke + 5 Hindi = all samples used
HUB_LANGUAGE = "hi"

SPOKE_LANGUAGES = [
    "bn", "mr", "te", "ta", "ur", "gu",
    "kn", "or", "ml", "pa", "as", "sd",
]

MANIFEST_PATH = Path("test/datasets/manifest.json")


# ── Load audio manifest ──────────────────────────────────────────────────────

def load_manifest() -> dict:
    """Load the test/datasets/manifest.json file."""
    with open(MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)


# ── Browser timing extraction ────────────────────────────────────────────────

def extract_browser_timing(timing: dict) -> dict:
    """
    Extract browser-side timing from Playwright's request.timing dict.

    Playwright timing fields:
      - startTime: absolute ms since epoch
      - All others (connectStart, connectEnd, requestStart, responseStart,
        responseEnd, etc.): offsets in ms relative to startTime (-1 if n/a)

    So total_ms = responseEnd (already relative to startTime).
    """
    connect_start  = timing.get("connectStart", -1)
    connect_end    = timing.get("connectEnd", -1)
    secure_start   = timing.get("secureConnectionStart", -1)
    request_start  = timing.get("requestStart", -1)
    response_start = timing.get("responseStart", -1)
    response_end   = timing.get("responseEnd", -1)

    def safe_diff(a, b):
        """Return a - b if both are valid (not -1), else None."""
        if a == -1 or b == -1:
            return None
        return round(float(a - b), 2)

    def safe_val(v):
        """Return v as float if valid, else None."""
        if v == -1:
            return None
        return round(float(v), 2)

    tcp_connect_ms   = safe_diff(connect_end, connect_start)
    tls_handshake_ms = safe_diff(connect_end, secure_start)
    # ttfb_ms = Time To First Byte = responseStart - requestStart.
    # This is the full round trip from request sent to first response byte.
    # It includes: upload + ASR + NMT + TTS + network latency.
    # Previously mislabelled as "upload_ms" which was misleading.
    ttfb_ms          = safe_diff(response_start, request_start)
    download_ms      = safe_diff(response_end, response_start)
    # responseEnd is already relative to startTime, so it IS the total duration
    total_ms         = safe_val(response_end)

    return {
        "tcp_connect_ms":   tcp_connect_ms,
        "tls_handshake_ms": tls_handshake_ms,
        "ttfb_ms":          ttfb_ms if ttfb_ms is not None else 0.0,
        "download_ms":      download_ms if download_ms is not None else 0.0,
        "total_ms":         total_ms if total_ms is not None else 0.0,
    }


def extract_server_timing(body: dict) -> dict:
    """
    Extract server-side timing from the response body.

    IMPORTANT: The exact key path should be confirmed against a real response
    payload before relying on it. Based on server.py analysis, the structure is:
        body["timing"]["server"]["total_ms"]
        body["timing"]["asr"]["total_ms"]    → mapped to asr_ms
        body["timing"]["nmt"]["total_ms"]    → mapped to nmt_ms
        body["timing"]["tts"]["total_ms"]    → mapped to tts_ms
        body["timing"]["asr"]["tcp_ms"]      → mapped to sarvam_tcp_ms (first API call)

    Using .get() with safe fallbacks so a missing key doesn't crash the run.
    """
    timing = body.get("timing", {})
    server = timing.get("server", {})
    asr    = timing.get("asr", {})
    nmt    = timing.get("nmt", {})
    tts    = timing.get("tts", {})

    return {
        "sarvam_tcp_ms":     float(asr.get("tcp_ms", 0)),
        "asr_ms":            float(asr.get("total_ms", 0)),
        "nmt_ms":            float(nmt.get("total_ms", 0)),
        "tts_ms":            float(tts.get("total_ms", 0)),
        "parse_ms":          float(asr.get("parse_ms", 0)),
        "session_load_ms":   float(server.get("session_load_ms", 0)),
        "pipeline_build_ms": float(server.get("pipeline_build_ms", 0)),
        "response_build_ms": float(server.get("response_build_ms", 0)),
        "state_save_ms":     float(server.get("state_save_ms", 0)),
        "log_write_ms":      float(server.get("log_write_ms", 0)),
        "total_ms":          float(server.get("total_ms", 0)),
    }


# ── Main test runner ──────────────────────────────────────────────────────────

async def run_test():
    """Execute the full 12-language × 5-turn baseline latency test."""

    # ── Connect to MongoDB ────────────────────────────────────────
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    try:
        client.server_info()
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB — {e}")
        sys.exit(1)

    db = client[MONGO_TEST_DB]
    print(f"  MongoDB connected → {MONGO_TEST_DB}")

    # ── Load audio manifest ───────────────────────────────────────
    manifest = load_manifest()
    print(f"  Manifest loaded: {len(manifest)} languages")

    if HUB_LANGUAGE not in manifest:
        print(f"ERROR: Hub language '{HUB_LANGUAGE}' not found in manifest.")
        sys.exit(1)

    # ── Generate session ID and insert test_sessions doc ──────────
    session_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    session_doc = {
        "session_id": session_id,
        "test_name":  f"baseline_12lang_to_{HUB_LANGUAGE}",
        "test_type":  "browser_playwright",
        "config": {
            "topology":        "spoke_to_hub",
            "hub_language":    HUB_LANGUAGE,
            "spoke_languages": SPOKE_LANGUAGES,
            "turns_per_pair":  TURNS_PER_CONVERSATION,
            "audio_source":    "fleurs_test_split",
        },
        "environment": {
            "target_url":         TARGET_URL,
            "browser":            "chromium",
            "playwright_version": "",  # filled below
            "test_mode_flag":     True,
        },
        "started_at":   started_at,
        "completed_at": None,
        "counts":       {},
        "cost":         {},
        "status":       "running",
        "notes":        None,
    }

    db["test_sessions"].insert_one(session_doc)
    print(f"\n  Session: {session_id}")
    print(f"  Status:  running")
    print(f"  URL:     {TARGET_URL}?testMode=true")

    # ── Launch browser ────────────────────────────────────────────
    total_successful = 0
    total_failed     = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        context = await browser.new_context()
        page    = await context.new_page()

        # Update playwright version in session doc
        db["test_sessions"].update_one(
            {"session_id": session_id},
            {"$set": {"environment.playwright_version": pw.chromium.name}},
        )

        # Navigate to the app with testMode
        await page.goto(f"{TARGET_URL}?testMode=true", wait_until="networkidle")
        print(f"  Page loaded with testMode=true\n")

        # ── Iterate over spoke languages ──────────────────────────
        for spoke_idx, spoke_lang in enumerate(SPOKE_LANGUAGES):
            conversation_id = str(uuid.uuid4())
            conv_started_at = datetime.now(timezone.utc)

            spoke_samples = manifest.get(spoke_lang, [])
            hub_samples   = manifest.get(HUB_LANGUAGE, [])

            if not spoke_samples or not hub_samples:
                print(f"  SKIP {spoke_lang}: no audio samples in manifest")
                continue

            # Insert conversation doc
            conv_doc = {
                "conversation_id": conversation_id,
                "session_id":      session_id,
                "language_a":      spoke_lang,
                "language_b":      HUB_LANGUAGE,
                "turn_count":      TURNS_PER_CONVERSATION,
                "started_at":      conv_started_at,
                "completed_at":    None,
                "status":          "running",
                "aggregates":      {},
            }
            db["conversations"].insert_one(conv_doc)

            print(f"{'─'*60}")
            print(f"  Conversation {spoke_idx+1}/12: {spoke_lang} ↔ {HUB_LANGUAGE}")
            print(f"  ID: {conversation_id}")
            print(f"{'─'*60}")

            conv_successful = 0
            conv_failed     = 0

            # ── 10 turns per conversation ─────────────────────────
            for turn in range(1, TURNS_PER_CONVERSATION + 1):

                # Turn topology (10 turns, alternating):
                #   odd  turns (1,3,5,7,9) → speaker "a", spoke language, samples 1-5
                #   even turns (2,4,6,8,10) → speaker "b", Hindi, samples 1-5
                if turn % 2 == 1:
                    speaker       = "a"
                    src_language  = spoke_lang
                    tgt_language  = HUB_LANGUAGE
                    sample_idx    = (turn // 2) % len(spoke_samples)
                    sample        = spoke_samples[sample_idx]
                else:
                    speaker       = "b"
                    src_language  = HUB_LANGUAGE
                    tgt_language  = spoke_lang
                    sample_idx    = (turn // 2 - 1) % len(hub_samples)
                    sample        = hub_samples[sample_idx]

                audio_file = sample["file"]
                event_id   = str(uuid.uuid4())

                print(f"    Turn {turn}/{TURNS_PER_CONVERSATION}: speaker={speaker}  "
                      f"{src_language}→{tgt_language}  "
                      f"file={Path(audio_file).name}")

                # Prepare the pipeline event document (filled in as we go)
                event_doc = {
                    "event_id":        event_id,
                    "session_id":      session_id,
                    "conversation_id": conversation_id,
                    "turn_number":     turn,
                    "speaker":         speaker,
                    "source_language": src_language,
                    "target_language": tgt_language,
                    "audio": {
                        "sample_file":  audio_file,
                        "duration_sec": float(sample.get("duration_sec", 0.0)),
                        "size_bytes":   int(sample.get("size_bytes", 0)),
                        "format":       "webm",
                    },
                    "browser":     {},
                    "server":      {},
                    "translation": {},
                    "cost_inr":    COST_PER_ACTIVATION_INR,
                    "success":     False,
                    "error":       None,
                    "timestamp":   datetime.now(timezone.utc),
                }

                try:
                    # Read the audio file
                    audio_path = Path(audio_file)
                    if not audio_path.exists():
                        raise FileNotFoundError(f"Audio file not found: {audio_file}")
                    audio_bytes = audio_path.read_bytes()

                    # Set up response capture — capture the /translate/ response
                    captured_response: Response | None = None

                    async def on_response(response: Response):
                        nonlocal captured_response
                        url = response.url
                        if "/translate/speaker_a" in url or "/translate/speaker_b" in url:
                            captured_response = response

                    page.on("response", on_response)

                    # Inject audio via Playwright evaluate
                    inject_result = await page.evaluate(
                        """([spk, arr]) => window.injectAudio(
                            spk,
                            new Blob([new Uint8Array(arr)], {type: 'audio/webm'})
                        )""",
                        [speaker, list(audio_bytes)],
                    )

                    # Wait a moment for the response to arrive
                    await page.wait_for_timeout(3000)

                    # Remove the listener to avoid stacking
                    page.remove_listener("response", on_response)

                    # Check injectAudio result
                    if isinstance(inject_result, dict) and inject_result.get("error"):
                        raise RuntimeError(f"injectAudio error: {inject_result['error']}")

                    if captured_response is None:
                        raise RuntimeError("No /translate/ response captured")

                    # ── Extract browser timing from request.timing ─────
                    req_timing = captured_response.request.timing
                    browser_timing = extract_browser_timing(req_timing)
                    event_doc["browser"] = browser_timing

                    # ── Extract server timing from response body ───────
                    try:
                        body = await captured_response.json()
                    except Exception:
                        body = {}

                    server_timing = extract_server_timing(body)
                    event_doc["server"] = server_timing

                    # ── Extract translation text ───────────────────────
                    event_doc["translation"] = {
                        "asr_transcript":  body.get("transcript", ""),
                        "nmt_translation": body.get("translation", ""),
                    }

                    event_doc["success"] = True
                    conv_successful += 1

                    print(f"      ✓  browser={browser_timing['total_ms']:.0f}ms  "
                          f"server={server_timing['total_ms']:.0f}ms  "
                          f"asr={server_timing['asr_ms']:.0f}ms")

                except Exception as exc:
                    event_doc["success"] = False
                    event_doc["error"]   = str(exc)
                    conv_failed += 1
                    print(f"      ✗  ERROR: {exc}")

                    # Remove listener if it was added but errored
                    try:
                        page.remove_listener("response", on_response)
                    except Exception:
                        pass

                # Insert the pipeline event into MongoDB
                db["pipeline_events"].insert_one(event_doc)

                # Rate-limit pacing — 7 seconds between turns
                if turn < TURNS_PER_CONVERSATION or spoke_idx < len(SPOKE_LANGUAGES) - 1:
                    print(f"      ⏳  waiting {INTER_TURN_DELAY_SEC}s (rate limit)...")
                    await asyncio.sleep(INTER_TURN_DELAY_SEC)

            # ── Update conversation doc ───────────────────────────
            total_successful += conv_successful
            total_failed     += conv_failed

            db["conversations"].update_one(
                {"conversation_id": conversation_id},
                {"$set": {
                    "completed_at": datetime.now(timezone.utc),
                    "status":       "completed",
                }},
            )
            print(f"  ✓  Conversation complete: {conv_successful} ok, {conv_failed} failed\n")

            # Reload the page between conversations to get a fresh session
            await page.goto(f"{TARGET_URL}?testMode=true", wait_until="networkidle")

        # ── Cleanup browser ───────────────────────────────────────
        await browser.close()

    # ── Update test_sessions with final stats ─────────────────────
    completed_at = datetime.now(timezone.utc)
    total_events = total_successful + total_failed

    db["test_sessions"].update_one(
        {"session_id": session_id},
        {"$set": {
            "completed_at": completed_at,
            "status":       "completed",
            "counts": {
                "total_conversations":   len(SPOKE_LANGUAGES),
                "total_pipeline_events": total_events,
                "successful_events":     total_successful,
                "failed_events":         total_failed,
            },
            "cost": {
                "total_cost_inr":          round(total_successful * COST_PER_ACTIVATION_INR, 2),
                "cost_per_activation_inr": COST_PER_ACTIVATION_INR,
            },
        }},
    )

    # ── Summary ───────────────────────────────────────────────────
    elapsed = (completed_at - started_at).total_seconds()
    print(f"\n{'='*60}")
    print(f"  TEST COMPLETE")
    print(f"  Session:     {session_id}")
    print(f"  Duration:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Events:      {total_events} total ({total_successful} ok, {total_failed} failed)")
    print(f"  Cost:        ₹{total_successful * COST_PER_ACTIVATION_INR:.2f}")
    print(f"{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_test())
