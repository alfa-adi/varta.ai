"""
test/ws_reliability_test/runner.py
────────────────────────────────────
V2 WebSocket Reliability + Latency Test — Main Entry Point

ONE test. ONE run. 120 translation turns. ONE dataset. ONE report.

Architecture under test:
  Browser → ws://HOST/ws/asr/{session_id}/{speaker}
    Binary frames: raw pcm_s16le, 16kHz mono, 640-byte chunks (20ms each)
    Text control:  {"type": "stop_recording"}
  Server → Browser:
    {"type": "transcript_partial"/"transcript_final"/"language_detected"}
    {"type": "audio_chunk", "data": "<base64 linear16 24kHz PCM>", "format": "mp3"}
    {"type": "audio_end"}

Test workload:
  11 source languages × 10 turns = 110 turns
  (sd excluded — not supported by Bulbul v3 TTS)
  Each language conversation uses one persistent WebSocket for all 10 turns.

Safety:
  The full 120-turn run requires RUN_EXPENSIVE_TEST=true in environment.
  Use --smoke or --conversations 1 --turns 1 for quick validation.

Usage:
  # Smoke test (1 conversation, 1 turn) — safe to run anytime
  python -m test.ws_reliability_test.runner --smoke

  # Small validation (1 conversation, 5 turns)
  python -m test.ws_reliability_test.runner --conversations 1 --turns 5

  # Full 120-turn baseline (EXPENSIVE — calls Sarvam APIs)
  RUN_EXPENSIVE_TEST=true python -m test.ws_reliability_test.runner

Environment:
  SARVAM_API_KEY    — needed by the server (not used directly by the test)
  TARGET_URL        — deployed server URL (default: http://localhost:8000)
  MONGO_URL         — MongoDB connection string (from .env)
  RUN_EXPENSIVE_TEST — must be "true" for the full run

Results:
  Local:   test/results/run_YYYYMMDD_HHMMSS/ (summary.json, report.html, etc.)
  MongoDB: varta_ws_test database (ws_test_sessions + ws_test_turns collections)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Add repository root to path so we can import production modules if needed
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

import httpx

from .schema import (
    TurnResult, ConversationResult, ExperimentResult,
    TurnStatus, FailureStage,
)
from .audio_convert import (
    get_pcm_chunks_cached, validate_pcm_audio, clear_pcm_cache,
    CHUNK_BYTES,
)
from .ws_client import WSConversationClient, _classify_turn
from .reporter import LocalReporter, MongoReporter, print_final_summary


# ── Configuration ─────────────────────────────────────────────────────────────

# 11 source languages → Hindi (sd excluded: not in Bulbul v3 TTS)
# Maps: short code → BCP-47 code (for Sarvam API)
LANGUAGE_MAP: dict[str, str] = {
    "bn": "bn-IN",
    "mr": "mr-IN",
    "te": "te-IN",
    "ta": "ta-IN",
    "ur": "ur-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "or": "od-IN",   # Odia: FLEURS code "or", Sarvam BCP-47 "od-IN"
    "ml": "ml-IN",
    "pa": "pa-IN",
    # "as": "as-IN",  # Skipped: Bulbul v3 TTS does not support Assamese
}

# Removed from language list (not supported by Bulbul v3)
EXCLUDED_LANGUAGES = {"sd"}

TARGET_LANGUAGE      = "hi-IN"
TURNS_PER_CONVO      = 10
DEFAULT_TARGET_URL   = "http://localhost:8000"

# Inter-turn pacing (backend enforces rate limits)
INTER_TURN_DELAY_SEC  = 7.0    # 7s between turns within a conversation
INTER_CONVO_DELAY_SEC = 5.0    # 5s between conversations

# Paths
MANIFEST_PATH  = _REPO_ROOT / "test" / "datasets" / "manifest.json"
RESULTS_DIR    = _REPO_ROOT / "test" / "results"


# ── Manifest loader ───────────────────────────────────────────────────────────

def load_manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_fixture_path(lang_code: str, sample_index: int, manifest: dict) -> Path | None:
    """
    Returns the absolute path to a FLEURS WebM fixture.
    sample_index is 1-based.
    """
    lang_entries = manifest.get(lang_code, [])
    for entry in lang_entries:
        if entry.get("sample_index") == sample_index:
            rel_path = entry.get("file", "")
            if rel_path:
                abs_path = _REPO_ROOT / rel_path
                return abs_path if abs_path.exists() else None
    # Fallback: try by position
    if sample_index <= len(lang_entries):
        entry = lang_entries[sample_index - 1]
        rel_path = entry.get("file", "")
        if rel_path:
            abs_path = _REPO_ROOT / rel_path
            return abs_path if abs_path.exists() else None
    return None


# ── Session creation ──────────────────────────────────────────────────────────

async def create_session(target_url: str, lang_a: str = "", lang_b: str = "") -> str:
    """
    Create a Varta session via POST /session/create.
    Returns session_id.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{target_url}/session/create",
            data={"lang_a": lang_a, "lang_b": lang_b},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["session_id"]


# ── Main runner ───────────────────────────────────────────────────────────────

async def run_experiment(
    target_url: str,
    n_conversations: int,
    turns_per_convo: int,
    verbose: bool = True,
    play_audio: bool = False,
) -> ExperimentResult:
    """
    Execute the full WebSocket reliability experiment.

    Args:
        target_url:      Server to test (e.g. "http://localhost:8000")
        n_conversations: How many source languages to test (max 11)
        turns_per_convo: Turns per conversation (max 10 — fixture dependent)
        verbose:         Print per-turn progress

    Returns:
        ExperimentResult with all 120 TurnResults and computed statistics.
    """
    run_id    = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = f"ws_reliability_{timestamp}"

    experiment = ExperimentResult(
        run_id    = run_id,
        run_label = run_label,
        started_at = time.time(),
        config = {
            "target_url":      target_url,
            "n_conversations": n_conversations,
            "turns_per_convo": turns_per_convo,
            "target_language": TARGET_LANGUAGE,
            "inter_turn_delay_sec":  INTER_TURN_DELAY_SEC,
            "inter_convo_delay_sec": INTER_CONVO_DELAY_SEC,
            "languages":       list(LANGUAGE_MAP.keys())[:n_conversations],
        },
    )

    manifest = load_manifest()
    language_items = list(LANGUAGE_MAP.items())[:n_conversations]

    print(f"\n{'═'*65}")
    print(f"  V2 WebSocket Reliability + Latency Test")
    print(f"{'═'*65}")
    print(f"  Target:       {target_url}")
    print(f"  Conversations:{n_conversations}  (languages)")
    print(f"  Turns each:   {turns_per_convo}")
    print(f"  Total turns:  {n_conversations * turns_per_convo}")
    print(f"  Run ID:       {run_id}")
    print(f"{'═'*65}\n")

    # ── Run each conversation ──────────────────────────────────────────────────
    for conv_idx, (lang_short, lang_bcp47) in enumerate(language_items):
        conv_num       = conv_idx + 1
        conversation_id = f"conv_{lang_short}_{conv_num:03d}_{run_id[:8]}"

        print(f"\n{'─'*60}")
        print(f"  Conversation {conv_num}/{n_conversations}:"
              f" {lang_short} ({lang_bcp47}) → {TARGET_LANGUAGE}")
        print(f"  ID: {conversation_id}")
        print(f"{'─'*60}")

        conv_result = ConversationResult(
            language        = lang_bcp47,
            session_id      = "",
            conversation_id = conversation_id,
        )

        # ── Create session ───────────────────────────────────────────────
        try:
            session_id = await create_session(target_url, lang_a=lang_bcp47, lang_b=TARGET_LANGUAGE)
            conv_result.session_id = session_id
            print(f"  Session created: {session_id[:16]}…")
        except Exception as exc:
            print(f"  ❌ Session creation failed: {exc}")
            # Mark all turns in this conversation as CONNECTION failures
            for tn in range(1, turns_per_convo + 1):
                tr = TurnResult(
                    session_id      = "",
                    conversation_id = conversation_id,
                    turn_id         = f"{conversation_id}_turn_{tn:02d}",
                    turn_number     = tn,
                    source_language = lang_bcp47,
                    target_language = TARGET_LANGUAGE,
                    status          = TurnStatus.FAILURE,
                    failure_stage   = FailureStage.CONNECTION,
                    failure_reason  = f"Session creation failed: {exc}",
                    turn_start_ts   = time.time(),
                    turn_end_ts     = time.time(),
                )
                conv_result.turns.append(tr)
            experiment.conversations.append(conv_result)
            if conv_idx < len(language_items) - 1:
                await asyncio.sleep(INTER_CONVO_DELAY_SEC)
            continue

        # ── Connect WebSockets ───────────────────────────────────────────
        ws_client_a = WSConversationClient(
            base_url   = target_url,
            session_id = session_id,
            speaker    = "a",
            verbose    = verbose,
        )
        ws_client_b = WSConversationClient(
            base_url   = target_url,
            session_id = session_id,
            speaker    = "b",
            verbose    = verbose,
        )

        try:
            connect_latency_ms_a = await ws_client_a.connect()
            connect_latency_ms_b = await ws_client_b.connect()
            conv_result.ws_connect_latency_ms = max(connect_latency_ms_a, connect_latency_ms_b)
        except Exception as exc:
            print(f"  ❌ WebSocket connect failed: {exc}")
            for tn in range(1, turns_per_convo + 1):
                is_speaker_a = (tn % 2 != 0)
                src_lang = lang_bcp47 if is_speaker_a else TARGET_LANGUAGE
                tgt_lang = TARGET_LANGUAGE if is_speaker_a else lang_bcp47
                tr = TurnResult(
                    session_id      = session_id,
                    conversation_id = conversation_id,
                    turn_id         = f"{conversation_id}_turn_{tn:02d}",
                    turn_number     = tn,
                    source_language = src_lang,
                    target_language = tgt_lang,
                    status          = TurnStatus.FAILURE,
                    failure_stage   = FailureStage.CONNECTION,
                    failure_reason  = f"WebSocket connect failed: {exc}",
                    turn_start_ts   = time.time(),
                    turn_end_ts     = time.time(),
                )
                conv_result.turns.append(tr)
            experiment.conversations.append(conv_result)
            if conv_idx < len(language_items) - 1:
                await asyncio.sleep(INTER_CONVO_DELAY_SEC)
            continue

        # ── Run turns ────────────────────────────────────────────────────
        for turn_num in range(1, turns_per_convo + 1):
            is_speaker_a = (turn_num % 2 != 0)
            active_client = ws_client_a if is_speaker_a else ws_client_b
            src_lang = lang_bcp47 if is_speaker_a else TARGET_LANGUAGE
            tgt_lang = TARGET_LANGUAGE if is_speaker_a else lang_bcp47
            
            fixture_lang_short = lang_short if is_speaker_a else "hi"
            fixture_sample_idx = (turn_num // 2) + 1 if is_speaker_a else (turn_num // 2)

            turn_id = f"{conversation_id}_turn_{turn_num:02d}"

            print(f"\n  ── Turn {turn_num}/{turns_per_convo} ─────────────────────────────")
            print(f"  Speaker: {'A' if is_speaker_a else 'B'} | {src_lang} → {tgt_lang}")

            # Build TurnResult skeleton
            tr = TurnResult(
                session_id      = session_id,
                conversation_id = conversation_id,
                turn_id         = turn_id,
                turn_number     = turn_num,
                source_language = src_lang,
                target_language = tgt_lang,
            )

            # Set connect latency on first turn only
            if turn_num == 1:
                tr.ws_connect_latency_ms = conv_result.ws_connect_latency_ms

            # ── Load and validate fixture ─────────────────────────────
            fixture_path = get_fixture_path(fixture_lang_short, fixture_sample_idx, manifest)
            if fixture_path is None:
                tr.fixture_valid  = False
                tr.status         = TurnStatus.FAILURE
                tr.failure_stage  = FailureStage.FIXTURE
                tr.failure_reason = f"Fixture not found for {fixture_lang_short} sample {fixture_sample_idx}"
                tr.audio_file     = f"test/datasets/{fixture_lang_short}/sample_{fixture_sample_idx}.webm"
                tr.turn_start_ts  = time.time()
                tr.turn_end_ts    = time.time()
                print(f"  ❌ Fixture missing: {tr.audio_file}")
                conv_result.turns.append(tr)
                if turn_num < turns_per_convo:
                    await asyncio.sleep(INTER_TURN_DELAY_SEC)
                continue

            tr.audio_file = str(fixture_path.relative_to(_REPO_ROOT))

            # ── Decode WebM → PCM chunks ──────────────────────────────
            try:
                pcm_chunks, duration_sec = get_pcm_chunks_cached(fixture_path)
            except Exception as exc:
                tr.fixture_valid  = False
                tr.status         = TurnStatus.FAILURE
                tr.failure_stage  = FailureStage.FIXTURE
                tr.failure_reason = f"WebM decode failed: {exc}"
                tr.turn_start_ts  = time.time()
                tr.turn_end_ts    = time.time()
                print(f"  ❌ WebM decode failed: {exc}")
                conv_result.turns.append(tr)
                if turn_num < turns_per_convo:
                    await asyncio.sleep(INTER_TURN_DELAY_SEC)
                continue

            # Validate PCM quality
            all_pcm = b"".join(pcm_chunks)
            fixture_size = fixture_path.stat().st_size if fixture_path.exists() else 0

            pcm_ok, pcm_err = validate_pcm_audio(all_pcm)
            if not pcm_ok:
                tr.fixture_valid  = False
                tr.fixture_size_bytes = fixture_size
                tr.pcm_bytes_total    = len(all_pcm)
                tr.status         = TurnStatus.FAILURE
                tr.failure_stage  = FailureStage.FIXTURE
                tr.failure_reason = f"PCM validation failed: {pcm_err}"
                tr.turn_start_ts  = time.time()
                tr.turn_end_ts    = time.time()
                print(f"  ❌ PCM invalid: {pcm_err}")
                conv_result.turns.append(tr)
                if turn_num < turns_per_convo:
                    await asyncio.sleep(INTER_TURN_DELAY_SEC)
                continue

            tr.fixture_valid        = True
            tr.fixture_size_bytes   = fixture_size
            tr.pcm_bytes_total      = len(all_pcm)
            tr.chunks_total         = len(pcm_chunks)
            tr.audio_duration_est_sec = duration_sec

            print(f"  Fixture: {tr.audio_file}")
            print(f"  PCM:     {len(pcm_chunks)} chunks × {CHUNK_BYTES}B = "
                  f"{len(all_pcm):,}B ({duration_sec:.1f}s est.)")

            # ── Execute the turn ──────────────────────────────────────
            await active_client.run_turn(turn_num, pcm_chunks, tr, play_audio=play_audio)

            conv_result.turns.append(tr)

            # Inter-turn delay (skip after last turn in conversation)
            if turn_num < turns_per_convo:
                print(f"  ⏳ Waiting {INTER_TURN_DELAY_SEC}s before next turn…")
                await asyncio.sleep(INTER_TURN_DELAY_SEC)

        # ── Close WebSocket after all turns ──────────────────────────────
        await ws_client_a.close()
        await ws_client_b.close()
        experiment.conversations.append(conv_result)

        # Print conversation summary
        conv_result.compute()
        print(f"\n  Conversation summary: {conv_result.clean_successes}/{conv_result.total_turns} clean, "
              f"reliability={conv_result.reliability_pct:.1f}%")

        # Inter-conversation delay
        if conv_idx < len(language_items) - 1:
            print(f"\n  ⏳ Waiting {INTER_CONVO_DELAY_SEC}s before next conversation…")
            await asyncio.sleep(INTER_CONVO_DELAY_SEC)

    experiment.finished_at = time.time()
    experiment.compute()
    return experiment


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V2 WebSocket Reliability + Latency Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--smoke",          action="store_true",
                   help="Quick smoke test: 1 conversation, 1 turn (bypasses RUN_EXPENSIVE_TEST check)")
    p.add_argument("--conversations",  type=int, default=None,
                   help="Number of conversations (languages) to test (default: 11 for full run)")
    p.add_argument("--turns",          type=int, default=None,
                   help="Turns per conversation (default: 10 for full run)")
    p.add_argument("--target",         type=str, default=None,
                   help="Target server URL (overrides TARGET_URL env var)")
    p.add_argument("--no-mongo",       action="store_true",
                   help="Skip MongoDB write even if MONGO_URL is set")
    p.add_argument("--quiet",          action="store_true",
                   help="Suppress per-turn verbose output")
    p.add_argument("--play-audio",     action="store_true",
                   help="Play the received TTS audio through speakers after each turn")
    return p.parse_args()


async def main():
    args = parse_args()

    # ── Resolve configuration ─────────────────────────────────────────────────
    target_url = (
        args.target
        or os.getenv("TARGET_URL")
        or os.getenv("RENDER_URL")
        or DEFAULT_TARGET_URL
    ).rstrip("/")

    mongo_url = os.getenv("MONGO_URL") if not args.no_mongo else None

    if args.smoke:
        n_conversations  = 1
        turns_per_convo  = 1
        require_expensive = False
        print("⚡ SMOKE MODE — 1 conversation, 1 turn (safe, no cost confirmation needed)")
    else:
        n_conversations  = min(args.conversations or 11, 11)   # cap at 11 languages
        turns_per_convo  = min(args.turns or 10, 10)           # cap at 10 (5 fixture samples per speaker)
        require_expensive = True

    total_turns = n_conversations * turns_per_convo

    # ── Cost safety gate ──────────────────────────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  V2 WebSocket Reliability Test — Pre-flight")
    print(f"{'━'*60}")
    print(f"  Target URL:           {target_url}")
    print(f"  Planned conversations: {n_conversations}")
    print(f"  Turns per convo:       {turns_per_convo}")
    print(f"  Total translation API calls: {total_turns}")
    print(f"  Retries by test runner: 0 (zero — by design)")
    print(f"  MongoDB:               {'disabled' if not mongo_url else 'varta_ws_test @ same cluster'}")
    print(f"{'━'*60}")

    if require_expensive:
        if os.getenv("RUN_EXPENSIVE_TEST", "").lower() != "true":
            print()
            print("  ⛔  STOPPED: full run requires explicit confirmation.")
            print("  Set RUN_EXPENSIVE_TEST=true in environment and re-run:")
            print()
            print("      RUN_EXPENSIVE_TEST=true python -m test.ws_reliability_test.runner")
            print()
            print("  For a safe smoke test (1 turn, no confirmation needed):")
            print()
            print("      python -m test.ws_reliability_test.runner --smoke")
            print()
            sys.exit(0)

    # ── Server health check ───────────────────────────────────────────────────
    print(f"\n  Checking server at {target_url}/health …")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{target_url}/health")
            resp.raise_for_status()
            print(f"  ✅ Server healthy: {resp.json()}")
    except Exception as exc:
        print(f"  ❌ Server not reachable: {exc}")
        print(f"  Make sure the varta.ai server is running at {target_url}")
        print(f"  Start it with:  uvicorn web.server:app --reload")
        sys.exit(1)

    # ── Manifest check ────────────────────────────────────────────────────────
    if not MANIFEST_PATH.exists():
        print(f"\n  ❌ Manifest not found: {MANIFEST_PATH}")
        print("  Run test/datasets/download_fleurs_samples.py first.")
        sys.exit(1)

    manifest = load_manifest()
    print(f"\n  Manifest: {len(manifest)} languages loaded")

    # Verify at least one fixture exists per language
    missing = []
    for lang_short in list(LANGUAGE_MAP.keys())[:n_conversations]:
        fixture = get_fixture_path(lang_short, 1, manifest)
        if fixture is None:
            missing.append(lang_short)

    if missing:
        print(f"  ⚠️  Fixtures missing for: {missing}")
        print("  These turns will be marked as FIXTURE failures.")

    print(f"\n  Starting experiment…\n")

    # ── Run experiment ────────────────────────────────────────────────────────
    t_start = time.time()
    experiment = await run_experiment(
        target_url      = target_url,
        n_conversations = n_conversations,
        turns_per_convo = turns_per_convo,
        verbose         = not args.quiet,
        play_audio      = args.play_audio,
    )
    t_elapsed = time.time() - t_start

    print(f"\n  Experiment complete in {t_elapsed/60:.1f} minutes")

    # ── Write local artifacts ─────────────────────────────────────────────────
    local_reporter = LocalReporter(RESULTS_DIR)
    local_reporter.write_all(experiment)

    # ── Write to MongoDB ──────────────────────────────────────────────────────
    if mongo_url:
        print("\n  Writing to MongoDB…")
        mongo_reporter = MongoReporter(mongo_url)
        connected = mongo_reporter.connect()
        if connected:
            mongo_reporter.write_all(experiment)
            mongo_reporter.close()
    else:
        print("\n  MongoDB: skipped (no MONGO_URL set or --no-mongo flag)")

    # ── Print final summary ───────────────────────────────────────────────────
    print_final_summary(experiment)

    print(f"\n  📁 Results in: {local_reporter.run_directory}")
    print(f"  🌐 Open:       {local_reporter.run_directory / 'report.html'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
