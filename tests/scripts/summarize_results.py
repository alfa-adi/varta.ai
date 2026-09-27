#!/usr/bin/env python3
"""
summarize_results.py
────────────────────
Prints a fast textual sanity check of one test session's results.

Usage:
    python summarize_results.py <session_id>

Reads from MongoDB collections: test_sessions, conversations, pipeline_events.
"""

import io
import os
import sys

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MONGO_URL     = os.getenv("MONGO_URL")
MONGO_TEST_DB = os.getenv("MONGO_TEST_DB", "varta_test_data")


def main():
    # ── Argument validation ───────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python summarize_results.py <session_id>")
        print("  Prints a summary of the baseline test results for the given session.")
        sys.exit(1)

    session_id = sys.argv[1]

    if not MONGO_URL:
        print("ERROR: MONGO_URL environment variable is not set.")
        sys.exit(1)

    # ── Connect to MongoDB ────────────────────────────────────────
    from pymongo import MongoClient

    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except Exception as e:
        print(f"ERROR: Could not connect to MongoDB — {e}")
        sys.exit(1)

    db = client[MONGO_TEST_DB]

    # ── Look up the session ───────────────────────────────────────
    session = db["test_sessions"].find_one({"session_id": session_id})
    if not session:
        print(f"ERROR: No test_sessions document found for session_id = {session_id}")
        sys.exit(1)

    # ── Session overview ──────────────────────────────────────────
    print("=" * 70)
    print("  SESSION SUMMARY")
    print("=" * 70)

    test_name = session.get("test_name", "n/a")
    status    = session.get("status", "n/a")
    counts    = session.get("counts", {})
    cost      = session.get("cost", {})

    started  = session.get("started_at")
    finished = session.get("completed_at")
    duration_str = "n/a"
    if started and finished:
        elapsed = (finished - started).total_seconds()
        duration_str = f"{elapsed:.0f}s ({elapsed/60:.1f} min)"

    print(f"  Test name:    {test_name}")
    print(f"  Session ID:   {session_id}")
    print(f"  Status:       {status}")
    print(f"  Duration:     {duration_str}")

    # ── Event counts ──────────────────────────────────────────────
    total_events = db["pipeline_events"].count_documents({"session_id": session_id})
    success_events = db["pipeline_events"].count_documents(
        {"session_id": session_id, "success": True}
    )
    failed_events = total_events - success_events
    success_pct = (success_events / total_events * 100) if total_events > 0 else 0.0

    print(f"\n  Events:       {total_events} total")
    print(f"  Successful:   {success_events} ({success_pct:.1f}%)")
    print(f"  Failed:       {failed_events}")

    if counts:
        conv_count = counts.get("total_conversations", "n/a")
        print(f"  Conversations: {conv_count}")

    if cost:
        total_cost = cost.get("total_cost_inr")
        per_act    = cost.get("cost_per_activation_inr")
        if total_cost is not None:
            print(f"  Cost:         INR {total_cost:.2f}"
                  + (f" (INR {per_act:.2f}/activation)" if per_act else ""))

    # ── Session-wide averages ─────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  SESSION-WIDE AVERAGES (successful events only)")
    print(f"{'─' * 70}")

    avg_pipeline = list(db["pipeline_events"].aggregate([
        {"$match": {"session_id": session_id, "success": True}},
        {"$group": {
            "_id": None,
            "avg_browser_total":         {"$avg": "$browser.total_ms"},
            "avg_browser_tcp_connect":   {"$avg": "$browser.tcp_connect_ms"},
            "avg_browser_tls_handshake": {"$avg": "$browser.tls_handshake_ms"},
            "avg_browser_ttfb":          {"$avg": "$browser.ttfb_ms"},
            "avg_browser_download":      {"$avg": "$browser.download_ms"},
            "avg_server_total":          {"$avg": "$server.total_ms"},
            "avg_server_sarvam_tcp":     {"$avg": "$server.sarvam_tcp_ms"},
            "avg_server_session_load":   {"$avg": "$server.session_load_ms"},
            "avg_server_parse":          {"$avg": "$server.parse_ms"},
            "avg_server_pipeline_build": {"$avg": "$server.pipeline_build_ms"},
            "avg_server_asr":            {"$avg": "$server.asr_ms"},
            "avg_server_nmt":            {"$avg": "$server.nmt_ms"},
            "avg_server_tts":            {"$avg": "$server.tts_ms"},
            "avg_server_response_build": {"$avg": "$server.response_build_ms"},
            "avg_server_state_save":     {"$avg": "$server.state_save_ms"},
            "avg_server_log_write":      {"$avg": "$server.log_write_ms"},
        }},
    ]))

    if avg_pipeline:
        a = avg_pipeline[0]

        def fmt(val, suffix="ms"):
            if val is None:
                return "n/a"
            return f"{val:.1f} {suffix}"

        # browser.tcp_connect_ms is often null (connections reused) — handle specially
        tcp_val = a.get("avg_browser_tcp_connect")
        tcp_str = "n/a (connections reused)" if tcp_val is None else fmt(tcp_val)

        tls_val = a.get("avg_browser_tls_handshake")
        tls_str = "n/a (connections reused)" if tls_val is None else fmt(tls_val)

        print(f"  Browser total:            {fmt(a.get('avg_browser_total'))}")
        print(f"  Browser TCP connect:      {tcp_str}")
        print(f"  Browser TLS handshake:    {tls_str}")
        print(f"  Browser upload:           {fmt(a.get('avg_browser_upload'))}")
        print(f"  Browser server wait:      {fmt(a.get('avg_browser_server_wait'))}")
        print(f"  Browser download:         {fmt(a.get('avg_browser_download'))}")
        print(f"  ─────────────────────────")
        print(f"  Server total:             {fmt(a.get('avg_server_total'))}")
        print(f"  Server Sarvam TCP:        {fmt(a.get('avg_server_sarvam_tcp'))}")
        print(f"  Server session load:      {fmt(a.get('avg_server_session_load'))}")
        print(f"  Server request parse:     {fmt(a.get('avg_server_parse'))}")
        print(f"  Server pipeline build:    {fmt(a.get('avg_server_pipeline_build'))}")
        print(f"  Server ASR (Sarvam API):  {fmt(a.get('avg_server_asr'))}")
        print(f"  Server NMT (Sarvam API):  {fmt(a.get('avg_server_nmt'))}")
        print(f"  Server TTS (Sarvam API):  {fmt(a.get('avg_server_tts'))}")
        print(f"  Server response build:    {fmt(a.get('avg_server_response_build'))}")
        print(f"  Server state save:        {fmt(a.get('avg_server_state_save'))}")
        print(f"  Server log write:         {fmt(a.get('avg_server_log_write'))}")
    else:
        print("  No successful events found — averages not available.")

    # ── Per-language breakdown ────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  PER-LANGUAGE BREAKDOWN (sorted by avg latency, descending)")
    print(f"{'─' * 70}")

    lang_pipeline = list(db["pipeline_events"].aggregate([
        {"$match": {"session_id": session_id, "success": True}},
        {"$group": {
            "_id": "$source_language",
            "avg_browser_total": {"$avg": "$browser.total_ms"},
            "turn_count":        {"$sum": 1},
        }},
        {"$sort": {"avg_browser_total": -1}},
    ]))

    if lang_pipeline:
        print(f"  {'Language':<12} {'Avg Latency':>14} {'Turns':>8}")
        print(f"  {'─'*12} {'─'*14} {'─'*8}")

        for row in lang_pipeline:
            lang = row["_id"] or "unknown"
            avg_ms = row.get("avg_browser_total")
            turns  = row.get("turn_count", 0)

            if avg_ms is not None:
                latency_str = f"{avg_ms:.1f} ms"
            else:
                latency_str = "n/a"

            print(f"  {lang:<12} {latency_str:>14} {turns:>8}")
    else:
        print("  No successful events found — per-language breakdown not available.")

    # ── Conversation summary ──────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  CONVERSATIONS")
    print(f"{'─' * 70}")

    conversations = list(db["conversations"].find(
        {"session_id": session_id},
        {"conversation_id": 1, "language_a": 1, "language_b": 1, "status": 1, "_id": 0},
    ))

    if conversations:
        print(f"  {'#':<4} {'Lang A':<10} {'Lang B':<10} {'Status':<12}")
        print(f"  {'─'*4} {'─'*10} {'─'*10} {'─'*12}")
        for i, conv in enumerate(conversations, 1):
            print(f"  {i:<4} {conv.get('language_a', 'n/a'):<10} "
                  f"{conv.get('language_b', 'n/a'):<10} "
                  f"{conv.get('status', 'n/a'):<12}")
    else:
        print("  No conversations found for this session.")

    print(f"\n{'=' * 70}")
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
