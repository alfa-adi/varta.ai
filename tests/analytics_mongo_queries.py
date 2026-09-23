"""
example_queries.py
──────────────────
Three aggregation snippets for the test-latency-tracking collections.

These are standalone examples — run them in a Python REPL or as a script
after setup_test_schema.py has created the collections and some test data
has been inserted.

Usage:
    python example_queries.py <session_id>
"""

import os
import sys
from pprint import pprint

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB  = os.getenv("MONGO_DB_NAME", "varta_metrics")

if not MONGO_URL:
    print("ERROR: MONGO_URL is not set.")
    sys.exit(1)

client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
db     = client[MONGO_DB]

# ── Accept session_id from command line (or use a placeholder) ───────────────

SESSION_ID = sys.argv[1] if len(sys.argv) > 1 else "YOUR-SESSION-UUID-HERE"
print(f"Querying session: {SESSION_ID}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Query 1 — Average browser.total_ms, server.asr_ms, server.nmt_ms,
#            server.tts_ms  grouped by source_language for one session.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Query 1: Avg latencies per source_language")
print("=" * 60)

pipeline_1 = [
    {"$match": {"session_id": SESSION_ID}},
    {"$group": {
        "_id": "$source_language",
        "avg_browser_total_ms": {"$avg": "$browser.total_ms"},
        "avg_server_asr_ms":    {"$avg": "$server.asr_ms"},
        "avg_server_nmt_ms":    {"$avg": "$server.nmt_ms"},
        "avg_server_tts_ms":    {"$avg": "$server.tts_ms"},
        "count":                {"$sum": 1},
    }},
    {"$sort": {"_id": 1}},
]

for doc in db["pipeline_events"].aggregate(pipeline_1):
    pprint(doc)
print()


# ─────────────────────────────────────────────────────────────────────────────
# Query 2 — Session-wide average across ALL browser.* and server.* fields.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Query 2: Session-wide averages (all stages)")
print("=" * 60)

pipeline_2 = [
    {"$match": {"session_id": SESSION_ID}},
    {"$group": {
        "_id": None,
        # Browser stages
        "avg_browser_tcp_connect_ms":   {"$avg": "$browser.tcp_connect_ms"},
        "avg_browser_tls_handshake_ms": {"$avg": "$browser.tls_handshake_ms"},
        "avg_browser_upload_ms":        {"$avg": "$browser.upload_ms"},
        "avg_browser_server_wait_ms":   {"$avg": "$browser.server_wait_ms"},
        "avg_browser_download_ms":      {"$avg": "$browser.download_ms"},
        "avg_browser_total_ms":         {"$avg": "$browser.total_ms"},
        # Server stages
        "avg_server_sarvam_tcp_ms":     {"$avg": "$server.sarvam_tcp_ms"},
        "avg_server_asr_ms":            {"$avg": "$server.asr_ms"},
        "avg_server_nmt_ms":            {"$avg": "$server.nmt_ms"},
        "avg_server_tts_ms":            {"$avg": "$server.tts_ms"},
        "avg_server_parse_ms":          {"$avg": "$server.parse_ms"},
        "avg_server_session_load_ms":   {"$avg": "$server.session_load_ms"},
        "avg_server_pipeline_build_ms": {"$avg": "$server.pipeline_build_ms"},
        "avg_server_response_build_ms": {"$avg": "$server.response_build_ms"},
        "avg_server_state_save_ms":     {"$avg": "$server.state_save_ms"},
        "avg_server_log_write_ms":      {"$avg": "$server.log_write_ms"},
        "avg_server_total_ms":          {"$avg": "$server.total_ms"},
        # Event count
        "total_events":                 {"$sum": 1},
    }},
]

for doc in db["pipeline_events"].aggregate(pipeline_2):
    doc.pop("_id", None)
    pprint(doc)
print()


# ─────────────────────────────────────────────────────────────────────────────
# Query 3 — Total cost_inr across all pipeline_events for one session.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Query 3: Total cost for session")
print("=" * 60)

pipeline_3 = [
    {"$match": {"session_id": SESSION_ID}},
    {"$group": {
        "_id": None,
        "total_cost_inr": {"$sum": "$cost_inr"},
        "event_count":    {"$sum": 1},
    }},
]

for doc in db["pipeline_events"].aggregate(pipeline_3):
    doc.pop("_id", None)
    pprint(doc)
print()
