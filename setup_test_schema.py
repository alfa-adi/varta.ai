"""
setup_test_schema.py
--------------------
Creates three MongoDB collections for the test-latency-tracking branch:

    test_sessions    - one document per test run
    conversations    - one document per language pair within a test run
    pipeline_events  - one document per conversational turn (ASR > NMT > TTS)

These collections live in a SEPARATE database ('varta_test_data') from the
main branch's production databases ('translation-data-cluste' and
'varta_metrics').  This script never touches any existing collections.

Usage:
    python setup_test_schema.py

Environment variables:
    MONGO_URL        - required  (e.g. mongodb+srv://user:pass@cluster...)
    MONGO_TEST_DB    - optional  (default: "varta_test_data")
"""

import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import CollectionInvalid

# ── Bootstrap ────────────────────────────────────────────────────────────────

load_dotenv()

MONGO_URL     = os.getenv("MONGO_URL")
MONGO_TEST_DB = os.getenv("MONGO_TEST_DB", "varta_test_data")

if not MONGO_URL:
    print("ERROR: MONGO_URL environment variable is not set.")
    print("Copy .env.example to .env and set MONGO_URL before running this script.")
    sys.exit(1)


# ── JSON Schema validators ──────────────────────────────────────────────────

TEST_SESSIONS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["session_id", "test_name", "test_type", "started_at", "status"],
        "properties": {
            "session_id": {
                "bsonType": "string",
                "description": "UUID4 identifying this test run",
            },
            "test_name": {
                "bsonType": "string",
                "description": "Human-readable name for this test run",
            },
            "test_type": {
                "bsonType": "string",
                "description": "Type of test, e.g. browser_playwright",
            },
            "config": {
                "bsonType": "object",
                "properties": {
                    "topology":        {"bsonType": "string"},
                    "hub_language":    {"bsonType": "string"},
                    "spoke_languages": {"bsonType": "array", "items": {"bsonType": "string"}},
                    "turns_per_pair":  {"bsonType": "int"},
                    "audio_source":    {"bsonType": "string"},
                },
            },
            "environment": {
                "bsonType": "object",
                "properties": {
                    "target_url":          {"bsonType": "string"},
                    "browser":             {"bsonType": "string"},
                    "playwright_version":  {"bsonType": "string"},
                    "test_mode_flag":      {"bsonType": "bool"},
                },
            },
            "started_at": {
                "bsonType": "date",
                "description": "When the test run started (UTC)",
            },
            "completed_at": {
                "bsonType": ["date", "null"],
            },
            "counts": {
                "bsonType": "object",
                "properties": {
                    "total_conversations":   {"bsonType": "int"},
                    "total_pipeline_events": {"bsonType": "int"},
                    "successful_events":     {"bsonType": "int"},
                    "failed_events":         {"bsonType": "int"},
                },
            },
            "cost": {
                "bsonType": "object",
                "properties": {
                    "total_cost_inr":          {"bsonType": "double"},
                    "cost_per_activation_inr": {"bsonType": "double"},
                },
            },
            "status": {
                "bsonType": "string",
                "enum": ["running", "completed", "failed", "partial"],
                "description": "Current status of the test run",
            },
            "notes": {
                "bsonType": ["string", "null"],
            },
        },
    }
}

CONVERSATIONS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["conversation_id", "session_id", "language_a", "language_b"],
        "properties": {
            "conversation_id": {
                "bsonType": "string",
                "description": "UUID4 identifying this conversation",
            },
            "session_id": {
                "bsonType": "string",
                "description": "FK → test_sessions.session_id",
            },
            "language_a": {
                "bsonType": "string",
                "description": "Language code for Speaker A, e.g. hi-IN",
            },
            "language_b": {
                "bsonType": "string",
                "description": "Language code for Speaker B, e.g. en-IN",
            },
            "turn_count": {
                "bsonType": "int",
            },
            "started_at": {
                "bsonType": "date",
            },
            "completed_at": {
                "bsonType": ["date", "null"],
            },
            "status": {
                "bsonType": "string",
                "enum": ["running", "completed"],
            },
            "aggregates": {
                "bsonType": "object",
                "properties": {
                    "avg_total_latency_ms": {"bsonType": ["double", "null"]},
                    "avg_asr_ms":           {"bsonType": ["double", "null"]},
                    "avg_nmt_ms":           {"bsonType": ["double", "null"]},
                    "avg_tts_ms":           {"bsonType": ["double", "null"]},
                    "avg_tcp_connect_ms":   {"bsonType": ["double", "null"]},
                    "success_rate":         {"bsonType": ["double", "null"]},
                },
            },
        },
    }
}

PIPELINE_EVENTS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["event_id", "session_id", "conversation_id", "success", "timestamp"],
        "properties": {
            "event_id": {
                "bsonType": "string",
                "description": "UUID4 identifying this pipeline event",
            },
            "session_id": {
                "bsonType": "string",
                "description": "FK → test_sessions.session_id (denormalised)",
            },
            "conversation_id": {
                "bsonType": "string",
                "description": "FK → conversations.conversation_id",
            },
            "turn_number": {
                "bsonType": "int",
            },
            "speaker": {
                "bsonType": "string",
                "enum": ["a", "b"],
                "description": "Which speaker sent audio this turn",
            },
            "source_language": {
                "bsonType": "string",
            },
            "target_language": {
                "bsonType": "string",
            },
            "audio": {
                "bsonType": "object",
                "properties": {
                    "sample_file":  {"bsonType": "string"},
                    "duration_sec": {"bsonType": "double"},
                    "size_bytes":   {"bsonType": "int"},
                    "format":       {"bsonType": "string"},
                },
            },
            "browser": {
                "bsonType": "object",
                "properties": {
                    "tcp_connect_ms":    {"bsonType": ["double", "null"]},
                    "tls_handshake_ms":  {"bsonType": ["double", "null"]},
                    "upload_ms":         {"bsonType": "double"},
                    "server_wait_ms":    {"bsonType": "double"},
                    "download_ms":       {"bsonType": "double"},
                    "total_ms":          {"bsonType": "double"},
                },
            },
            "server": {
                "bsonType": "object",
                "properties": {
                    "sarvam_tcp_ms":     {"bsonType": "double"},
                    "asr_ms":            {"bsonType": "double"},
                    "nmt_ms":            {"bsonType": "double"},
                    "tts_ms":            {"bsonType": "double"},
                    "parse_ms":          {"bsonType": "double"},
                    "session_load_ms":   {"bsonType": "double"},
                    "pipeline_build_ms": {"bsonType": "double"},
                    "response_build_ms": {"bsonType": "double"},
                    "state_save_ms":     {"bsonType": "double"},
                    "log_write_ms":      {"bsonType": "double"},
                    "total_ms":          {"bsonType": "double"},
                },
            },
            "translation": {
                "bsonType": "object",
                "properties": {
                    "asr_transcript":   {"bsonType": "string"},
                    "nmt_translation":  {"bsonType": "string"},
                },
            },
            "cost_inr": {
                "bsonType": "double",
            },
            "success": {
                "bsonType": "bool",
                "description": "Whether this pipeline event completed successfully",
            },
            "error": {
                "bsonType": ["string", "null"],
            },
            "timestamp": {
                "bsonType": "date",
                "description": "When this event was recorded (UTC)",
            },
        },
    }
}


# ── Collection + Index setup ─────────────────────────────────────────────────

def create_collection_if_missing(db, name, validator):
    """Create a collection with a JSON Schema validator — skip if it already exists."""
    existing = db.list_collection_names()
    if name in existing:
        print(f"  >>  {name:20s}  already exists -- skipped")
        return
    try:
        db.create_collection(name, validator=validator)
        print(f"  OK  {name:20s}  created with schema validator")
    except CollectionInvalid as e:
        # Should not happen due to the check above, but guard anyway
        print(f"  !!  {name:20s}  {e}")


def setup_indexes(db):
    """Create all required indexes. Index creation is naturally idempotent."""

    # test_sessions
    db["test_sessions"].create_index(
        [("session_id", ASCENDING)],
        unique=True,
        name="idx_session_id_unique",
    )

    # conversations
    db["conversations"].create_index(
        [("conversation_id", ASCENDING)],
        unique=True,
        name="idx_conversation_id_unique",
    )
    db["conversations"].create_index(
        [("session_id", ASCENDING), ("language_a", ASCENDING), ("language_b", ASCENDING)],
        name="idx_session_lang_pair",
    )

    # pipeline_events
    db["pipeline_events"].create_index(
        [("event_id", ASCENDING)],
        unique=True,
        name="idx_event_id_unique",
    )
    db["pipeline_events"].create_index(
        [("conversation_id", ASCENDING)],
        name="idx_conversation_id",
    )
    db["pipeline_events"].create_index(
        [("session_id", ASCENDING), ("source_language", ASCENDING)],
        name="idx_session_source_lang",
    )
    db["pipeline_events"].create_index(
        [("timestamp", DESCENDING)],
        name="idx_timestamp_desc",
    )

    print("  OK  All 7 indexes ready")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("-" * 55)
    print("  varta.ai -- Test Schema Setup")
    print(f"  Database: {MONGO_TEST_DB}  (isolated from main branch)")
    print("-" * 55)

    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)

    # Fail fast if the cluster is unreachable
    try:
        client.server_info()
    except Exception as e:
        print(f"  ERROR: Could not connect to MongoDB -- {e}")
        sys.exit(1)

    db = client[MONGO_TEST_DB]

    print("\n  Collections:")
    create_collection_if_missing(db, "test_sessions",   TEST_SESSIONS_VALIDATOR)
    create_collection_if_missing(db, "conversations",   CONVERSATIONS_VALIDATOR)
    create_collection_if_missing(db, "pipeline_events", PIPELINE_EVENTS_VALIDATOR)

    print("\n  Indexes:")
    setup_indexes(db)

    # Confirm collections
    all_cols = sorted(db.list_collection_names())
    print(f"\n  All collections in {MONGO_TEST_DB}: {all_cols}")
    print("-" * 55)
    print("  Done.")


if __name__ == "__main__":
    main()
