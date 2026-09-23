#!/usr/bin/env python3
"""
preflight_check.py
──────────────────
Validates every prerequisite before launching the baseline latency test.

Run this immediately before `run_baseline_test.py` so that a missing env
var or empty audio folder is discovered before turn 40 of 60:

    python preflight_check.py && python run_baseline_test.py

Exit code:
    0 — all checks passed
    1 — one or more checks failed
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MONGO_URL      = os.getenv("MONGO_URL")
MONGO_TEST_DB  = os.getenv("MONGO_TEST_DB", "varta_test_data")
RENDER_URL     = os.getenv("RENDER_URL")

MANIFEST_PATH  = Path("test/datasets/manifest.json")

EXPECTED_LANGS = sorted([
    "hi", "bn", "mr", "te", "ta", "ur", "gu",
    "kn", "or", "ml", "pa", "as", "sd",
])
EXPECTED_TOTAL_SAMPLES = 65

REQUIRED_COLLECTIONS = ["test_sessions", "conversations", "pipeline_events"]

# ── State ─────────────────────────────────────────────────────────────────────

_failures = 0


def _report(passed: bool, label: str, detail: str = ""):
    """Print a [PASS] or [FAIL] line and track failures."""
    global _failures
    tag = "[PASS]" if passed else "[FAIL]"
    msg = f"  {tag}  {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    if not passed:
        _failures += 1


# ── Checks ────────────────────────────────────────────────────────────────────

def check_mongo_url():
    """Check 1: MONGO_URL environment variable is set."""
    _report(
        bool(MONGO_URL),
        "MONGO_URL is set",
        "" if MONGO_URL else "Environment variable MONGO_URL is not set",
    )
    return bool(MONGO_URL)


def check_mongo_connection():
    """Check 2: MongoDB is reachable with a 5s timeout."""
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        _report(True, "MongoDB connection")
        return client
    except Exception as e:
        _report(False, "MongoDB connection", str(e))
        return None


def check_collections(client):
    """Check 3: All three required collections exist."""
    try:
        db = client[MONGO_TEST_DB]
        existing = db.list_collection_names()
        missing = [c for c in REQUIRED_COLLECTIONS if c not in existing]
        if missing:
            _report(False, "Required collections exist", f"Missing: {missing}")
        else:
            _report(True, "Required collections exist", ", ".join(REQUIRED_COLLECTIONS))
    except Exception as e:
        _report(False, "Required collections exist", str(e))


def check_event_id_index(client):
    """Check 4: pipeline_events has an index referencing event_id."""
    try:
        db = client[MONGO_TEST_DB]
        indexes = db["pipeline_events"].index_information()
        has_event_id_index = any(
            any(field == "event_id" for field, _ in idx.get("key", []))
            for idx in indexes.values()
        )
        if has_event_id_index:
            _report(True, "pipeline_events has event_id index")
        else:
            _report(False, "pipeline_events has event_id index",
                    f"Indexes found: {list(indexes.keys())}")
    except Exception as e:
        _report(False, "pipeline_events has event_id index", str(e))


def check_manifest_exists():
    """Check 5a: manifest.json exists and parses as JSON."""
    if not MANIFEST_PATH.exists():
        _report(False, "Manifest file exists", f"{MANIFEST_PATH} not found")
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
            manifest = json.load(f)
        _report(True, "Manifest file exists and parses")
        return manifest
    except Exception as e:
        _report(False, "Manifest file exists and parses", str(e))
        return None


def check_manifest_languages(manifest: dict):
    """Check 5b: Manifest has all 13 expected language keys."""
    present = sorted(manifest.keys())
    missing = [l for l in EXPECTED_LANGS if l not in present]
    extra   = [l for l in present if l not in EXPECTED_LANGS]
    if missing:
        _report(False, "Manifest has all 13 language keys",
                f"Missing: {missing}" + (f", Extra: {extra}" if extra else ""))
    else:
        _report(True, "Manifest has all 13 language keys")


def check_manifest_sample_count(manifest: dict):
    """Check 5c: Total sample count across all languages is 65."""
    total = sum(len(samples) for samples in manifest.values())
    if total == EXPECTED_TOTAL_SAMPLES:
        _report(True, f"Total sample count = {total}")
    else:
        per_lang = {k: len(v) for k, v in manifest.items()}
        _report(False, f"Total sample count = {total} (expected {EXPECTED_TOTAL_SAMPLES})",
                f"Per-language: {per_lang}")


def check_manifest_files_exist(manifest: dict):
    """Check 5d: Every audio file referenced in the manifest exists on disk."""
    missing_files = []
    total_checked = 0
    for lang_code, samples in manifest.items():
        for sample in samples:
            file_path = Path(sample.get("file", ""))
            total_checked += 1
            if not file_path.exists():
                missing_files.append(str(file_path))

    if missing_files:
        shown = missing_files[:5]
        suffix = f" (and {len(missing_files) - 5} more)" if len(missing_files) > 5 else ""
        _report(False, f"All {total_checked} manifest audio files exist on disk",
                f"{len(missing_files)} missing: {shown}{suffix}")
    else:
        _report(True, f"All {total_checked} manifest audio files exist on disk")


def check_render_url():
    """Check 6a: RENDER_URL environment variable is set."""
    _report(
        bool(RENDER_URL),
        "RENDER_URL is set",
        "" if RENDER_URL else "Environment variable RENDER_URL is not set",
    )
    return bool(RENDER_URL)


def check_health_endpoint():
    """Check 6b: GET {RENDER_URL}/health returns HTTP 200."""
    try:
        import requests
        url = f"{RENDER_URL.rstrip('/')}/health"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            _report(True, f"Health endpoint ({url})", f"HTTP {resp.status_code}")
        else:
            _report(False, f"Health endpoint ({url})",
                    f"HTTP {resp.status_code} (expected 200)")
    except Exception as e:
        _report(False, "Health endpoint", str(e))


def check_test_mode():
    """Check 7: GET {RENDER_URL}/?testMode=true contains test-banner and injectAudio."""
    try:
        import requests
        url = f"{RENDER_URL.rstrip('/')}/?testMode=true"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            _report(False, "Test mode page loads",
                    f"HTTP {resp.status_code} (expected 200)")
            return

        html = resp.text
        has_banner = "test-banner" in html
        has_inject = "injectAudio" in html

        if has_banner and has_inject:
            _report(True, "Test mode page contains test-banner and injectAudio")
        else:
            missing = []
            if not has_banner:
                missing.append("test-banner")
            if not has_inject:
                missing.append("injectAudio")
            _report(False, "Test mode page contains test-banner and injectAudio",
                    f"Missing: {missing}")
    except Exception as e:
        _report(False, "Test mode page loads", str(e))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  varta.ai — Preflight Check")
    print("=" * 60)

    # ── MongoDB checks ────────────────────────────────────────────
    print("\n  MongoDB:")
    mongo_ok = check_mongo_url()
    client = None
    if mongo_ok:
        client = check_mongo_connection()
        if client:
            check_collections(client)
            check_event_id_index(client)
    else:
        _report(False, "MongoDB connection", "skipped (MONGO_URL not set)")
        _report(False, "Required collections exist", "skipped")
        _report(False, "pipeline_events has event_id index", "skipped")

    # ── Manifest checks ──────────────────────────────────────────
    print("\n  Audio Manifest:")
    manifest = check_manifest_exists()
    if manifest:
        check_manifest_languages(manifest)
        check_manifest_sample_count(manifest)
        check_manifest_files_exist(manifest)
    else:
        _report(False, "Manifest has all 13 language keys", "skipped")
        _report(False, "Total sample count", "skipped")
        _report(False, "All manifest audio files exist on disk", "skipped")

    # ── Render deployment checks ─────────────────────────────────
    print("\n  Render Deployment:")
    render_ok = check_render_url()
    if render_ok:
        check_health_endpoint()
        check_test_mode()
    else:
        _report(False, "Health endpoint", "skipped (RENDER_URL not set)")
        _report(False, "Test mode page", "skipped (RENDER_URL not set)")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if _failures == 0:
        print("  OK  All checks passed")
    else:
        print(f"  XX  {_failures} check(s) failed")
    print("=" * 60)

    sys.exit(1 if _failures > 0 else 0)


if __name__ == "__main__":
    main()
