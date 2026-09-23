"""
web/storage/conversation_repository.py────────────────────────────────────────
All functions are synchronous (*_sync suffix) and intended to be called
via asyncio.to_thread from async code.  Never call from the event loop
directly.
"""
import logging
import time as _time
from datetime import datetime, timezone
from typing import Optional

import pymongo
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from web.storage.mongo import get_db

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Raised when reserve_turn_sync cannot find the parent session document."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Session ───────────────────────────────────────────────────────────────────

def create_session_doc_sync(
    user_id:     str,
    session_id:  str,
    customer_id: str,
) -> None:
    db = get_db()
    if db is None:
        return
    db["translation_sessions"].insert_one({
        "session_id":    session_id,
        "user_id":       user_id,
        "customer_id":   customer_id,
        "created_at":    _now(),
        "next_sequence": 0,       # first turn increments to 1 (ReturnDocument.AFTER)
        "turn_count":    0,
        "participants":  {},
    })


# ── Turns ─────────────────────────────────────────────────────────────────────

def reserve_turn_sync(
    user_id:     str,
    session_id:  str,
    turn_id:     str,
    speaker_id:  str,
    fingerprint: Optional[str],   # None is valid for WS turns (no audio fingerprint)
) -> tuple[dict, bool]:
    """
    Atomically allocate a sequence number and insert a new turn document.

    Returns (document, is_new).
    is_new=False means the turn already exists (idempotent hit).

    P2 fix — check-before-increment:
    The function first checks whether a document with this turn_id already exists.
    If so, it returns early WITHOUT incrementing next_sequence, preventing sequence
    gaps that would occur if the $inc ran before the DuplicateKeyError was caught.

    DuplicateKeyError on session_id+seq   → retry $inc (sequence collision, rare).
    Missing session document              → raises SessionNotFoundError.
    """
    db    = get_db()
    if db is None:
        return None, False
    turns = db["translation_turns"]

    # P2 — pre-check: avoid incrementing sequence for already-reserved turns.
    existing = turns.find_one({"turn_id": turn_id, "user_id": user_id})
    if existing is not None:
        return existing, False

    while True:
        # Increment sequence on the AFTER image so seq=1 for the first turn.
        session_doc = db["translation_sessions"].find_one_and_update(
            {"session_id": session_id, "user_id": user_id},
            {"$inc": {"next_sequence": 1}},
            return_document=ReturnDocument.AFTER,
        )
        if session_doc is None:
            raise SessionNotFoundError(
                f"Session {session_id!r} not found for user {user_id!r}"
            )
        seq = session_doc["next_sequence"]

        doc = {
            "turn_id":           turn_id,
            "user_id":           user_id,
            "session_id":        session_id,
            "speaker_id":        speaker_id,
            "sequence":          seq,
            "status":            "received",
            "audio_fingerprint": fingerprint,
            "created_at":        _now(),
            "updated_at":        _now(),
            "metadata":          {},
        }
        try:
            print(f">>> reserve_turn_sync ATTEMPTING insert_one doc: {doc}")
            turns.insert_one(doc)
            print(f">>> reserve_turn_sync SUCCESS inserting doc: {doc}")
            return doc, True
        except DuplicateKeyError as exc:
            print(f">>> reserve_turn_sync DUPLICATE KEY ERROR: {exc}")
            details = getattr(exc, "details", None) or {}
            kp = details.get("keyPattern", {})
            err_str = str(exc)
            if "turn_id" in kp or "turn_id" in err_str:
                # Lost race: another writer inserted between our find_one and insert_one.
                existing = turns.find_one({"turn_id": turn_id, "user_id": user_id})
                # Rollback the sequence increment: this turn won't use the slot we took.
                # The gap is documented as an accepted invariant (monotonic, non-contiguous).
                logger.warning(
                    f"[Repo] Race on turn_id={turn_id}: seq={seq} was wasted (gap accepted)."
                    " Returning existing turn."
                )
                return existing, False
            elif ("session_id" in kp and "sequence" in kp) or ("session_id" in err_str and "sequence" in err_str):
                # Sequence collision (two workers raced).  Retry the increment.
                logger.warning(
                    f"[Repo] Sequence collision session={session_id} seq={seq}; retrying"
                )
                continue
            raise  # Unexpected index — propagate


def update_asr_sync(
    user_id:    str,
    turn_id:    str,
    transcript: str,
    src_lang:   str,
) -> None:
    """
    Transition turn received → transcribed.
    Matches only status=received to prevent double-write.
    src_lang must NOT be "auto" (Sarvam 403 otherwise); callers resolve first.
    Raises ValueError (not AssertionError) on bad src_lang so the persist
    worker can catch it and call mark_turn_failed_sync instead of crashing.
    """
    if not src_lang or src_lang == "auto":
        raise ValueError(
            f"update_asr_sync: src_lang must be a BCP-47 code, got {src_lang!r}"
        )
    db = get_db()
    if db is None:
        return
    db["translation_turns"].update_one(
        {"turn_id": turn_id, "user_id": user_id, "status": "received"},
        {"$set": {
            "status":          "transcribed",
            "transcript":      transcript,
            "source_language": src_lang,
            "updated_at":      _now(),
        }},
    )


# ── Replica-set probe (called once in init_mongo) ────────────────────────────

_replica_set_available: bool = False


def probe_replica_set(client) -> None:
    """
    Probe whether the connected MongoDB deployment supports multi-document
    transactions.  Called from init_mongo after the client is created.

    Strategy: run `hello` and check `isWritablePrimary` plus the absence of
    `msg: isdbgrid` (mongos).  Standalone mongod and mongomock do NOT support
    transactions; replica sets and Atlas do.

    This avoids catching specific OperationFailure codes (which differ between
    mongod versions and mongomock) in the hot path.
    """
    global _replica_set_available
    try:
        result = client.admin.command("hello")
        is_primary  = result.get("isWritablePrimary", False)
        is_mongos   = result.get("msg") == "isdbgrid"
        has_set_name = "setName" in result
        _replica_set_available = is_primary and has_set_name and not is_mongos
    except Exception as exc:
        logger.warning(f"[Repo] Replica-set probe failed, disabling transactions: {exc}")
        _replica_set_available = False


def _do_complete_turn(
    db,
    user_id:     str,
    turn_id:     str,
    session_id:  str,
    translation: Optional[dict],
    timing:      Optional[dict],
    mongo_session=None,
) -> None:
    set_fields: dict = {"status": "completed", "updated_at": _now()}
    if translation:
        set_fields["translation"] = translation
    if timing:
        set_fields["timing"] = timing

    res = db["translation_turns"].update_one(
        {
            "turn_id": turn_id,
            "user_id": user_id,
            # Fix 4: accept both received and transcribed.
            # When src_lang is unknown and update_asr was skipped, the turn stays
            # at "received".  complete_turn must still advance it to "completed".
            "status":  {"$in": ["received", "transcribed"]},
        },
        {"$set": set_fields},
        session=mongo_session,
    )
    if res.modified_count > 0:
        db["translation_sessions"].update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$inc": {"turn_count": 1}},
            session=mongo_session,
        )


def complete_turn_sync(
    user_id:     str,
    turn_id:     str,
    session_id:  str,
    translation: Optional[dict] = None,
    timing:      Optional[dict] = None,
) -> None:
    """
    Transition received|transcribed → completed.
    Uses a multi-document transaction when _replica_set_available=True
    (set by probe_replica_set at startup).  Falls back to non-transactional
    ordered writes for standalone mongod and mongomock.

    Non-transactional note: if the process dies between the turns update and
    the sessions $inc, turn_count will under-count.  No reconciliation daemon
    is specified in this revision; operators can recount with:
        db.translation_sessions.aggregate([{$lookup: ...}])
    """
    db = get_db()
    if db is None:
        return
    if _replica_set_available:
        with db.client.start_session() as s:
            with s.start_transaction():
                _do_complete_turn(db, user_id, turn_id, session_id, translation, timing, mongo_session=s)
    else:
        _do_complete_turn(db, user_id, turn_id, session_id, translation, timing)


def mark_turn_failed_sync(
    user_id:  str,
    turn_id:  str,
    error:    str,
) -> None:
    """Update status to failed, but NEVER overwrite completed or cancelled."""
    db = get_db()
    db["translation_turns"].update_one(
        {
            "turn_id": turn_id,
            "user_id": user_id,
            "status":  {"$nin": ["completed", "cancelled"]},
        },
        {"$set": {"status": "failed", "error_message": error, "updated_at": _now()}},
    )


def get_turn_sync(user_id: str, turn_id: str) -> Optional[dict]:
    db = get_db()
    if db is None:
        return None
    return db["translation_turns"].find_one({"turn_id": turn_id, "user_id": user_id})
