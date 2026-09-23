"""
Create indexes and schema validators for varta_conversations.
Safe to re-run (check status before applying).
Uses an owner-token lock with expiry to prevent concurrent migration (P1-B).
"""
import logging
import os
import time
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)


def up():
    from pymongo import MongoClient, ASCENDING
    mongo_url  = os.getenv("MONGO_URL")
    db_name    = os.getenv("MONGO_CONVERSATION_DB")
    if not mongo_url or not db_name:
        raise RuntimeError("MONGO_URL and MONGO_CONVERSATION_DB required for migration")

    client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
    db     = client[db_name]
    locks  = db["_migration_locks"]

    now   = time.time()
    owner = str(__import__('uuid').uuid4())   # P1-B: owner token for safe release

    # Acquire lock: either unlocked OR previous lock has expired (60s TTL).
    # P1-B: store `owner` in the lock document so only the acquiring runner can release it.
    try:
        result = locks.find_one_and_update(
            {
                "_id": "001",
                "$or": [
                    {"locked": False},
                    {"expires_at": {"$lt": now}},
                ],
            },
            {"$set": {
                "locked":     True,
                "owner":      owner,          # P1-B: identity of lock holder
                "expires_at": now + 60,
                "status":     "running",
            }},
            upsert=True,
            return_document=True,    # AFTER
        )
    except Exception as e:
        if isinstance(e, DuplicateKeyError):
            result = None
        else:
            raise

    if result and result.get("status") == "completed":
        logger.info("Migration 001 already applied.")
        client.close()
        return

    # A second worker that lost the race should wait, not die.
    if result is None or result.get("owner") != owner:
        logger.info("Migration 001 locked by another worker; waiting...")
        for _ in range(30):   # wait up to 30s
            time.sleep(1)
            doc = locks.find_one({"_id": "001"})
            if doc and doc.get("status") == "completed":
                logger.info("Migration 001 completed by other worker.")
                client.close()
                return
        raise RuntimeError("Migration 001: timed out waiting for other worker")

    try:
        # Indexes
        db["translation_turns"].create_index(
            [("turn_id", ASCENDING), ("user_id", ASCENDING)],
            unique=True, name="idx_turns_id_user_unique",
        )
        db["translation_turns"].create_index(
            [("session_id", ASCENDING), ("sequence", ASCENDING)],
            unique=True, name="idx_turns_session_seq_unique",
        )
        db["translation_sessions"].create_index(
            [("session_id", ASCENDING), ("user_id", ASCENDING)],
            unique=True, name="idx_sessions_id_user_unique",
        )

        # P1-B: release by owner token — prevent another runner from unlocking our lease.
        locks.update_one(
            {"_id": "001", "owner": owner},
            {"$set": {"status": "completed", "locked": False}},
        )
        logger.info("Migration 001 applied.")
    except Exception:
        # P1-B: only release if we still hold the lock.
        locks.update_one(
            {"_id": "001", "owner": owner},
            {"$set": {"locked": False}},
        )
        raise

    client.close()
