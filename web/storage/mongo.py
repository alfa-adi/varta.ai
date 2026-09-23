"""
MongoDB client initialised in the FastAPI startup event (post-fork).
Not at module import — MongoClient is not fork-safe before gunicorn fork.

CONVERSATION_PERSISTENCE_ENABLED: if false, _conv_store_ready stays False
and all repository calls are no-ops.  Anonymous live routes are never gated
on _conv_store_ready.
"""
import logging
import os
from typing import Optional

from pymongo import MongoClient

logger = logging.getLogger(__name__)

_PERSISTENCE_ENABLED: bool = (
    os.getenv("CONVERSATION_PERSISTENCE_ENABLED", "false").lower() == "true"
)
_MONGO_CONV_DB_NAME: Optional[str] = os.getenv("MONGO_CONVERSATION_DB")

_client: Optional[MongoClient] = None
_conv_db = None          # pymongo Database handle
_conv_store_ready: bool  = False


def init_mongo(mongo_url: Optional[str]) -> None:
    """
    Called from FastAPI startup event, after the gunicorn fork.
    mongo_url is the same MONGO_URL already used by the server (not a new var).

    Fix 8 — replica-set probe:
    After creating the client, call probe_replica_set so complete_turn_sync
    knows whether transactions are safe without catching OperationFailure code
    numbers at runtime (codes differ between mongod versions and mongomock).
    """
    global _client, _conv_db
    if not _PERSISTENCE_ENABLED:
        return
    if not mongo_url or not _MONGO_CONV_DB_NAME:
        logger.warning(
            "[Persist] CONVERSATION_PERSISTENCE_ENABLED=true but MONGO_URL or "
            "MONGO_CONVERSATION_DB is missing — persistence disabled."
        )
        return
    _client = MongoClient(mongo_url, serverSelectionTimeoutMS=3000)
    _conv_db = _client[_MONGO_CONV_DB_NAME]
    # Probe once at startup — never in the hot path.
    from web.storage.conversation_repository import probe_replica_set
    probe_replica_set(_client)


def check_readiness() -> None:
    """Ping the server and set _conv_store_ready. Never raises."""
    global _conv_store_ready
    if _client is None:
        _conv_store_ready = False
        return
    try:
        _client.admin.command("ping")
        _conv_store_ready = True
    except Exception as exc:
        logger.error(f"[Persist] MongoDB readiness check failed: {exc}")
        _conv_store_ready = False


def get_db():
    return _conv_db


def is_ready() -> bool:
    return _conv_store_ready


def is_enabled() -> bool:
    return _PERSISTENCE_ENABLED
