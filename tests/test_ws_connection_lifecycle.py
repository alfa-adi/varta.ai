"""
tests/test_connection_lifecycle.py
────────────────────────────────────
Tests for web/connection_manager.py — connection ownership, Redis lease logic,
backpressure, duplicate rejection, and deterministic release().

Run:
    pytest tests/test_connection_lifecycle.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_ws():
    """Minimal FastAPI WebSocket mock."""
    ws = MagicMock()
    ws.send_json  = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.close      = AsyncMock()
    return ws


def _make_adapter():
    """Minimal SarvamLiveASRAdapter mock."""
    adapter = MagicMock()
    adapter.close = AsyncMock()
    return adapter


def _make_redis(set_returns=True, get_returns=None):
    """Minimal redis client mock."""
    r = MagicMock()
    r.set     = MagicMock(return_value=set_returns)
    r.get     = MagicMock(return_value=get_returns)
    r.expire  = MagicMock()
    r.delete  = MagicMock()
    return r


# Reset the process-local registry before each test so tests are isolated.
@pytest.fixture(autouse=True)
def clear_registry():
    import web.connection_manager as cm
    cm._registry.clear()
    yield
    cm._registry.clear()


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_acquire_returns_owner_for_new_connection():
    from web.connection_manager import acquire_connection

    ws      = _make_ws()
    adapter = _make_adapter()
    owner   = await acquire_connection("sess1", "a", ws, adapter, redis=None)

    assert owner is not None
    assert owner.session_id  == "sess1"
    assert owner.speaker     == "a"
    assert owner.generation  == 1
    assert owner._closed     is False

    await owner.release()


@pytest.mark.asyncio
async def test_acquire_rejects_process_local_duplicate():
    from web.connection_manager import acquire_connection

    ws1 = _make_ws()
    ws2 = _make_ws()
    adapter1 = _make_adapter()
    adapter2 = _make_adapter()

    owner1 = await acquire_connection("sess2", "a", ws1, adapter1, redis=None)
    assert owner1 is not None

    owner2 = await acquire_connection("sess2", "a", ws2, adapter2, redis=None)
    assert owner2 is None
    # Second websocket should have been closed with 4409
    ws2.close.assert_awaited()

    await owner1.release()


@pytest.mark.asyncio
async def test_acquire_rejects_redis_duplicate():
    from web.connection_manager import acquire_connection

    redis = _make_redis(set_returns=False)   # lease already held
    ws    = _make_ws()
    adapter = _make_adapter()

    owner = await acquire_connection("sess3", "b", ws, adapter, redis=redis)
    assert owner is None
    ws.close.assert_awaited()


@pytest.mark.asyncio
async def test_acquire_acquires_redis_lease_with_correct_args():
    from web.connection_manager import acquire_connection
    from web.protocol import REDIS_LEASE_TTL_SEC

    redis   = _make_redis(set_returns=True)
    ws      = _make_ws()
    adapter = _make_adapter()

    owner = await acquire_connection("sess4", "a", ws, adapter, redis=redis)
    assert owner is not None

    # Redis SET must have been called with NX=True and EX=TTL
    call_kwargs = redis.set.call_args[1]
    assert call_kwargs.get("nx")  is True
    assert call_kwargs.get("ex")  == REDIS_LEASE_TTL_SEC

    await owner.release()


@pytest.mark.asyncio
async def test_release_calls_adapter_close_exactly_once():
    from web.connection_manager import acquire_connection

    ws      = _make_ws()
    adapter = _make_adapter()
    owner   = await acquire_connection("sess5", "a", ws, adapter, redis=None)

    await owner.release()
    await owner.release()   # idempotent

    adapter.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_release_removes_from_registry():
    from web.connection_manager import acquire_connection, get_owner

    ws      = _make_ws()
    adapter = _make_adapter()
    owner   = await acquire_connection("sess6", "a", ws, adapter, redis=None)

    assert get_owner("sess6", "a") is not None
    await owner.release()
    assert get_owner("sess6", "a") is None


@pytest.mark.asyncio
async def test_release_deletes_redis_lease_with_correct_token():
    from web.connection_manager import acquire_connection

    token = None

    def fake_get(key):
        nonlocal token
        return token

    redis = MagicMock()
    redis.set = MagicMock(return_value=True)
    redis.get = MagicMock(side_effect=fake_get)
    redis.expire = MagicMock()
    redis.delete = MagicMock()

    ws      = _make_ws()
    adapter = _make_adapter()
    owner   = await acquire_connection("sess7", "b", ws, adapter, redis=redis)

    # Capture the token the owner wrote
    token = owner._lease_token

    await owner.release()
    redis.delete.assert_called()


@pytest.mark.asyncio
async def test_enqueue_writer_delivers_messages():
    from web.connection_manager import acquire_connection

    ws      = _make_ws()
    adapter = _make_adapter()
    owner   = await acquire_connection("sess8", "a", ws, adapter, redis=None)

    msg = {"type": "test_message", "turn_id": "t1"}
    await owner.enqueue(msg)

    # Give the writer task a cycle to run
    await asyncio.sleep(0.05)

    ws.send_json.assert_awaited_with(msg)

    await owner.release()


@pytest.mark.asyncio
async def test_two_different_speakers_can_coexist():
    from web.connection_manager import acquire_connection, get_owner

    ws_a  = _make_ws(); adapter_a = _make_adapter()
    ws_b  = _make_ws(); adapter_b = _make_adapter()

    owner_a = await acquire_connection("sess9", "a", ws_a, adapter_a, redis=None)
    owner_b = await acquire_connection("sess9", "b", ws_b, adapter_b, redis=None)

    assert owner_a is not None
    assert owner_b is not None
    assert get_owner("sess9", "a") is owner_a
    assert get_owner("sess9", "b") is owner_b

    await owner_a.release()
    await owner_b.release()

    assert get_owner("sess9", "a") is None
    assert get_owner("sess9", "b") is None


@pytest.mark.asyncio
async def test_send_audio_end_message_shape():
    from web.connection_manager import acquire_connection
    from web.protocol import MSG_AUDIO_END

    ws      = _make_ws()
    adapter = _make_adapter()
    owner   = await acquire_connection("sess10", "a", ws, adapter, redis=None)

    await owner.send_audio_end("turn-abc", reason="completed")
    await asyncio.sleep(0.05)

    sent = ws.send_json.call_args[0][0]
    assert sent["type"]    == MSG_AUDIO_END
    assert sent["turn_id"] == "turn-abc"
    assert sent["reason"]  == "completed"

    await owner.release()


@pytest.mark.asyncio
async def test_send_turn_error_message_shape():
    from web.connection_manager import acquire_connection
    from web.protocol import MSG_TURN_ERROR, TurnErrorCode

    ws      = _make_ws()
    adapter = _make_adapter()
    owner   = await acquire_connection("sess11", "a", ws, adapter, redis=None)

    await owner.send_turn_error("turn-xyz", TurnErrorCode.NMT_ERROR, "translation failed", retryable=True)
    await asyncio.sleep(0.05)

    sent = ws.send_json.call_args[0][0]
    assert sent["type"]     == MSG_TURN_ERROR
    assert sent["turn_id"]  == "turn-xyz"
    assert sent["code"]     == TurnErrorCode.NMT_ERROR
    assert sent["retryable"] is True

    await owner.release()
