"""Tests for the conversation-level single-speaker turn lock."""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def clear_turn_claims():
    import web.connection_manager as cm

    cm._session_turns.clear()
    yield
    cm._session_turns.clear()


@pytest.mark.asyncio
async def test_only_one_speaker_can_claim_a_session_turn():
    from web.connection_manager import acquire_session_turn, release_session_turn

    assert await acquire_session_turn("session-1", "a", "turn-a") is True
    assert await acquire_session_turn("session-1", "b", "turn-b") is False

    await release_session_turn("session-1", "a", "turn-a")
    assert await acquire_session_turn("session-1", "b", "turn-b") is True


@pytest.mark.asyncio
async def test_concurrent_speakers_have_one_winner():
    from web.connection_manager import acquire_session_turn

    results = await asyncio.gather(
        acquire_session_turn("session-2", "a", "turn-a"),
        acquire_session_turn("session-2", "b", "turn-b"),
    )

    assert sorted(results) == [False, True]
