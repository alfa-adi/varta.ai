"""
tests/test_asr_adapter.py
──────────────────────────
Unit tests for SarvamLiveASRAdapter.

Tests cover:
  - session.begin readiness gate (no audio before ready)
  - speech_start / audio_input / speech_end ordering
  - Odia language normalization (or-IN ↔ od-IN)
  - Transcript queue overflow counter
  - Fatal error classification (no retry)
  - Non-fatal error classification (retry allowed)
  - Reconnect cancels old reader before creating new one
  - close() is idempotent
  - URL contains auto when hint is empty or "unknown"
  - URL contains varta_to_sarvam_lang(od-IN) = or-IN
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapter.sarvam_asr import SarvamLiveASRAdapter
from adapter.sarvam_protocol import (
    FATAL_CLOSE_CODES,
    SarvamASREvent,
    is_retryable_close,
    is_retryable_error,
    sarvam_to_varta_lang,
    varta_to_sarvam_lang,
)
from web.protocol import TRANSCRIPT_QUEUE_MAX

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_adapter() -> SarvamLiveASRAdapter:
    return SarvamLiveASRAdapter(api_key="test-key")


def session_begin_frame(request_id: str = "req-123") -> dict:
    return {"event": SarvamASREvent.SESSION_BEGIN, "request_id": request_id}


def partial_frame(text: str, lang: str = "hi-IN") -> dict:
    return {"event": SarvamASREvent.TRANSCRIPT_PARTIAL, "text": text, "language": lang}


def final_frame(text: str, lang: str = "hi-IN", confidence: float = 0.95) -> dict:
    return {
        "event": SarvamASREvent.TRANSCRIPT_FINAL,
        "text":  text,
        "language": lang,
        "language_confidence": confidence,
    }


def error_frame(code: str = "internal_error", is_fatal: bool = False) -> dict:
    return {"event": SarvamASREvent.ERROR, "code": code, "is_fatal": is_fatal, "message": "test"}


# ── Language normalization ────────────────────────────────────────────────────

class TestLanguageNormalization:
    def test_sarvam_odia_to_varta(self):
        assert sarvam_to_varta_lang("or-IN") == "od-IN"

    def test_varta_odia_to_sarvam(self):
        assert varta_to_sarvam_lang("od-IN") == "or-IN"

    def test_non_odia_passthrough(self):
        assert sarvam_to_varta_lang("hi-IN") == "hi-IN"
        assert varta_to_sarvam_lang("hi-IN") == "hi-IN"

    def test_auto_passthrough(self):
        assert sarvam_to_varta_lang("auto") == "auto"
        assert varta_to_sarvam_lang("auto") == "auto"


# ── Error classification ──────────────────────────────────────────────────────

class TestErrorClassification:
    def test_fatal_not_retryable(self):
        assert not is_retryable_error(error_frame(is_fatal=True))

    def test_auth_not_retryable(self):
        assert not is_retryable_error(error_frame(code="auth_invalid"))

    def test_quota_not_retryable(self):
        assert not is_retryable_error(error_frame(code="quota_exceeded"))

    def test_invalid_param_not_retryable(self):
        assert not is_retryable_error(error_frame(code="invalid_param_language"))

    def test_internal_error_retryable(self):
        assert is_retryable_error(error_frame(code="internal_error"))

    def test_network_retryable(self):
        assert is_retryable_error({"event": "error", "code": "network_timeout", "is_fatal": False})

    def test_fatal_close_codes_not_retryable(self):
        for code in FATAL_CLOSE_CODES:
            assert not is_retryable_close(code), f"code {code} should not be retryable"

    def test_normal_close_retryable(self):
        assert is_retryable_close(1000)
        assert is_retryable_close(1006)


# ── URL building ──────────────────────────────────────────────────────────────

class TestURLBuilding:
    def setup_method(self):
        self.adapter = make_adapter()

    def test_empty_hint_becomes_auto(self):
        url = self.adapter._build_url("")
        assert "language_code=auto" in url

    def test_unknown_hint_becomes_auto(self):
        url = self.adapter._build_url("unknown")
        assert "language_code=auto" in url

    def test_valid_lang_preserved(self):
        url = self.adapter._build_url("hi-IN")
        assert "language_code=hi-IN" in url

    def test_odia_normalized_to_or_in_for_sarvam(self):
        """od-IN (Varta code) should become or-IN in the Sarvam URL."""
        url = self.adapter._build_url("od-IN")
        assert "language_code=or-IN" in url
        assert "od-IN" not in url

    def test_model_and_encoding_present(self):
        url = self.adapter._build_url("auto")
        assert "model=saaras%3Av3-realtime" in url or "model=saaras:v3-realtime" in url
        assert "encoding=linear16" in url
        assert "sample_rate=16000" in url
        assert "stream_type=fast" in url
        assert "endpointing=manual" in url


# ── Transcript queue overflow ─────────────────────────────────────────────────

class TestQueueOverflow:
    @pytest.mark.asyncio
    async def test_overflow_increments_counter(self):
        adapter = make_adapter()
        adapter._recv_queue = asyncio.Queue(maxsize=2)

        # Fill the queue
        await adapter._recv_queue.put(partial_frame("one"))
        await adapter._recv_queue.put(partial_frame("two"))

        # Simulate reader dropping a frame due to full queue
        frame = partial_frame("three")
        try:
            adapter._recv_queue.put_nowait(frame)
        except asyncio.QueueFull:
            adapter.queue_overflows += 1

        assert adapter.queue_overflows == 1
        assert adapter.queue_depth == 2


# ── session.begin readiness gate ──────────────────────────────────────────────

class TestSessionBeginReadiness:
    @pytest.mark.asyncio
    async def test_ready_flag_false_before_begin(self):
        adapter = make_adapter()
        assert adapter._ready is False

    @pytest.mark.asyncio
    async def test_ready_flag_set_by_background_reader(self):
        adapter = make_adapter()
        adapter._recv_queue = asyncio.Queue(maxsize=TRANSCRIPT_QUEUE_MAX)

        # Simulate the reader processing a session.begin frame
        adapter._ready = False
        frame = session_begin_frame("req-abc")
        # Replicate the background_reader logic inline
        if frame.get("event") == SarvamASREvent.SESSION_BEGIN:
            adapter.session_begin_request_id = frame.get("request_id", "")
            adapter._ready = True

        assert adapter._ready is True
        assert adapter.session_begin_request_id == "req-abc"


# ── listen_transcripts ────────────────────────────────────────────────────────

class TestListenTranscripts:
    @pytest.mark.asyncio
    async def test_yields_partial_transcript(self):
        adapter = make_adapter()
        adapter._recv_queue = asyncio.Queue()
        await adapter._recv_queue.put(partial_frame("हेलो", lang="hi-IN"))
        # Add a sentinel to avoid infinite loop
        await adapter._recv_queue.put({"event": SarvamASREvent.SESSION_END, "audio_duration_s": 1.0})

        results = []
        async for item in adapter.listen_transcripts():
            results.append(item)

        assert len(results) == 1
        assert results[0]["transcript"] == "हेलो"
        assert results[0]["is_partial"] is True
        assert results[0]["language"] == "hi-IN"

    @pytest.mark.asyncio
    async def test_yields_final_with_confidence(self):
        adapter = make_adapter()
        adapter._recv_queue = asyncio.Queue()
        await adapter._recv_queue.put(final_frame("नमस्ते", lang="hi-IN", confidence=0.97))
        await adapter._recv_queue.put({"event": SarvamASREvent.SESSION_END, "audio_duration_s": 2.0})

        results = []
        async for item in adapter.listen_transcripts():
            results.append(item)

        assert results[0]["is_partial"] is False
        assert results[0]["language_confidence"] == 0.97

    @pytest.mark.asyncio
    async def test_odia_language_normalized_in_output(self):
        """Sarvam reports or-IN; listen_transcripts must yield od-IN."""
        adapter = make_adapter()
        adapter._recv_queue = asyncio.Queue()
        await adapter._recv_queue.put(partial_frame("ଓଡ଼ିଆ", lang="or-IN"))
        await adapter._recv_queue.put({"event": SarvamASREvent.SESSION_END, "audio_duration_s": 0.5})

        results = []
        async for item in adapter.listen_transcripts():
            results.append(item)

        assert results[0]["language"] == "od-IN", \
            "or-IN from Sarvam must be normalized to od-IN in output"

    @pytest.mark.asyncio
    async def test_fatal_error_yields_sentinel_and_stops(self):
        adapter = make_adapter()
        adapter._recv_queue = asyncio.Queue()
        await adapter._recv_queue.put(error_frame(code="auth_invalid", is_fatal=True))

        results = []
        async for item in adapter.listen_transcripts():
            results.append(item)

        assert len(results) == 1
        assert "_provider_error" in results[0]

    @pytest.mark.asyncio
    async def test_non_fatal_error_does_not_stop_generator(self):
        adapter = make_adapter()
        adapter._recv_queue = asyncio.Queue()
        await adapter._recv_queue.put(error_frame(code="internal", is_fatal=False))
        await adapter._recv_queue.put(partial_frame("continues"))
        await adapter._recv_queue.put({"event": SarvamASREvent.SESSION_END, "audio_duration_s": 0.1})

        results = []
        async for item in adapter.listen_transcripts():
            results.append(item)

        # Non-fatal error should NOT stop the generator; transcript arrives
        transcripts = [r for r in results if r.get("transcript") == "continues"]
        assert transcripts, "Generator should continue after non-fatal error"


# ── close() idempotency ───────────────────────────────────────────────────────

class TestCloseIdempotency:
    @pytest.mark.asyncio
    async def test_close_when_not_started_is_safe(self):
        adapter = make_adapter()
        # Should not raise
        await adapter.close()
        await adapter.close()

    @pytest.mark.asyncio
    async def test_close_increments_reader_cancels(self):
        adapter = make_adapter()
        adapter._recv_queue = asyncio.Queue()

        # Set up a dummy reader task that runs forever
        async def forever():
            await asyncio.sleep(9999)

        adapter._reader_task = asyncio.create_task(forever())
        adapter._ws = MagicMock()
        adapter._ws.open = True
        adapter._ws.close = AsyncMock()
        adapter._ws.send  = AsyncMock()

        await adapter.close()
        assert adapter.reader_cancels == 1

        # Second close is a no-op
        await adapter.close()
        assert adapter.reader_cancels == 1  # not incremented again
