import os
import pytest
from unittest.mock import AsyncMock

os.environ["CONVERSATION_PERSISTENCE_ENABLED"] = "true"
os.environ["MONGO_CONVERSATION_DB"] = "test_conv"
os.environ["AUTH_SECRET"] = "test-secret"
os.environ["SARVAM_API_KEY"] = "test"
os.environ["MONGO_URL"] = "mongodb://localhost:27017"

import mongomock
from fastapi.testclient import TestClient
import fakeredis

from adapter.base import ASROutput, NMTOutput, TTSOutput
from web import server
from web.storage import mongo as mongo_store
from pipeline.single import SinglePipeline

@pytest.fixture(autouse=True)
def mock_mongo(monkeypatch):
    client = mongomock.MongoClient()
    db = client.test_conv
    def mock_get_db(): return db
    monkeypatch.setattr(mongo_store, "get_db", mock_get_db)
    monkeypatch.setattr(server, "_conv_store_ready", lambda: True)
    monkeypatch.setattr(server, "_conv_store_enabled", lambda: True)
    yield db

@pytest.fixture
def mock_redis(monkeypatch):
    r = fakeredis.FakeStrictRedis()
    monkeypatch.setattr(server, "_redis", r)
    return r

@pytest.fixture
def client():
    with TestClient(server.app) as c:
        yield c

class StubASR:
    async def transcribe(self, input_obj):
        return ASROutput(transcript="hello", detected_language="hi-IN", confidence=1.0, latency_ms=10, model_id="stub", tcp_ms=1, api_ms=5, parse_ms=1)
        
    async def transcribe_dual(self, input_a, input_b):
        return (
            ASROutput(transcript="hello", detected_language="hi-IN", confidence=1.0, latency_ms=10, model_id="stub", tcp_ms=1, api_ms=5, parse_ms=1),
            ASROutput(transcript="hello", detected_language="hi-IN", confidence=1.0, latency_ms=10, model_id="stub", tcp_ms=1, api_ms=5, parse_ms=1)
        )

class StubNMT:
    async def translate(self, input_obj):
        return NMTOutput(translated_text="hello", src_language="hi-IN", tgt_language="en-IN", latency_ms=10, model_id="stub", tcp_ms=1, api_ms=5, parse_ms=1)

class StubTTS:
    async def synthesise(self, input_obj):
        return TTSOutput(audio_bytes=b"123", audio_format="pcm", language="en-IN", latency_ms=10, model_id="stub", tcp_ms=1, api_ms=5, parse_ms=1)
        
    async def synthesise_streaming(self, input_obj):
        yield b"chunk1"
        yield b"chunk2"

@pytest.fixture(autouse=True)
def mock_adapters(monkeypatch):
    monkeypatch.setattr(server, "_asr", StubASR())
    monkeypatch.setattr(server, "_nmt", StubNMT())
    monkeypatch.setattr(server, "_tts", StubTTS())
    # clear cache so get_or_create_session picks up new stubs
    if hasattr(server, "_session_pipelines"):
        server._session_pipelines.clear()
