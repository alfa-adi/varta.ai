import pytest
import asyncio
from fastapi import WebSocketDisconnect
from web import server
from web.services import conversation_service

def test_anonymous_create_no_mongo(client, monkeypatch):
    monkeypatch.setattr(server, "_conv_store_enabled", lambda: False)
    monkeypatch.setattr(server, "MONGO_URL", None)
    response = client.post("/session/create", data={"lang_a": "hi-IN", "lang_b": "en-IN"})
    assert response.status_code == 200

def test_create_durable_requires_redis(client, monkeypatch):
    monkeypatch.setattr(server, "_redis", None)
    from web.auth import AuthenticatedUser
    server.app.dependency_overrides[server.get_current_user] = lambda: AuthenticatedUser(user_id="user123")
    try:
        response = client.post(
            "/session/create_durable",
            data={"customer_id": "cust1"},
            headers={"Authorization": "Bearer fake_token"}
        )
        assert response.status_code == 503
    finally:
        server.app.dependency_overrides.clear()

def test_create_durable_requires_bearer(client, mock_redis):
    response = client.post("/session/create_durable", data={"customer_id": "cust1"})
    assert response.status_code == 401

def test_save_session_retains_durable(mock_redis):
    server.save_session("sess1", "hi", "en", is_durable=True, durable_user_id="u1")
    class DummyState:
        lang_a = "hi"
        lang_b = "en"
        pending_transcript_a = None
        pending_transcript_b = None
    class DummyPipeline:
        state = DummyState()
    pipeline = DummyPipeline()
    server.update_pipeline_state("sess1", pipeline)
    state = server.load_session("sess1")
    assert state["is_durable"] is True
    assert state["durable_user_id"] == "u1"

def test_save_session_clears_pending_turn_id(mock_redis):
    server.save_session("sess2", "hi", "en", pending_turn_id_a="turn1")
    server.save_session("sess2", "hi", "en", pending_turn_id_a=None)
    state = server.load_session("sess2")
    assert state.get("pending_turn_id_a") is None

def test_save_session_pending_a_none_clears_transcript(mock_redis):
    server.save_session("sess3", "hi", "en", pending_a="text")
    server.save_session("sess3", "hi", "en", pending_a=None)
    state = server.load_session("sess3")
    assert state.get("pending_transcript_a") is None

def test_dual_get_turn_fail_open(client, monkeypatch, mock_redis):
    server.save_session("sess4", "hi-IN", "en-IN")
    
    def mock_get_turn(*args):
        raise RuntimeError("DB down")
        
    monkeypatch.setattr(conversation_service.conv_svc, "get_turn_sync", mock_get_turn)
    
    files = {
        "audio_a": ("a.wav", b"123", "audio/wav"),
        "audio_b": ("b.wav", b"123", "audio/wav")
    }
    response = client.post(
        "/translate/dual",
        data={"session_id": "sess4"},
        files=files
    )
    assert response.status_code == 200

def test_buffered_a_sets_pending_turn_id(client, mock_redis, mock_mongo):
    server.save_session("sess5", "hi-IN", None)
    files = {
        "audio": ("a.wav", b"123", "audio/wav")
    }
    response = client.post(
        "/translate/speaker_a",
        data={"session_id": "sess5"},
        files=files
    )
    assert response.status_code == 200
    state = server.load_session("sess5")
    assert state.get("pending_turn_id_a") is not None

@pytest.mark.asyncio
async def test_persist_queue_worker_order():
    q = asyncio.Queue()
    q.put_nowait({"op": "reserve", "turn_id": "t1", "fp": None})
    q.put_nowait({"op": "update_asr", "turn_id": "t1", "text": "hello", "src": "hi"})
    q.put_nowait({"op": "complete", "turn_id": "t1", "trans_doc": {}})
    q.put_nowait(None)
    
    calls = []
    class DummySvc:
        def reserve_turn_sync(self, *args): calls.append("reserve")
        def update_asr_sync(self, *args): calls.append("asr")
        def complete_turn_sync(self, *args): calls.append("complete")
        
    class DummyOwner:
        _closed = False
        async def enqueue(self, msg): pass
        
    await server.persist_queue_worker(
        q, user_id="u1", session_id="s1", speaker="a",
        conv_svc=DummySvc(), owner=DummyOwner()
    )
    assert calls == ["reserve", "asr", "complete"]

@pytest.mark.asyncio
async def test_persist_worker_skips_after_reserve_fail():
    q = asyncio.Queue()
    q.put_nowait({"op": "reserve", "turn_id": "t1", "fp": None})
    q.put_nowait({"op": "update_asr", "turn_id": "t1", "text": "hello", "src": "hi"})
    q.put_nowait({"op": "complete", "turn_id": "t1", "trans_doc": {}})
    q.put_nowait(None)
    
    calls = []
    class DummySvc:
        def reserve_turn_sync(self, *args): 
            calls.append("reserve")
            raise Exception("fail")
        def update_asr_sync(self, *args): calls.append("asr")
        def complete_turn_sync(self, *args): calls.append("complete")
        
    class DummyOwner:
        _closed = False
        async def enqueue(self, msg): pass
        
    await server.persist_queue_worker(
        q, user_id="u1", session_id="s1", speaker="a",
        conv_svc=DummySvc(), owner=DummyOwner()
    )
    assert calls == ["reserve"]

def test_durable_ws_without_token_raises(mock_redis):
    server.save_session("sess6", "hi-IN", "en-IN", is_durable=True, durable_user_id="u1")
    
    from fastapi.testclient import TestClient
    c = TestClient(server.app, raise_server_exceptions=False)
    
    with pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/ws/asr/sess6/a") as ws:
            ws.receive_text()
    assert exc.value.code == 1008

def test_mark_failed_does_not_overwrite_completed(mock_mongo):
    db = mock_mongo
    db.translation_turns.insert_one({
        "turn_id": "t1",
        "user_id": "u1",
        "status": "completed"
    })
    
    from web.storage import conversation_repository as repo
    repo.mark_turn_failed_sync("u1", "t1", "err")
    
    doc = db.translation_turns.find_one({"turn_id": "t1"})
    assert doc["status"] == "completed"
