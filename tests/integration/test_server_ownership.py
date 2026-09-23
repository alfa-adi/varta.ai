"""
Backend Integration Test for Redis active_turn_id Ownership
"""
import asyncio
import json
import socket
import subprocess
import sys
import time

import pytest
import redis
import websockets
import httpx

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port

def check_redis():
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=2)
        r.ping()
        return True
    except Exception:
        return False

@pytest.fixture(scope="module")
def server_port():
    if not check_redis():
        pytest.fail("Redis must be running locally to test Redis turn lease ownership.")
        
    port = get_free_port()
    proc = subprocess.Popen(
        [sys.executable, "tests/integration/launch_test_server.py", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to start
    url = f"http://127.0.0.1:{port}/health"
    started = False
    for _ in range(30):
        try:
            with httpx.Client() as client:
                res = client.get(url, timeout=0.5)
                if res.status_code == 200:
                    started = True
                    break
        except Exception:
            pass
        time.sleep(0.1)
        
    if not started:
        proc.kill()
        pytest.fail("Failed to start fake server.")
        
    yield port
    
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()

@pytest.mark.asyncio
async def test_redis_active_turn_lease_released_at_audio_end(server_port):
    # Setup test session
    async with httpx.AsyncClient() as client:
        res = await client.post(f"http://127.0.0.1:{server_port}/session/create", data={"lang_a": "en-IN", "lang_b": "hi-IN"})
        assert res.status_code == 200
        session_id = res.json()["session_id"]
        
    uri = f"ws://127.0.0.1:{server_port}/ws/asr/{session_id}/a"
    
    async with websockets.connect(uri) as ws:
        # Wait for server_ready
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "server_ready":
                break

        # 1. Start Turn 1
        turn_1_id = "turn-1111-1111"
        await ws.send(json.dumps({"type": "turn_start", "turn_id": turn_1_id}))
        
        # 2. End Turn 1 (triggers final transcript -> pipeline -> audio chunks -> audio_end)
        await ws.send(json.dumps({"type": "stop_recording", "turn_id": turn_1_id}))
        
        # 3. Wait for audio_end
        received_audio_end = False
        while not received_audio_end:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "audio_end" and msg.get("turn_id") == turn_1_id:
                received_audio_end = True
                
        # 4. Immediate Start Turn 2
        turn_2_id = "turn-2222-2222"
        await ws.send(json.dumps({"type": "turn_start", "turn_id": turn_2_id}))
        
        # Wait a bit to ensure no TURN_IN_PROGRESS error arrives
        error_msg = None
        try:
            # wait 0.5 seconds for any incoming error message
            while True:
                msg_str = await asyncio.wait_for(ws.recv(), timeout=0.5)
                msg = json.loads(msg_str)
                if msg.get("type") == "turn_error" and msg.get("code") == "TURN_IN_PROGRESS":
                    error_msg = msg
                    break
        except asyncio.TimeoutError:
            pass # No error received, this is the expected successful path
            
        assert error_msg is None, f"Turn 2 was rejected: {error_msg}"
