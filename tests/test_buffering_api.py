"""
Test script for the Speaker A / Speaker B buffering logic.
This script tests the API endpoints directly without needing a browser or microphone.

Usage:
  1. Ensure the server is running (`python web/server.py` or `uvicorn web.server:app`)
  2. Run this script: `python tests/test_buffering_api.py`
"""

import sys
import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_AUDIO_PATH = Path(__file__).parent / "fixtures" / "TestingAudio_AamirKhan_6.wav"

def check_server():
    """Fail fast if the server isn't running."""
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"❌ Server is not running at {BASE_URL}. Please start it first.")
        sys.exit(1)

def test_speaker_a_first():
    print("\n" + "="*50)
    print("🧪 TEST 1: Speaker A speaks first (Language B unknown)")
    print("="*50)

    # 1. Create a new session with no languages
    print("\n1️⃣  Creating new session...")
    resp = requests.post(f"{BASE_URL}/session/create", data={"lang_a": "", "lang_b": ""})
    resp.raise_for_status()
    session_id = resp.json()["session_id"]
    print(f"   ✅ Session created: {session_id}")

    # 2. Speaker A sends audio
    print("\n2️⃣  Speaker A sending audio...")
    with open(TEST_AUDIO_PATH, "rb") as f:
        files = {"audio": ("test.wav", f, "audio/wav")}
        data = {"session_id": session_id}
        resp_a = requests.post(f"{BASE_URL}/translate/speaker_a", files=files, data=data)
    
    resp_a.raise_for_status()
    result_a = resp_a.json()
    
    print("   Response from Speaker A:")
    print(f"   {result_a}")
    
    if result_a.get("status") == "buffered":
        print("   ✅ SUCCESS: Speaker A's audio was buffered because Speaker B's language is unknown.")
    else:
        print("   ❌ FAILURE: Expected a 'buffered' status!")
        sys.exit(1)

    # 3. Speaker B sends audio
    print("\n3️⃣  Speaker B sending audio (this should unlock A's buffered audio)...")
    with open(TEST_AUDIO_PATH, "rb") as f:
        files = {"audio": ("test.wav", f, "audio/wav")}
        data = {"session_id": session_id}
        resp_b = requests.post(f"{BASE_URL}/translate/speaker_b", files=files, data=data)
    
    resp_b.raise_for_status()
    result_b = resp_b.json()
    
    print("   Response from Speaker B:")
    print(f"   - Target translation ready: {'translation' in result_b}")
    print(f"   - Deferred result ready: {'deferred' in result_b}")
    
    if "deferred" in result_b:
        def_res = result_b["deferred"]
        print(f"   ✅ SUCCESS: Received deferred result for speaker '{def_res['speaker']}'.")
        print(f"      Original language: {def_res['src_language']} | Translated to: {def_res['tgt_language']}")
    else:
        print("   ❌ FAILURE: Did not receive Speaker A's deferred result in Speaker B's response!")
        sys.exit(1)


def test_speaker_b_first():
    print("\n" + "="*50)
    print("🧪 TEST 2: Speaker B speaks first (Language A unknown)")
    print("="*50)

    # 1. Create a new session with no languages
    print("\n1️⃣  Creating new session...")
    resp = requests.post(f"{BASE_URL}/session/create", data={"lang_a": "", "lang_b": ""})
    resp.raise_for_status()
    session_id = resp.json()["session_id"]
    print(f"   ✅ Session created: {session_id}")

    # 2. Speaker B sends audio
    print("\n2️⃣  Speaker B sending audio...")
    with open(TEST_AUDIO_PATH, "rb") as f:
        files = {"audio": ("test.wav", f, "audio/wav")}
        data = {"session_id": session_id}
        resp_b = requests.post(f"{BASE_URL}/translate/speaker_b", files=files, data=data)
    
    resp_b.raise_for_status()
    result_b = resp_b.json()
    
    print("   Response from Speaker B:")
    print(f"   {result_b}")
    
    if result_b.get("status") == "buffered":
        print("   ✅ SUCCESS: Speaker B's audio was buffered because Speaker A's language is unknown.")
    else:
        print("   ❌ FAILURE: Expected a 'buffered' status!")
        sys.exit(1)

    # 3. Speaker A sends audio
    print("\n3️⃣  Speaker A sending audio (this should unlock B's buffered audio)...")
    with open(TEST_AUDIO_PATH, "rb") as f:
        files = {"audio": ("test.wav", f, "audio/wav")}
        data = {"session_id": session_id}
        resp_a = requests.post(f"{BASE_URL}/translate/speaker_a", files=files, data=data)
    
    resp_a.raise_for_status()
    result_a = resp_a.json()
    
    print("   Response from Speaker A:")
    print(f"   - Target translation ready: {'translation' in result_a}")
    print(f"   - Deferred result ready: {'deferred' in result_a}")
    
    if "deferred" in result_a:
        def_res = result_a["deferred"]
        print(f"   ✅ SUCCESS: Received deferred result for speaker '{def_res['speaker']}'.")
        print(f"      Original language: {def_res['src_language']} | Translated to: {def_res['tgt_language']}")
    else:
        print("   ❌ FAILURE: Did not receive Speaker B's deferred result in Speaker A's response!")
        sys.exit(1)


if __name__ == "__main__":
    if not TEST_AUDIO_PATH.exists():
        print(f"❌ Test audio file not found at {TEST_AUDIO_PATH}")
        sys.exit(1)
        
    print("Starting integration tests for Speaker A/B buffering logic...")
    check_server()
    
    test_speaker_a_first()
    time.sleep(1) # Breathe
    test_speaker_b_first()
    
    print("\n🎉 ALL TESTS PASSED! The cross-buffering logic works perfectly in both directions.")
