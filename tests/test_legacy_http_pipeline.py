import asyncio
from fastapi.testclient import TestClient
from web.server import app
import json

client = TestClient(app)

def test():
    print("Creating session...")
    resp = client.post("/session/create", data={"lang_a": "hi-IN", "lang_b": "ta-IN"})
    print(resp.status_code, resp.text)
    session_id = resp.json().get("session_id")
    
    print("Sending audio to speaker A...")
    with open("audio.wav", "wb") as f:
        f.write(b"dummy audio data")
        
    with open("audio.wav", "rb") as f:
        # We expect a 500 or potentially the specific Sarvam error since it's dummy data
        resp = client.post(
            "/translate/speaker_a", 
            data={"session_id": session_id},
            files={"audio": ("audio.wav", f, "audio/wav")}
        )
    print(resp.status_code)
    try:
        print(json.dumps(resp.json(), indent=2))
    except:
        print(resp.text)

if __name__ == "__main__":
    test()
