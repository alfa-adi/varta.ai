import asyncio
import os
import sys

# Fake credentials
os.environ["SARVAM_API_KEY"] = "fake"
os.environ["MONGODB_URI"] = "mongodb://localhost:27017" # mock URI

class FakeAdapter:
    def __init__(self, *args, **kwargs):
        self.q = asyncio.Queue()

    async def start_session(self, *args, **kwargs):
        pass

    async def stream_chunk(self, *args, **kwargs):
        pass

    async def signal_speech_end(self, *args, **kwargs):
        self.q.put_nowait({
            "transcript": "fake final",
            "language": "hi-IN",
            "is_partial": False
        })
        self.q.put_nowait(None)

    async def close(self, *args, **kwargs):
        pass

    async def listen_transcripts(self):
        while True:
            msg = await self.q.get()
            if msg is None:
                break
            yield msg

class FakePipeline:
    def __init__(self, *args, **kwargs):
        pass

    async def run_from_transcript(self, *args, **kwargs):
        yield b"fake_audio_chunk"

# Inject the fakes before importing web.server
import adapter.sarvam_asr
adapter.sarvam_asr.SarvamLiveASRAdapter = FakeAdapter

import pipeline.single
pipeline.single.SinglePipeline = FakePipeline

from web.server import app
import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1])
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")
