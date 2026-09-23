import asyncio
import os
from dotenv import load_dotenv
import math
import struct

load_dotenv()
API_KEY = os.environ.get("SARVAM_API_KEY")

async def test_live_asr():
    print("--- Testing SarvamLiveASRAdapter ---")
    from adapter.sarvam_asr import SarvamLiveASRAdapter
    
    if not API_KEY:
        print("Error: SARVAM_API_KEY not found in environment.")
        return

    adapter = SarvamLiveASRAdapter(api_key=API_KEY)
    
    # 1. Connect
    print("1. Connecting to session...")
    await adapter.start_session(language_hint="hi-IN")
    
    # 2. Generate 1 second of 16kHz dummy PCM data (sine wave) to simulate speech
    print("2. Generating and sending dummy PCM audio (1 sec)...")
    sample_rate = 16000
    frequency = 440.0 # A4 note
    duration = 1.0
    num_samples = int(sample_rate * duration)
    
    # Pack as 16-bit signed little-endian (pcm_s16le)
    pcm_bytes = bytearray()
    for i in range(num_samples):
        val = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
        pcm_bytes.extend(struct.pack('<h', val))
        
    # Send in chunks of 3200 bytes (100ms)
    chunk_size = 3200
    for i in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[i:i+chunk_size]
        await adapter.stream_chunk(chunk)
        await asyncio.sleep(0.1) # Simulate real-time streaming
        
    # 3. Flush utterance and wait for response
    print("3. Flushing utterance and waiting for final transcript...")
    async for frame in adapter.flush_utterance():
        print(f"   Received frame: {frame}")
        
    # 4. Close
    print("4. Closing session...")
    await adapter.close()
    print("ASR Test Completed.\n")


async def test_tts():
    print("--- Testing SarvamTTSAdapter ---")
    from adapter.sarvam_tts import SarvamTTSAdapter
    from pipeline.types import TTSInput
    
    if not API_KEY:
        print("Error: SARVAM_API_KEY not found in environment.")
        return
        
    adapter = SarvamTTSAdapter(api_key=API_KEY)
    tts_input = TTSInput(
        text="Hello world, this is a test of the text to speech system.",
        language="en-IN"
    )
    
    print(f"1. Synthesizing streaming for text: '{tts_input.text}'")
    total_bytes = 0
    chunk_count = 0
    
    try:
        async for audio_chunk in adapter.synthesise_streaming(tts_input):
            chunk_count += 1
            total_bytes += len(audio_chunk)
            print(f"   Received MP3 chunk {chunk_count}: {len(audio_chunk)} bytes")
            
        print(f"2. Total bytes received: {total_bytes}")
        if chunk_count > 0 and total_bytes > 0:
            print("TTS Test Passed! Successfully received MP3 chunks.")
        else:
            print("TTS Test Failed: No audio chunks received.")
    except Exception as e:
        print(f"TTS Test Failed with exception: {e}")

async def main():
    await test_live_asr()
    await test_tts()

if __name__ == "__main__":
    asyncio.run(main())
