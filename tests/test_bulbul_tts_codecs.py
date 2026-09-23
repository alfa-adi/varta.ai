import asyncio
import os
from sarvamai import AsyncSarvamAI

async def test_bulbul_codecs():
    client = AsyncSarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY", ""))
    for codec in ["linear16", "mp3", "pcm", "wav", "mulaw"]:
        try:
            ws_ctx = client.text_to_speech_streaming.connect(
                model="bulbul:v3",
                send_completion_event=True,
            )
            ws = await ws_ctx.__aenter__()
            try:
                await ws.configure(
                    target_language_code="hi-IN",
                    speaker="priya",
                    output_audio_codec=codec,
                    speech_sample_rate=24000,
                    pace=1.0,
                )
                print(f"✅ Codec '{codec}' is SUPPORTED")
            except Exception as e:
                print(f"❌ Codec '{codec}' failed: {e}")
            finally:
                await ws_ctx.__aexit__(None, None, None)
        except Exception as e:
            print(f"Could not connect: {e}")

if __name__ == "__main__":
    asyncio.run(test_bulbul_codecs())
