"""
adapters/sarvam_tts.py
──────────────────────
Sarvam Bulbul v3 — Text-to-Speech adapter.

API details (confirmed from docs.sarvam.ai):
  POST https://api.sarvam.ai/text-to-speech
  Auth: Header  api-subscription-key: <key>
  Body: application/json
    {
      "inputs":              [{"text": "text to speak"}],
      "target_language_code": "hi-IN",
      "model":               "bulbul:v3",
      "speaker":             "meera",
      "pitch":               0,
      "pace":                1.0,
      "loudness":            1.5,
      "speech_sample_rate":  22050,
      "enable_preprocessing": true,
      "enc_format":          "mp3"
    }
  Response JSON:
    { "audios": ["<base64-encoded-audio>"] }

  Supported languages: hi-IN bn-IN ta-IN te-IN gu-IN kn-IN ml-IN mr-IN od-IN pa-IN en-IN
  Note: Bulbul v3 only covers 11 languages. For the remaining 11 scheduled languages,
        a fallback to Bhashini IITM TTS would be needed in a full system.
"""

import base64
import time
import httpx

from adapter.base import BaseTTSAdapter
from pipeline.types import TTSInput, TTSOutput


SARVAM_TTS_ENDPOINT = "https://api.sarvam.ai/text-to-speech"
MODEL_ID = "sarvam/bulbul-v3"

# Languages Bulbul v3 actually supports
BULBUL_SUPPORTED = {
    "hi-IN", "bn-IN", "ta-IN", "te-IN", "gu-IN",
    "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "en-IN"
}

# Voice selection: pick the most natural-sounding default per language
# These are Sarvam's actual speaker names
DEFAULT_VOICES = {
    "female": {
        "hi-IN": "priya",     "bn-IN": "priya",    "ta-IN": "priya",
        "te-IN": "priya",     "gu-IN": "priya",    "kn-IN": "priya",
        "ml-IN": "priya",     "mr-IN": "priya",    "od-IN": "priya",
        "pa-IN": "priya",     "en-IN": "priya",
    },
    "male": {
        "hi-IN": "amit",      "bn-IN": "amit",     "ta-IN": "amit",
        "te-IN": "amit",      "gu-IN": "amit",     "kn-IN": "amit",
        "ml-IN": "amit",      "mr-IN": "rahul",    "od-IN": "amit",
        "pa-IN": "amit",      "en-IN": "amit",
    },
}


class SarvamTTSAdapter(BaseTTSAdapter):
    """
    Wraps Sarvam's /text-to-speech endpoint.
    Returns decoded audio bytes — the caller never sees base64 or Sarvam JSON.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def synthesise(self, input: TTSInput) -> TTSOutput:
        """
        Convert text to speech in the target language.

        Key behaviours:
          - Automatically picks the best voice for the language+gender combo
          - Decodes base64 response → returns raw bytes (caller-ready)
          - enable_preprocessing handles Indian-language numbers, addresses,
            dates — critical for natural-sounding output
          - Raises ValueError if language isn't supported by Bulbul v3
            (in a full system, this triggers fallback to Bhashini TTS)
        """
        if input.language not in BULBUL_SUPPORTED:
            raise ValueError(
                f"Bulbul v3 does not support '{input.language}'. "
                f"Supported: {sorted(BULBUL_SUPPORTED)}"
            )

        start_ms = int(time.time() * 1000)

        # Select the right voice name for this language + gender
        voice = (
            DEFAULT_VOICES
            .get(input.voice_gender, DEFAULT_VOICES["female"])
            .get(input.language, "priya")
        )

        payload = {
            "text":                 input.text,
            "target_language_code": input.language,
            "model":                "bulbul:v3",
            "speaker":              voice,
            "pace":                 input.pace,          # 0.5–2.0
            "speech_sample_rate":   input.sample_rate,
            "enable_preprocessing": True,
            "enc_format":           input.audio_format,  # "mp3" | "wav" etc.
        }

        # ── Timing hooks ─────────────────────────────────────────────
        timing = {}

        def on_request(request):
            timing['request_sent'] = int(time.time() * 1000)

        def on_response(response):
            timing['response_received'] = int(time.time() * 1000)

        async with httpx.AsyncClient(
            timeout=30.0,
            event_hooks={
                'request':  [on_request],
                'response': [on_response],
            }
        ) as client:
            tcp_start = int(time.time() * 1000)
            response = await client.post(
                SARVAM_TTS_ENDPOINT,
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type":         "application/json",
                },
                json=payload,
            )
            tcp_ms = timing.get('request_sent', tcp_start) - tcp_start
            api_ms = timing.get('response_received', int(time.time() * 1000)) - timing.get('request_sent', tcp_start)

        if response.status_code >= 400:
            try:
                err_body = response.json()
            except Exception:
                err_body = response.text
            raise RuntimeError(
                f"Sarvam TTS API error {response.status_code}: {err_body}"
            )

        parse_start = int(time.time() * 1000)
        body = response.json()

        # Sarvam returns { "audios": ["<base64string>"] }
        # We decode the base64 immediately so the caller gets raw bytes.
        raw_audio = base64.b64decode(body["audios"][0])

        latency_ms = int(time.time() * 1000) - start_ms

        return TTSOutput(
            audio_bytes  = raw_audio,
            audio_format = input.audio_format,
            language     = input.language,
            latency_ms   = latency_ms,
            model_id     = MODEL_ID,
            tcp_ms       = tcp_ms,
            api_ms       = api_ms,
            parse_ms     = int(time.time() * 1000) - parse_start,
        )