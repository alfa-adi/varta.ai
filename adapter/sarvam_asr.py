"""
adapters/sarvam_asr.py
──────────────────────
Sarvam Saaras v3 — Speech-to-Text adapter.

API details (confirmed from docs.sarvam.ai):
  POST https://api.sarvam.ai/speech-to-text
  Auth: Header  api-subscription-key: <key>
  Body: multipart/form-data
    - file          : audio bytes (wav/mp3/ogg/webm/flac etc.)
    - model         : "saaras:v3"
    - language_code : BCP-47 string e.g. "hi-IN"  (optional — omit for auto-detect)
    - mode          : "transcribe" | "codemix" | "verbatim" | "translit"
  Response JSON:
    { "transcript": "...", "language_code": "hi-IN" }
"""

import time
import httpx

from adapter.base import BaseASRAdapter
from pipeline.types import ASRInput, ASROutput


SARVAM_ASR_ENDPOINT = "https://api.sarvam.ai/speech-to-text"
MODEL_ID = "sarvam/saaras-v3"


class SarvamASRAdapter(BaseASRAdapter):
    """
    Wraps Sarvam's /speech-to-text endpoint.
    The caller never sees Sarvam's multipart format — only ASRInput/ASROutput.
    """

    def __init__(self, api_key: str):
        # Store the key. We use a custom header — NOT a Bearer token.
        # This is a Sarvam-specific detail hidden from the rest of the system.
        self._api_key = api_key

    async def transcribe(self, input: ASRInput) -> ASROutput:
        """
        Send audio to Saaras v3 and return a clean ASROutput.

        Key behaviours:
          - If input.language_hint is None, we omit language_code entirely
            → Saaras auto-detects the language (our cross-pipeline needs this)
          - The response always includes language_code so we know what was spoken
          - We measure wall-clock latency for the registry feedback loop
        """
        start_ms = int(time.time() * 1000)

        # Build the multipart form fields
        form_data: dict = {
            "model": (None, "saaras:v3"),        # Always use v3
            "mode":  (None, input.mode),          # "transcribe" in our case
        }

        # language_code is OPTIONAL — only send it if the caller provides a hint.
        # Omitting it triggers Saaras's auto-detection, which is exactly what
        # the dual pipeline needs on the first utterance of each speaker.
        if input.language_hint:
            form_data["language_code"] = (None, input.language_hint)

        # The audio file must be a tuple: (filename, bytes, content_type)
        # Sarvam uses the filename extension to help codec detection.
        audio_filename = f"audio.{input.audio_format}"
        mime_map = {
            "wav":  "audio/wav",
            "mp3":  "audio/mpeg",
            "ogg":  "audio/ogg",
            "webm": "audio/webm",
            "flac": "audio/flac",
            "m4a":  "audio/mp4",
        }
        content_type = mime_map.get(input.audio_format, "audio/wav")

        form_data["file"] = (audio_filename, input.audio_bytes, content_type)

        # ── Timing hooks ─────────────────────────────────────────────
        timing = {}

        def on_request(request):
            timing['request_sent'] = int(time.time() * 1000)

        def on_response(response):
            timing['response_received'] = int(time.time() * 1000)

        # Make the async HTTP request
        async with httpx.AsyncClient(
            timeout=60.0,
            event_hooks={
                'request':  [on_request],
                'response': [on_response],
            }
        ) as client:
            tcp_start = int(time.time() * 1000)
            response = await client.post(
                SARVAM_ASR_ENDPOINT,
                headers={"api-subscription-key": self._api_key},
                files=form_data,
            )
            tcp_ms = timing.get('request_sent', tcp_start) - tcp_start
            api_ms = timing.get('response_received', int(time.time() * 1000)) - timing.get('request_sent', tcp_start)

        # Raise immediately on HTTP errors so callers see clean exceptions
        response.raise_for_status()

        parse_start = int(time.time() * 1000)
        body = response.json()

        latency_ms = int(time.time() * 1000) - start_ms

        # Sarvam returns { "transcript": "...", "language_code": "hi-IN" }
        # We normalise this into our universal ASROutput type.
        detected_lang = body.get("language_code", input.language_hint or "unknown")

        return ASROutput(
            transcript        = body.get("transcript", ""),
            detected_language = detected_lang,
            confidence        = body.get("language_confidence", 1.0),
            latency_ms        = latency_ms,
            model_id          = MODEL_ID,
            tcp_ms            = tcp_ms,
            api_ms            = api_ms,
            parse_ms          = int(time.time() * 1000) - parse_start,
        )