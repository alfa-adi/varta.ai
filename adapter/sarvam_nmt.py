"""
adapters/sarvam_nmt.py
──────────────────────
Sarvam Translate — Text-to-Text Translation adapter.

API details (confirmed from docs.sarvam.ai):
  POST https://api.sarvam.ai/translate
  Auth: Header  api-subscription-key: <key>
  Body: application/json
    {
      "input":                  "text to translate",
      "source_language_code":   "hi-IN",   # optional — API auto-detects if omitted
      "target_language_code":   "ta-IN",
      "model":                  "sarvam-translate",
      "enable_preprocessing":   true
    }
  Response JSON:
    { "translated_text": "..." }

  Note from changelog: "Translation APIs now automatically detect input language"
  → source_language_code is optional but providing it improves accuracy.
"""

import time
import httpx

from adapter.base import BaseNMTAdapter
from pipeline.types import NMTInput, NMTOutput


SARVAM_TRANSLATE_ENDPOINT = "https://api.sarvam.ai/translate"
MODEL_ID = "sarvam/sarvam-translate:v1"


class SarvamNMTAdapter(BaseNMTAdapter):
    """
    Wraps Sarvam's /translate endpoint.
    The caller never sees JSON body construction — only NMTInput/NMTOutput.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    async def translate(self, input: NMTInput) -> NMTOutput:
        """
        Translate text from src_language to tgt_language.

        Key behaviours:
          - enable_preprocessing=True lets Sarvam handle numbers, dates,
            proper nouns, and addresses natively — reduces context loss
          - We pass src_language even though it's optional, because we
            always know it (the ASR layer detected it for us)
          - If for some reason src == tgt (same language conversation),
            we short-circuit and return the original text immediately
        """
        # Short-circuit: no point calling the API if languages are identical
        if input.src_language == input.tgt_language:
            return NMTOutput(
                translated_text = input.text,
                src_language    = input.src_language,
                tgt_language    = input.tgt_language,
                latency_ms      = 0,
                model_id        = MODEL_ID + "/passthrough",
            )

        start_ms = int(time.time() * 1000)

        payload = {
            "input":                input.text,
            "source_language_code": input.src_language,
            "target_language_code": input.tgt_language,
            "model":                "sarvam-translate:v1",
            "enable_preprocessing": True,    # handles numerals, dates, addresses
        }

        # ── Timing hooks ─────────────────────────────────────────────
        timing = {}

        async def on_request(request):
            timing['request_sent'] = int(time.time() * 1000)

        async def on_response(response):
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
                SARVAM_TRANSLATE_ENDPOINT,
                headers={
                    "api-subscription-key": self._api_key,
                    "Content-Type":         "application/json",
                },
                json=payload,
            )
            tcp_ms = timing.get('request_sent', tcp_start) - tcp_start
            api_ms = timing.get('response_received', int(time.time() * 1000)) - timing.get('request_sent', tcp_start)

        if response.status_code >= 400:
            # Log the error body for debugging
            try:
                err_body = response.json()
            except Exception:
                err_body = response.text
            raise RuntimeError(
                f"Sarvam NMT API error {response.status_code}: {err_body}"
            )

        parse_start = int(time.time() * 1000)
        body = response.json()

        latency_ms = int(time.time() * 1000) - start_ms

        # Sarvam returns { "translated_text": "..." }
        return NMTOutput(
            translated_text = body.get("translated_text", ""),
            src_language    = input.src_language,
            tgt_language    = input.tgt_language,
            latency_ms      = latency_ms,
            model_id        = MODEL_ID,
            tcp_ms          = tcp_ms,
            api_ms          = api_ms,
            parse_ms        = int(time.time() * 1000) - parse_start,
        )