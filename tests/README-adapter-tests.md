# Adapter Unit Tests

This suite contains focused unit tests that validate the behavior of the abstraction layers (adapters) which wrap external services (like ASR, NMT, and TTS models). 

## Files in this suite

- **`test_ws_asr_adapter.py`**
  - **Purpose:** Validates the `SarvamLiveASRAdapter`, which is the core WebSocket abstraction used in the real-time streaming pivot.
  - **Context:** Mocks the backend connection and tests readiness gating, speech start/end ordering, dialect normalization (e.g., Odia `or-IN` ↔ `od-IN`), and fatal/non-fatal error categorization.

- **`test_bulbul_tts_codecs.py`**
  - **Purpose:** Validates the audio format compatibility of the Sarvam TTS engine.
  - **Context:** Tests multiple audio codecs (`linear16`, `mp3`, `pcm`, `wav`, `mulaw`) against the `bulbul:v3` model to ensure the integration successfully receives the expected audio formats.

- **`test_legacy_http_adapters.py`**
  - **Purpose:** Contains tests for the older, synchronous HTTP adapters used prior to the real-time WebSocket pivot.
  - **Context:** Retained for backward compatibility verification of the REST pipeline.
