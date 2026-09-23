# Backend Probes & Smoke Tests

This suite contains standalone scripts designed to test the live backend APIs and remote services from the command line. They are incredibly useful for verifying that external dependencies (like the Sarvam API or MongoDB) are reachable and behaving as expected.

## Files in this suite

- **`probe_saaras_ws_asr.py`**
  - **Purpose:** A direct WebSocket probe to test the Sarvam Saaras v3 ASR endpoint. 
  - **Context:** Connects to `wss://api.sarvam.ai`, streams binary audio frames, and verifies the incoming real-time JSON transcription responses.

- **`smoke_test_http_api.py`**
  - **Purpose:** An end-to-end smoke test against the deployed HTTP pipeline (legacy).
  - **Context:** Can be pointed at `localhost` or a remote deployment URL (e.g., Render) to fire off test audio and ensure the HTTP infrastructure is alive.

- **`analytics_mongo_queries.py`**
  - **Purpose:** A collection of aggregation snippets for the latency-tracking database collections.
  - **Context:** Tests and demonstrates how to query the MongoDB cluster for latency metrics (upload MS, decode MS, server wait MS, etc.) gathered during test runs.
