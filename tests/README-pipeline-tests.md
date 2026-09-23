# Pipeline & State Machine Tests

This suite contains integration and unit tests that validate the overarching state machines, locking mechanisms, and lifecycle states of a conversation turn in Varta.ai.

## Files in this suite

- **`test_ws_connection_lifecycle.py`**
  - **Purpose:** Tests the lifecycle and robustness of the backend real-time streaming WebSocket connection.
  - **Context:** Mocks the client/server handshake, verifying that connections open, transition states correctly when data flows, and safely teardown without leaking resources.

- **`test_ws_speaker_lock.py`**
  - **Purpose:** Validates the "active speaker" mutex lock during a turn.
  - **Context:** Ensures that when Speaker A initiates a recording, Speaker B is securely locked out of starting a conflicting turn, preserving the half-duplex conversational design.

- **`test_legacy_http_pipeline.py`**
  - **Purpose:** Validates the FastAPI `TestClient` flow for the legacy HTTP architecture.
  - **Context:** Creates a test session and pushes dummy audio data to verify the old synchronous routing behavior.
