# V2 WebSocket Reliability + Latency Test

This test suite executes a comprehensive 120-turn reliability and latency experiment on the V2 WebSocket translation pipeline.

## Overview

The test uses persistent WebSocket connections to simulate the exact behavior of the V2 browser client (`pcm-processor.js`). It reads audio from FLEURS WebM fixtures, decodes them to 16kHz mono `pcm_s16le`, chunks them into 640-byte binary frames (representing 20ms of audio), and streams them over the `/ws/asr/{session_id}/{speaker}` endpoint at real-time speeds.

It then waits for the server to process the audio (ASR, NMT, and TTS) and measures the latencies of each step, capturing the final `audio_chunk` messages containing the TTS output.

### Constraints & Features

- **Cost-efficient:** It conducts the experiment by gathering all necessary metrics (WebSocket stability, ASR success, pipeline latency, TTS delivery) in a single run, reducing redundant API costs.
- **Reporting:** Generates detailed HTML reports and JSON artifacts in the `test/results/` directory, and automatically inserts metrics into MongoDB (if `MONGO_URL` is set).
- **Rate-limit pacing:** Pauses for 7 seconds between turns and 5 seconds between conversations to avoid hitting the server's 10 req/min rate limit.

## Usage

### 1. Requirements

Ensure `pydub` is installed, as it is required to decode the WebM fixtures.

```bash
pip install -r requirements.txt
```

You also need `ffmpeg` installed on your system for `pydub` to handle WebM/Opus.

### 2. Running Tests

You can run the tests in three modes:

**Smoke Test (1 conversation, 1 turn)**
Safe to run anytime. Quick validation to ensure the test infrastructure works.
```bash
python -m test.ws_reliability_test.runner --smoke
```

**Small Validation (1 conversation, 5 turns)**
Validates WebSocket persistence across multiple turns.
```bash
python -m test.ws_reliability_test.runner --conversations 1 --turns 5
```

**Full 120-Turn Experiment**
Runs the full suite. Because this incurs actual API costs, you must explicitly approve it by setting `RUN_EXPENSIVE_TEST=true`.
```bash
RUN_EXPENSIVE_TEST=true python -m test.ws_reliability_test.runner
```

### Environment Variables

| Variable | Description |
|---|---|
| `TARGET_URL` | The URL of the Varta backend (default: `http://localhost:8000`) |
| `MONGO_URL` | MongoDB connection string for storing results in the `varta_ws_test` DB. |
| `RUN_EXPENSIVE_TEST` | Set to `true` to run the full test. |

## Results

After execution, results are stored in two places:

1. **Local Filesystem:** `test/results/run_YYYYMMDD_HHMMSS/`
    - `summary.json`: Top-level experiment summary and metrics.
    - `pipeline_events.jsonl`: One JSON object per turn, https://127.0.0.1:50630/static/artifacts/ea07a966-c2b0-429c-91d2-748fa0634545/.user_uploaded/media_1786711557694.png?csrf=2e6753aa-52e0-430c-8d8f-483d0f56940bwith detailed timings.
    - `failures.jsonl`: Filtered list of turns that failed or were only partially successful.
    - `report.html`: An interactive, easy-to-read HTML summary of the entire run.

2. **MongoDB:** If `MONGO_URL` is provided, metrics are written to the `varta_ws_test` database:
    - `ws_test_sessions`: Collection containing session-level aggregates.
    - `ws_test_turns`: Collection containing individual turn records.
