"""
tests/e2e/stub_server.py
─────────────────────────
Self-contained deterministic test server for Varta.ai browser E2E tests.

Does NOT import web.server (which requires SARVAM_API_KEY and MongoDB).
Instead, re-implements the minimal surface: /session/create, /ws/asr/{session_id}/{speaker},
/metrics/browser, and static file serving.

Usage:
    python -m tests.e2e.stub_server          # starts on port 8000
    python -m tests.e2e.stub_server --port 8001

Admin endpoints:
    POST /_stub/configure   Configure stub behavior for upcoming turns.
    POST /_stub/reset       Reset to default happy-path behavior.
    GET  /_stub/history     Return all observed upstream events (contract check).
"""

import asyncio
import base64
import json
import math
import os
import struct
import sys
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Import just the protocol constants (no side effects, no API key needed)
sys.path.insert(0, str(PROJECT_ROOT))
from web.protocol import (
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    MSG_AUDIO_CHUNK,
    MSG_AUDIO_END,
    MSG_LANGUAGE_DETECTED,
    MSG_SERVER_READY,
    MSG_TRANSCRIPT_FINAL,
    MSG_TRANSCRIPT_PARTIAL,
    MSG_TURN_ERROR,
    MSG_TURN_CANCELLED,
    PROTOCOL_VERSION,
    TurnErrorCode,
)


# ══════════════════════════════════════════════════════════════════════════════
# Stub configuration — controlled by /_stub/configure
# ══════════════════════════════════════════════════════════════════════════════

class StubConfig:
    """Mutable configuration for the deterministic stub."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.partial_delay_s = 0.1
        self.final_delay_s = 0.3
        self.final_text = "Hello, this is a test transcript."
        self.detected_language = "hi-IN"
        self.language_confidence = 0.95
        self.partial_count = 2
        self.first_audio_delay_s = 0.2
        self.audio_chunk_count = 5
        self.audio_chunk_samples = 4800   # 200ms at 24kHz
        self.inter_chunk_delay_s = 0.05
        self.translated_text = "Hello, this is a test translation."
        # Failure injection
        self.fail_mode = None   # None | "nmt_error" | "tts_error" | "no_speech" | "delay_audio" | "drop_connection"
        self.delay_audio_s = 0.0
        self.drop_asr_after_chunks = 0   # 0 = don't drop

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


_config = StubConfig()

# History of upstream events (for contract assertions)
_upstream_history: list[dict] = []
_history_lock = asyncio.Lock()


async def _record_event(event: dict):
    async with _history_lock:
        _upstream_history.append({**event, "timestamp": time.time()})


# ══════════════════════════════════════════════════════════════════════════════
# In-memory session store (minimal)
# ══════════════════════════════════════════════════════════════════════════════

_sessions: dict[str, dict] = {}
_active_sockets: dict[str, WebSocket] = {}


# ══════════════════════════════════════════════════════════════════════════════
# Deterministic audio generator
# ══════════════════════════════════════════════════════════════════════════════

def _generate_pcm_chunk(
    num_samples: int,
    frequency: float = 440.0,
    sample_rate: int = 24_000,
) -> bytes:
    """Generate a pure sine wave PCM chunk (signed 16-bit LE, mono)."""
    buf = bytearray(num_samples * 2)
    two_pi_f = 2.0 * math.pi * frequency
    for i in range(num_samples):
        t = i / sample_rate
        val = int(16_000 * math.sin(two_pi_f * t))
        struct.pack_into("<h", buf, i * 2, max(-32768, min(32767, val)))
    return bytes(buf)


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Varta Stub Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static files ──────────────────────────────────────────────────────────────
static_dir = PROJECT_ROOT / "web" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")
else:
    print(f"[Stub] WARNING: {static_dir} does not exist — run 'npm run build' first")


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.post("/_stub/configure")
async def stub_configure(config: dict = {}):
    for key, val in config.items():
        if hasattr(_config, key):
            setattr(_config, key, val)
    return JSONResponse({"status": "ok", "config": _config.to_dict()})


@app.post("/_stub/reset")
async def stub_reset():
    _config.reset()
    async with _history_lock:
        _upstream_history.clear()
    return JSONResponse({"status": "reset"})


@app.post("/_stub/inject")
async def stub_inject(payload: dict):
    """Inject an arbitrary WebSocket frame into an active connection."""
    session_id = payload.get("session_id")
    speaker = payload.get("speaker")
    message = payload.get("message")
    
    key = f"{session_id}:{speaker}"
    ws = _active_sockets.get(key)
    if ws is None:
        return JSONResponse({"status": "error", "reason": "no_active_socket"}, status_code=404)
        
    await ws.send_json(message)
    return JSONResponse({"status": "ok"})


@app.get("/_stub/history")
async def stub_history():
    async with _history_lock:
        return JSONResponse({"events": list(_upstream_history)})


# ── Session ───────────────────────────────────────────────────────────────────

@app.post("/session/create")
async def create_session(
    lang_a: str = Form(default=""),
    lang_b: str = Form(default=""),
):
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "lang_a": lang_a or None,
        "lang_b": lang_b or None,
    }
    return {"session_id": session_id, "lang_a": lang_a or None, "lang_b": lang_b or None}


# ── Metrics (sink — just accept, don't store) ─────────────────────────────────

@app.post("/metrics/browser")
async def metrics_browser():
    return {"status": "ok"}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Root → redirect to static ────────────────────────────────────────────────

@app.get("/")
async def root():
    return HTMLResponse(
        '<html><body><a href="/static/">Go to app</a></body></html>'
    )


# ── WebSocket: Deterministic live ASR relay ───────────────────────────────────

@app.websocket("/ws/asr/{session_id}/{speaker}")
async def ws_asr_stub(websocket: WebSocket, session_id: str, speaker: str):
    if speaker not in ("a", "b"):
        await websocket.close(code=1003, reason="speaker must be 'a' or 'b'")
        return

    await websocket.accept()
    key = f"{session_id}:{speaker}"
    _active_sockets[key] = websocket
    print(f"[Stub] Accepted: session={session_id} speaker={speaker}")

    # Check session exists
    if session_id not in _sessions:
        await websocket.send_json({
            "type":      MSG_TURN_ERROR,
            "turn_id":   None,
            "code":      TurnErrorCode.SESSION_NOT_FOUND,
            "message":   "Session not found.",
            "retryable": False,
        })
        await websocket.close(code=1008, reason="SESSION_NOT_FOUND")
        return

    # Send server_ready
    await websocket.send_json({
        "type":             MSG_SERVER_READY,
        "protocol_version": PROTOCOL_VERSION,
        "session_id":       session_id,
        "input_speaker":    speaker,
        "asr_model":        "saaras:v3-realtime",
        "encoding":         "linear16",
        "sample_rate_hz":   16_000,
    })

    await _record_event({
        "event":      "server_ready_sent",
        "session_id": session_id,
        "speaker":    speaker,
    })

    active_turn_id = None
    chunks_received = 0

    try:
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))

            # Binary frame: PCM audio chunk from browser
            if msg.get("bytes"):
                if active_turn_id is not None:
                    chunks_received += 1
                    await _record_event({
                        "event":       "audio_chunk_received",
                        "turn_id":     active_turn_id,
                        "chunk_index": chunks_received,
                        "byte_count":  len(msg["bytes"]),
                    })

                    # Failure: drop connection mid-stream
                    if (
                        _config.drop_asr_after_chunks > 0
                        and chunks_received >= _config.drop_asr_after_chunks
                    ):
                        await _record_event({
                            "event":   "connection_dropped_by_stub",
                            "turn_id": active_turn_id,
                        })
                        await websocket.close(code=1011, reason="stub_drop")
                        return

            # Text frame: control message
            elif msg.get("text"):
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = ctrl.get("type")

                if msg_type == "turn_start":
                    turn_id = ctrl.get("turn_id") or str(uuid.uuid4())
                    active_turn_id = turn_id
                    chunks_received = 0

                    await _record_event({
                        "event":          "turn_start_received",
                        "turn_id":        turn_id,
                        "input_speaker":  ctrl.get("input_speaker"),
                        "output_speaker": ctrl.get("output_speaker"),
                    })

                elif msg_type == "stop_recording":
                    tid = ctrl.get("turn_id") or active_turn_id
                    if tid == active_turn_id and active_turn_id is not None:
                        await _record_event({
                            "event":           "stop_recording_received",
                            "turn_id":         active_turn_id,
                            "chunks_received": chunks_received,
                        })

                        # Run the deterministic turn pipeline
                        await _run_stub_pipeline(
                            websocket, session_id, speaker, active_turn_id
                        )
                        active_turn_id = None

    except WebSocketDisconnect:
        print(f"[Stub] Browser disconnected: {session_id}:{speaker}")
        if active_turn_id:
            await _record_event({
                "event":   "browser_disconnected",
                "turn_id": active_turn_id,
            })
    except Exception as exc:
        print(f"[Stub] Error: {exc}")
    finally:
        _active_sockets.pop(key, None)
        print(f"[Stub] Handler done: {session_id}:{speaker}")


async def _run_stub_pipeline(
    websocket: WebSocket,
    session_id: str,
    speaker: str,
    turn_id: str,
):
    """Simulate the ASR → NMT → TTS pipeline with deterministic timing."""
    cfg = _config

    # ── Partial transcripts ──
    for i in range(cfg.partial_count):
        await asyncio.sleep(cfg.partial_delay_s)
        frac = (i + 1) / (cfg.partial_count + 1)
        partial_text = cfg.final_text[: int(len(cfg.final_text) * frac)]
        await websocket.send_json({
            "type":          MSG_TRANSCRIPT_PARTIAL,
            "turn_id":       turn_id,
            "text":          partial_text,
            "language_code": cfg.detected_language,
        })

    # ── Final transcript ──
    await asyncio.sleep(cfg.final_delay_s)

    # Failure: no speech
    if cfg.fail_mode == "no_speech":
        await websocket.send_json({
            "type":      MSG_TURN_ERROR,
            "turn_id":   turn_id,
            "code":      TurnErrorCode.FINAL_TRANSCRIPT_TIMEOUT,
            "message":   "No speech detected. Please try again.",
            "retryable": True,
        })
        await _record_event({"event": "no_speech_sent", "turn_id": turn_id})
        return

    await websocket.send_json({
        "type":                MSG_TRANSCRIPT_FINAL,
        "turn_id":             turn_id,
        "text":                cfg.final_text,
        "language_code":       cfg.detected_language,
        "language_confidence": cfg.language_confidence,
    })
    await _record_event({"event": "transcript_final_sent", "turn_id": turn_id})

    # Language detected
    if cfg.detected_language:
        await websocket.send_json({
            "type":          MSG_LANGUAGE_DETECTED,
            "turn_id":       turn_id,
            "language_code": cfg.detected_language,
        })

    # ── Failure: NMT error ──
    if cfg.fail_mode == "nmt_error":
        await websocket.send_json({
            "type":      MSG_TURN_ERROR,
            "turn_id":   turn_id,
            "code":      TurnErrorCode.NMT_ERROR,
            "message":   "Translation service unavailable (stub).",
            "retryable": True,
        })
        await _record_event({"event": "nmt_error_sent", "turn_id": turn_id})
        return

    # ── Failure: TTS error ──
    if cfg.fail_mode == "tts_error":
        await websocket.send_json({
            "type":      MSG_TURN_ERROR,
            "turn_id":   turn_id,
            "code":      TurnErrorCode.TTS_ERROR,
            "message":   "Speech synthesis failed (stub).",
            "retryable": True,
        })
        await _record_event({"event": "tts_error_sent", "turn_id": turn_id})
        return

    # ── Optional delay (stale audio test) ──
    if cfg.fail_mode == "delay_audio" and cfg.delay_audio_s > 0:
        await asyncio.sleep(cfg.delay_audio_s)

    # ── TTS audio chunks ──
    await asyncio.sleep(cfg.first_audio_delay_s)

    for i in range(cfg.audio_chunk_count):
        pcm = _generate_pcm_chunk(cfg.audio_chunk_samples, frequency=440.0 + i * 50)
        await websocket.send_json({
            "type":           MSG_AUDIO_CHUNK,
            "turn_id":        turn_id,
            "format":         AUDIO_FORMAT,
            "sample_rate_hz": AUDIO_SAMPLE_RATE,
            "channels":       AUDIO_CHANNELS,
            "data":           base64.b64encode(pcm).decode(),
        })
        if i < cfg.audio_chunk_count - 1:
            await asyncio.sleep(cfg.inter_chunk_delay_s)

    await _record_event({
        "event":   "audio_chunks_sent",
        "turn_id": turn_id,
        "count":   cfg.audio_chunk_count,
    })

    # ── audio_end ──
    await websocket.send_json({
        "type":               MSG_AUDIO_END,
        "turn_id":            turn_id,
        "reason":             "completed",
        "server_completed_at": int(time.time() * 1000),
    })
    await _record_event({"event": "audio_end_sent", "turn_id": turn_id})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"[Stub] Starting deterministic test server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
