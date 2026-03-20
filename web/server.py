"""
web/server.py
─────────────
FastAPI server exposing the translation pipelines via REST endpoints.

Endpoints:
  GET  /                  → Serves the web UI (index.html)
  POST /translate/single  → One-way translation (audio_in → translated audio)
  POST /translate/dual    → Both speakers simultaneously
  POST /translate/speaker_a → Speaker A's turn only
  POST /translate/speaker_b → Speaker B's turn only

Sessions are stored in-memory (dict keyed by session_id).
For production, use Redis. For the PoC, memory is fine.
"""

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from adapter.sarvam_asr import SarvamASRAdapter
from adapter.sarvam_nmt import SarvamNMTAdapter
from adapter.sarvam_tts import SarvamTTSAdapter
from pipeline.dual import DualPipeline
from pipeline.single import SinglePipeline

# ── Bootstrap ────────────────────────────────────────────────────────────────

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")
if not API_KEY:
    raise RuntimeError("SARVAM_API_KEY not found in environment. Copy .env.example to .env and set it.")

# Build one set of adapter instances — shared across all pipelines and sessions
# (They are stateless, so sharing is safe and saves connection pool resources)
_asr = SarvamASRAdapter(API_KEY)
_nmt = SarvamNMTAdapter(API_KEY)
_tts = SarvamTTSAdapter(API_KEY)

# In-memory session store: session_id → DualPipeline instance
# Each session has its own SessionState (detected languages per speaker)
_sessions: dict[str, DualPipeline] = {}

app = FastAPI(title="Sarvam Translation PoC", version="0.1.0")

# Serve static files (index.html, app.js)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helper ───────────────────────────────────────────────────────────────────

def _get_or_create_session(
    session_id: str,
    lang_a: Optional[str] = None,
    lang_b: Optional[str] = None,
) -> DualPipeline:
    """Return existing session or create a new one."""
    if session_id not in _sessions:
        _sessions[session_id] = DualPipeline(
            asr=_asr, nmt=_nmt, tts=_tts,
            initial_lang_a=lang_a,
            initial_lang_b=lang_b,
        )
    return _sessions[session_id]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main web interface."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/session/create")
async def create_session(
    lang_a: str = Form(default=""),
    lang_b: str = Form(default=""),
):
    """
    Create a new translation session.
    lang_a and lang_b are optional — if empty, ASR will auto-detect language
    from the first audio clip each speaker sends. The UI calls this automatically
    on the first record press; no manual "Start Session" step required.
    """
    session_id = str(uuid.uuid4())[:8]
    # Convert empty strings to None so SessionState treats them as unknown
    _get_or_create_session(
        session_id,
        lang_a or None,
        lang_b or None,
    )
    return {"session_id": session_id, "lang_a": lang_a or None, "lang_b": lang_b or None}


@app.post("/translate/single")
async def translate_single(
    audio:        UploadFile = File(...),
    src_language: str        = Form(...),
    tgt_language: str        = Form(...),
):
    """
    One-way single translation.
    Upload audio, get back a JSON with transcript + translation + base64 audio.
    No session needed — stateless endpoint.
    """
    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1].lower() if audio.filename else "wav"

    pipeline = SinglePipeline(
        asr_adapter  = _asr,
        nmt_adapter  = _nmt,
        tts_adapter  = _tts,
        src_language = src_language,
        tgt_language = tgt_language,
    )

    try:
        result = await pipeline.run(audio_bytes, audio_format=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "transcript":       result.source_transcript,
        "translation":      result.translated_text,
        "src_language":     result.src_language,
        "tgt_language":     result.tgt_language,
        "audio_b64":        base64.b64encode(result.audio_bytes).decode(),
        "audio_format":     result.audio_format,
        "total_latency_ms": result.total_latency_ms,
    })


@app.post("/translate/dual")
async def translate_dual(
    audio_a:    UploadFile = File(...),
    audio_b:    UploadFile = File(...),
    session_id: str        = Form(...),
):
    """
    Simultaneous two-way translation.
    Both speakers upload their audio in the same request.
    Both translations are returned together.
    Both ASR, NMT, and TTS calls run in parallel internally.
    """
    bytes_a = await audio_a.read()
    bytes_b = await audio_b.read()
    fmt_a = audio_a.filename.rsplit(".", 1)[-1].lower() if audio_a.filename else "wav"

    pipeline = _get_or_create_session(session_id)

    try:
        dual_result = await pipeline.process_both(bytes_a, bytes_b, audio_format=fmt_a)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    r_a = dual_result.for_speaker_a
    r_b = dual_result.for_speaker_b

    return JSONResponse({
        "for_speaker_a": {
            "transcript":       r_a.source_transcript,
            "translation":      r_a.translated_text,
            "src_language":     r_a.src_language,
            "tgt_language":     r_a.tgt_language,
            "audio_b64":        base64.b64encode(r_a.audio_bytes).decode(),
            "total_latency_ms": r_a.total_latency_ms,
        },
        "for_speaker_b": {
            "transcript":       r_b.source_transcript,
            "translation":      r_b.translated_text,
            "src_language":     r_b.src_language,
            "tgt_language":     r_b.tgt_language,
            "audio_b64":        base64.b64encode(r_b.audio_bytes).decode(),
            "total_latency_ms": r_b.total_latency_ms,
        },
    })


@app.post("/translate/speaker_a")
async def translate_speaker_a(
    audio:      UploadFile = File(...),
    session_id: str        = Form(...),
):
    """Turn-by-turn: process only Speaker A's audio."""
    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1].lower() if audio.filename else "wav"
    pipeline = _get_or_create_session(session_id)

    try:
        result = await pipeline.process_speaker_a(audio_bytes, audio_format=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        return JSONResponse({"status": "buffered", "message": "Waiting for Speaker B language detection"})

    return JSONResponse({
        "transcript":       result.source_transcript,
        "translation":      result.translated_text,
        "src_language":     result.src_language,
        "tgt_language":     result.tgt_language,
        "audio_b64":        base64.b64encode(result.audio_bytes).decode(),
        "audio_format":     result.audio_format,
        "total_latency_ms": result.total_latency_ms,
    })


@app.post("/translate/speaker_b")
async def translate_speaker_b(
    audio:      UploadFile = File(...),
    session_id: str        = Form(...),
):
    """Turn-by-turn: process only Speaker B's audio."""
    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1].lower() if audio.filename else "wav"
    pipeline = _get_or_create_session(session_id)

    try:
        result = await pipeline.process_speaker_b(audio_bytes, audio_format=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        return JSONResponse({"status": "buffered", "message": "Waiting for Speaker A language detection"})

    return JSONResponse({
        "transcript":       result.source_transcript,
        "translation":      result.translated_text,
        "src_language":     result.src_language,
        "tgt_language":     result.tgt_language,
        "audio_b64":        base64.b64encode(result.audio_bytes).decode(),
        "audio_format":     result.audio_format,
        "total_latency_ms": result.total_latency_ms,
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=True)