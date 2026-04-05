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
  POST /metrics/browser   → Receive browser-side timing data
  POST /session/create    → Create a new translation session

Sessions are stored in-memory (dict keyed by session_id).
For production, use Redis. For the PoC, memory is fine.
"""

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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
# ── Session Store ────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL")
SESSION_TTL = 60 * 60 * 2   # 2 hours in seconds

_local_sessions: dict[str, dict] = {}

_redis = None
if REDIS_URL:
    try:
        import redis as redis_lib
        _redis = redis_lib.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
        print("  Redis connected — sessions persist across restarts")
    except Exception as e:
        print(f"  Redis connection failed ({e}) — falling back to in-memory")
        _redis = None
else:
    print("  No REDIS_URL — using in-memory sessions (local mode)")

# ── MongoDB Logging + Metrics ────────────────────────────────────────────────
# Two databases on the same cluster:
#   translation-data-cluste → existing translation logs (UNCHANGED)
#   varta_metrics → new latency tracking data

MONGO_URL    = os.getenv("MONGO_URL")
MONGO_DB     = os.getenv("MONGO_DB_NAME", "translation-data-cluste")

_mongo         = None   # the database handle — None means logging is disabled
_mongo_metrics = None   # latency metrics database handle

if MONGO_URL:
    try:
        from pymongo import MongoClient
        from datetime import datetime
        _mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
        _mongo_client.server_info()   # fail fast if connection is broken
        _mongo         = _mongo_client[MONGO_DB]
        _mongo_metrics = _mongo_client["varta_metrics"]
        print("✅  MongoDB connected")
        print(f"    logs    → {MONGO_DB}")
        print(f"    metrics → varta_metrics")
    except Exception as e:
        print(f"⚠️   MongoDB failed ({e}) — logging disabled")
        _mongo         = None
        _mongo_metrics = None
else:
    print("ℹ️   No MONGO_URL — logging disabled (local mode)")

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# get_remote_address reads the caller's IP from the incoming request
# This is the "key" — one counter per IP address per time window
limiter = Limiter(key_func=get_remote_address, default_limits=["15/minute"])

app = FastAPI(title="Sarvam Translation PoC", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
from fastapi.middleware.cors import CORSMiddleware

# ALLOWED_ORIGINS in your .env controls which websites can call this API
# Default is localhost:8000 for local development
# On Render, set ALLOWED_ORIGINS=https://your-frontend-domain.com
# Multiple origins: ALLOWED_ORIGINS=https://site1.com,https://site2.com
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8000,http://localhost:3000,https://varta-ai-22b3.onrender.com,https://varta-ai-1-7wwd.onrender.com"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = allowed_origins,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
    allow_credentials = False,
)

# Serve static files (index.html, app.js)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Session functions ─────────────────────────────────────────────────────────
# These four functions are the ONLY things that touch Redis or _local_sessions.
# Everything else in the file calls these functions — never Redis directly.
# This means if you ever switch from Redis to something else,
# you only change these four functions, nothing else.

def save_session(session_id: str, lang_a, lang_b, pending_a=None, pending_b=None):
    # Converts session data into a string
    # because Redis can only store strings, not Python objects
    data = json.dumps({
        "lang_a": lang_a,
        "lang_b": lang_b,
        "pending_transcript_a": pending_a,
        "pending_transcript_b": pending_b,
    })

    if _redis:
        # setex = "set with expiry"
        # saves the string to Redis and marks it to auto-delete after SESSION_TTL
        _redis.setex(f"session:{session_id}", SESSION_TTL, data)
    else:
        # No Redis — save to the in-memory dict instead
        _local_sessions[session_id] = {
            "lang_a": lang_a,
            "lang_b": lang_b,
            "pending_transcript_a": pending_a,
            "pending_transcript_b": pending_b,
        }


def load_session(session_id: str):
    if _redis:
        raw = _redis.get(f"session:{session_id}")
        # json.loads converts the string back into a Python dict
        # if raw is None (session expired or never existed), return None
        return json.loads(raw) if raw else None
    else:
        return _local_sessions.get(session_id)


def get_pipeline(session_id: str):
    # Load the session state from storage
    state = load_session(session_id)
    if state is None:
        return None

    # Rebuild the full pipeline object from saved state
    # We can't store the pipeline object itself in Redis — it's a complex
    # Python object with network connections inside it, not a simple string.
    # But rebuilding it from saved state takes microseconds.
    return DualPipeline(
        asr                  = _asr,
        nmt                  = _nmt,
        tts                  = _tts,
        initial_lang_a       = state.get("lang_a"),
        initial_lang_b       = state.get("lang_b"),
        pending_transcript_a = state.get("pending_transcript_a"),
        pending_transcript_b = state.get("pending_transcript_b"),
    )


def update_pipeline_state(session_id: str, pipeline: DualPipeline):
    # After a translation runs, the ASR may have detected a language
    # that wasn't known before. Save those updated values back to storage
    # so the next request starts with the correct language codes.
    # Also persist any pending (buffered) transcripts.
    save_session(
        session_id,
        pipeline.state.lang_a,
        pipeline.state.lang_b,
        pending_a=pipeline.state.pending_transcript_a,
        pending_b=pipeline.state.pending_transcript_b,
    )


# ── Logging (varta_logs — UNCHANGED) ─────────────────────────────────────────

def log_translation(
    session_id: str,
    endpoint:   str,
    src_lang:   str,
    tgt_lang:   str,
    latency_ms: int,
    char_count: int,
):
    # Fire-and-forget — if this fails for any reason,
    # the exception is swallowed silently.
    # MongoDB being down must never surface as a 500 error.
    if _mongo is None:
        return   # logging disabled — exit immediately, do nothing

    try:
        from datetime import datetime
        _mongo["translation_logs"].insert_one({
            "session_id":  session_id,
            "endpoint":    endpoint,
            "src_language": src_lang,
            "tgt_language": tgt_lang,
            "latency_ms":  latency_ms,
            "char_count":  char_count,
            "timestamp":   datetime.utcnow(),
        })
    except Exception:
        pass   # silent failure — translation already completed successfully


# ── Metrics (varta_metrics — NEW) ─────────────────────────────────────────────

def log_metrics(
    session_id: str,
    endpoint:   str,
    timing:     dict,
    src_lang:   str,
    tgt_lang:   str,
    char_count: int,
):
    """Write latency data to varta_metrics. Completely separate from log_translation."""
    if _mongo_metrics is None:
        return

    try:
        from datetime import datetime
        now = datetime.utcnow()

        # Write to request_latency collection
        _mongo_metrics["request_latency"].insert_one({
            "session_id": session_id,
            "endpoint":   endpoint,
            "timestamp":  now,
            "browser":    timing.get("browser", {}),
            "server":     timing.get("server", {}),
            "asr":        timing.get("asr", {}),
            "nmt":        timing.get("nmt", {}),
            "tts":        timing.get("tts", {}),
            "src_language": src_lang,
            "tgt_language": tgt_lang,
            "char_count":   char_count,
        })

        # Write to model_performance collection — 3 documents
        model_entries = [
            {
                "session_id":   session_id,
                "timestamp":    now,
                "model_id":     "sarvam/saaras-v3",
                "model_type":   "ASR",
                "src_language": src_lang,
                "tgt_language": tgt_lang,
                "char_count":   char_count,
                **timing.get("asr", {}),
            },
            {
                "session_id":   session_id,
                "timestamp":    now,
                "model_id":     "sarvam/sarvam-translate",
                "model_type":   "NMT",
                "src_language": src_lang,
                "tgt_language": tgt_lang,
                "char_count":   char_count,
                **timing.get("nmt", {}),
            },
            {
                "session_id":   session_id,
                "timestamp":    now,
                "model_id":     "sarvam/bulbul-v3",
                "model_type":   "TTS",
                "src_language": src_lang,
                "tgt_language": tgt_lang,
                "char_count":   char_count,
                **timing.get("tts", {}),
            },
        ]
        _mongo_metrics["model_performance"].insert_many(model_entries)

    except Exception:
        pass   # fire-and-forget — never affect translation response


# ── Timing helper ─────────────────────────────────────────────────────────────

def _build_timing(result, server_timing):
    """Assemble the nested timing dict from pipeline result + server-side measurements."""
    t = result.timing if result.timing else {}
    return {
        "server": server_timing,
        "asr": {
            "total_ms": t.get("asr_total_ms", 0),
            "tcp_ms":   t.get("asr_tcp_ms", 0),
            "api_ms":   t.get("asr_api_ms", 0),
            "parse_ms": t.get("asr_parse_ms", 0),
        },
        "nmt": {
            "total_ms": t.get("nmt_total_ms", 0),
            "tcp_ms":   t.get("nmt_tcp_ms", 0),
            "api_ms":   t.get("nmt_api_ms", 0),
            "parse_ms": t.get("nmt_parse_ms", 0),
        },
        "tts": {
            "total_ms": t.get("tts_total_ms", 0),
            "tcp_ms":   t.get("tts_tcp_ms", 0),
            "api_ms":   t.get("tts_api_ms", 0),
            "parse_ms": t.get("tts_parse_ms", 0),
        },
        "browser": {},   # populated by browser JS via /metrics/browser
    }


# ── Helper ───────────────────────────────────────────────────────────────────

def _get_or_create_session(session_id, lang_a=None, lang_b=None):
    t0 = int(time.time() * 1000)
    existing = get_pipeline(session_id)
    session_load_ms = int(time.time() * 1000) - t0

    if existing:
        return existing, session_load_ms, 0

    # New session — save initial state and return a fresh pipeline
    save_session(session_id, lang_a, lang_b)
    build_start = int(time.time() * 1000)
    pipe = DualPipeline(
        asr=_asr, nmt=_nmt, tts=_tts,
        initial_lang_a=lang_a,
        initial_lang_b=lang_b,
    )
    pipeline_build_ms = int(time.time() * 1000) - build_start
    return pipe, session_load_ms, pipeline_build_ms


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main web interface."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "version": "0.1.0"}


# ── Startup summary ───────────────────────────────────────────
# Printed once at startup — shows connection status at a glance
# Green = connected and working
# Warning = fallback mode (still works, just not persistent)
print("─" * 50)
print("  Sarvam Translation PoC — ready")
print(f"  Redis:   {'✅ connected' if _redis else '⚠️  in-memory fallback'}")
print(f"  MongoDB: {'✅ logs + metrics' if _mongo is not None else '⚠️  disabled'}")
print(f"  CORS:    {allowed_origins}")
print("─" * 50)


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
    session_id = str(uuid.uuid4())
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
    server_start = int(time.time() * 1000)

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

    # ── Measure log_translation time ─────────────────────────────
    log_start = int(time.time() * 1000)
    log_translation(
        session_id = "single",
        endpoint   = "single",
        src_lang   = result.src_language,
        tgt_lang   = result.tgt_language,
        latency_ms = result.total_latency_ms,
        char_count = len(result.source_transcript),
    )
    log_write_ms = int(time.time() * 1000) - log_start

    # ── Build response with timing ───────────────────────────────
    resp_start = int(time.time() * 1000)

    server_timing = {
        "total_ms":          0,   # filled below
        "session_load_ms":   0,
        "pipeline_build_ms": 0,
        "response_build_ms": 0,
        "state_save_ms":     0,
        "log_write_ms":      log_write_ms,
    }
    timing = _build_timing(result, server_timing)

    response_data = {
        "transcript":       result.source_transcript,
        "translation":      result.translated_text,
        "src_language":     result.src_language,
        "tgt_language":     result.tgt_language,
        "audio_b64":        base64.b64encode(result.audio_bytes).decode(),
        "audio_format":     result.audio_format,
        "total_latency_ms": result.total_latency_ms,
        "timing":           timing,
    }

    response_build_ms = int(time.time() * 1000) - resp_start
    total_server_ms = int(time.time() * 1000) - server_start

    timing["server"]["response_build_ms"] = response_build_ms
    timing["server"]["total_ms"] = total_server_ms

    # ── Write metrics (fire-and-forget, after response is built) ──
    metrics_start = int(time.time() * 1000)
    log_metrics(
        session_id = "single",
        endpoint   = "single",
        timing     = timing,
        src_lang   = result.src_language,
        tgt_lang   = result.tgt_language,
        char_count = len(result.source_transcript),
    )

    return JSONResponse(response_data)


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
    server_start = int(time.time() * 1000)

    bytes_a = await audio_a.read()
    bytes_b = await audio_b.read()
    fmt_a = audio_a.filename.rsplit(".", 1)[-1].lower() if audio_a.filename else "wav"

    pipeline, session_load_ms, pipeline_build_ms = _get_or_create_session(session_id)

    try:
        dual_result = await pipeline.process_both(bytes_a, bytes_b, audio_format=fmt_a)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    r_a = dual_result.for_speaker_a
    r_b = dual_result.for_speaker_b

    resp_start = int(time.time() * 1000)

    # Build timing for both directions
    server_timing_a = {
        "total_ms": 0, "session_load_ms": session_load_ms,
        "pipeline_build_ms": pipeline_build_ms, "response_build_ms": 0,
        "state_save_ms": 0, "log_write_ms": 0,
    }
    server_timing_b = {
        "total_ms": 0, "session_load_ms": session_load_ms,
        "pipeline_build_ms": pipeline_build_ms, "response_build_ms": 0,
        "state_save_ms": 0, "log_write_ms": 0,
    }
    timing_a = _build_timing(r_a, server_timing_a)
    timing_b = _build_timing(r_b, server_timing_b)

    response_data = {
        "for_speaker_a": {
            "transcript":       r_a.source_transcript,
            "translation":      r_a.translated_text,
            "src_language":     r_a.src_language,
            "tgt_language":     r_a.tgt_language,
            "audio_b64":        base64.b64encode(r_a.audio_bytes).decode(),
            "total_latency_ms": r_a.total_latency_ms,
            "timing":           timing_a,
        },
        "for_speaker_b": {
            "transcript":       r_b.source_transcript,
            "translation":      r_b.translated_text,
            "src_language":     r_b.src_language,
            "tgt_language":     r_b.tgt_language,
            "audio_b64":        base64.b64encode(r_b.audio_bytes).decode(),
            "total_latency_ms": r_b.total_latency_ms,
            "timing":           timing_b,
        },
    }

    response_build_ms = int(time.time() * 1000) - resp_start
    total_server_ms = int(time.time() * 1000) - server_start

    timing_a["server"]["response_build_ms"] = response_build_ms
    timing_a["server"]["total_ms"] = total_server_ms
    timing_b["server"]["response_build_ms"] = response_build_ms
    timing_b["server"]["total_ms"] = total_server_ms

    # Write metrics for both directions
    log_metrics(session_id=session_id, endpoint="dual_a", timing=timing_a,
                src_lang=r_a.src_language, tgt_lang=r_a.tgt_language,
                char_count=len(r_a.source_transcript))
    log_metrics(session_id=session_id, endpoint="dual_b", timing=timing_b,
                src_lang=r_b.src_language, tgt_lang=r_b.tgt_language,
                char_count=len(r_b.source_transcript))

    return JSONResponse(response_data)


@app.post("/translate/speaker_a")
async def translate_speaker_a(
    audio:      UploadFile = File(...),
    session_id: str        = Form(...),
):
    """Turn-by-turn: process only Speaker A's audio."""
    server_start = int(time.time() * 1000)

    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1].lower() if audio.filename else "wav"
    pipeline, session_load_ms, pipeline_build_ms = _get_or_create_session(session_id)

    try:
        speaker_result = await pipeline.process_speaker_a(audio_bytes, audio_format=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Measure state save ────────────────────────────────────────
    state_start = int(time.time() * 1000)
    update_pipeline_state(session_id, pipeline)
    state_save_ms = int(time.time() * 1000) - state_start

    if speaker_result.buffered:
        return JSONResponse({"status": "buffered", "message": "Waiting for Speaker B language detection"})

    result = speaker_result.result

    # ── Measure log_translation ──────────────────────────────────
    log_start = int(time.time() * 1000)
    log_translation(
        session_id = session_id,
        endpoint   = "speaker_a",
        src_lang   = result.src_language,
        tgt_lang   = result.tgt_language,
        latency_ms = result.total_latency_ms,
        char_count = len(result.source_transcript),
    )
    log_write_ms = int(time.time() * 1000) - log_start

    # ── Build response with timing ───────────────────────────────
    resp_start = int(time.time() * 1000)

    server_timing = {
        "total_ms":          0,
        "session_load_ms":   session_load_ms,
        "pipeline_build_ms": pipeline_build_ms,
        "response_build_ms": 0,
        "state_save_ms":     state_save_ms,
        "log_write_ms":      log_write_ms,
    }
    timing = _build_timing(result, server_timing)

    response_data = {
        "transcript":       result.source_transcript,
        "translation":      result.translated_text,
        "src_language":     result.src_language,
        "tgt_language":     result.tgt_language,
        "audio_b64":        base64.b64encode(result.audio_bytes).decode(),
        "audio_format":     result.audio_format,
        "total_latency_ms": result.total_latency_ms,
        "timing":           timing,
    }

    # If Speaker B had a buffered transcript, include the deferred result
    if speaker_result.deferred_result:
        dr = speaker_result.deferred_result
        dr_timing = _build_timing(dr, {
            "total_ms": 0, "session_load_ms": 0, "pipeline_build_ms": 0,
            "response_build_ms": 0, "state_save_ms": 0, "log_write_ms": 0,
        })
        response_data["deferred"] = {
            "speaker":          "b",
            "transcript":       dr.source_transcript,
            "translation":      dr.translated_text,
            "src_language":     dr.src_language,
            "tgt_language":     dr.tgt_language,
            "audio_b64":        base64.b64encode(dr.audio_bytes).decode(),
            "audio_format":     dr.audio_format,
            "total_latency_ms": dr.total_latency_ms,
            "timing":           dr_timing,
        }
        log_translation(
            session_id = session_id,
            endpoint   = "speaker_b_deferred",
            src_lang   = dr.src_language,
            tgt_lang   = dr.tgt_language,
            latency_ms = dr.total_latency_ms,
            char_count = len(dr.source_transcript),
        )
        log_metrics(
            session_id = session_id,
            endpoint   = "speaker_b_deferred",
            timing     = dr_timing,
            src_lang   = dr.src_language,
            tgt_lang   = dr.tgt_language,
            char_count = len(dr.source_transcript),
        )

    response_build_ms = int(time.time() * 1000) - resp_start
    total_server_ms = int(time.time() * 1000) - server_start

    timing["server"]["response_build_ms"] = response_build_ms
    timing["server"]["total_ms"] = total_server_ms

    # ── Write metrics ─────────────────────────────────────────────
    log_metrics(
        session_id = session_id,
        endpoint   = "speaker_a",
        timing     = timing,
        src_lang   = result.src_language,
        tgt_lang   = result.tgt_language,
        char_count = len(result.source_transcript),
    )

    return JSONResponse(response_data)


@app.post("/translate/speaker_b")
async def translate_speaker_b(
    audio:      UploadFile = File(...),
    session_id: str        = Form(...),
):
    """Turn-by-turn: process only Speaker B's audio."""
    server_start = int(time.time() * 1000)

    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1].lower() if audio.filename else "wav"
    pipeline, session_load_ms, pipeline_build_ms = _get_or_create_session(session_id)

    try:
        speaker_result = await pipeline.process_speaker_b(audio_bytes, audio_format=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Measure state save ────────────────────────────────────────
    state_start = int(time.time() * 1000)
    update_pipeline_state(session_id, pipeline)
    state_save_ms = int(time.time() * 1000) - state_start

    if speaker_result.buffered:
        return JSONResponse({"status": "buffered", "message": "Waiting for Speaker A language detection"})

    result = speaker_result.result

    # ── Measure log_translation ──────────────────────────────────
    log_start = int(time.time() * 1000)
    log_translation(
        session_id = session_id,
        endpoint   = "speaker_b",
        src_lang   = result.src_language,
        tgt_lang   = result.tgt_language,
        latency_ms = result.total_latency_ms,
        char_count = len(result.source_transcript),
    )
    log_write_ms = int(time.time() * 1000) - log_start

    # ── Build response with timing ───────────────────────────────
    resp_start = int(time.time() * 1000)

    server_timing = {
        "total_ms":          0,
        "session_load_ms":   session_load_ms,
        "pipeline_build_ms": pipeline_build_ms,
        "response_build_ms": 0,
        "state_save_ms":     state_save_ms,
        "log_write_ms":      log_write_ms,
    }
    timing = _build_timing(result, server_timing)

    response_data = {
        "transcript":       result.source_transcript,
        "translation":      result.translated_text,
        "src_language":     result.src_language,
        "tgt_language":     result.tgt_language,
        "audio_b64":        base64.b64encode(result.audio_bytes).decode(),
        "audio_format":     result.audio_format,
        "total_latency_ms": result.total_latency_ms,
        "timing":           timing,
    }

    # If Speaker A had a buffered transcript, include the deferred result
    if speaker_result.deferred_result:
        dr = speaker_result.deferred_result
        dr_timing = _build_timing(dr, {
            "total_ms": 0, "session_load_ms": 0, "pipeline_build_ms": 0,
            "response_build_ms": 0, "state_save_ms": 0, "log_write_ms": 0,
        })
        response_data["deferred"] = {
            "speaker":          "a",
            "transcript":       dr.source_transcript,
            "translation":      dr.translated_text,
            "src_language":     dr.src_language,
            "tgt_language":     dr.tgt_language,
            "audio_b64":        base64.b64encode(dr.audio_bytes).decode(),
            "audio_format":     dr.audio_format,
            "total_latency_ms": dr.total_latency_ms,
            "timing":           dr_timing,
        }
        log_translation(
            session_id = session_id,
            endpoint   = "speaker_a_deferred",
            src_lang   = dr.src_language,
            tgt_lang   = dr.tgt_language,
            latency_ms = dr.total_latency_ms,
            char_count = len(dr.source_transcript),
        )
        log_metrics(
            session_id = session_id,
            endpoint   = "speaker_a_deferred",
            timing     = dr_timing,
            src_lang   = dr.src_language,
            tgt_lang   = dr.tgt_language,
            char_count = len(dr.source_transcript),
        )

    response_build_ms = int(time.time() * 1000) - resp_start
    total_server_ms = int(time.time() * 1000) - server_start

    timing["server"]["response_build_ms"] = response_build_ms
    timing["server"]["total_ms"] = total_server_ms

    # ── Write metrics ─────────────────────────────────────────────
    log_metrics(
        session_id = session_id,
        endpoint   = "speaker_b",
        timing     = timing,
        src_lang   = result.src_language,
        tgt_lang   = result.tgt_language,
        char_count = len(result.source_transcript),
    )

    return JSONResponse(response_data)


# ── Browser Metrics Endpoint ─────────────────────────────────────────────────

@app.post("/metrics/browser")
async def receive_browser_metrics(
    session_id:      str = Form(...),
    upload_ms:       int = Form(...),
    server_wait_ms:  int = Form(...),
    parse_ms:        int = Form(...),
    audio_decode_ms: int = Form(...),
    total_ms:        int = Form(...),
):
    """
    Receive browser-side timing data and merge it into the most recent
    request_latency document for this session.
    Not rate-limited — this is a metrics-only endpoint.
    """
    if _mongo_metrics is not None:
        try:
            from datetime import datetime, timedelta
            _mongo_metrics["request_latency"].update_one(
                {"session_id": session_id,
                 "timestamp": {"$gte": datetime.utcnow() - timedelta(minutes=1)}},
                {"$set": {"browser": {
                    "upload_ms":       upload_ms,
                    "server_wait_ms":  server_wait_ms,
                    "parse_ms":        parse_ms,
                    "audio_decode_ms": audio_decode_ms,
                    "total_ms":        total_ms,
                }}},
            )
        except Exception:
            pass
    return {"status": "ok"}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=True)