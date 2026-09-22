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
  POST /metrics/browser   → Receive browser-side lifecycle event stream
  POST /session/create    → Create a new translation session
  WS   /ws/asr/{session_id}/{speaker} → Live ASR/NMT/TTS relay

Deployment notes:
  - Two-worker mode (gunicorn -w 2) requires REDIS_URL.
    Session config and cross-worker connection leases are stored in Redis.
    Startup will fail loudly if two-worker mode is enabled without Redis.
  - Live adapters (SarvamLiveASRAdapter, Bulbul WS) are worker-local.
    They cannot be serialized to Redis.
"""


import asyncio
import base64
import hashlib
from web.services import conversation_service as cs_mod
import json
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from adapter.sarvam_asr import SarvamASRAdapter, SarvamLiveASRAdapter
from adapter.sarvam_nmt import SarvamNMTAdapter
from adapter.sarvam_tts import SarvamTTSAdapter
from pipeline.dual import DualPipeline
from pipeline.single import SinglePipeline
from web.connection_manager import (
    acquire_connection,
    acquire_session_turn,
    release_session_turn,
)
from web.protocol import (
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    MSG_AUDIO_CHUNK,
    MSG_LANGUAGE_DETECTED,
    MSG_SERVER_READY,
    MSG_STOP_RECORDING,
    MSG_TRANSCRIPT_FINAL,
    MSG_TRANSCRIPT_PARTIAL,
    MSG_TURN_ERROR,
    MSG_TURN_START,
    PROTOCOL_VERSION,
    TurnErrorCode,
    WSCloseCode,
    redis_asr_lang_key,
)

# ── Bootstrap ────────────────────────────────────────────────────────────────

load_dotenv()

API_KEY = os.getenv("SARVAM_API_KEY")
if not API_KEY:
    raise RuntimeError("SARVAM_API_KEY not found in environment. Copy .env.example to .env and set it.")

# Worker count — set by gunicorn via WEB_CONCURRENCY or explicit env var
_WORKER_COUNT = int(os.getenv("WEB_CONCURRENCY", "1"))

# Shared stateless adapter instances (safe to share across sessions)
_asr = SarvamASRAdapter(API_KEY)
_nmt = SarvamNMTAdapter(API_KEY)
_tts = SarvamTTSAdapter(API_KEY)

# NOTE: _live_asr_sessions has been removed.
# Live adapters are now owned exclusively by LiveConnectionOwner instances
# in web/connection_manager.py. An adapter is created fresh per connection
# and closed deterministically when the connection owner is released.

# ── Session Store ────────────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL")
SESSION_TTL = 60 * 60 * 2   # 2 hours in seconds

_local_sessions: dict[str, dict] = {}
_redis = None

if REDIS_URL:
    try:
        import redis as redis_lib
        _redis = redis_lib.from_url(REDIS_URL, decode_responses=True)
        _redis.ping()
        print("  Redis connected - sessions and leases persist across restarts")
    except Exception as e:
        print(f"  Redis connection failed ({e})")
        _redis = None
else:
    print("  No REDIS_URL - using in-memory sessions (local mode)")

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
        _mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=3000)
        _mongo_client.server_info()   # fail fast if connection is broken
        _mongo         = _mongo_client[MONGO_DB]
        _mongo_metrics = _mongo_client["varta_metrics"]
        print("  MongoDB connected")
        print(f"    logs    -> {MONGO_DB}")
        print("    metrics -> varta_metrics")
    except Exception as e:
        print(f"[WARN] MongoDB failed ({e}) - logging disabled")
        _mongo         = None
        _mongo_metrics = None
else:
    print("[INFO] No MONGO_URL - logging disabled (local mode)")

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

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

# Sentinel — module-level; do NOT export.
_UNSET = object()

def save_session(
    session_id:  str,
    lang_a,
    lang_b,
    pending_a=None,
    pending_b=None,
    pending_turn_id_a=_UNSET,
    pending_turn_id_b=_UNSET,
    is_durable=_UNSET,
    durable_user_id=_UNSET,
    durable_customer_id=_UNSET,
    durable_session_id=_UNSET,
):
    if _redis is None:
        _local_sessions[session_id] = {
            "lang_a": lang_a,
            "lang_b": lang_b,
            "pending_transcript_a": pending_a,
            "pending_transcript_b": pending_b,
        }
        return

    # Load existing to retain durable identity fields (lost-update race is an accepted limitation)
    existing = load_session(session_id) or {}
    
    # Resolve pending turns (legacy positional vs explicit kwargs)
    val_a = pending_a if pending_a is not None else existing.get("pending_transcript_a")
    if pending_turn_id_a is not _UNSET:
        val_a = pending_turn_id_a

    val_b = pending_b if pending_b is not None else existing.get("pending_transcript_b")
    if pending_turn_id_b is not _UNSET:
        val_b = pending_turn_id_b

    data = {
        "lang_a": lang_a,
        "lang_b": lang_b,
        "pending_transcript_a": val_a or "",
        "pending_transcript_b": val_b or "",
        "is_durable": existing.get("is_durable", False) if is_durable is _UNSET else is_durable,
        "durable_user_id": existing.get("durable_user_id") if durable_user_id is _UNSET else durable_user_id,
        "durable_customer_id": existing.get("durable_customer_id") if durable_customer_id is _UNSET else durable_customer_id,
        "durable_session_id": existing.get("durable_session_id") if durable_session_id is _UNSET else durable_session_id,
    }
    _redis.setex(f"session:{session_id}", SESSION_TTL, json.dumps(data))



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
    # ── DETACHED on test-latency-tracking branch ──────────────────────────────
    # Writing to translation_logs is disabled on this branch to avoid
    # conflicting with the main branch's collection.
    # Re-enable once new branch-specific collections are defined.
    return
    # ─────────────────────────────────────────────────────────────────────────
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
    # ── Re-enabled on test-latency-tracking branch ────────────────────────────
    # Previously disabled to avoid conflicting with main branch collections.
    # Now re-enabled so benchmark data flows to MongoDB.
    # ─────────────────────────────────────────────────────────────────────────
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
print("-" * 50)
print("  Sarvam Translation PoC - ready")
print(f"  Redis:   {'connected' if _redis else 'in-memory fallback'}")
print(f"  MongoDB: {'logs + metrics' if _mongo is not None else 'disabled'}")
print(f"  CORS:    {allowed_origins}")
print("-" * 50)



from web.auth import get_current_user, AuthenticatedUser

@app.post("/session/create_durable")
async def create_durable_session(
    customer_id: str = Form(...),
    token_payload: AuthenticatedUser = Depends(get_current_user)
):
    if _redis is None:
        raise HTTPException(503, "Durable sessions require Redis")
        
    session_id = str(uuid.uuid4())
    user_id = token_payload.user_id
    
    # 1. Write ledger entry to Mongo inline (fail closed)
    from web.services import conversation_service as cs_mod
    try:
        await asyncio.to_thread(
            cs_mod.conv_svc.create_session_doc_sync,
            user_id, session_id, customer_id
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error(f"[Persist] create_durable failed Mongo insert: {exc}")
        raise HTTPException(500, "Could not initialize durable session")
        
    # 2. Write runtime state to Redis
    save_session(
        session_id,
        "hi-IN", "en-IN",  # Defaults, overridden by first WS
        is_durable=True,
        durable_user_id=user_id,
        durable_customer_id=customer_id,
        durable_session_id=session_id
    )
    return {"session_id": session_id}

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
    audio_a:       UploadFile = File(...),
    audio_b:       UploadFile = File(...),
    session_id:    str        = Form(...),
    x_request_id:  str | None = Header(None, alias="X-Request-Id"),
    bg_tasks: BackgroundTasks = BackgroundTasks(),
):
    server_start = int(time.time() * 1000)

    bytes_a = await audio_a.read()
    bytes_b = await audio_b.read()
    fmt_a   = audio_a.filename.rsplit(".", 1)[-1].lower() if audio_a.filename else "wav"

    s          = load_session(session_id) or {}
    is_durable = s.get("is_durable", False)
    durable_uid = s.get("durable_user_id")

    if x_request_id:
        turn_id_a = f"{x_request_id}:a"
        turn_id_b = f"{x_request_id}:b"
    else:
        turn_id_a = f"dual_{uuid.uuid4().hex[:8]}:a"
        turn_id_b = f"dual_{uuid.uuid4().hex[:8]}:b"

    fp_a = hashlib.sha256(bytes_a).hexdigest()
    fp_b = hashlib.sha256(bytes_b).hexdigest()

    if is_durable and durable_uid:
        import asyncio
        try:
            try:
                existing_a = await asyncio.to_thread(cs_mod.conv_svc.get_turn_sync, durable_uid, turn_id_a)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"get_turn_sync A failed: {e}")
                existing_a = None

            if existing_a:
                try:
                    existing_b = await asyncio.to_thread(cs_mod.conv_svc.get_turn_sync, durable_uid, turn_id_b)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"get_turn_sync B failed: {e}")
                    existing_b = None
                if existing_b and existing_b.get("audio_fingerprint") == fp_a and existing_b.get("status") == "completed" and existing_a.get("audio_fingerprint") == fp_b and existing_a.get("status") == "completed":
                    t_a = existing_a.get("translation", {})
                    t_b = existing_b.get("translation", {})
                    return JSONResponse({
                        "for_speaker_a": {
                            "transcript":       existing_a.get("transcript"),
                            "translation":      t_a.get("translated_text"),
                            "src_language":     existing_a.get("source_language"),
                            "tgt_language":     t_a.get("tgt_language"),
                            "audio_b64":        None,
                            "total_latency_ms": 0,
                            "timing":           existing_a.get("timing", {}),
                            "cached":           True,
                        },
                        "for_speaker_b": {
                            "transcript":       existing_b.get("transcript"),
                            "translation":      t_b.get("translated_text"),
                            "src_language":     existing_b.get("source_language"),
                            "tgt_language":     t_b.get("tgt_language"),
                            "audio_b64":        None,
                            "total_latency_ms": 0,
                            "timing":           existing_b.get("timing", {}),
                            "cached":           True,
                        },
                    })
                raise HTTPException(409, "Request still in progress; retry after completion")
        except HTTPException:
            raise
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(f"[Persist] Fail-open: get_turn error: {exc}")

    pipeline, session_load_ms, pipeline_build_ms = _get_or_create_session(session_id)
    try:
        dual_result = await pipeline.process_both(bytes_a, bytes_b, audio_format=fmt_a)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

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

    if is_durable and durable_uid:
        import asyncio
        async def _persist_dual_both():
            for (tid, fp, res, spk) in [
                (turn_id_a, fp_b, r_a, "b"),
                (turn_id_b, fp_a, r_b, "a"),
            ]:
                try:
                    _, is_new = await asyncio.to_thread(cs_mod.conv_svc.reserve_turn_sync, durable_uid, session_id, tid, spk, fp)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Reserve failed: {e}")
                    continue
                if not is_new:
                    continue
                src = res.src_language or ""
                if src and src != "auto":
                    try:
                        await asyncio.to_thread(cs_mod.conv_svc.update_asr_sync, durable_uid, tid, res.source_transcript, src)
                    except ValueError:
                        pass
                try:
                    trans_doc = {"translated_text": res.translated_text, "tgt_language": res.tgt_language}
                    await asyncio.to_thread(cs_mod.conv_svc.complete_turn_sync, durable_uid, tid, session_id, trans_doc, None)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"Complete sync failed: {e}")

        bg_tasks.add_task(_persist_dual_both)

    return JSONResponse(response_data)


@app.post("/translate/speaker_a")
async def translate_speaker_a(
    audio:      UploadFile = File(...),
    session_id: str        = Form(...),
    bg_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Turn-by-turn: process only Speaker A's audio."""
    import hashlib
    server_start = int(time.time() * 1000)

    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1].lower() if audio.filename else "wav"
    pipeline, session_load_ms, pipeline_build_ms = _get_or_create_session(session_id)
    
    s = load_session(session_id) or {}
    is_durable = s.get("is_durable", False)
    durable_uid = s.get("durable_user_id")
    
    turn_id = f"spa_{uuid.uuid4().hex[:8]}"
    fp = hashlib.sha256(audio_bytes).hexdigest()

    try:
        speaker_result = await pipeline.process_speaker_a(audio_bytes, audio_format=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Measure state save ────────────────────────────────────────
    state_start = int(time.time() * 1000)
    update_pipeline_state(session_id, pipeline)
    state_save_ms = int(time.time() * 1000) - state_start

    if speaker_result.buffered:
        if is_durable and durable_uid:
            from web.services import conversation_service as cs_mod
            import asyncio
            try:
                await asyncio.to_thread(cs_mod.conv_svc.reserve_turn_sync, durable_uid, session_id, turn_id, "a", fp)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Reserve fail: {e}")
            save_session(session_id, pending_turn_id_a=turn_id)
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
    
    if is_durable and durable_uid:
        import asyncio
        async def _persist_spa():
            try:
                from web.services import conversation_service as cs_mod
                _, is_new = await asyncio.to_thread(cs_mod.conv_svc.reserve_turn_sync, durable_uid, session_id, turn_id, "a", fp)
                if not is_new: return
                src = result.src_language or ""
                if src and src != "auto":
                    try: await asyncio.to_thread(cs_mod.conv_svc.update_asr_sync, durable_uid, turn_id, result.source_transcript, src)
                    except ValueError: pass
                trans_doc = {"translated_text": result.translated_text, "tgt_language": result.tgt_language}
                await asyncio.to_thread(cs_mod.conv_svc.complete_turn_sync, durable_uid, turn_id, session_id, trans_doc, None)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"[_persist_spa] failed: {e}")
        bg_tasks.add_task(_persist_spa)

    # If Speaker B had a buffered transcript, include the deferred result
    if speaker_result.deferred_result:
        dr = speaker_result.deferred_result
        dr_timing = _build_timing(dr, {
            "total_ms": dr.total_latency_ms, "session_load_ms": 0, "pipeline_build_ms": 0,
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
        
        deferred_turn_id = s.get("pending_turn_id_b")
        if is_durable and durable_uid and deferred_turn_id:
            import asyncio
            async def _persist_deferred_spb():
                try:
                    from web.services import conversation_service as cs_mod
                    src = dr.src_language or ""
                    if src and src != "auto":
                        try: await asyncio.to_thread(cs_mod.conv_svc.update_asr_sync, durable_uid, deferred_turn_id, dr.source_transcript, src)
                        except ValueError: pass
                    trans_doc = {"translated_text": dr.translated_text, "tgt_language": dr.tgt_language}
                    await asyncio.to_thread(cs_mod.conv_svc.complete_turn_sync, durable_uid, deferred_turn_id, session_id, trans_doc, None)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"[_persist_deferred_spb] failed: {e}")
            bg_tasks.add_task(_persist_deferred_spb)

    response_build_ms = int(time.time() * 1000) - resp_start
    total_server_ms = int(time.time() * 1000) - server_start

    timing["server"]["response_build_ms"] = response_build_ms
    timing["server"]["total_ms"] = total_server_ms

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
    bg_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Turn-by-turn: process only Speaker B's audio."""
    import hashlib
    server_start = int(time.time() * 1000)

    audio_bytes = await audio.read()
    ext = audio.filename.rsplit(".", 1)[-1].lower() if audio.filename else "wav"
    pipeline, session_load_ms, pipeline_build_ms = _get_or_create_session(session_id)
    
    s = load_session(session_id) or {}
    is_durable = s.get("is_durable", False)
    durable_uid = s.get("durable_user_id")
    
    turn_id = f"spb_{uuid.uuid4().hex[:8]}"
    fp = hashlib.sha256(audio_bytes).hexdigest()

    try:
        speaker_result = await pipeline.process_speaker_b(audio_bytes, audio_format=ext)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Measure state save ────────────────────────────────────────
    state_start = int(time.time() * 1000)
    update_pipeline_state(session_id, pipeline)
    state_save_ms = int(time.time() * 1000) - state_start

    if speaker_result.buffered:
        if is_durable and durable_uid:
            from web.services import conversation_service as cs_mod
            import asyncio
            try:
                await asyncio.to_thread(cs_mod.conv_svc.reserve_turn_sync, durable_uid, session_id, turn_id, "b", fp)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Reserve fail: {e}")
            save_session(session_id, pending_turn_id_b=turn_id)
        return JSONResponse({"status": "buffered", "message": "Waiting for Speaker A language detection"})

    result = speaker_result.result

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
    
    if is_durable and durable_uid:
        import asyncio
        async def _persist_spb():
            try:
                from web.services import conversation_service as cs_mod
                _, is_new = await asyncio.to_thread(cs_mod.conv_svc.reserve_turn_sync, durable_uid, session_id, turn_id, "b", fp)
                if not is_new: return
                src = result.src_language or ""
                if src and src != "auto":
                    try: await asyncio.to_thread(cs_mod.conv_svc.update_asr_sync, durable_uid, turn_id, result.source_transcript, src)
                    except ValueError: pass
                trans_doc = {"translated_text": result.translated_text, "tgt_language": result.tgt_language}
                await asyncio.to_thread(cs_mod.conv_svc.complete_turn_sync, durable_uid, turn_id, session_id, trans_doc, None)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"[_persist_spb] failed: {e}")
        bg_tasks.add_task(_persist_spb)

    # If Speaker A had a buffered transcript, include the deferred result
    if speaker_result.deferred_result:
        dr = speaker_result.deferred_result
        dr_timing = _build_timing(dr, {
            "total_ms": dr.total_latency_ms, "session_load_ms": 0, "pipeline_build_ms": 0,
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
        
        deferred_turn_id = s.get("pending_turn_id_a")
        if is_durable and durable_uid and deferred_turn_id:
            import asyncio
            async def _persist_deferred_spa():
                try:
                    from web.services import conversation_service as cs_mod
                    src = dr.src_language or ""
                    if src and src != "auto":
                        try: await asyncio.to_thread(cs_mod.conv_svc.update_asr_sync, durable_uid, deferred_turn_id, dr.source_transcript, src)
                        except ValueError: pass
                    trans_doc = {"translated_text": dr.translated_text, "tgt_language": dr.tgt_language}
                    await asyncio.to_thread(cs_mod.conv_svc.complete_turn_sync, durable_uid, deferred_turn_id, session_id, trans_doc, None)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).error(f"[_persist_deferred_spa] failed: {e}")
            bg_tasks.add_task(_persist_deferred_spa)

    response_build_ms = int(time.time() * 1000) - resp_start
    total_server_ms = int(time.time() * 1000) - server_start

    timing["server"]["response_build_ms"] = response_build_ms
    timing["server"]["total_ms"] = total_server_ms

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


# ── Live transcript Redis helpers ────────────────────────────────────────────
# These store partial/final ASR transcript frames while the user is recording.
# Key: "asr:{speaker}:{session_id}"  Type: List  TTL: 2h

def _push_transcript(session_id: str, speaker: str, text: str, is_partial: bool) -> None:
    """Append a transcript frame to the Redis list (or in-memory fallback)."""
    entry = json.dumps({
        "text": text,
        "is_partial": is_partial,
        "ts": int(time.time() * 1000),
    })
    key = f"asr:{speaker}:{session_id}"
    if _redis:
        _redis.rpush(key, entry)
        _redis.expire(key, SESSION_TTL)
    else:
        # In-memory fallback: piggyback on _local_sessions using a sub-key
        bucket = _local_sessions.setdefault(f"__asr_{speaker}_{session_id}", [])
        bucket.append(json.loads(entry))


def _pop_final_transcript(session_id: str, speaker: str) -> tuple[str, str]:
    """
    Read all buffered transcript frames, clear the list, and return
    (final_text, detected_language). Final text is the last non-empty entry.
    """
    key = f"asr:{speaker}:{session_id}"
    lang_key = f"asr:lang:{speaker}:{session_id}"

    if _redis:
        raw_entries = _redis.lrange(key, 0, -1)
        _redis.delete(key)
        detected_lang = _redis.get(lang_key) or ""
        entries = [json.loads(e) for e in raw_entries] if raw_entries else []
    else:
        entries = _local_sessions.pop(f"__asr_{speaker}_{session_id}", [])
        detected_lang = _local_sessions.pop(f"__asr_lang_{speaker}_{session_id}", "")

    # Take the last non-empty transcript text
    final_text = ""
    for entry in reversed(entries):
        if entry.get("text"):
            final_text = entry["text"]
            break

    return final_text, detected_lang


def _save_detected_language(session_id: str, speaker: str, language: str) -> None:
    """Persist the ASR-detected language so it survives between turns."""
    if not language:
        return
    if _redis:
        _redis.setex(f"asr:lang:{speaker}:{session_id}", SESSION_TTL, language)
    else:
        _local_sessions[f"__asr_lang_{speaker}_{session_id}"] = language


# ── WebSocket: Live ASR relay ─────────────────────────────────────────────────

@app.websocket("/ws/asr/{session_id}/{speaker}")
async def ws_asr_live(websocket: WebSocket, session_id: str, speaker: str):
    """
    Live ASR WebSocket endpoint — one connection owner per (session_id, speaker).
    """
    if speaker not in ("a", "b"):
        await websocket.close(code=1003, reason="speaker must be 'a' or 'b'")
        return

    token = None
    if "sec-websocket-protocol" in websocket.headers:
        protocols = websocket.headers.get("sec-websocket-protocol", "").split(",")
        for p in protocols:
            p = p.strip()
            if p.startswith("token-"):
                token = p[len("token-"):]
                break

    if token:
        try:
            from web.auth import get_current_user
            await get_current_user(token)
            await websocket.accept(subprotocol=f"token-{token}")
        except Exception:
            await websocket.close(code=1008, reason="Authentication failed")
            return
    else:
        await websocket.accept()
    other_speaker = "b" if speaker == "a" else "a"
    print(f"[WS/ASR] Accepted: session={session_id} speaker={speaker}")

    state = load_session(session_id)
    if state is None:
        await websocket.send_json({
            "type":    MSG_TURN_ERROR,
            "turn_id": None,
            "code":    TurnErrorCode.UPSTREAM_RECONNECT_FAILED,
            "message": "Session not found",
            "retry":   False,
        })
        await websocket.close(code=4404, reason="session_not_found")
        return

    durable_uid = None
    if token:
        try:
            from web.auth import get_current_user
            u = await get_current_user(token)
            durable_uid = u.user_id
        except Exception:
            pass
    if not durable_uid:
        durable_uid = state.get("durable_user_id")

    persist_queue = asyncio.Queue()
    
    async def persist_queue_worker(q: asyncio.Queue):
        from web.services import conversation_service as cs_mod
        while True:
            msg = await q.get()
            if msg is None:
                q.task_done()
                break
            try:
                op = msg.get("op")
                tid = msg.get("turn_id")
                if op == "reserve":
                    await asyncio.to_thread(cs_mod.conv_svc.reserve_turn_sync, durable_uid, session_id, tid, speaker, msg.get("fp"))
                elif op == "update_asr":
                    src = msg.get("src") or ""
                    if src and src != "auto":
                        try: await asyncio.to_thread(cs_mod.conv_svc.update_asr_sync, durable_uid, tid, msg.get("text"), src)
                        except ValueError: pass
                elif op == "complete":
                    await asyncio.to_thread(cs_mod.conv_svc.complete_turn_sync, durable_uid, tid, session_id, msg.get("trans_doc"), None)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"[WS Persist] Error: {e}")
            finally:
                q.task_done()

    persist_worker_task = asyncio.create_task(
        persist_queue_worker(persist_queue), name=f"ws-persist:{session_id}:{speaker}"
    )

    if _redis:
        detected_lang = _redis.get(f"asr:lang:{speaker}:{session_id}")
        if detected_lang:
            detected_lang = detected_lang.decode("utf-8")
    else:
        detected_lang = _local_sessions.get(f"__asr_lang_{speaker}_{session_id}", "")

    adapter = SarvamLiveASRAdapter(API_KEY)
    await adapter.start_session(language_hint=detected_lang or "")

    owner = await acquire_connection(session_id, speaker, websocket, adapter, redis=_redis)
    if owner is None:
        return

    await owner.enqueue({
        "type":             MSG_SERVER_READY,
        "protocol_version": PROTOCOL_VERSION,
        "session_id":       session_id,
        "input_speaker":    speaker,
        "asr_model":        "saaras:v3-realtime",
        "encoding":         "linear16",
        "sample_rate_hz":   16_000,
    })

    active_turn_id: str | None = None
    turn_lock     = asyncio.Lock()
    terminal_sent = False

    async def _release_active_turn(turn_id: str) -> None:
        nonlocal active_turn_id
        async with turn_lock:
            if active_turn_id == turn_id:
                active_turn_id = None
            if owner.active_turn_id == turn_id:
                owner.active_turn_id = None
        await release_session_turn(
            session_id,
            speaker,
            turn_id,
            redis=_redis,
        )

    def _get_tgt_lang() -> str:
        s = load_session(session_id)
        if not s:
            return "en-IN"
        return s.get(f"lang_{other_speaker}") or "en-IN"

    async def run_turn_pipeline(turn_id: str, final_text: str, src_lang: str) -> None:
        nonlocal terminal_sent
        try:
            tgt_lang = _get_tgt_lang()
            pipeline = SinglePipeline(
                asr_adapter  = _asr,
                nmt_adapter  = _nmt,
                tts_adapter  = _tts,
                src_language = src_lang or "auto",
                tgt_language = tgt_lang,
            )
            
            if durable_uid:
                persist_queue.put_nowait({
                    "op": "update_asr",
                    "turn_id": turn_id,
                    "text": final_text,
                    "src": src_lang
                })

            chunk_count = 0
            async for audio_chunk in pipeline.run_from_transcript(
                transcript=final_text, src_language=src_lang or "auto",
            ):
                await owner.enqueue({
                    "type":           MSG_AUDIO_CHUNK,
                    "turn_id":        turn_id,
                    "format":         AUDIO_FORMAT,
                    "sample_rate_hz": AUDIO_SAMPLE_RATE,
                    "channels":       AUDIO_CHANNELS,
                    "data":           base64.b64encode(audio_chunk).decode(),
                })
                chunk_count += 1

            if durable_uid and hasattr(pipeline, "last_nmt_output"):
                trans_doc = {
                    "translated_text": pipeline.last_nmt_output.translated_text,
                    "tgt_language": tgt_lang
                }
                persist_queue.put_nowait({
                    "op": "complete", "turn_id": turn_id, "trans_doc": trans_doc
                })

            if not terminal_sent:
                terminal_sent = True
                await owner.send_audio_end(turn_id, reason="completed")
                print(f"[WS/ASR] turn={turn_id} audio_end chunks={chunk_count}")

        except asyncio.CancelledError:
            if not terminal_sent:
                terminal_sent = True
                try:
                    await owner.send_turn_cancelled(turn_id, reason="pipeline_cancelled")
                except Exception:
                    pass
        except Exception as exc:
            print(f"[WS/ASR] Pipeline error turn={turn_id}: {exc}")
            if not terminal_sent:
                terminal_sent = True
                try:
                    await owner.send_turn_error(
                        turn_id, TurnErrorCode.NMT_ERROR, str(exc), retryable=True
                    )
                except Exception:
                    pass

    async def transcript_reader():
        nonlocal terminal_sent
        try:
            async for frame in adapter.receive_stream():
                if owner.is_closed:
                    break
                
                turn_id = owner.active_turn_id
                if not turn_id:
                    continue
                
                ftype = frame.get("type")
                if ftype == "transcript_partial":
                    text = frame.get("text", "")
                    await owner.enqueue({
                        "type":    MSG_TRANSCRIPT_PARTIAL,
                        "turn_id": turn_id,
                        "text":    text,
                    })
                    _push_transcript(session_id, speaker, text, is_partial=True)

                elif ftype == "transcript_final":
                    text = frame.get("text", "")
                    _push_transcript(session_id, speaker, text, is_partial=False)

                    lang = frame.get("detected_language")
                    _save_detected_language(session_id, speaker, lang)

                    confidence = frame.get("language_confidence")
                    await owner.enqueue({
                        "type":                MSG_TRANSCRIPT_FINAL,
                        "turn_id":             turn_id,
                        "text":                text,
                        "language_code":       lang,
                        "language_confidence": confidence,
                    })
                    if lang:
                        await owner.enqueue({
                            "type":          MSG_LANGUAGE_DETECTED,
                            "turn_id":       turn_id,
                            "language_code": lang,
                        })
                        s = load_session(session_id)
                        if s is not None:
                            s[f"lang_{speaker}"] = lang
                            save_session(
                                session_id,
                                s.get("lang_a"), s.get("lang_b"),
                                s.get("pending_transcript_a"),
                                s.get("pending_transcript_b"),
                            )

                    if text:
                        owner._turn_task = asyncio.create_task(
                            run_turn_pipeline(turn_id, text, lang or adapter.detected_language),
                            name=f"turn-pipeline:{turn_id}",
                        )
                    else:
                        if not terminal_sent:
                            terminal_sent = True
                            await owner.send_turn_error(
                                turn_id,
                                TurnErrorCode.FINAL_TRANSCRIPT_TIMEOUT,
                                "No speech detected. Please try again.",
                                retryable=True,
                            )
                        await _release_active_turn(turn_id)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"[WS/ASR] Transcript reader error: {exc}")
            if active_turn_id and not terminal_sent:
                terminal_sent = True
                try:
                    await owner.send_turn_error(
                        active_turn_id,
                        TurnErrorCode.UPSTREAM_RECONNECT_FAILED,
                        f"Transcript stream error: {exc}",
                        retryable=True,
                    )
                except Exception:
                    pass
            if active_turn_id:
                await _release_active_turn(active_turn_id)

    reader_task = asyncio.create_task(
        transcript_reader(),
        name=f"transcript-reader:{session_id}:{speaker}",
    )

    try:
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(msg.get("code", 1000))

            if msg.get("bytes"):
                if active_turn_id is not None:
                    await adapter.stream_chunk(msg["bytes"])

            elif msg.get("text"):
                try:
                    ctrl = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                msg_type = ctrl.get("type")

                if msg_type == MSG_TURN_START:
                    new_turn_id = ctrl.get("turn_id")
                    if not new_turn_id:
                        new_turn_id = str(uuid.uuid4())
                        print(f"[WS/ASR] WARN: missing turn_id - generated {new_turn_id} (legacy client)")

                    turn_acquired = await acquire_session_turn(
                        session_id,
                        speaker,
                        new_turn_id,
                        redis=_redis,
                    )
                    if not turn_acquired:
                        await owner.send_turn_error(
                            new_turn_id,
                            TurnErrorCode.TURN_IN_PROGRESS,
                            "The other speaker is currently using the microphone.",
                            retryable=False,
                        )
                        continue

                    async with turn_lock:
                        if owner.active_turn_id is not None:
                            await release_session_turn(
                                session_id,
                                speaker,
                                new_turn_id,
                                redis=_redis,
                            )
                            await owner.send_turn_error(
                                new_turn_id,
                                TurnErrorCode.TURN_IN_PROGRESS,
                                "Another turn is already in progress.",
                                retryable=False,
                            )
                            continue
                        owner.active_turn_id = new_turn_id
                        active_turn_id       = new_turn_id
                        terminal_sent        = False
                        
                    if durable_uid:
                        persist_queue.put_nowait({"op": "reserve", "turn_id": new_turn_id, "fp": ""})
                        
                    print(f"[WS/ASR] Turn started: {new_turn_id}")

                elif msg_type == MSG_STOP_RECORDING:
                    tid = ctrl.get("turn_id") or active_turn_id
                    if tid == active_turn_id and active_turn_id is not None:
                        await adapter.signal_speech_end()
                        print(f"[WS/ASR] stop_recording for turn={tid}")

    except WebSocketDisconnect:
        print(f"[WS/ASR] Browser disconnected: {session_id}:{speaker}")
        if active_turn_id and not terminal_sent:
            try:
                await owner.send_turn_cancelled(active_turn_id, reason="browser_disconnected")
            except Exception:
                pass
    except Exception as exc:
        print(f"[WS/ASR] Unexpected error {session_id}:{speaker}: {exc}")
        if active_turn_id and not terminal_sent:
            try:
                await owner.send_turn_error(
                    active_turn_id, TurnErrorCode.TURN_TOTAL_TIMEOUT, str(exc), retryable=True
                )
            except Exception:
                pass
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
        if active_turn_id:
            await _release_active_turn(active_turn_id)
        await owner.release()
        persist_queue.put_nowait(None)
        print(f"[WS/ASR] Handler done: {session_id}:{speaker}")

