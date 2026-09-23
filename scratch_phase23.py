import os
import re

with open("web/server.py", "r", encoding="utf-8") as f:
    server = f.read()

# 1. Imports at the top
if "import logging" not in server[:500]:
    server = server.replace(
        "from web.protocol import (",
        "import logging\nfrom web.protocol import ("
    )
if "MSG_TURN_PERSISTED" not in server:
    server = server.replace(
        'MSG_TURN_CANCELLED,',
        'MSG_TURN_CANCELLED, MSG_TURN_PERSISTED,'
    )


# 2. Startup Persistence
# Search for `app = FastAPI(...)`
startup_code = """
from web.storage.mongo import (
    init_mongo as _conv_init_mongo,
    check_readiness as _conv_check_readiness,
    is_ready as _conv_store_ready,
    is_enabled as _conv_store_enabled,
)

@app.on_event("startup")
async def _startup_persistence() -> None:
    _conv_init_mongo(MONGO_URL)
    _conv_check_readiness()
    if _conv_store_ready():
        print("[INFO] Conversation persistence ready")
    else:
        print("[INFO] Conversation persistence disabled or unavailable")
"""
if "def _startup_persistence" not in server:
    server = re.sub(
        r'(app = FastAPI[^\n]*\n)',
        r'\1' + startup_code,
        server,
        count=1
    )

# 3. Replace save_session
save_session_new = """_UNSET = object()

def save_session(
    session_id: str,
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
    existing = load_session(session_id) or {}

    def _merge(kwarg, key, default=None):
        if kwarg is _UNSET:
            return existing.get(key, default)
        return kwarg

    payload = {
        "lang_a": lang_a,
        "lang_b": lang_b,
        "pending_transcript_a": pending_a,
        "pending_transcript_b": pending_b,
        "pending_turn_id_a": _merge(pending_turn_id_a, "pending_turn_id_a"),
        "pending_turn_id_b": _merge(pending_turn_id_b, "pending_turn_id_b"),
        "is_durable": _merge(is_durable, "is_durable", False),
        "durable_user_id": _merge(durable_user_id, "durable_user_id"),
        "durable_customer_id": _merge(durable_customer_id, "durable_customer_id"),
        "durable_session_id": _merge(durable_session_id, "durable_session_id"),
    }

    if payload.get("is_durable") and _redis is None:
        raise RuntimeError("Durable session snapshot requires Redis")

    data = json.dumps(payload)
    if _redis:
        _redis.setex(f"session:{session_id}", SESSION_TTL, data)
    else:
        _local_sessions[session_id] = json.loads(data)
"""
server = re.sub(
    r'def save_session\(.*?(?=\n\n(?:@app|def load_session|#))',
    save_session_new,
    server,
    flags=re.DOTALL,
    count=1
)

# 4. Replace POST /session/create_durable
create_durable_new = """@app.post("/session/create_durable")
async def create_durable_session(
    customer_id: str = Form(...),
    lang_a: str = Form(default=""),
    lang_b: str = Form(default=""),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    if not _conv_store_enabled() or not _conv_store_ready():
        raise HTTPException(503, "Durable persistence is not available")
    if _redis is None:
        raise HTTPException(503, "Durable sessions require Redis")

    session_id = str(uuid.uuid4())
    try:
        await asyncio.to_thread(
            cs_mod.conv_svc.create_session_doc_sync,
            current_user.user_id,
            session_id,
            customer_id,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("[Persist] create_durable Mongo insert: %s", exc)
        raise HTTPException(500, "Could not initialize durable session")

    save_session(
        session_id,
        lang_a or None,
        lang_b or None,
        is_durable=True,
        durable_user_id=current_user.user_id,
        durable_customer_id=customer_id,
        durable_session_id=session_id,
    )
    return JSONResponse({
        "session_id": session_id,
        "is_durable": True,
        "customer_id": customer_id,
    })"""
server = re.sub(
    r'@app\.post\("/session/create_durable"\).*?(?=\n\n@|\n#)',
    create_durable_new,
    server,
    flags=re.DOTALL,
    count=1
)

# 5. REST Speaker Endpoints (pending ids)
server = server.replace(
    '        save_session(session_id, pending_turn_id_a=turn_id)',
    '        st = load_session(session_id) or {}\n        save_session(\n            session_id,\n            st.get("lang_a"), st.get("lang_b"),\n            st.get("pending_transcript_a"), st.get("pending_transcript_b"),\n            pending_turn_id_a=turn_id,\n        )'
)
server = server.replace(
    '        save_session(session_id, pending_turn_id_b=turn_id)',
    '        st = load_session(session_id) or {}\n        save_session(\n            session_id,\n            st.get("lang_a"), st.get("lang_b"),\n            st.get("pending_transcript_a"), st.get("pending_transcript_b"),\n            pending_turn_id_b=turn_id,\n        )'
)
server = server.replace(
    '                save_session(session_id, pending_a=None)',
    '                st = load_session(session_id) or {}\n                save_session(\n                    session_id,\n                    st.get("lang_a"), st.get("lang_b"),\n                    None, st.get("pending_transcript_b"),\n                    pending_turn_id_a=None,\n                )'
)
server = server.replace(
    '                save_session(session_id, pending_b=None)',
    '                st = load_session(session_id) or {}\n                save_session(\n                    session_id,\n                    st.get("lang_a"), st.get("lang_b"),\n                    st.get("pending_transcript_a"), None,\n                    pending_turn_id_b=None,\n                )'
)

# 6. log_metrics turn_id
if "turn_id: str | None = None" not in server:
    server = server.replace(
        'def log_metrics(session_id: str, payload: dict, direction: str):',
        'def log_metrics(session_id: str, payload: dict, direction: str, turn_id: str | None = None):'
    )
    server = server.replace(
        '                {"session_id": session_id, "timestamp": now, direction: payload}',
        '                {"session_id": session_id, "timestamp": now, direction: payload, **({"turn_id": turn_id} if turn_id else {})}'
    )
    server = server.replace(
        '        log_metrics(session_id, t_a.get("timing", {}), "dual_a")',
        '        log_metrics(session_id, t_a.get("timing", {}), "dual_a", turn_id=turn_id_a)'
    )
    server = server.replace(
        '        log_metrics(session_id, t_b.get("timing", {}), "dual_b")',
        '        log_metrics(session_id, t_b.get("timing", {}), "dual_b", turn_id=turn_id_b)'
    )

with open("web/server.py", "w", encoding="utf-8") as f:
    f.write(server)
