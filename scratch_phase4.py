import os
import re

with open("web/server.py", "r", encoding="utf-8") as f:
    server = f.read()

worker_code = """
async def persist_queue_worker(
    queue: asyncio.Queue,
    *,
    user_id: str,
    session_id: str,
    speaker: str,
    conv_svc,
    owner,
) -> None:
    \"\"\"Serial reserve → asr → complete|fail. Sentinel None drains. Do not cancel from WS finally.\"\"\"
    failed_reserve: set[str] = set()
    while True:
        item = await queue.get()
        try:
            if item is None:
                break
            op = item.get("op")
            tid = item.get("turn_id")
            if op == "reserve":
                try:
                    await asyncio.to_thread(
                        conv_svc.reserve_turn_sync,
                        user_id, session_id, tid, speaker, item.get("fp"),
                    )
                except Exception as exc:
                    logging.getLogger(__name__).error("[WS Persist] reserve %s: %s", tid, exc)
                    failed_reserve.add(tid)
                    if owner is not None and not owner._closed:
                        await owner.enqueue({
                            "type": MSG_TURN_PERSISTED, "turn_id": tid,
                            "status": "persist_failed", "error": str(exc),
                        })
            elif op == "update_asr":
                if tid in failed_reserve:
                    continue
                src = item.get("src") or ""
                if src and src != "auto":
                    try:
                        await asyncio.to_thread(
                            conv_svc.update_asr_sync, user_id, tid, item.get("text"), src,
                        )
                    except Exception as exc:
                        logging.getLogger(__name__).warning("[WS Persist] asr %s: %s", tid, exc)
            elif op == "complete":
                if tid in failed_reserve:
                    continue
                await asyncio.to_thread(
                    conv_svc.complete_turn_sync,
                    user_id, tid, session_id, item.get("trans_doc"), None,
                )
                if owner is not None and not owner._closed:
                    await owner.enqueue({
                        "type": MSG_TURN_PERSISTED, "turn_id": tid, "status": "completed",
                    })
            elif op == "fail":
                if tid not in failed_reserve:
                    await asyncio.to_thread(
                        conv_svc.mark_turn_failed_sync, user_id, tid, item.get("error", ""),
                    )
        except Exception as exc:
            logging.getLogger(__name__).error("[WS Persist] %s", exc)
        finally:
            queue.task_done()
"""

if "async def persist_queue_worker" not in server[:server.find("async def ws_asr_live")]:
    server = server.replace(
        "async def ws_asr_live(websocket: WebSocket, session_id: str, speaker: str):",
        worker_code + "\nasync def ws_asr_live(websocket: WebSocket, session_id: str, speaker: str):"
    )

# Now, we need to extract the inside of ws_asr_live and modify it.
# Instead of complex regex, let's replace exact strings found in ws_asr_live.txt

# 1. Replace handshake block (lines 9-83 in txt)
# from "token = None" down to "persist_worker_task = ..."
old_handshake_start = '    token = None'
old_handshake_end = '    persist_worker_task = asyncio.create_task(\n        persist_queue_worker(persist_queue), name=f"ws-persist:{session_id}:{speaker}"\n    )'
# wait, it's better to just regex everything from `    token = None` to `    persist_worker_task = asyncio.create_task(\n        persist_queue_worker(persist_queue), name=f"ws-persist:{session_id}:{speaker}"\n    )\n`
pattern1 = r'    token = None\n.*?persist_worker_task = asyncio\.create_task\([^)]+\)\n'

new_handshake = """    from web.auth import authenticate_websocket as _ws_auth
    from web.protocol import WSCloseCode

    state = load_session(session_id)
    is_durable = bool(state and state.get("is_durable"))
    durable_uid = (state or {}).get("durable_user_id")
    selected_proto = None

    raw_protocols = websocket.headers.get("sec-websocket-protocol", "")
    if is_durable:
        user, selected_proto = await _ws_auth(raw_protocols)
        if user is None or user.user_id != durable_uid:
            await websocket.close(code=WSCloseCode.POLICY_VIOLATION, reason="UNAUTHORIZED")
            return
        await websocket.accept(subprotocol=selected_proto)
    else:
        await websocket.accept()

    other_speaker = "b" if speaker == "a" else "a"
    print(f"[WS/ASR] Accepted: session={session_id} speaker={speaker}")

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
"""
server = re.sub(pattern1, new_handshake, server, flags=re.DOTALL)


# 2. Add owner logic & persist_worker initialization
# After `owner = await acquire_connection(session_id, speaker, websocket, adapter, redis=_redis)`
# `if owner is None:\n        return`
# We append the worker start code.

old_owner_code = """    owner = await acquire_connection(session_id, speaker, websocket, adapter, redis=_redis)
    if owner is None:
        return"""

new_owner_code = """    persist_queue = None
    persist_worker_task = None

    owner = await acquire_connection(session_id, speaker, websocket, adapter, redis=_redis)
    if owner is None:
        return

    if is_durable and durable_uid and _conv_store_ready():
        persist_queue = asyncio.Queue()
        from web.services import conversation_service as cs_mod
        persist_worker_task = asyncio.create_task(
            persist_queue_worker(
                persist_queue,
                user_id=durable_uid,
                session_id=session_id,
                speaker=speaker,
                conv_svc=cs_mod.conv_svc,
                owner=owner,
            ),
            name=f"ws-persist:{session_id}:{speaker}",
        )"""

server = server.replace(old_owner_code, new_owner_code)

# 3. Update run_turn_pipeline
old_put_asr = """            if durable_uid:
                persist_queue.put_nowait({
                    "op": "update_asr",
                    "turn_id": turn_id,
                    "text": final_text,
                    "src": src_lang
                })"""
new_put_asr = """            if persist_queue is not None:
                persist_queue.put_nowait({
                    "op": "update_asr",
                    "turn_id": turn_id,
                    "text": final_text,
                    "src": src_lang
                })"""
server = server.replace(old_put_asr, new_put_asr)

old_put_complete = """            if durable_uid and hasattr(pipeline, "last_nmt_output"):
                trans_doc = {
                    "translated_text": pipeline.last_nmt_output.translated_text,
                    "tgt_language": tgt_lang
                }
                persist_queue.put_nowait({
                    "op": "complete", "turn_id": turn_id, "trans_doc": trans_doc
                })"""
new_put_complete = """            if persist_queue is not None and hasattr(pipeline, "last_nmt_output"):
                trans_doc = {
                    "translated_text": pipeline.last_nmt_output.translated_text,
                    "tgt_language": tgt_lang
                }
                persist_queue.put_nowait({
                    "op": "complete", "turn_id": turn_id, "trans_doc": trans_doc
                })"""
server = server.replace(old_put_complete, new_put_complete)

old_pipeline_except1 = """        except asyncio.CancelledError:
            if not terminal_sent:
                terminal_sent = True
                try:
                    await owner.send_turn_cancelled(turn_id, reason="pipeline_cancelled")
                except Exception:
                    pass"""
new_pipeline_except1 = """        except asyncio.CancelledError as exc:
            if persist_queue is not None:
                persist_queue.put_nowait({"op": "fail", "turn_id": turn_id, "error": str(exc)})
            if not terminal_sent:
                terminal_sent = True
                try:
                    await owner.send_turn_cancelled(turn_id, reason="pipeline_cancelled")
                except Exception:
                    pass"""
server = server.replace(old_pipeline_except1, new_pipeline_except1)

old_pipeline_except2 = """        except Exception as exc:
            print(f"[WS/ASR] Pipeline error turn={turn_id}: {exc}")
            if not terminal_sent:
                terminal_sent = True
                try:
                    await owner.send_turn_error(
                        turn_id, TurnErrorCode.NMT_ERROR, str(exc), retryable=True
                    )
                except Exception:
                    pass"""
new_pipeline_except2 = """        except Exception as exc:
            if persist_queue is not None:
                persist_queue.put_nowait({"op": "fail", "turn_id": turn_id, "error": str(exc)})
            print(f"[WS/ASR] Pipeline error turn={turn_id}: {exc}")
            if not terminal_sent:
                terminal_sent = True
                try:
                    await owner.send_turn_error(
                        turn_id, TurnErrorCode.NMT_ERROR, str(exc), retryable=True
                    )
                except Exception:
                    pass"""
server = server.replace(old_pipeline_except2, new_pipeline_except2)

# 4. Update MSG_TURN_START
old_turn_start = """                    if durable_uid:
                        persist_queue.put_nowait({"op": "reserve", "turn_id": new_turn_id, "fp": ""})"""
new_turn_start = """                    if persist_queue is not None:
                        persist_queue.put_nowait({"op": "reserve", "turn_id": new_turn_id, "fp": None})"""
server = server.replace(old_turn_start, new_turn_start)

# 5. finally block
old_finally = """    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
        if active_turn_id:
            await _release_active_turn(active_turn_id)
        await owner.release()
        persist_queue.put_nowait(None)
        print(f"[WS/ASR] Handler done: {session_id}:{speaker}")"""

new_finally = """    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
        if active_turn_id:
            await _release_active_turn(active_turn_id)

        if persist_queue is not None:
            persist_queue.put_nowait(None)
            if persist_worker_task is not None:
                done, _ = await asyncio.wait({persist_worker_task}, timeout=5.0)

        await owner.release()
        print(f"[WS/ASR] Handler done: {session_id}:{speaker}")"""

server = server.replace(old_finally, new_finally)

with open("web/server.py", "w", encoding="utf-8") as f:
    f.write(server)
