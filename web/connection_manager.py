"""
web/connection_manager.py
──────────────────────────
Server-side live connection owner and global connection manager.

Design invariants:
  - At most one LiveConnectionOwner per (session_id, speaker) across all workers,
    enforced by a Redis SET NX EX lease.
  - One writer task handles all browser sends; transcript reading and NMT/TTS
    work never call websocket.send_* directly.
  - Outbound queue is bounded (OUTBOUND_QUEUE_MAX). Backpressure that persists
    beyond OUTBOUND_BACKPRESSURE_GRACE seconds terminates the turn with
    OUTBOUND_BACKPRESSURE.
  - On disconnect/cancel/error the owner cancels + awaits all tasks, calls
    adapter.close() exactly once, releases the Redis lease, and removes its
    registry entry only if the generation still matches.
  - The upstream adapter (_live_asr_sessions / _live_asr) is owned exclusively
    by the connection owner; it must never be reused by a later connection.

Usage (in server.py):

    owner = await LiveConnectionManager.acquire(
        session_id, speaker, websocket, adapter, redis=_redis, worker_id=WORKER_ID
    )
    if owner is None:
        # duplicate — already closed with 4409
        return
    try:
        await owner.run(handle_turn_fn)
    finally:
        await owner.release()
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from fastapi import WebSocket

from adapter.sarvam_asr import SarvamLiveASRAdapter
from web.protocol import (
    MSG_AUDIO_END,
    MSG_TURN_CANCELLED,
    MSG_TURN_ERROR,
    OUTBOUND_QUEUE_MAX,
    REDIS_LEASE_RENEW_SEC,
    REDIS_LEASE_TTL_SEC,
    SESSION_TURN_LEASE_TTL_SEC,
    Timeout,
    TurnErrorCode,
    WSCloseCode,
    redis_lease_key,
    redis_session_turn_key,
)

# Process-unique ID for Redis lease ownership
WORKER_ID = str(uuid.uuid4())


# ── Registry (process-local) ──────────────────────────────────────────────────

# Maps "session_id:speaker" → LiveConnectionOwner
_registry: dict[str, "LiveConnectionOwner"] = {}
_registry_lock = asyncio.Lock()

# Session-level turn ownership. Connection ownership is per speaker, but the
# conversation microphone is intentionally exclusive across both speakers.
# Value: (speaker, turn_id, lease_token)
_session_turns: dict[str, tuple[str, str, str]] = {}
_session_turn_lock = asyncio.Lock()


async def acquire_session_turn(
    session_id: str,
    speaker: str,
    turn_id: str,
    redis=None,
) -> bool:
    """Claim the single active-turn slot for a conversation session."""
    token = f"{WORKER_ID}:{speaker}:{turn_id}"
    redis_key = redis_session_turn_key(session_id)
    redis_acquired = False

    if redis is not None:
        try:
            redis_acquired = bool(redis.set(
                redis_key,
                token,
                nx=True,
                ex=SESSION_TURN_LEASE_TTL_SEC,
            ))
        except Exception as exc:
            print(f"[ConnManager] Session turn lease error: {exc}")
            # Continue with the process-local guard for degraded local mode.
            redis_acquired = True

        if not redis_acquired:
            return False

    async with _session_turn_lock:
        existing = _session_turns.get(session_id)
        if existing is not None:
            if redis is not None:
                try:
                    if redis.get(redis_key) == token:
                        redis.delete(redis_key)
                except Exception:
                    pass
            return False
        _session_turns[session_id] = (speaker, turn_id, token)

    return True


async def release_session_turn(
    session_id: str,
    speaker: str,
    turn_id: str,
    redis=None,
) -> None:
    """Release a turn slot only if this speaker still owns that turn."""
    token = None
    async with _session_turn_lock:
        existing = _session_turns.get(session_id)
        if existing is not None and existing[:2] == (speaker, turn_id):
            token = existing[2]
            del _session_turns[session_id]

    if redis is not None and token is not None:
        try:
            if redis.get(redis_session_turn_key(session_id)) == token:
                redis.delete(redis_session_turn_key(session_id))
        except Exception as exc:
            print(f"[ConnManager] Session turn release error: {exc}")


# ── Connection owner ──────────────────────────────────────────────────────────

@dataclass
class LiveConnectionOwner:
    """
    Owns one live browser WebSocket for a single (session_id, speaker) pair.

    Fields:
        session_id          Session UUID.
        speaker             "a" or "b".
        websocket           The accepted FastAPI WebSocket.
        adapter             SarvamLiveASRAdapter for this connection.
        generation          Monotonic integer; incremented on each new owner.
        outbound_queue      Bounded asyncio.Queue for browser-bound messages.
        active_turn_id      Currently active turn UUID, or None.
        _writer_task        Background task that drains outbound_queue.
        _lease_task         Background task that renews the Redis lease.
        _redis              Redis client or None.
        _lease_key          Redis key for the ownership lease.
        _lease_token        The value stored under _lease_key (WORKER_ID:generation).
        _closed             True once release() has been called.
    """
    session_id:      str
    speaker:         str
    websocket:       WebSocket
    adapter:         SarvamLiveASRAdapter
    generation:      int

    outbound_queue:  asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=OUTBOUND_QUEUE_MAX))
    active_turn_id:  str | None = None

    _writer_task:    asyncio.Task | None = field(default=None, repr=False)
    _lease_task:     asyncio.Task | None = field(default=None, repr=False)
    _turn_task:      asyncio.Task | None = field(default=None, repr=False)
    _redis:          object = field(default=None, repr=False)
    _lease_key:      str    = field(default="", repr=False)
    _lease_token:    str    = field(default="", repr=False)
    _closed:         bool   = field(default=False, repr=False)
    _backpressure_since: float | None = field(default=None, repr=False)

    # ── Outbound helpers ──────────────────────────────────────────────────────

    async def enqueue(self, msg: dict) -> None:
        """
        Put a message on the outbound queue.
        Tracks how long we've been at capacity; raises after the grace period.
        """
        try:
            self.outbound_queue.put_nowait(msg)
            self._backpressure_since = None
        except asyncio.QueueFull:
            now = time.monotonic()
            if self._backpressure_since is None:
                self._backpressure_since = now
            elif now - self._backpressure_since > Timeout.OUTBOUND_BACKPRESSURE_GRACE:
                raise RuntimeError(TurnErrorCode.OUTBOUND_BACKPRESSURE)
            # Brief grace — try a blocking put with a short timeout
            try:
                await asyncio.wait_for(
                    self.outbound_queue.put(msg),
                    timeout=0.1,
                )
                self._backpressure_since = None
            except asyncio.TimeoutError:
                pass   # caller will retry or the grace timer will fire next call

    async def send_turn_error(self, turn_id: str, code: str, message: str, retryable: bool = True) -> None:
        """Enqueue a turn_error terminal event."""
        await self.enqueue({
            "type":     MSG_TURN_ERROR,
            "turn_id":  turn_id,
            "code":     code,
            "message":  message,
            "retryable": retryable,
        })

    async def send_audio_end(self, turn_id: str, reason: str = "completed") -> None:
        """Enqueue an audio_end terminal event."""
        await self.enqueue({
            "type":               MSG_AUDIO_END,
            "turn_id":            turn_id,
            "reason":             reason,
            "server_completed_at": int(time.time() * 1000),
        })

    async def send_turn_cancelled(self, turn_id: str, reason: str = "disconnected") -> None:
        """Enqueue a turn_cancelled terminal event."""
        await self.enqueue({
            "type":    MSG_TURN_CANCELLED,
            "turn_id": turn_id,
            "reason":  reason,
        })

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start_background_tasks(self) -> None:
        """Start the writer and lease-renewal tasks. Called once after construction."""
        self._writer_task = asyncio.create_task(
            self._writer_loop(), name=f"ws-writer:{self.session_id}:{self.speaker}"
        )
        if self._redis is not None:
            self._lease_task = asyncio.create_task(
                self._lease_renew_loop(), name=f"lease-renew:{self.session_id}:{self.speaker}"
            )

    async def release(self) -> None:
        """
        Deterministic cleanup. Safe to call multiple times.

        Order:
        1. Mark closed.
        2. Cancel + await active turn task.
        3. Cancel + await writer task.
        4. Close adapter.
        5. Release Redis lease.
        6. Remove registry entry (if generation still matches).
        """
        if self._closed:
            return
        self._closed = True
        cleanup_start = time.monotonic()

        # 1. Cancel turn pipeline task
        await self._cancel_task(self._turn_task, "turn-pipeline")
        self._turn_task = None

        # 2. Drain / cancel lease renewal
        await self._cancel_task(self._lease_task, "lease-renew")
        self._lease_task = None

        # 3. Signal writer to stop and await it
        try:
            self.outbound_queue.put_nowait(None)   # sentinel
        except asyncio.QueueFull:
            pass
        await self._cancel_task(self._writer_task, "ws-writer")
        self._writer_task = None

        # 4. Close adapter
        try:
            await self.adapter.close()
        except Exception as exc:
            print(f"[ConnOwner] adapter.close() error: {exc}")

        # 5. Release Redis lease
        await self._release_lease()

        # 6. Remove from registry
        key = _owner_key(self.session_id, self.speaker)
        async with _registry_lock:
            existing = _registry.get(key)
            if existing is not None and existing.generation == self.generation:
                del _registry[key]

        elapsed = int((time.monotonic() - cleanup_start) * 1000)
        print(f"[ConnOwner] Released {self.session_id}:{self.speaker} "
              f"gen={self.generation} cleanup_ms={elapsed}")

    # ── Private ───────────────────────────────────────────────────────────────

    async def _writer_loop(self) -> None:
        """Drains outbound_queue and sends JSON to the browser WebSocket."""
        try:
            while True:
                msg = await self.outbound_queue.get()
                if msg is None:   # sentinel — shut down
                    break
                try:
                    await self.websocket.send_json(msg)
                except Exception as exc:
                    print(f"[ConnOwner] Writer send error: {exc}")
                    break
        except asyncio.CancelledError:
            pass
        finally:
            print(f"[ConnOwner] Writer stopped for {self.session_id}:{self.speaker}")

    async def _lease_renew_loop(self) -> None:
        """Renews the Redis ownership lease every REDIS_LEASE_RENEW_SEC seconds."""
        try:
            while True:
                await asyncio.sleep(REDIS_LEASE_RENEW_SEC)
                await self._renew_lease()
        except asyncio.CancelledError:
            pass

    async def _renew_lease(self) -> None:
        if self._redis is None or not self._lease_key:
            return
        try:
            current = self._redis.get(self._lease_key)
            if current == self._lease_token:
                self._redis.expire(self._lease_key, REDIS_LEASE_TTL_SEC)
        except Exception as exc:
            print(f"[ConnOwner] Lease renewal error: {exc}")

    async def _release_lease(self) -> None:
        if self._redis is None or not self._lease_key:
            return
        try:
            current = self._redis.get(self._lease_key)
            if current == self._lease_token:
                self._redis.delete(self._lease_key)
                print(f"[ConnOwner] Redis lease released: {self._lease_key}")
            else:
                print("[ConnOwner] Lease token mismatch — not releasing foreign lease")
        except Exception as exc:
            print(f"[ConnOwner] Lease release error: {exc}")

    @staticmethod
    async def _cancel_task(task: asyncio.Task | None, name: str) -> None:
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception) as exc:
            if not isinstance(exc, asyncio.CancelledError):
                print(f"[ConnOwner] {name} task error during cancel: {exc}")


# ── Global factory ────────────────────────────────────────────────────────────

def _owner_key(session_id: str, speaker: str) -> str:
    return f"{session_id}:{speaker}"


async def acquire_connection(
    session_id: str,
    speaker:    str,
    websocket:  WebSocket,
    adapter:    SarvamLiveASRAdapter,
    redis=None,
) -> LiveConnectionOwner | None:
    """
    Attempt to register a new connection owner for (session_id, speaker).

    Returns a LiveConnectionOwner on success (background tasks already started).
    Returns None if a duplicate connection is detected; the new websocket is
    closed with code 4409 DUPLICATE_CONNECTION before returning.

    Redis lease:
      If redis is provided, acquires `SET live-owner:{sid}:{sp} {token} NX EX 30`.
      If the key already exists, another worker owns this session-speaker and
      the new connection is rejected.
    """
    key   = _owner_key(session_id, speaker)
    token = f"{WORKER_ID}:{id(websocket)}"

    # ── Try Redis lease first (cross-worker duplicate check) ───────────────────
    if redis is not None:
        lease_key = redis_lease_key(session_id, speaker)
        try:
            acquired = redis.set(lease_key, token, nx=True, ex=REDIS_LEASE_TTL_SEC)
        except Exception as exc:
            print(f"[ConnManager] Redis lease error: {exc}")
            acquired = None   # proceed without Redis lease if Redis is down

        if not acquired:
            print(f"[ConnManager] Redis lease held by another worker for {key} — rejecting")
            try:
                await websocket.close(
                    code=WSCloseCode.DUPLICATE_CONNECTION,
                    reason=TurnErrorCode.DUPLICATE_CONNECTION,
                )
            except Exception:
                pass
            return None
    else:
        lease_key = ""
        token     = ""

    # ── Try process-local registry ────────────────────────────────────────────
    async with _registry_lock:
        existing = _registry.get(key)
        if existing is not None:
            print(f"[ConnManager] Process-local duplicate for {key} gen={existing.generation} — rejecting")
            try:
                await websocket.close(
                    code=WSCloseCode.DUPLICATE_CONNECTION,
                    reason=TurnErrorCode.DUPLICATE_CONNECTION,
                )
            except Exception:
                pass
            # Release the Redis lease we just acquired (we're not taking over)
            if redis is not None and lease_key:
                try:
                    redis.delete(lease_key)
                except Exception:
                    pass
            return None

        generation = (existing.generation + 1) if existing else 1
        owner = LiveConnectionOwner(
            session_id=session_id,
            speaker=speaker,
            websocket=websocket,
            adapter=adapter,
            generation=generation,
            _redis=redis,
            _lease_key=lease_key,
            _lease_token=token,
        )
        _registry[key] = owner

    await owner.start_background_tasks()
    print(f"[ConnManager] Acquired {key} gen={generation} worker={WORKER_ID[:8]}")
    return owner


def get_owner(session_id: str, speaker: str) -> LiveConnectionOwner | None:
    """Return the current owner for (session_id, speaker), or None."""
    return _registry.get(_owner_key(session_id, speaker))
