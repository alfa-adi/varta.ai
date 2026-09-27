"""
test/ws_reliability_test/ws_client.py
──────────────────────────────────────
Persistent WebSocket conversation client for the V2 reliability test.

One WSConversationClient instance manages a single WebSocket connection
that stays alive for all turns in a conversation (one per language).

V2 WebSocket endpoint: ws://HOST/ws/asr/{session_id}/{speaker}
Speaker: always "a" for the test runner (we are the one speaker)

Binary frames (browser → server):
    Raw pcm_s16le bytes, 640 bytes per frame (320 Int16 samples = 20ms at 16kHz)
    Same format as pcm-processor.js AudioWorklet output.

Text frames (browser → server):
    {"type": "stop_recording"}  — signals end of utterance

Text JSON messages (server → browser):
    {"type": "transcript_partial", "transcript": "..."}
    {"type": "transcript_final", "transcript": "..."}
    {"type": "language_detected", "language": "hi-IN", "speaker": "a"}
    {"type": "audio_chunk", "data": "<base64 linear16 24kHz>", "format": "mp3"}
    {"type": "audio_end"}
    {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

import websockets
import websockets.exceptions

from .schema import TurnResult, TurnStatus, FailureStage
from .audio_convert import CHUNK_BYTES, decode_received_audio

# ── Timing constants ──────────────────────────────────────────────────────────

# How long to sleep between sending each 20ms PCM chunk.
# 18ms is slightly faster than real-time (20ms) to account for network overhead.
# Set to 0 to send as fast as possible (not recommended — server buffers may overflow).
CHUNK_SEND_INTERVAL_SEC = 0.018   # 18ms between 20ms chunks

# How long to wait for a single WebSocket message before timeout
MESSAGE_RECV_TIMEOUT_SEC = 15.0   # per-message timeout during turn

# Maximum time to wait for audio_end after stop_recording
AUDIO_END_TIMEOUT_SEC = 45.0      # TTS synthesis can take time for long texts

# Maximum time for the entire turn from PCM send start to audio_end
TURN_TOTAL_TIMEOUT_SEC = 90.0


class WSConversationClient:
    """
    Manages a single persistent WebSocket connection for one conversation.

    Usage:
        client = WSConversationClient(base_url, session_id)
        connect_latency_ms = await client.connect()

        for turn_number in range(1, 11):
            result = TurnResult(...)
            await client.run_turn(turn_number, pcm_chunks, result)

        await client.close()
    """

    def __init__(
        self,
        base_url: str,
        session_id: str,
        speaker: str = "a",
        verbose: bool = True,
    ):
        """
        Args:
            base_url:   HTTP/HTTPS server URL (e.g. "http://localhost:8000").
                        Converted to ws:// or wss:// automatically.
            session_id: Varta session ID (from POST /session/create).
            speaker:    "a" or "b" (always "a" for single-speaker test).
            verbose:    Print per-turn progress.
        """
        self._session_id  = session_id
        self._speaker     = speaker
        self._verbose     = verbose
        self._ws          = None
        self._ws_open_ts  = 0.0       # wall time when WS opened

        # Convert HTTP base URL to WebSocket URL
        if base_url.startswith("https://"):
            ws_base = base_url.replace("https://", "wss://", 1)
        elif base_url.startswith("http://"):
            ws_base = base_url.replace("http://", "ws://", 1)
        else:
            ws_base = base_url

        self._ws_url = f"{ws_base}/ws/asr/{session_id}/{speaker}"

    # ── Connection management ──────────────────────────────────────────────────

    def is_open(self) -> bool:
        """Check if the WebSocket is currently connected and open."""
        if self._ws is None:
            return False
        try:
            # websockets library may expose .open or .state depending on version
            if hasattr(self._ws, "open"):
                return bool(self._ws.open)
            from websockets.connection import State as _WSState
            return self._ws.state == _WSState.OPEN
        except Exception:
            return False

    async def connect(self) -> int:
        """
        Open the WebSocket connection.

        Returns:
            connect_latency_ms (int): Time from connect() call to open event.

        Raises:
            RuntimeError: If connection fails.
        """
        t0 = time.perf_counter()
        try:
            self._ws = await websockets.connect(
                self._ws_url,
                ping_interval=20,
                ping_timeout=10,
                open_timeout=15,
                max_size=10 * 1024 * 1024,   # 10MB — large enough for audio
            )
            self._ws_open_ts = time.time()
            latency_ms = int((time.perf_counter() - t0) * 1000)
            if self._verbose:
                print(f"  [WS] Connected → {self._ws_url}  ({latency_ms}ms)")
            return latency_ms
        except Exception as exc:
            raise RuntimeError(f"WebSocket connect failed: {exc}") from exc

    async def close(self):
        """Gracefully close the WebSocket connection."""
        if self._ws and self.is_open():
            try:
                await self._ws.close()
                if self._verbose:
                    print(f"  [WS] Closed connection for session {self._session_id[:8]}…")
            except Exception:
                pass
        self._ws = None

    # ── Turn execution ─────────────────────────────────────────────────────────

    async def run_turn(
        self,
        turn_number: int,
        pcm_chunks: list[bytes],
        result: TurnResult,
        play_audio: bool = False,
    ) -> None:
        """
        Execute one translation turn and populate result with all observations.

        Lifecycle:
          1. Check WS is open
          2. Send PCM chunks as binary frames (paced at ~real-time)
          3. Send {"type": "stop_recording"} JSON control message
          4. Receive messages until audio_end or timeout
          5. Validate received audio bytes as PCM

        All timings and observations are written directly into result.
        Errors are recorded but never re-raised — the caller sees them in result.status.
        """
        result.turn_start_ts = time.time()
        result.turn_number   = turn_number

        # ── 1. WebSocket health check ─────────────────────────────────
        result.ws_was_open_at_turn_start = self.is_open()
        result.connection_age_ms = int((time.time() - self._ws_open_ts) * 1000)

        if not result.ws_was_open_at_turn_start:
            result.status        = TurnStatus.FAILURE
            result.failure_stage = FailureStage.CONNECTION
            result.failure_reason = "WebSocket not open at turn start"
            result.turn_end_ts   = time.time()
            if self._verbose:
                print(f"  [Turn {turn_number}] FAIL — WS not open")
            return

        # ── 2. Send PCM chunks ────────────────────────────────────────
        result.pcm_bytes_sent = 0
        result.chunks_sent    = 0
        result.send_start_ts  = time.time()

        try:
            for i, chunk in enumerate(pcm_chunks):
                if not self.is_open():
                    result.errors.append(f"WS closed during send at chunk {i}")
                    result.status        = TurnStatus.FAILURE
                    result.failure_stage = FailureStage.SEND
                    result.failure_reason = f"WebSocket closed while sending chunk {i}/{len(pcm_chunks)}"
                    result.turn_end_ts   = time.time()
                    result.send_end_ts   = result.turn_end_ts
                    result.send_latency_ms = int((result.send_end_ts - result.send_start_ts) * 1000)
                    return

                await self._ws.send(chunk)  # binary frame — raw PCM bytes
                result.pcm_bytes_sent += len(chunk)
                result.chunks_sent    += 1

                # Pace the send to approximate real-time recording
                if i < len(pcm_chunks) - 1:   # no delay after last chunk
                    await asyncio.sleep(CHUNK_SEND_INTERVAL_SEC)

        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException) as exc:
            result.errors.append(f"WS error during PCM send: {exc}")
            result.status        = TurnStatus.FAILURE
            result.failure_stage = FailureStage.SEND
            result.failure_reason = f"WebSocket closed during send"
            result.failure_message = str(exc)
            result.turn_end_ts   = time.time()
            result.send_end_ts   = result.turn_end_ts
            result.send_latency_ms = int((result.send_end_ts - result.send_start_ts) * 1000)
            return

        result.send_end_ts   = time.time()
        result.send_latency_ms = int((result.send_end_ts - result.send_start_ts) * 1000)

        if self._verbose:
            pct = int(result.pcm_bytes_sent / max(result.pcm_bytes_total, 1) * 100)
            print(f"  [Turn {turn_number}] Sent {result.chunks_sent} chunks "
                  f"({result.pcm_bytes_sent:,} bytes, {pct}% of fixture) "
                  f"in {result.send_latency_ms}ms")

        # ── 3. Send stop_recording ────────────────────────────────────
        try:
            stop_msg = json.dumps({"type": "stop_recording"})
            await self._ws.send(stop_msg)
            result.stop_recording_ts = time.time()
        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException) as exc:
            result.errors.append(f"WS error sending stop_recording: {exc}")
            result.status        = TurnStatus.FAILURE
            result.failure_stage = FailureStage.SEND
            result.failure_reason = "WebSocket closed when sending stop_recording"
            result.failure_message = str(exc)
            result.turn_end_ts   = time.time()
            return

        # ── 4. Receive messages until audio_end ───────────────────────
        audio_bytes_accumulator: list[bytes] = []
        turn_deadline = time.time() + TURN_TOTAL_TIMEOUT_SEC

        try:
            while True:
                remaining = turn_deadline - time.time()
                if remaining <= 0:
                    result.errors.append("Turn total timeout exceeded")
                    result.status        = TurnStatus.FAILURE
                    result.failure_stage = FailureStage.TIMEOUT
                    result.failure_reason = f"Turn exceeded {TURN_TOTAL_TIMEOUT_SEC}s"
                    result.turn_end_ts   = time.time()
                    return

                try:
                    raw_msg = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=min(MESSAGE_RECV_TIMEOUT_SEC, remaining),
                    )
                except asyncio.TimeoutError:
                    result.errors.append(
                        f"Message receive timeout after {MESSAGE_RECV_TIMEOUT_SEC}s "
                        f"(partial:{result.asr_partial_count}, "
                        f"asr_ok:{result.asr_success}, "
                        f"audio_chunks:{result.audio_chunks_received})"
                    )
                    result.status        = TurnStatus.FAILURE
                    result.failure_stage = FailureStage.TIMEOUT
                    result.failure_reason = "Message receive timeout"
                    result.turn_end_ts   = time.time()
                    # Classify what we got before timeout
                    if result.asr_success and result.audio_chunks_received > 0:
                        result.status        = TurnStatus.PARTIAL_SUCCESS
                        result.failure_stage = FailureStage.RECEIVE
                        result.failure_reason = "Timeout waiting for audio_end (partial stream received)"
                    elif result.asr_success:
                        result.failure_stage = FailureStage.NMT_TTS
                        result.failure_reason = "Timeout waiting for audio after ASR succeeded"
                    elif result.pipeline_triggered:
                        result.failure_stage = FailureStage.ASR
                    return

                # Parse the JSON message
                try:
                    msg = json.loads(raw_msg) if isinstance(raw_msg, str) else json.loads(raw_msg)
                except json.JSONDecodeError as exc:
                    result.errors.append(f"JSON parse error: {exc} | raw={str(raw_msg)[:100]}")
                    continue

                msg_type = msg.get("type", "")
                recv_ts  = time.time()

                # ── Dispatch on message type ───────────────────────
                if msg_type == "transcript_partial":
                    result.asr_partial_count += 1
                    result.pipeline_triggered = True   # server is processing

                elif msg_type == "transcript_final":
                    result.pipeline_triggered  = True
                    result.asr_success         = True
                    result.asr_transcript      = msg.get("transcript", "")
                    result.asr_latency_ms      = int((recv_ts - result.stop_recording_ts) * 1000)
                    if self._verbose:
                        transcript_preview = (result.asr_transcript or "")[:60]
                        print(f"  [Turn {turn_number}] ASR ✓ in {result.asr_latency_ms}ms: "
                              f"'{transcript_preview}'")

                elif msg_type == "language_detected":
                    result.asr_detected_language = msg.get("language", "")

                elif msg_type == "audio_chunk":
                    encoded = msg.get("data", "")
                    if not encoded:
                        result.errors.append("audio_chunk received with empty data field")
                        continue

                    try:
                        chunk_bytes = base64.b64decode(encoded)
                    except Exception as exc:
                        result.errors.append(f"base64 decode failed on audio_chunk: {exc}")
                        continue

                    if result.audio_chunks_received == 0:
                        result.first_audio_chunk_ts = recv_ts
                        if result.stop_recording_ts > 0:
                            result.pipeline_latency_ms = int(
                                (recv_ts - result.stop_recording_ts) * 1000
                            )
                        if self._verbose:
                            print(f"  [Turn {turn_number}] First audio chunk received "
                                  f"(pipeline: {result.pipeline_latency_ms}ms)")

                    audio_bytes_accumulator.append(chunk_bytes)
                    result.audio_chunks_received += 1
                    result.audio_bytes_received  += len(chunk_bytes)
                    result.last_audio_chunk_ts    = recv_ts

                elif msg_type == "audio_end":
                    result.audio_end_received = True
                    end_ts = recv_ts
                    if result.last_audio_chunk_ts and result.first_audio_chunk_ts:
                        result.audio_delivery_ms = int(
                            (result.last_audio_chunk_ts - result.first_audio_chunk_ts) * 1000
                        )
                    result.total_e2e_latency_ms = int((end_ts - result.turn_start_ts) * 1000)
                    if self._verbose:
                        print(f"  [Turn {turn_number}] audio_end received | "
                              f"chunks={result.audio_chunks_received} | "
                              f"bytes={result.audio_bytes_received:,} | "
                              f"e2e={result.total_e2e_latency_ms}ms")
                    break  # turn complete

                elif msg_type == "error":
                    err_msg = msg.get("message", "unknown server error")
                    result.server_error_received = True
                    result.server_error_message  = err_msg
                    result.errors.append(f"Server error: {err_msg}")
                    if self._verbose:
                        print(f"  [Turn {turn_number}] Server error: {err_msg}")
                    # Server still sends audio_end even on errors — wait for it
                    # so we don't leave the WebSocket in a dirty state

                else:
                    # Unknown message type — log and continue
                    result.errors.append(f"Unknown message type: {msg_type!r}")

        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.WebSocketException) as exc:
            result.errors.append(f"WS error during receive: {exc}")
            if result.audio_chunks_received > 0:
                result.status        = TurnStatus.PARTIAL_SUCCESS
                result.failure_stage = FailureStage.RECEIVE
            elif result.asr_success:
                result.status        = TurnStatus.FAILURE
                result.failure_stage = FailureStage.NMT_TTS
            else:
                result.status        = TurnStatus.FAILURE
                result.failure_stage = FailureStage.ASR
            result.failure_reason  = "WebSocket closed during receive"
            result.failure_message = str(exc)
            result.turn_end_ts     = time.time()
            return

        result.turn_end_ts = time.time()

        # ── 5. Validate received PCM audio ────────────────────────────
        if audio_bytes_accumulator:
            all_audio = b"".join(audio_bytes_accumulator)
            result.decode_attempted  = True
            ok, err, samples, dur = decode_received_audio(all_audio)
            result.decode_success    = ok
            result.decode_error      = err
            result.decoded_samples   = samples
            result.decoded_duration_sec = dur

            if not ok:
                result.errors.append(f"PCM decode validation failed: {err}")
            
            if play_audio:
                try:
                    # Attempt to play the audio synchronously
                    from pydub import AudioSegment
                    from pydub.playback import play
                    
                    seg = AudioSegment(
                        data=all_audio,
                        sample_width=2,
                        frame_rate=24000,
                        channels=1
                    )
                    if self._verbose:
                        print(f"  [Turn {turn_number}] 🔊 Playing received audio...")
                    play(seg)
                except Exception as exc:
                    if self._verbose:
                        print(f"  [Turn {turn_number}] ❌ Failed to play audio: {exc}")

        # ── 6. Classify final turn status ─────────────────────────────
        _classify_turn(result)

        if self._verbose:
            _print_turn_summary(result)


# ── Classification helper ─────────────────────────────────────────────────────

def _classify_turn(r: TurnResult) -> None:
    """Assign r.status and r.failure_stage based on what was observed."""

    # Already classified (e.g. connection failure set status early)
    if r.status not in (TurnStatus.NOT_RUN,):
        if r.failure_stage is not None:
            return   # already fully classified

    if not r.ws_was_open_at_turn_start:
        r.status        = TurnStatus.FAILURE
        r.failure_stage = FailureStage.CONNECTION
        r.failure_reason = r.failure_reason or "WS not open at turn start"
        return

    if not r.fixture_valid:
        r.status        = TurnStatus.FAILURE
        r.failure_stage = FailureStage.FIXTURE
        r.failure_reason = r.failure_reason or "Fixture invalid or missing"
        return

    send_ratio = r.pcm_bytes_sent / max(r.pcm_bytes_total, 1)
    if send_ratio < 0.5:
        r.status        = TurnStatus.FAILURE
        r.failure_stage = FailureStage.SEND
        r.failure_reason = r.failure_reason or f"Only {send_ratio:.0%} of PCM bytes sent"
        return

    if not r.asr_success or not r.asr_transcript:
        r.status        = TurnStatus.FAILURE
        r.failure_stage = FailureStage.ASR
        r.failure_reason = r.failure_reason or "No ASR transcript_final received"
        return

    if r.audio_bytes_received == 0:
        r.status        = TurnStatus.FAILURE
        r.failure_stage = FailureStage.NMT_TTS
        r.failure_reason = r.failure_reason or "No audio_chunk received (NMT/TTS failed or timed out)"
        return

    if r.decode_attempted and not r.decode_success:
        r.status        = TurnStatus.PARTIAL_SUCCESS
        r.failure_stage = FailureStage.DECODE
        r.failure_reason = r.failure_reason or f"PCM decode validation failed: {r.decode_error}"
        return

    if not r.audio_end_received:
        r.status        = TurnStatus.PARTIAL_SUCCESS
        r.failure_stage = FailureStage.RECEIVE
        r.failure_reason = r.failure_reason or "Audio stream incomplete (no audio_end received)"
        return

    # All required stages passed
    r.status        = TurnStatus.CLEAN_SUCCESS
    r.failure_stage = None
    r.failure_reason = None


def _print_turn_summary(r: TurnResult) -> None:
    status_icon = {
        TurnStatus.CLEAN_SUCCESS:   "✅",
        TurnStatus.PARTIAL_SUCCESS: "⚠️ ",
        TurnStatus.FAILURE:         "❌",
        TurnStatus.NOT_RUN:         "⏭️ ",
    }.get(r.status, "?")

    stage = f" [{r.failure_stage.value}]" if r.failure_stage else ""
    reason = f": {r.failure_reason}" if r.failure_reason else ""
    print(f"  {status_icon} Turn {r.turn_number:2d} | {r.source_language} → {r.target_language} | "
          f"e2e={r.total_e2e_latency_ms or '—'}ms | "
          f"asr={r.asr_latency_ms or '—'}ms | "
          f"pipe={r.pipeline_latency_ms or '—'}ms | "
          f"audio={r.audio_bytes_received:,}B | "
          f"{r.status.value}{stage}{reason}")
