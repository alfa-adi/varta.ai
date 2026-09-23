"""
test/ws_reliability_test/schema.py
────────────────────────────────────
Data models for the 120-turn V2 WebSocket reliability + latency test.

Every TurnResult captures the full lifecycle of one translation turn
through the V2 WebSocket pipeline:
  PCM send → ASR → NMT+TTS (server-side) → audio_chunk delivery → audio_end

One TurnResult per turn (120 total).
One ConversationResult per language (11 total, since sd excluded).
One ExperimentResult for the full run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Status enum ───────────────────────────────────────────────────────────────

class TurnStatus(str, Enum):
    CLEAN_SUCCESS     = "CLEAN_SUCCESS"      # All stages succeeded
    PARTIAL_SUCCESS   = "PARTIAL_SUCCESS"    # Audio arrived but stream incomplete or PCM invalid
    FAILURE           = "FAILURE"            # Pipeline failed before full audio delivery
    NOT_RUN           = "NOT_RUN"            # Turn was never attempted (e.g. fixture missing)


class FailureStage(str, Enum):
    CONNECTION   = "CONNECTION"    # WS not open at turn start
    FIXTURE      = "FIXTURE"       # Audio file missing or invalid
    SEND         = "SEND"          # PCM send failed or incomplete
    ASR          = "ASR"           # No transcript_final received
    NMT_TTS      = "NMT_TTS"       # No audio_chunk received after stop_recording
    RECEIVE      = "RECEIVE"       # Partial audio stream (no audio_end)
    DECODE       = "DECODE"        # audio_bytes not valid PCM
    TIMEOUT      = "TIMEOUT"       # Stage timed out
    PROTOCOL     = "PROTOCOL"      # Unexpected WS message / sequence error
    UNKNOWN      = "UNKNOWN"       # Unclassified


# ── Per-turn result ───────────────────────────────────────────────────────────

@dataclass
class TurnResult:
    """
    Complete record for one translation turn.
    Populated incrementally as the turn progresses.
    """

    # ── Identity ──────────────────────────────────────────────────
    session_id:          str = ""
    conversation_id:     str = ""          # e.g. "conv_hi_003"
    turn_id:             str = ""          # e.g. "conv_hi_003_turn_07"
    turn_number:         int = 0           # 1-based within conversation
    source_language:     str = ""          # BCP-47 e.g. "bn-IN"
    target_language:     str = "hi-IN"     # always Hindi for this test
    audio_file:          str = ""          # relative path to fixture

    # ── Fixture validation ─────────────────────────────────────────
    fixture_valid:       bool  = False
    fixture_size_bytes:  int   = 0
    pcm_bytes_total:     int   = 0         # total PCM bytes after WebM decode
    chunks_total:        int   = 0         # number of 640-byte chunks
    audio_duration_est_sec: float = 0.0   # pcm_bytes / (16000 * 2)

    # ── WebSocket health ───────────────────────────────────────────
    ws_was_open_at_turn_start: bool = False
    ws_connect_latency_ms:     Optional[int] = None   # only set on turn 1
    connection_age_ms:         int   = 0              # ms since WS was opened

    # ── PCM audio send ─────────────────────────────────────────────
    pcm_bytes_sent:  int   = 0
    chunks_sent:     int   = 0
    send_start_ts:   float = 0.0
    send_end_ts:     float = 0.0
    send_latency_ms: int   = 0

    # ── ASR ────────────────────────────────────────────────────────
    asr_success:           bool           = False
    asr_transcript:        Optional[str]  = None
    asr_detected_language: Optional[str]  = None
    asr_partial_count:     int            = 0
    asr_latency_ms:        Optional[int]  = None  # stop_recording → transcript_final
    stop_recording_ts:     float          = 0.0

    # ── NMT + TTS (server-side, measured indirectly) ───────────────
    pipeline_triggered:      bool          = False   # server got final transcript
    first_audio_chunk_ts:    Optional[float] = None
    last_audio_chunk_ts:     Optional[float] = None
    audio_end_received:      bool          = False
    audio_chunks_received:   int           = 0
    audio_bytes_received:    int           = 0

    # Latencies derived from timestamps
    pipeline_latency_ms:  Optional[int] = None   # stop_recording → first audio_chunk (NMT+TTS combined)
    audio_delivery_ms:    Optional[int] = None   # first audio_chunk → audio_end

    # ── PCM decode validation (Python-side) ───────────────────────
    decode_attempted:    bool           = False
    decode_success:      Optional[bool] = None
    decode_error:        Optional[str]  = None
    decoded_samples:     int            = 0       # Int16 samples count
    decoded_duration_sec: float         = 0.0     # decoded_samples / 24000

    # ── Error event from server ────────────────────────────────────
    server_error_received: bool          = False
    server_error_message:  Optional[str] = None

    # ── Result classification ──────────────────────────────────────
    status:           TurnStatus           = TurnStatus.NOT_RUN
    failure_stage:    Optional[FailureStage] = None
    failure_reason:   Optional[str]          = None
    failure_message:  Optional[str]          = None
    errors:           list[str]              = field(default_factory=list)

    # ── Timestamps ─────────────────────────────────────────────────
    turn_start_ts:  float = 0.0
    turn_end_ts:    float = 0.0
    total_e2e_latency_ms: Optional[int] = None   # turn_start → audio_end

    def to_dict(self) -> dict:
        """Convert to plain dict for JSON serialisation / MongoDB storage."""
        return {
            # Identity
            "session_id":           self.session_id,
            "conversation_id":      self.conversation_id,
            "turn_id":              self.turn_id,
            "turn_number":          self.turn_number,
            "source_language":      self.source_language,
            "target_language":      self.target_language,
            "audio_file":           self.audio_file,

            # Fixture
            "fixture_valid":        self.fixture_valid,
            "fixture_size_bytes":   self.fixture_size_bytes,
            "pcm_bytes_total":      self.pcm_bytes_total,
            "chunks_total":         self.chunks_total,
            "audio_duration_est_sec": self.audio_duration_est_sec,

            # WebSocket health
            "ws_was_open_at_turn_start": self.ws_was_open_at_turn_start,
            "ws_connect_latency_ms":     self.ws_connect_latency_ms,
            "connection_age_ms":         self.connection_age_ms,

            # Audio send
            "pcm_bytes_sent":   self.pcm_bytes_sent,
            "chunks_sent":      self.chunks_sent,
            "send_latency_ms":  self.send_latency_ms,

            # ASR
            "asr_success":           self.asr_success,
            "asr_transcript":        self.asr_transcript,
            "asr_detected_language": self.asr_detected_language,
            "asr_partial_count":     self.asr_partial_count,
            "asr_latency_ms":        self.asr_latency_ms,

            # NMT+TTS pipeline
            "pipeline_triggered":    self.pipeline_triggered,
            "first_audio_chunk_ts":  self.first_audio_chunk_ts,
            "audio_end_received":    self.audio_end_received,
            "audio_chunks_received": self.audio_chunks_received,
            "audio_bytes_received":  self.audio_bytes_received,
            "pipeline_latency_ms":   self.pipeline_latency_ms,
            "audio_delivery_ms":     self.audio_delivery_ms,

            # Decode
            "decode_attempted":      self.decode_attempted,
            "decode_success":        self.decode_success,
            "decode_error":          self.decode_error,
            "decoded_samples":       self.decoded_samples,
            "decoded_duration_sec":  self.decoded_duration_sec,

            # Server error
            "server_error_received": self.server_error_received,
            "server_error_message":  self.server_error_message,

            # Classification
            "status":          self.status.value,
            "failure_stage":   self.failure_stage.value if self.failure_stage else None,
            "failure_reason":  self.failure_reason,
            "failure_message": self.failure_message,
            "errors":          self.errors,

            # Timestamps
            "turn_start_ts":        self.turn_start_ts,
            "turn_end_ts":          self.turn_end_ts,
            "total_e2e_latency_ms": self.total_e2e_latency_ms,
        }


# ── Per-conversation result ───────────────────────────────────────────────────

@dataclass
class ConversationResult:
    """Aggregated result for one language conversation (10 turns)."""

    language:       str
    session_id:     str
    conversation_id: str
    ws_connect_latency_ms: Optional[int] = None
    turns:          list[TurnResult] = field(default_factory=list)

    # ── Computed properties (call compute() after all turns complete)
    total_turns:        int = 0
    clean_successes:    int = 0
    partial_successes:  int = 0
    failures:           int = 0
    not_run:            int = 0

    ws_disconnects:      int = 0
    asr_failures:        int = 0
    nmt_tts_failures:    int = 0
    receive_failures:    int = 0
    decode_failures:     int = 0
    timeout_failures:    int = 0
    fixture_failures:    int = 0

    reliability_pct:         float = 0.0   # clean_successes / total_turns
    pipeline_success_pct:    float = 0.0   # turns with asr_success / total_turns
    median_e2e_latency_ms:   Optional[float] = None
    p95_e2e_latency_ms:      Optional[float] = None
    median_asr_latency_ms:   Optional[float] = None
    median_pipeline_latency_ms: Optional[float] = None

    primary_failure_stage: Optional[str] = None

    def compute(self):
        """Compute aggregate statistics from self.turns."""
        import statistics

        self.total_turns     = len(self.turns)
        self.clean_successes = sum(1 for t in self.turns if t.status == TurnStatus.CLEAN_SUCCESS)
        self.partial_successes = sum(1 for t in self.turns if t.status == TurnStatus.PARTIAL_SUCCESS)
        self.failures        = sum(1 for t in self.turns if t.status == TurnStatus.FAILURE)
        self.not_run         = sum(1 for t in self.turns if t.status == TurnStatus.NOT_RUN)

        self.ws_disconnects    = sum(1 for t in self.turns if t.failure_stage == FailureStage.CONNECTION)
        self.asr_failures      = sum(1 for t in self.turns if t.failure_stage == FailureStage.ASR)
        self.nmt_tts_failures  = sum(1 for t in self.turns if t.failure_stage == FailureStage.NMT_TTS)
        self.receive_failures  = sum(1 for t in self.turns if t.failure_stage == FailureStage.RECEIVE)
        self.decode_failures   = sum(1 for t in self.turns if t.failure_stage == FailureStage.DECODE)
        self.timeout_failures  = sum(1 for t in self.turns if t.failure_stage == FailureStage.TIMEOUT)
        self.fixture_failures  = sum(1 for t in self.turns if t.failure_stage == FailureStage.FIXTURE)

        if self.total_turns > 0:
            self.reliability_pct = round(self.clean_successes / self.total_turns * 100, 1)
            asr_ok = sum(1 for t in self.turns if t.asr_success)
            self.pipeline_success_pct = round(asr_ok / self.total_turns * 100, 1)

        e2e_vals = [t.total_e2e_latency_ms for t in self.turns
                    if t.total_e2e_latency_ms is not None]
        if e2e_vals:
            self.median_e2e_latency_ms = round(statistics.median(e2e_vals), 1)
            sorted_e2e = sorted(e2e_vals)
            idx = int(len(sorted_e2e) * 0.95)
            self.p95_e2e_latency_ms = sorted_e2e[min(idx, len(sorted_e2e)-1)]

        asr_vals = [t.asr_latency_ms for t in self.turns if t.asr_latency_ms is not None]
        if asr_vals:
            self.median_asr_latency_ms = round(statistics.median(asr_vals), 1)

        pipe_vals = [t.pipeline_latency_ms for t in self.turns if t.pipeline_latency_ms is not None]
        if pipe_vals:
            self.median_pipeline_latency_ms = round(statistics.median(pipe_vals), 1)

        # Most common failure stage among failed turns
        failed = [t for t in self.turns if t.failure_stage is not None]
        if failed:
            stage_counts: dict[str, int] = {}
            for t in failed:
                s = t.failure_stage.value
                stage_counts[s] = stage_counts.get(s, 0) + 1
            self.primary_failure_stage = max(stage_counts, key=stage_counts.get)

    def to_dict(self) -> dict:
        return {
            "language":        self.language,
            "session_id":      self.session_id,
            "conversation_id": self.conversation_id,
            "ws_connect_latency_ms": self.ws_connect_latency_ms,
            "total_turns":     self.total_turns,
            "clean_successes": self.clean_successes,
            "partial_successes": self.partial_successes,
            "failures":        self.failures,
            "not_run":         self.not_run,
            "reliability_pct": self.reliability_pct,
            "pipeline_success_pct": self.pipeline_success_pct,
            "median_e2e_latency_ms": self.median_e2e_latency_ms,
            "p95_e2e_latency_ms": self.p95_e2e_latency_ms,
            "median_asr_latency_ms": self.median_asr_latency_ms,
            "median_pipeline_latency_ms": self.median_pipeline_latency_ms,
            "primary_failure_stage": self.primary_failure_stage,
            "ws_disconnects":  self.ws_disconnects,
            "asr_failures":    self.asr_failures,
            "nmt_tts_failures": self.nmt_tts_failures,
            "receive_failures": self.receive_failures,
            "decode_failures": self.decode_failures,
            "timeout_failures": self.timeout_failures,
            "fixture_failures": self.fixture_failures,
        }


# ── Full experiment result ────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Full result of the 120-turn experiment."""

    run_id:       str = ""
    run_label:    str = ""       # e.g. "ws_reliability_2026-08-13_10-30"
    started_at:   float = 0.0
    finished_at:  float = 0.0

    config: dict = field(default_factory=dict)
    conversations: list[ConversationResult] = field(default_factory=list)

    # ── Top-level aggregates (call compute() to populate) ──────────
    total_turns:        int   = 0
    clean_successes:    int   = 0
    partial_successes:  int   = 0
    failures:           int   = 0
    not_run:            int   = 0

    clean_reliability_pct:       float = 0.0
    user_visible_success_pct:    float = 0.0   # clean + partial
    pipeline_success_pct:        float = 0.0

    # Failure breakdown
    connection_failures: int = 0
    fixture_failures:    int = 0
    send_failures:       int = 0
    asr_failures:        int = 0
    nmt_tts_failures:    int = 0
    receive_failures:    int = 0
    decode_failures:     int = 0
    timeout_failures:    int = 0
    unknown_failures:    int = 0

    # Stage latency statistics (over all successful turns)
    latency_stats: dict = field(default_factory=dict)

    # Turn-by-turn latency (turn_number 1–10, averaged across all languages)
    latency_by_turn: dict = field(default_factory=dict)

    def all_turns(self) -> list[TurnResult]:
        """Flat list of all TurnResult objects."""
        turns = []
        for conv in self.conversations:
            turns.extend(conv.turns)
        return turns

    def compute(self):
        """Compute all aggregate statistics."""
        import statistics

        for conv in self.conversations:
            conv.compute()

        turns = self.all_turns()
        self.total_turns     = len(turns)
        self.clean_successes = sum(1 for t in turns if t.status == TurnStatus.CLEAN_SUCCESS)
        self.partial_successes = sum(1 for t in turns if t.status == TurnStatus.PARTIAL_SUCCESS)
        self.failures        = sum(1 for t in turns if t.status == TurnStatus.FAILURE)
        self.not_run         = sum(1 for t in turns if t.status == TurnStatus.NOT_RUN)

        if self.total_turns > 0:
            self.clean_reliability_pct    = round(self.clean_successes / self.total_turns * 100, 1)
            self.user_visible_success_pct = round(
                (self.clean_successes + self.partial_successes) / self.total_turns * 100, 1
            )
            asr_ok = sum(1 for t in turns if t.asr_success)
            self.pipeline_success_pct = round(asr_ok / self.total_turns * 100, 1)

        # Failure breakdown by stage
        self.connection_failures = sum(1 for t in turns if t.failure_stage == FailureStage.CONNECTION)
        self.fixture_failures    = sum(1 for t in turns if t.failure_stage == FailureStage.FIXTURE)
        self.send_failures       = sum(1 for t in turns if t.failure_stage == FailureStage.SEND)
        self.asr_failures        = sum(1 for t in turns if t.failure_stage == FailureStage.ASR)
        self.nmt_tts_failures    = sum(1 for t in turns if t.failure_stage == FailureStage.NMT_TTS)
        self.receive_failures    = sum(1 for t in turns if t.failure_stage == FailureStage.RECEIVE)
        self.decode_failures     = sum(1 for t in turns if t.failure_stage == FailureStage.DECODE)
        self.timeout_failures    = sum(1 for t in turns if t.failure_stage == FailureStage.TIMEOUT)
        self.unknown_failures    = sum(1 for t in turns
                                       if t.status == TurnStatus.FAILURE
                                       and t.failure_stage in (None, FailureStage.UNKNOWN))

        # Stage latency statistics
        def _stats(vals: list) -> dict:
            if not vals:
                return {"count": 0, "available": False}
            sv = sorted(vals)
            return {
                "available": True,
                "count":  len(sv),
                "min":    sv[0],
                "mean":   round(sum(sv) / len(sv), 1),
                "median": round(statistics.median(sv), 1),
                "p95":    sv[min(int(len(sv) * 0.95), len(sv)-1)],
                "p99":    sv[min(int(len(sv) * 0.99), len(sv)-1)],
                "max":    sv[-1],
            }

        e2e_vals   = [t.total_e2e_latency_ms for t in turns if t.total_e2e_latency_ms is not None]
        asr_vals   = [t.asr_latency_ms for t in turns if t.asr_latency_ms is not None]
        pipe_vals  = [t.pipeline_latency_ms for t in turns if t.pipeline_latency_ms is not None]
        del_vals   = [t.audio_delivery_ms for t in turns if t.audio_delivery_ms is not None]
        send_vals  = [t.send_latency_ms for t in turns if t.send_latency_ms > 0]

        self.latency_stats = {
            "e2e":             _stats(e2e_vals),
            "asr":             _stats(asr_vals),
            "pipeline_nmt_tts": _stats(pipe_vals),
            "audio_delivery":  _stats(del_vals),
            "send":            _stats(send_vals),
        }

        # Latency by turn number (1–10)
        max_turn = max((t.turn_number for t in turns), default=0)
        by_turn: dict[int, dict] = {}
        for tn in range(1, max_turn + 1):
            tn_turns = [t for t in turns if t.turn_number == tn]
            tn_e2e   = [t.total_e2e_latency_ms for t in tn_turns if t.total_e2e_latency_ms is not None]
            by_turn[str(tn)] = {
                "turn": tn,
                "count": len(tn_turns),
                "success_count": sum(1 for t in tn_turns if t.status == TurnStatus.CLEAN_SUCCESS),
                **_stats(tn_e2e),
            }
        self.latency_by_turn = by_turn

    def to_dict(self) -> dict:
        return {
            "run_id":       self.run_id,
            "run_label":    self.run_label,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "config":       self.config,
            "total_turns":        self.total_turns,
            "clean_successes":    self.clean_successes,
            "partial_successes":  self.partial_successes,
            "failures":           self.failures,
            "not_run":            self.not_run,
            "clean_reliability_pct":    self.clean_reliability_pct,
            "user_visible_success_pct": self.user_visible_success_pct,
            "pipeline_success_pct":     self.pipeline_success_pct,
            "failure_breakdown": {
                "connection": self.connection_failures,
                "fixture":    self.fixture_failures,
                "send":       self.send_failures,
                "asr":        self.asr_failures,
                "nmt_tts":    self.nmt_tts_failures,
                "receive":    self.receive_failures,
                "decode":     self.decode_failures,
                "timeout":    self.timeout_failures,
                "unknown":    self.unknown_failures,
            },
            "latency_stats":  self.latency_stats,
            "latency_by_turn": self.latency_by_turn,
            "conversations": [c.to_dict() for c in self.conversations],
        }
