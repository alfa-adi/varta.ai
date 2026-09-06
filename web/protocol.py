"""
web/protocol.py
───────────────
Varta live WebSocket protocol constants.

All message type strings, stable error codes, and WebSocket close codes
are defined here. Import from this module only — never hard-code strings
in server.py, connection_manager.py, or tests.

Browser → Server messages
─────────────────────────
  turn_start      Begin a new turn; sent before the first binary audio frame.
  stop_recording  End of capture; triggers ASR flush and NMT/TTS pipeline.
  (binary)        Raw PCM audio chunks (s16le, 16 kHz, mono, 20 ms each).

Server → Browser messages
─────────────────────────
  server_ready        Sent once per connection after session config is loaded.
  transcript_partial  Live partial ASR result for the active turn.
  transcript_final    Confirmed final ASR result; triggers NMT/TTS.
  language_detected   ASR-detected BCP-47 language code.
  audio_chunk         A TTS audio fragment (pcm_s16le, 24 kHz, mono).
  audio_end           Server has finished sending all audio for this turn.
                      Browser playback may still be active.
  turn_error          Terminal error — mutually exclusive with audio_end.
  turn_cancelled      Terminal cancellation — mutually exclusive with audio_end.
"""

# ── Browser → Server message types ───────────────────────────────────────────

MSG_TURN_START       = "turn_start"
MSG_STOP_RECORDING   = "stop_recording"

# ── Server → Browser message types ───────────────────────────────────────────

MSG_SERVER_READY        = "server_ready"
MSG_TRANSCRIPT_PARTIAL  = "transcript_partial"
MSG_TRANSCRIPT_FINAL    = "transcript_final"
MSG_LANGUAGE_DETECTED   = "language_detected"
MSG_AUDIO_CHUNK         = "audio_chunk"
MSG_AUDIO_END           = "audio_end"
MSG_TURN_ERROR          = "turn_error"
MSG_TURN_CANCELLED      = "turn_cancelled"

# ── Audio format fields (carried on every audio_chunk message) ────────────────

AUDIO_FORMAT     = "pcm_s16le"
AUDIO_SAMPLE_RATE = 24_000
AUDIO_CHANNELS   = 1

# ── Protocol version ─────────────────────────────────────────────────────────

PROTOCOL_VERSION = "1"

# ── Stable turn error codes ───────────────────────────────────────────────────
# These strings are part of the public wire contract.
# Do not rename without a versioned migration.

class TurnErrorCode:
    # Connection lifecycle
    CONNECTION_OPEN_TIMEOUT     = "CONNECTION_OPEN_TIMEOUT"
    DUPLICATE_CONNECTION        = "DUPLICATE_CONNECTION"       # also used as WS close reason

    # ASR / upstream
    ASR_SESSION_START_TIMEOUT   = "ASR_SESSION_START_TIMEOUT"
    UPSTREAM_IDLE_TIMEOUT       = "UPSTREAM_IDLE_TIMEOUT"
    UPSTREAM_RECONNECT_FAILED   = "UPSTREAM_RECONNECT_FAILED"
    TRANSCRIPT_BACKPRESSURE     = "TRANSCRIPT_BACKPRESSURE"
    FINAL_TRANSCRIPT_TIMEOUT    = "FINAL_TRANSCRIPT_TIMEOUT"

    # NMT
    NMT_TIMEOUT                 = "NMT_TIMEOUT"
    NMT_ERROR                   = "NMT_ERROR"

    # TTS
    TTS_FIRST_BYTE_TIMEOUT      = "TTS_FIRST_BYTE_TIMEOUT"
    TTS_COMPLETION_TIMEOUT      = "TTS_COMPLETION_TIMEOUT"
    TTS_ERROR                   = "TTS_ERROR"

    # Turn management
    TURN_IN_PROGRESS            = "TURN_IN_PROGRESS"
    TURN_TOTAL_TIMEOUT          = "TURN_TOTAL_TIMEOUT"
    OUTBOUND_BACKPRESSURE       = "OUTBOUND_BACKPRESSURE"

    # Session / deployment
    SESSION_NOT_FOUND           = "SESSION_NOT_FOUND"

    # Audio / capture (browser-side; included for server log parity)
    AUDIO_BACKPRESSURE          = "AUDIO_BACKPRESSURE"
    AUDIO_SAMPLE_RATE_UNSUPPORTED = "AUDIO_SAMPLE_RATE_UNSUPPORTED"


# ── WebSocket close codes ─────────────────────────────────────────────────────
# Standard range: 1000–1015 (RFC 6455)
# Application range: 4000–4999 (private use)

class WSCloseCode:
    NORMAL_CLOSURE      = 1000
    POLICY_VIOLATION    = 1008   # auth / policy failure
    DUPLICATE_CONNECTION = 4409  # second owner attempt for same session-speaker
    # Browser must NOT auto-reconnect on 4409.


# ── Timeouts (seconds) ────────────────────────────────────────────────────────

class Timeout:
    BROWSER_OPEN        = 10.0
    ASR_SESSION_BEGIN   = 8.0
    UPSTREAM_IDLE       = 30.0
    FINAL_TRANSCRIPT    = 45.0
    NMT                 = 8.0
    TTS_FIRST_BYTE      = 10.0
    TTS_COMPLETION      = 30.0
    TURN_TOTAL          = 90.0
    OUTBOUND_BACKPRESSURE_GRACE = 1.0  # seconds before OUTBOUND_BACKPRESSURE is raised

# ── Queue / buffer limits ─────────────────────────────────────────────────────

OUTBOUND_QUEUE_MAX  = 64    # server outbound queue to browser
TRANSCRIPT_QUEUE_MAX = 256  # ASR adapter's inbound transcript queue

# ── Redis key helpers ─────────────────────────────────────────────────────────

REDIS_LEASE_TTL_SEC    = 30   # absolute TTL for live-owner lease
REDIS_LEASE_RENEW_SEC  = 15   # renew heartbeat interval
SESSION_TURN_LEASE_TTL_SEC = 120  # stale cross-speaker turn lease safety limit


def redis_session_key(session_id: str) -> str:
    return f"session:{session_id}"


def redis_lease_key(session_id: str, speaker: str) -> str:
    return f"live-owner:{session_id}:{speaker}"


def redis_session_turn_key(session_id: str) -> str:
    """Redis key that permits only one active speaker turn per session."""
    return f"live-turn:{session_id}"


def redis_asr_lang_key(session_id: str, speaker: str) -> str:
    return f"asr:lang:{speaker}:{session_id}"
