"""
adapter/sarvam_protocol.py
──────────────────────────
Sarvam realtime API protocol constants and event parsing helpers.

References:
  ASR realtime: wss://api.sarvam.ai/speech-to-text-realtime/ws
  TTS streaming: Bulbul v3 WebSocket

These constants are used only by the adapter layer. Nothing outside
adapter/ should import this module — the rest of Varta works with
the Varta protocol types in web/protocol.py and pipeline/types.py.
"""

# ── ASR realtime endpoint ─────────────────────────────────────────────────────

REALTIME_ASR_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"

ASR_MODEL        = "saaras:v3-realtime"
ASR_ENCODING     = "linear16"
ASR_SAMPLE_RATE  = 16_000          # only 8000 or 16000 accepted
ASR_STREAM_TYPE  = "fast"          # "fast" for conversational; "balanced" for accuracy
ASR_ENDPOINTING  = "manual"        # we control speech_start / speech_end

# language_code for auto-detection — send exactly this string, never "unknown"
ASR_LANGUAGE_AUTO = "auto"

# ── Sarvam ASR → server event names ──────────────────────────────────────────

class SarvamASREvent:
    SESSION_BEGIN       = "session.begin"
    SESSION_END         = "session.end"
    TRANSCRIPT_PARTIAL  = "transcript.partial"
    TRANSCRIPT_FINAL    = "transcript.final"
    ERROR               = "error"
    PONG                = "pong"

# ── Sarvam ASR client → server event names ────────────────────────────────────

class SarvamASRCommand:
    SPEECH_START = "speech_start"
    AUDIO_INPUT  = "audio_input"
    SPEECH_END   = "speech_end"
    FLUSH        = "flush"      # watchdog forced-finalization only
    PING         = "ping"
    END          = "end"        # graceful adapter shutdown

# ── Odia language code normalization ─────────────────────────────────────────
# Sarvam realtime ASR reports Odia as "or-IN".
# Varta, Bulbul TTS, and the rest of the stack use "od-IN".
# Normalize at the adapter boundary — nothing else changes.

_SARVAM_TO_VARTA_LANG: dict[str, str] = {
    "or-IN": "od-IN",
}

_VARTA_TO_SARVAM_LANG: dict[str, str] = {
    "od-IN": "or-IN",
}


def sarvam_to_varta_lang(code: str) -> str:
    """Convert a Sarvam-reported language code to the Varta/TTS code."""
    return _SARVAM_TO_VARTA_LANG.get(code, code)


def varta_to_sarvam_lang(code: str) -> str:
    """Convert a Varta language code to what the Sarvam realtime endpoint accepts."""
    return _VARTA_TO_SARVAM_LANG.get(code, code)


# ── Error classification ──────────────────────────────────────────────────────

# These Sarvam error categories should NOT be retried.
# Everything else (network drops, internal errors) may be retried with backoff.
NO_RETRY_ERROR_PREFIXES = (
    "auth",          # authentication failure
    "quota",         # quota exceeded
    "invalid_param", # bad request parameters
    "model_not_found",
)

FATAL_CLOSE_CODES = {
    4000,  # application error (generic fatal)
    4001,  # auth
    4003,  # forbidden / quota
    4008,  # policy violation
}


def is_retryable_error(error_event: dict) -> bool:
    """
    Return True if a Sarvam structured error event warrants a reconnect attempt.
    Return False for fatal, auth, quota, or invalid-parameter errors.

    error_event expected shape:
        { "event": "error", "code": str, "is_fatal": bool, "message": str }
    """
    if error_event.get("is_fatal"):
        return False
    code: str = (error_event.get("code") or "").lower()
    for prefix in NO_RETRY_ERROR_PREFIXES:
        if code.startswith(prefix):
            return False
    return True


def is_retryable_close(close_code: int) -> bool:
    """
    Return True if a WebSocket close code warrants a reconnect attempt.
    """
    return close_code not in FATAL_CLOSE_CODES


# ── Reconnect backoff ─────────────────────────────────────────────────────────

RECONNECT_BASE_DELAY_SEC  = 0.5
RECONNECT_MAX_DELAY_SEC   = 16.0
RECONNECT_MAX_ATTEMPTS    = 5

# ── Keepalive ─────────────────────────────────────────────────────────────────

# Send a Sarvam-protocol ping this many seconds before the idle timeout.
# The idle timeout is undocumented but assumed ~60 s; ping at 45 s.
ASR_PING_INTERVAL_SEC = 45.0

# ── TTS / Bulbul ─────────────────────────────────────────────────────────────

TTS_AUDIO_CODEC    = "linear16"
TTS_SAMPLE_RATE    = 24_000
TTS_MODEL          = "bulbul:v3"

# Languages supported by Bulbul v3
BULBUL_SUPPORTED_LANGS: frozenset[str] = frozenset({
    "hi-IN", "bn-IN", "ta-IN", "te-IN", "gu-IN",
    "kn-IN", "ml-IN", "mr-IN", "pa-IN", "od-IN", "en-IN",
})
