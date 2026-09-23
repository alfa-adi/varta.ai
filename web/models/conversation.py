from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class TurnStatus(str, Enum):
    received    = "received"
    transcribed = "transcribed"
    completed   = "completed"
    failed      = "failed"
    cancelled   = "cancelled"

# Note: "translated" is intentionally omitted.  The WS path goes:
#   received → transcribed → completed  (NMT result stored inline on complete_turn)
# There is no intermediate "translated" state written to the DB.


class TranslationTurn(BaseModel):
    turn_id:           str
    user_id:           str
    session_id:        str
    speaker_id:        str
    sequence:          int
    status:            TurnStatus
    audio_fingerprint: Optional[str] = None   # None on WS turns (stream has no fingerprint)
    created_at:        datetime
    updated_at:        datetime
    transcript:        Optional[str]  = None
    source_language:   Optional[str]  = None
    translation:       Optional[dict] = None   # {"translated_text": "...", "tgt_language": "..."}
    timing:            Optional[dict] = None
    error_message:     Optional[str]  = None
    metadata:          dict[str, Any] = Field(default_factory=dict)
