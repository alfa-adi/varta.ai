"""
Thin façade that exposes synchronous repository functions under clean names.
All methods are sync; callers use asyncio.to_thread.
"""
from typing import Optional

from web.storage import conversation_repository as _repo
from web.storage.conversation_repository import SessionNotFoundError  # re-export


class ConversationService:
    def create_session_doc_sync(self, user_id: str, session_id: str, customer_id: str) -> None:
        _repo.create_session_doc_sync(user_id, session_id, customer_id)

    def reserve_turn_sync(self, user_id, session_id, turn_id, speaker_id, fingerprint):
        return _repo.reserve_turn_sync(user_id, session_id, turn_id, speaker_id, fingerprint)

    def update_asr_sync(self, user_id, turn_id, transcript, src_lang):
        _repo.update_asr_sync(user_id, turn_id, transcript, src_lang)

    def complete_turn_sync(self, user_id, turn_id, session_id, translation=None, timing=None):
        _repo.complete_turn_sync(user_id, turn_id, session_id, translation, timing)

    def mark_turn_failed_sync(self, user_id, turn_id, error):
        _repo.mark_turn_failed_sync(user_id, turn_id, error)

    def get_turn_sync(self, user_id, turn_id):
        return _repo.get_turn_sync(user_id, turn_id)


conv_svc = ConversationService()
