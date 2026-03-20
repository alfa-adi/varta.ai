
"""
adapters/base.py
────────────────
Abstract base classes that define the contract every adapter must honour.
No adapter is allowed to deviate from these signatures.
"""
 
from abc import ABC, abstractmethod
from pipeline.types import ASRInput, ASROutput, NMTInput, NMTOutput, TTSInput, TTSOutput
 
 
class BaseASRAdapter(ABC):
    @abstractmethod
    async def transcribe(self, input: ASRInput) -> ASROutput:
        """Convert audio bytes → transcript + detected language."""
        ...
 
 
class BaseNMTAdapter(ABC):
    @abstractmethod
    async def translate(self, input: NMTInput) -> NMTOutput:
        """Translate text from src_language to tgt_language."""
        ...
 
 
class BaseTTSAdapter(ABC):
    @abstractmethod
    async def synthesise(self, input: TTSInput) -> TTSOutput:
        """Convert text → audio bytes in the target language."""
        ...