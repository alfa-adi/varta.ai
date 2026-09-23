import os

# 1. pipeline/single.py
with open("pipeline/single.py", "r", encoding="utf-8") as f:
    single = f.read()

single = single.replace(
    'async def run_from_transcript(self, transcript: str, src_language: str, voice_gender="female") -> AsyncIterator[bytes]:',
    'async def run_from_transcript(self, transcript: str, src_language: str, voice_gender="female", on_nmt_complete=None) -> AsyncIterator[bytes]:'
)
single = single.replace(
    'self.last_nmt_output = nmt_output\n\n        # ── Step 2: TTS streaming — yield chunks as they arrive ───────',
    'self.last_nmt_output = nmt_output\n        if on_nmt_complete is not None:\n            on_nmt_complete(nmt_output)\n\n        # ── Step 2: TTS streaming — yield chunks as they arrive ───────'
)

with open("pipeline/single.py", "w", encoding="utf-8") as f:
    f.write(single)

# 2. frontend/src/wsClient.js
with open("frontend/src/wsClient.js", "r", encoding="utf-8") as f:
    wsjs = f.read()

wsjs = wsjs.replace(
    '  async open() {',
    '  async open(token = null) {'
)
wsjs = wsjs.replace(
    '    this._ws = new WebSocket(url);',
    '    const protocols = token ? [`token-${token}`] : [];\n    this._ws = new WebSocket(url, protocols);'
)

with open("frontend/src/wsClient.js", "w", encoding="utf-8") as f:
    f.write(wsjs)
