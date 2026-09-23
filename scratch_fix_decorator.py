import re

with open("web/server.py", "r", encoding="utf-8") as f:
    text = f.read()

pattern = r'(@app\.websocket\("/ws/asr/\{session_id\}/\{speaker\}"\)\n)(async def persist_queue_worker.*?)(async def ws_asr_live)'
text = re.sub(pattern, r'\2\1\3', text, flags=re.DOTALL)

with open("web/server.py", "w", encoding="utf-8") as f:
    f.write(text)
