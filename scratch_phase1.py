import os
import re

# 1. .gitignore
with open(".gitignore", "r", encoding="utf-8") as f:
    gi = f.read()
gi = re.sub(r"^tests/\n", "", gi, flags=re.MULTILINE)
with open(".gitignore", "w", encoding="utf-8") as f:
    f.write(gi)

# 2. .env.example
with open(".env.example", "r", encoding="utf-8") as f:
    env = f.read()

env = re.sub(r"CONVERSATION_PERSISTENCE_ENABLED=true", "CONVERSATION_PERSISTENCE_ENABLED=false", env)
env = re.sub(r"AUTH_SECRET=supersecretkey", "AUTH_SECRET=", env)

with open(".env.example", "w", encoding="utf-8") as f:
    f.write(env)

# 3. web/protocol.py
with open("web/protocol.py", "r", encoding="utf-8") as f:
    proto = f.read()
proto = proto.replace(
    'MSG_TURN_CANCELLED      = "turn_cancelled"',
    'MSG_TURN_CANCELLED      = "turn_cancelled"\nMSG_TURN_PERSISTED      = "turn_persisted"'
)
with open("web/protocol.py", "w", encoding="utf-8") as f:
    f.write(proto)

# 4. web/storage/conversation_repository.py
with open("web/storage/conversation_repository.py", "r", encoding="utf-8") as f:
    repo = f.read()
repo = re.sub(r'print\(">>> reserve_turn_sync \.\.\."\)\n?', "", repo)
repo = repo.replace(
    "def mark_turn_failed_sync(user_id: str, turn_id: str, error: str) -> None:\n    db = get_db()\n",
    "def mark_turn_failed_sync(user_id: str, turn_id: str, error: str) -> None:\n    db = get_db()\n    if db is None:\n        return\n"
)
with open("web/storage/conversation_repository.py", "w", encoding="utf-8") as f:
    f.write(repo)
