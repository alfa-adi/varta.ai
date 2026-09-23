import os
with open('web/server.py', 'r', encoding='utf-8') as f:
    text = f.read()

# We need to swap the decorator and the worker
dec = '@app.websocket("/ws/asr/{session_id}/{speaker}")\n'

# If the decorator is right before the worker, it's wrong.
idx = text.find(dec + 'async def persist_queue_worker')
if idx != -1:
    print("Found exact bad pattern!")
    # the worker ends at `async def ws_asr_live`
    worker_start = idx + len(dec)
    ws_asr_idx = text.find('async def ws_asr_live', worker_start)
    worker_code = text[worker_start:ws_asr_idx]
    
    new_text = text[:idx] + worker_code + dec + text[ws_asr_idx:]
    with open('web/server.py', 'w', encoding='utf-8') as f:
        f.write(new_text)
else:
    # Maybe there are some blank lines?
    dec2 = '@app.websocket("/ws/asr/{session_id}/{speaker}")'
    idx2 = text.find(dec2)
    worker_idx = text.find('async def persist_queue_worker', idx2)
    if worker_idx != -1 and worker_idx < text.find('async def ws_asr_live', idx2):
        print("Found with spaces")
        ws_asr_idx = text.find('async def ws_asr_live', worker_idx)
        worker_code = text[worker_idx:ws_asr_idx]
        new_text = text[:idx2] + worker_code + text[idx2:worker_idx] + text[ws_asr_idx:]
        with open('web/server.py', 'w', encoding='utf-8') as f:
            f.write(new_text)
    else:
        print("Pattern not found!")
