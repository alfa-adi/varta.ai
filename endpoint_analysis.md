# Varta.ai — Open Endpoint Analysis
## Pipeline → Main Database Integration Surface

**Source branch:** `test-latency-tracking` @ `38d40c6`  
**Server file:** [`web/server.py`](file:///c:/Users/admin/OneDrive/Desktop/varta.com(master%20folder)/varta.ai/web/server.py) (1 307 lines)

---

## 1. Endpoint Inventory (What Exists Today)

### REST Endpoints

| # | Method | Path | Auth | Session-aware | DB writes today | Status |
|---|---|---|---|---|---|---|
| R1 | `GET` | `/` | None | No | None | ✅ Stable |
| R2 | `GET` | `/health` | None | No | None | ✅ Stable |
| R3 | `POST` | `/session/create` | None | Creates | None | ✅ Stable |
| R4 | `POST` | `/translate/single` | None | No | `varta_metrics` (fire-and-forget) | ✅ Stable |
| R5 | `POST` | `/translate/dual` | None | Reads | `varta_metrics` (fire-and-forget) | ✅ Stable |
| R6 | `POST` | `/translate/speaker_a` | None | Reads/Writes | `varta_metrics` (fire-and-forget) | ✅ Stable |
| R7 | `POST` | `/translate/speaker_b` | None | Reads/Writes | `varta_metrics` (fire-and-forget) | ✅ Stable |
| R8 | `POST` | `/metrics/browser` | None | No | `varta_metrics.request_latency` (update) | ✅ Stable |

### WebSocket Endpoints

| # | Path | Auth | Per-connection state | DB writes today |
|---|---|---|---|---|
| W1 | `WS /ws/asr/{session_id}/{speaker}` | None | Redis lease + turn lock | `varta_metrics` (via `log_metrics` in `run_turn_pipeline`) — **currently missing in WS path** |

### Gaps (currently no endpoint at all)

| # | Planned endpoint | Why it's missing |
|---|---|---|
| G1 | `POST /session/create_durable` | Defined in plan §9 — not yet in `server.py` |
| G2 | `GET /session/{session_id}` | No read-back of session state exposed to caller |
| G3 | `GET /turns/{session_id}` | No turn history query |
| G4 | `GET /turns/{session_id}/{turn_id}` | No single-turn retrieval |
| G5 | `DELETE /session/{session_id}` | No explicit session deletion / GDPR erasure path |
| G6 | `POST /admin/migrate` | Migration currently CLI-only (`migrations/apply.py`) |

---

## 2. Data Flow Map

```
Browser / Client
       │
       │ REST (multipart audio)       WebSocket (PCM binary frames)
       ▼                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      web/server.py (FastAPI)                    │
│                                                                 │
│  /session/create ──────────────────────────────────────────┐   │
│  /session/create_durable (PLANNED) ────────────────────┐   │   │
│                                                         │   │   │
│  /translate/single ─── SinglePipeline ─────────────────┼───┼─┐ │
│  /translate/dual   ─── DualPipeline   ─────────────────┼───┼─┤ │
│  /translate/speaker_a ─ DualPipeline  ─────────────────┼───┼─┤ │
│  /translate/speaker_b ─ DualPipeline  ─────────────────┼───┼─┤ │
│  /ws/asr/{id}/{spk} ── SarvamLiveASR ─────────────────┼───┼─┤ │
│                                       │               │   │ │ │
└───────────────────────────────────────┼───────────────┼───┼─┼─┘
                                        │               │   │ │
            ┌───────────────────────────┘               │   │ │
            ▼                                           │   │ │
    ┌───────────────┐    adapter layer                  │   │ │
    │ SarvamASR     │◄── ASRInput ──────────────────────┘   │ │
    │ SarvamNMT     │◄── NMTInput                           │ │
    │ SarvamTTS     │◄── TTSInput                           │ │
    └───────┬───────┘                                       │ │
            │ PipelineResult / ASROutput / NMTOutput        │ │
            ▼                                               │ │
    ┌────────────────────────────────────────────────────┐  │ │
    │                  Storage Layer                     │  │ │
    │                                                    │  │ │
    │  Redis ◄──────────── session snapshot ─────────────┘  │ │
    │  (lang_a/b, pending, TTL=2h)                          │ │
    │                                                        │ │
    │  MongoDB: varta_metrics ◄───── log_metrics() ──────────┘ │
    │  ├── request_latency                                      │
    │  └── model_performance                                    │
    │                                                           │
    │  MongoDB: translation-data-cluste ◄── DETACHED ───────────┘
    │  └── translation_logs  (disabled on test-latency-tracking)│
    │                                                           │
    │  MongoDB: varta_conversations ◄── PLANNED (plan §4–§7)    │
    │  ├── translation_sessions                                 │
    │  └── translation_turns                                    │
    └────────────────────────────────────────────────────────┘
```

---

## 3. Open Ends — What Each Endpoint Does NOT Do (Yet)

### R3 — `POST /session/create`

| Gap | Impact | Plan fix |
|---|---|---|
| Returns `session_id` but writes **nothing to MongoDB** | No durable record of session creation | `create_session_doc_sync` called on `/session/create_durable` |
| No caller identity — session is fully anonymous | Cannot attribute turns to a user | `create_durable` route adds JWT + `durable_user_id` |
| No `customer_id` binding | Cannot scope data to a tenant/org | `create_durable` accepts `customer_id: str = Form(...)` |
| Session TTL = 2h Redis only — lost on restart without Redis | Gaps in turn sequence across restarts | Durable sessions persist to Mongo independently |

---

### R4 — `POST /translate/single`

| Gap | Impact | Plan fix |
|---|---|---|
| `log_translation()` is dead code (`return` on line 268) | **Zero archival of any translation** in `translation-data-cluste` | Re-enable or move to `varta_conversations` |
| `session_id` is hardcoded `"single"` — no real session | All single turns share one metrics row | Pass caller-supplied `session_id` via Form |
| No idempotency — retry creates second Mongo document | Billing duplicates possible | Not in scope for single (stateless) |
| No turn_id → metrics join key | Cannot correlate translation to latency row | Add `turn_id` kwarg to `log_metrics` (plan §10) |

---

### R5 — `POST /translate/dual`

| Gap | Impact | Plan fix |
|---|---|---|
| No `X-Request-Id` header → new UUIDs on every call | Retry creates two new turns, doubles billing | Plan §10: stable `turn_id = f"{x_request_id}:a/b"` when header present |
| No `Authorization` header on durable sessions | Anyone with a `session_id` can submit audio | Plan §10: `_authorize_durable_session()` |
| Persist is **fire-and-forget background task** | Mongo failure silently drops turn record | Plan §10 P0-B: `await _persist_dual_both()` inline |
| No idempotency short-circuit | Duplicate request calls Sarvam API again (costs money) | Plan §10: cache-hit path returns existing completed turn |
| Fingerprint pairing was reversed | Wrong audio indexed per speaker | Fixed in plan §10 (fp_b for turn_id_a) |

---

### R6/R7 — `POST /translate/speaker_a` / `speaker_b`

| Gap | Impact | Plan fix |
|---|---|---|
| Buffered speaker_a: `pending_turn_id_a` not written to Redis before returning 200 | Speaker_b arrives before Redis is updated → skips deferred completion | Plan Fix 3: inline `await` |
| Non-buffered: persist is `asyncio.create_task` (fire-and-forget) | Lost on deploy/restart | Plan P0-B: inline `await` for durable sessions |
| No `Authorization` check | Anyone can write to a durable session | Plan P0-A: `_authorize_durable_session()` |
| `log_translation()` is dead code | Zero archival | Same as R4 above |
| Deferred result logging calls dead `log_translation` + dead-code path | Speaker_b deferred result never archived | Re-enable or replace with new collection |

---

### R8 — `POST /metrics/browser`

| Gap | Impact | Plan fix |
|---|---|---|
| Matches by `session_id + timestamp ≥ now-1min` — can hit wrong document if two requests in 1 min | Browser timing merged into wrong turn | Match by `turn_id` once it is in the metrics doc |
| Not authenticated | Any caller can corrupt metrics for any session | Low priority — metrics only |
| No `turn_id` in the update | Cannot join browser → server latency at turn granularity | Add `turn_id: str = Form(...)` and update the filter |

---

### W1 — `WS /ws/asr/{session_id}/{speaker}`

| Gap | Impact | Plan fix |
|---|---|---|
| Fully anonymous — durable session WS is unprotected | Anyone with `session_id` can inject audio into durable session | Plan §11a: `authenticate_websocket` before `accept()` |
| `log_metrics()` is NOT called anywhere in `run_turn_pipeline` | WS turn latency never recorded | Plan §11d: `_on_nmt_complete` callback + persist queue |
| No `turn_persisted` message | Client cannot confirm durable write | Plan §13: `MSG_TURN_PERSISTED` overlay |
| Worker continues after `reserve` fails → emits `turn_persisted: completed` with no doc | False confirmation | Plan §11b P0-C: `_failed_reserve` set |
| Persist worker timeout is `asyncio.wait_for` (cancels task) | Mongo write killed mid-way | Fixed to `asyncio.wait` (non-cancelling) |
| No graceful-shutdown hook | In-flight workers lost on SIGTERM | Plan P0-B: `_persist_task_registry` WeakSet |
| `save_session()` during language update overwrites `is_durable` and `pending_turn_id_*` | Durable identity silently erased each turn | Plan §8: sentinel `_UNSET` merge |

---

## 4. Endpoints Needed for "Bigger Service" Layer

These are the new endpoints that must exist for the persistence pipeline to be complete and queryable by external services (analytics, admin, mobile apps, partner APIs).

### 4A — Identity & Session Management

| Endpoint | Method | Auth | Body | Returns | Priority |
|---|---|---|---|---|---|
| `/session/create_durable` | POST | Bearer JWT | `customer_id`, `lang_a?`, `lang_b?` | `{session_id, is_durable, customer_id}` | **P0** (plan §9) |
| `/session/{session_id}` | GET | Bearer JWT (owner) | — | Session metadata from Redis + Mongo | P1 |
| `/session/{session_id}` | DELETE | Bearer JWT (owner) | — | Soft-delete / GDPR erasure | P2 |

### 4B — Turn Retrieval

| Endpoint | Method | Auth | Query params | Returns | Priority |
|---|---|---|---|---|---|
| `/turns/{session_id}` | GET | Bearer JWT (owner) | `?page=1&limit=20&status=completed` | Paginated list of `translation_turns` docs | P1 |
| `/turns/{session_id}/{turn_id}` | GET | Bearer JWT (owner) | — | Single turn doc (transcript, translation, timing, status) | P1 |
| `/turns/{session_id}/{turn_id}/audio` | GET | Bearer JWT (owner) | — | **Not in scope** — TEXT_ONLY per §17 | — |

### 4C — Admin & Operations

| Endpoint | Method | Auth | Notes | Priority |
|---|---|---|---|---|
| `/admin/migrate` | POST | Internal secret | Triggers `migrations/apply.py` via HTTP instead of CLI | P2 |
| `/admin/health/deep` | GET | Internal secret | Returns Mongo ready, Redis ready, replica-set probe result | P1 |
| `/admin/sessions/{customer_id}` | GET | Internal secret | All sessions for a customer — analytics/support query | P2 |

### 4D — Metrics & Analytics (new)

| Endpoint | Method | Auth | Notes | Priority |
|---|---|---|---|---|
| `/metrics/latency/{session_id}` | GET | Bearer JWT (owner) | Read back `request_latency` docs for a session | P2 |
| `/metrics/summary` | GET | Internal secret | Aggregate P50/P95/P99 latency by model, language pair | P2 |

---

## 5. Database Collection → Endpoint Mapping

```
MongoDB: varta_conversations
├── translation_sessions
│   ├── WRITTEN BY:  POST /session/create_durable
│   ├── QUERIED BY:  GET /session/{session_id}
│   ├── DELETED BY:  DELETE /session/{session_id}
│   └── UPDATED BY:  reserve_turn_sync ($inc next_sequence, $inc turn_count)
│
└── translation_turns
    ├── WRITTEN BY:  reserve_turn_sync (all /translate/* + WS MSG_TURN_START)
    ├── UPDATED BY:  update_asr_sync, complete_turn_sync, mark_turn_failed_sync
    ├── QUERIED BY:  GET /turns/{session_id}, GET /turns/{session_id}/{turn_id}
    └── IDEMPOTENCY: get_turn_sync (used by /translate/dual cache-hit short-circuit)

MongoDB: varta_metrics
├── request_latency
│   ├── WRITTEN BY:  log_metrics() from all REST + (planned) WS turns
│   ├── UPDATED BY:  POST /metrics/browser (browser timing merge)
│   └── QUERIED BY:  GET /metrics/latency/{session_id}  (planned)
│
└── model_performance
    ├── WRITTEN BY:  log_metrics() — 3 docs per request (ASR + NMT + TTS)
    └── QUERIED BY:  GET /metrics/summary  (planned)

MongoDB: translation-data-cluste  ← DETACHED on this branch
└── translation_logs  ← log_translation() currently returns immediately (line 268)
    └── ACTION NEEDED: either re-enable or permanently redirect to varta_conversations

Redis
├── session:{session_id}           ← save_session() / load_session()
├── live-owner:{session_id}:{spk}  ← acquire_connection() lease
├── live-turn:{session_id}         ← acquire_session_turn() cross-speaker lock
└── asr:lang:{spk}:{session_id}   ← _save_detected_language()
```

---

## 6. Wire Contract Gaps (Additive Fields Not Yet Sent)

The following fields exist in the plan or are implied but are **not yet present in the live JSON responses**:

| Response | Field missing | Why needed |
|---|---|---|
| All `/translate/*` | `turn_id` | Client needs it to subscribe to `turn_persisted` WS overlay |
| `/translate/dual` | `cached: true` | Client must suppress duplicate TTS playback on idempotency hit |
| `/translate/dual` | `audio_b64: null` on cache hit | Client must not try to decode null |
| `WS server_ready` | `is_durable: bool` | Client needs to know whether `turn_persisted` messages will arrive |
| `WS audio_end` | No change | Deliberately unchanged (protocol v1, §17 non-goal #3) |
| `WS turn_persisted` | `{type, turn_id, status, error?}` | **New non-terminal overlay** — plan §13 |

---

## 7. Redis Key Gaps

The following Redis keys are written by `server.py` but have **no corresponding cleanup path**:

| Redis key | Written by | TTL set? | Cleanup gap |
|---|---|---|---|
| `session:{id}` | `save_session()` | ✅ 2h | Cleared automatically |
| `asr:{spk}:{id}` (transcript list) | `_push_transcript()` | ✅ 2h | `_pop_final_transcript()` deletes on read |
| `asr:lang:{spk}:{id}` | `_save_detected_language()` | ✅ 2h | No explicit delete — expires only |
| `live-owner:{id}:{spk}` | `acquire_connection()` | ✅ 30s (renewed) | `owner.release()` deletes it |
| `live-turn:{id}` | `acquire_session_turn()` | ✅ 120s | `release_session_turn()` deletes it |
| `pending_turn_id_a` / `pending_turn_id_b` | `save_session()` | Via session TTL | Must be explicitly cleared after B completes A — plan Fix 1 sentinel |

---

## 8. Implementation Priority Matrix

| Item | Endpoint(s) | Blocking? | Commit |
|---|---|---|---|
| `POST /session/create_durable` | G1 | Yes — no durable sessions without this | Commit 2 |
| `_authorize_durable_session()` on R5/R6/R7 | R5, R6, R7 | Yes (P0-A) — unprotected data | Commit 2 |
| Inline `await` persist on R5/R6/R7 | R5, R6, R7 | Yes (P0-B) — silent data loss | Commit 3 |
| WS auth before `accept()` | W1 | Yes (P0-A) | Commit 4 |
| Persist worker `_failed_reserve` | W1 | Yes (P0-C) — false success | Commit 4 |
| `MSG_TURN_PERSISTED` overlay | W1 | No — additive | Commit 4 |
| `GET /turns/{session_id}` | G3 | No — read-only, post-launch | Post-MVP |
| `GET /turns/{session_id}/{turn_id}` | G4 | No — read-only, post-launch | Post-MVP |
| `POST /metrics/browser` turn_id join | R8 | No — metrics quality | Post-MVP |
| `DELETE /session/{session_id}` | G5 | No — compliance/GDPR | Post-MVP |
| Re-enable `translation_logs` | R4–R7 | No — archival only | Post-MVP |
