# Translation and transcription persistence implementation plan

## Purpose and scope

This is an execution-ready implementation plan for the test-latency-tracking branch. It is based on the supplied database design and a review of the current repository.

It is written so that another LLM or engineer can implement the feature without inventing a different data model, breaking the existing latency reports, or replacing the current Redis-based live-session architecture.

This document is documentation only. It must not be interpreted as permission to modify application code, create MongoDB collections, delete databases, or install dependencies during the planning step.

The first implementation slice covers:

- Durable user, customer/patient, session, and ordered-turn records.
- ASR transcript and NMT translation on every completed translation turn.
- Stable session-scoped speaker identity.
- Gemini-ready structured transcription and diarization contracts.
- Repository-level ownership checks, idempotency, sequence allocation, and pagination.
- Gradual REST, WebSocket, and latency-harness integration.

The first implementation slice does not include:

- Deleting the existing MongoDB databases or collections.
- Moving live Redis state into MongoDB.
- Storing raw audio bytes in MongoDB.
- Adding a live Gemini provider dependency.
- Replacing the existing frontend protocol.
- Production healthcare compliance certification.

## Instructions for the implementing LLM

1. Work only on the test-latency-tracking branch unless the user explicitly selects another branch.
2. Read the current versions of every file named in this plan before editing.
3. Preserve existing REST response shapes and WebSocket message types.
4. Keep Redis responsible for live buffers, connection ownership, leases, and expiring runtime state.
5. Keep the existing latency-test collections until their reports are migrated and verified.
6. Do not drop a database or collection from a setup script.
7. Use UTC-aware timestamps.
8. Use Pydantic 2 conventions because the repository specifies pydantic>=2.12.5.
9. Do not add a Gemini SDK in the first persistence slice. Add only the provider-neutral Pydantic contract.
10. Do not make optional metrics failures block an otherwise successful translation.
11. Do not report durable completion when the durable turn write has silently failed.
12. Add tests before changing route wiring.
13. Do not store secrets, raw audio bytes, or unbounded provider payloads in MongoDB.
14. Report changed files, tests run, tests skipped, and any database operation performed.

## Codebase review

### Branch and deployment

The current branch is test-latency-tracking.

The backend entry point is web.server:app. The service is FastAPI running through Uvicorn/Gunicorn.

The current deployment configuration is inconsistent:

- render.yaml starts one Gunicorn worker while Redis is not configured.
- Procfile specifies two workers.
- web/server.py documents Redis as required for two-worker operation but allows an in-memory fallback.

The implementation must not alter worker count or Redis behavior as part of database persistence. That is a separate deployment decision.

### Dependencies

requirements.txt already contains:

- pymongo==4.7.1
- redis==5.0.1
- pydantic>=2.12.5
- FastAPI, HTTPX, requests, and the existing provider/audio packages

No new database driver is required for the first slice.

PyMongo is synchronous while FastAPI routes are asynchronous. The repository must not execute blocking PyMongo calls directly on the event loop. Choose one explicit implementation:

- Keep PyMongo and run repository calls through a bounded thread handoff such as asyncio.to_thread.
- Introduce an approved async MongoDB driver in a separate dependency change.

Do not mix both approaches.

### Current MongoDB behavior

web/server.py currently:

- Reads MONGO_URL.
- Reads MONGO_DB_NAME, defaulting to translation-data-cluste.
- Creates an optional application-log database handle.
- Creates a varta_metrics handle on the same MongoDB client.
- Continues serving when MongoDB is absent or unavailable.
- Has log_translation disabled by an early return on this branch.
- Writes request_latency and model_performance metrics with exceptions swallowed.
- Merges browser timing into the latest request_latency record.

These are optional logs and metrics. They are not a durable conversation history.

The new conversation store must use a separate setting:

~~~text
MONGO_CONVERSATION_DB=varta_test_data
~~~

Keep MONGO_DB_NAME during migration because existing code and documentation still refer to it.

### Current runtime session state

The session helpers in web/server.py persist only:

- lang_a
- lang_b
- pending_transcript_a
- pending_transcript_b

They use Redis with a two-hour TTL when REDIS_URL is set, otherwise a process-local dictionary. Pipeline objects are reconstructed from this state.

This is temporary coordination state. It is not a conversation history and must not be used as the durable source of truth.

### Current translation routes

The current backend exposes:

- POST /session/create
- POST /translate/single
- POST /translate/dual
- POST /translate/speaker_a
- POST /translate/speaker_b
- POST /metrics/browser
- WS /ws/asr/{session_id}/{speaker}

The REST responses contain transcript, translation, languages, audio, and timing.

The single endpoint is stateless and uses the literal single value for its metrics/log session ID.

The speaker endpoints process one side at a time and may return a deferred result for the other speaker.

The dual endpoint returns two directional results in a single request.

The WebSocket flow already has a turn_id. It accepts a client ID or generates a UUID, sends the ID with outbound events, receives final ASR text, then runs NMT/TTS. It currently does not persist a turn.

### Current pipeline contracts

pipeline/types.py defines dataclasses for ASRInput, ASROutput, NMTInput, NMTOutput, TTSInput, TTSOutput, and PipelineResult.

ASROutput already contains transcript, detected language, confidence, latency, model ID, and timing values.

NMTOutput already contains translated text, source/target languages, latency, and model ID.

PipelineResult contains source transcript, translated text, audio bytes, audio format, source/target languages, total latency, and a timing dictionary.

New fields must be optional with defaults so existing adapter constructors and fake adapters remain compatible.

pipeline/single.py run_from_transcript currently streams only TTS chunks after NMT. It does not expose the translated text or NMT metadata to the WebSocket caller. The persistence implementation must add a compatible callback or result channel without changing the existing audio chunk protocol.

### Current latency-test schema

tests/scripts/setup_test_schema.py creates the isolated varta_test_data database with:

- test_sessions: one benchmark run
- conversations: one language-pair conversation inside a run
- pipeline_events: one latency event per tested turn

tests/scripts/run_baseline_test.py writes these documents directly from the Playwright test runner.

A pipeline_events document already contains:

- event_id
- session_id
- conversation_id
- turn_number
- speaker
- source_language and target_language
- audio fixture metadata
- browser timing
- server timing
- translation.asr_transcript
- translation.nmt_translation
- cost, success, error, and timestamp

Existing preflight, summary, visualization, and analytics scripts depend on these names. The new durable schema must not overwrite them in phase 1.

### Current gaps

1. There is no durable user or customer identity in the runtime session API.
2. /session/create stores language hints only.
3. No route creates a durable conversation session.
4. No route writes one durable record per logical turn.
5. Existing speaker values are only a and b.
6. There is no mapping from provider speaker labels to known participants.
7. The live pipeline does not expose NMT result metadata to the persistence layer.
8. Current live ASR output is not a diarized utterance contract.
9. The latency scripts use MONGO_TEST_DB with a default of varta_test_data, but that setting is not currently documented in .env.example.
10. MongoDB is optional, so durable persistence needs an explicit feature flag and failure policy.

## Target architecture

~~~text
Main platform or test client
          |
          | versioned API or compatibility route
          v
Translation/transcription service
          |
          +-- Redis: live state, ASR buffers, connection leases
          |
          +-- MongoDB: users, customers, sessions, durable turns
          |
          +-- Object storage: input/output audio and optional raw provider JSON
          |
          +-- varta_metrics: latency and model-performance telemetry
~~~

MongoDB is the durable source of truth for completed, failed, or cancelled turns that have been accepted by the persistence service.

Redis remains disposable operational state. A Redis TTL or worker restart must not delete completed conversation history.

## Database boundary

Use a service-owned database configured through:

~~~text
MONGO_CONVERSATION_DB=varta_test_data
~~~

Production must set an explicit database name. Do not silently reuse MONGO_DB_NAME for the new conversation store.

Initial collections:

~~~text
users
customers
translation_sessions
translation_turns
~~~

The existing test_sessions, conversations, and pipeline_events collections remain available for benchmark reporting until migrated.

If a completely clean database is desired, use a new explicit name such as varta_conversations_test. Do not drop varta_test_data as part of application startup or schema setup.

## Canonical data model

### users

One document per platform user or tenant.

Required:

~~~json
{
  "user_id": "usr_123",
  "created_at": "UTC datetime",
  "updated_at": "UTC datetime"
}
~~~

Optional:

~~~json
{
  "external_user_id": "platform-user-456",
  "display_name": "Example Clinic",
  "metadata": {}
}
~~~

Rules:

- user_id is immutable and is the ownership boundary.
- Every child document repeats user_id for authorization and query efficiency.
- Repository methods must verify the user/customer/session relationship.
- Do not place authentication secrets or access tokens in metadata.

### customers

One document per patient or customer belonging to a user.

Required:

~~~json
{
  "customer_id": "cus_123",
  "user_id": "usr_123",
  "customer_type": "patient",
  "created_at": "UTC datetime",
  "updated_at": "UTC datetime"
}
~~~

customer_type must be customer or patient.

Optional fields:

- external_customer_id
- display_name
- bounded non-sensitive metadata

External customer IDs are unique only within a user unless the main platform guarantees global uniqueness.

### translation_sessions

One document represents one conversation instance for exactly one user and one customer.

Required:

~~~json
{
  "session_id": "ses_123",
  "user_id": "usr_123",
  "customer_id": "cus_123",
  "customer_snapshot": {
    "customer_id": "cus_123",
    "customer_type": "patient",
    "display_name": "Name at session time"
  },
  "status": "active",
  "started_at": "UTC datetime",
  "created_at": "UTC datetime",
  "updated_at": "UTC datetime"
}
~~~

Allowed statuses:

~~~text
active | completed | failed | cancelled
~~~

Optional or derived fields:

- participants
- source_language and target_language defaults
- turn_count as a reconciled summary
- last_sequence as the atomic allocator state
- ended_at
- bounded metadata

Participant shape:

~~~json
{
  "speaker_id": "speaker_00",
  "diarization_label": "SPEAKER_00",
  "participant_id": "cus_123",
  "role": "patient",
  "display_name": "Patient"
}
~~~

Allowed roles:

~~~text
user | customer | patient | agent | unknown
~~~

participants maps provider labels to known or unknown people for the session. Unknown speakers are valid.

The customer snapshot is intentionally denormalized. It preserves historical session meaning if the customer is renamed or reclassified later.

### translation_turns

One document represents one logical conversational turn.

Required:

~~~json
{
  "turn_id": "turn_123",
  "session_id": "ses_123",
  "user_id": "usr_123",
  "customer_id": "cus_123",
  "sequence": 2,
  "speaker_id": "speaker_00",
  "asr": {
    "text": "source text"
  },
  "translation": {
    "text": "target text"
  },
  "status": "completed",
  "created_at": "UTC datetime",
  "updated_at": "UTC datetime"
}
~~~

Required semantic rules:

- sequence is 1-based, allocated once, and never changed.
- (session_id, sequence) is unique.
- turn_id is unique and is the idempotency key.
- speaker_id is stable only within a session.
- Two adjacent turns may have the same speaker_id.
- asr.text is the source transcript.
- translation.text is the NMT output.
- Failed or cancelled turns remain in the timeline.

ASR subdocument:

~~~json
{
  "text": "Source-language text",
  "language": "hi-IN",
  "provider": "sarvam",
  "model": "sarvam/saaras-v3",
  "confidence": 0.96,
  "utterances": [],
  "raw_response_ref": null
}
~~~

For Gemini diarization, utterances contains validated provider utterances with utterance_id, speaker_id, optional speaker label, text, start/end milliseconds, and confidence.

Translation subdocument:

~~~json
{
  "text": "Target-language text",
  "language": "en-IN",
  "provider": "sarvam",
  "model": "sarvam/sarvam-translate",
  "confidence": null,
  "raw_response_ref": null
}
~~~

Other turn fields:

- speaker_label
- speaker_role
- source_language
- target_language
- audio object keys, hash, format, duration, and size
- timing for ASR, NMT, TTS, persistence, and total latency
- started_at and ended_at
- safe error code/message when failed
- bounded metadata

Allowed turn statuses:

~~~text
received | transcribed | translated | completed | failed | cancelled
~~~

## Turn ordering and idempotency

### Atomic sequence allocation

Use an atomic update on the parent session:

~~~text
find_one_and_update(
  {session_id: S, user_id: U, status: active},
  {$inc: {last_sequence: 1}, $set: {updated_at: now}},
  return_document=AFTER
)
~~~

The returned last_sequence becomes the immutable sequence.

Sequence gaps are acceptable if a worker crashes after allocation and before turn insert. Correct chronological ordering is more important than gapless numbering.

If gapless numbers are later required, introduce a transaction-backed allocator. Never renumber already persisted turns.

### Idempotent turn writes

The repository must accept a caller-provided turn_id when available.

Retry behavior:

1. Look up turn_id with user_id.
2. If identity matches, return the existing turn.
3. If the same turn_id is reused for a different session or speaker, reject it as a conflict.
4. Never allocate a second sequence for a retry of an existing turn.

Unique indexes are the last line of defense, not the primary idempotency mechanism.

### Same-speaker behavior

Never infer a turn boundary from a speaker change. Each accepted turn start, accepted REST turn, or provider utterance selected as a logical turn receives its own durable record.

Example:

~~~text
sequence  speaker_id  asr.text
1         speaker_00  First sentence
2         speaker_00  Follow-up sentence
3         speaker_01  Response
~~~

## Pydantic contract plan

Create web/schemas/conversation.py with Pydantic 2 models for API and repository boundaries.

Required models:

~~~text
UserRecord
CustomerRecord
CustomerSnapshot
SessionParticipant
TranslationSessionRecord
DiarizedUtterance
AsrRecord
TranslationRecord
AudioReference
TurnTiming
TranslationTurnRecord
CreateSessionRequest
CreateSessionResponse
TurnPersistenceStatus
~~~

Use ConfigDict(extra="forbid", populate_by_name=True) unless a model explicitly needs provider extension fields.

Validation requirements:

- IDs are non-empty strings.
- sequence is at least 1.
- confidence is between 0 and 1 when present.
- start/end timestamps cannot be reversed.
- user_id and customer_id are required on sessions and turns.
- asr.text and translation.text may be empty only before completion.
- participant speaker_id values are unique within a session.
- unknown speakers are accepted and retain their provider label.
- status transitions are validated by the service layer, not only by Pydantic.

Do not use Pydantic models as the only MongoDB validator. Keep MongoDB JSON Schema validators in the setup/migration layer and test both against the same fixtures.

## Gemini structured-output plan

Create web/schemas/gemini_transcription.py with provider-facing models:

~~~python
class GeminiDiarizedUtterance(BaseModel):
    utterance_id: str
    speaker_id: str
    speaker_label: str | None = None
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    confidence: float | None = None


class GeminiTranscriptionResponse(BaseModel):
    language: str | None = None
    full_transcript: str
    utterances: list[GeminiDiarizedUtterance]
    audio_duration_ms: int | None = None
~~~

The contract must be usable through model_json_schema() or the selected Gemini SDK's native Pydantic response-schema support.

The Gemini instruction must require:

- chronological utterances;
- stable speaker_id values throughout one audio item;
- one object per logical utterance/turn;
- no merging solely because adjacent utterances have the same speaker;
- timestamps when available;
- confidence only when supported;
- no invented words, speakers, timestamps, or languages;
- exact JSON matching the response model.

Validate Gemini output before mapping it to the normalized persistence model. Do not expose provider-specific JSON as the public platform API contract.

## Proposed file changes

### New files

#### web/schemas/__init__.py

Export public Pydantic contracts without importing MongoDB or provider clients.

#### web/schemas/conversation.py

Implement the normalized user/customer/session/turn models and field validation.

#### web/schemas/gemini_transcription.py

Implement Gemini response models and helpers for structured response schema and prompt requirements. Keep it independent of the Gemini SDK.

#### web/storage/mongo.py

Implement:

- environment-driven database configuration;
- one process-wide MongoClient per application process;
- connection ping and health state;
- collection handle access;
- explicit close hook for tests/shutdown;
- credential-safe startup diagnostics.

The module must not make the entire legacy FastAPI app fail when optional MongoDB is absent. It must expose clear persistence state so durable APIs can reject or report unavailable persistence.

#### web/storage/conversation_repository.py

Own every MongoDB operation for users, customers, sessions, and turns. Route handlers must not call MongoDB collections directly.

Required behavior:

~~~python
ensure_user(user)
ensure_customer(user_id, customer)
create_session(session)
get_session(user_id, session_id)
reserve_turn(...)
get_turn(user_id, turn_id)
update_asr(...)
update_translation(...)
mark_turn_failed(...)
complete_session(...)
list_turns(user_id, session_id, cursor, limit)
~~~

Exact method names may vary only if ownership, idempotency, ordering, and pagination behavior are preserved.

#### web/services/conversation_service.py

Implement domain orchestration:

- verify user/customer ownership;
- create immutable customer snapshots;
- reserve sequences;
- map ASR/NMT/provider output;
- enforce status transitions;
- handle idempotent retries;
- update session summaries;
- return structured persistence errors.

This layer must not know whether the caller is REST, WebSocket, or the test harness.

#### tests/scripts/setup_conversation_schema.py

Create an idempotent setup script for the four new collections.

It must:

- read MONGO_URL and MONGO_CONVERSATION_DB;
- ping MongoDB before changes;
- create strict validators with validationAction=error;
- create named indexes idempotently;
- print results without credentials;
- never drop databases or collections.

#### tests/unit/test_conversation_schemas.py

Test Pydantic validation, same-speaker turns, timestamp/confidence rules, unknown speakers, and Gemini fixtures.

#### tests/unit/test_conversation_service.py

Test ownership, legal status transitions, idempotency, sequence allocation, and failure handling with a fake repository.

#### tests/integration/test_conversation_repository.py

Add opt-in MongoDB tests. Skip cleanly when MONGO_URL is absent. Use generated IDs and a dedicated test database or cleanup scope.

### Existing files to modify

#### pipeline/types.py

Add optional normalized metadata while preserving current constructors:

- ASRUtterance dataclass;
- optional utterances on ASROutput;
- optional provider metadata or raw-response reference;
- optional structured NMT/TTS metadata on PipelineResult if needed by the mapper.

Do not add raw audio bytes to persisted metadata.

#### pipeline/single.py

Expose NMT result metadata to the live WebSocket persistence path. The current run_from_transcript only yields TTS chunks.

Add a compatible callback or result channel invoked after NMT completes and before TTS streaming. It must expose translated text, source/target languages, model ID, and NMT timing.

Keep existing TTS chunk format and send order unchanged.

#### adapter/sarvam_asr.py

Populate optional normalized metadata for the existing non-Gemini ASR path. Keep current live WebSocket frames compatible. Sarvam output may have an empty diarization list; never invent speaker segments.

#### web/server.py

Use the service layer at explicit lifecycle points.

REST:

- Extend session creation through versioned fields or backward-compatible optional user/customer fields.
- Create/update a durable session when identity is available.
- Persist successful speaker A/B outputs as separate turns.
- Persist deferred output exactly once.
- Persist both dual results as separate turns with explicit order metadata.
- Leave stateless /translate/single unpersisted unless a durable session is supplied.

WebSocket:

- At accepted turn_start, reserve/create a received turn.
- On final ASR, update asr and set transcribed.
- When NMT metadata is available, update translation and set translated.
- After TTS completion, set completed.
- On provider failure, timeout, or disconnect, set failed or cancelled with safe error details.
- Reuse the existing turn_id for correlation and idempotency.

Do not add raw collection calls to route handlers.

#### .env.example

Document:

~~~text
MONGO_URL=
MONGO_CONVERSATION_DB=varta_test_data
MONGO_METRICS_DB=varta_metrics
CONVERSATION_PERSISTENCE_ENABLED=false
~~~

Keep MONGO_DB_NAME documented during migration. Never commit secrets.

#### tests/scripts/preflight_check.py

Add a separate optional check for the four new collections. Do not remove the existing latency collection checks.

#### tests/scripts/run_baseline_test.py

Keep writing the current pipeline_events documents. Add dual-write behind an explicit flag such as PERSIST_CONVERSATIONS=true.

The dual-write mode must:

- create or ensure one test user;
- apply an explicit test customer policy;
- create a translation_sessions document;
- write one translation_turns document per logical test turn;
- preserve the current benchmark documents and reports;
- use the shared repository/service mapper rather than duplicate document construction.

#### docs/backend_connectors.md

Document new versioned session and conversation-read endpoints without changing the existing frontend connector contract until migration.

## API rollout

### Phase 1: internal contracts

No public API change.

Add schemas, repository, validators, service, and fake-repository tests. Keep durable persistence disabled by default.

### Phase 2: versioned ownership-aware API

Add:

~~~text
POST /v1/users/{user_id}/customers
POST /v1/users/{user_id}/customers/{customer_id}/sessions
GET  /v1/users/{user_id}/sessions/{session_id}
GET  /v1/users/{user_id}/sessions/{session_id}/turns
~~~

Session creation requires customer identity and accepts optional language configuration.

Reads require authenticated tenant context or an equivalent user-scoped path. A session ID alone must never retrieve patient/customer data.

Use cursor pagination. The first implementation may use sequence greater than cursor with ascending order. Do not return an unbounded full conversation by default.

### Phase 3: REST dual-write

Enable persistence for speaker A, speaker B, and dual routes when durable identity is present. Keep old response JSON unchanged.

Use one turn per logical result. A deferred result must use a stable ID and cannot be inserted twice.

### Phase 4: WebSocket durable turns

Persist final turns from the live flow. Partial frames remain transient unless a separate partial-transcript audit requirement is approved.

### Phase 5: Gemini adapter

Add Gemini only after the normalized contract and repository tests are stable. Map Gemini utterances to the same asr, speaker, and translation fields. Do not create a Gemini-specific database schema.

## MongoDB validators and indexes

Create strict validators for all four new collections.

Required indexes:

~~~text
users:
  unique(user_id)

customers:
  unique(customer_id)
  (user_id, created_at desc)
  partial unique(user_id, external_customer_id)

translation_sessions:
  unique(session_id)
  (user_id, customer_id, started_at desc)
  (user_id, status, updated_at desc)

translation_turns:
  unique(turn_id)
  unique(session_id, sequence)
  (session_id, sequence)
  (user_id, customer_id, created_at desc)
~~~

Use explicit index names and idempotent creation. The external customer ID index must exclude null or missing values.

Do not create indexes for every nested timing field. Analytics belongs in varta_metrics or a separate analytics pipeline.

## Persistence failure policy

Distinguish:

1. MongoDB disabled: local/demo behavior remains available.
2. MongoDB unavailable before a durable operation: versioned durable APIs return a controlled persistence-unavailable response.
3. MongoDB fails after a turn is accepted: retain a retryable pending state or outbox record; never claim durable completion silently.
4. Optional metrics write fails: record the failure and continue the translation response, as the current branch does.

Feature flag:

~~~text
CONVERSATION_PERSISTENCE_ENABLED=false
~~~

When false, no durable conversation writes occur.

When true, startup should ping the conversation database before accepting versioned durable traffic. Legacy translation routes need not become unavailable solely because the feature is off.

The implementation must choose one durable acknowledgement rule:

- Wait for the durable turn write before reporting completed.
- Or report persistence_status=pending and guarantee a retryable outbox path.

Do not combine an asynchronous best-effort write with a completed acknowledgement.

## Security and privacy

Before production use:

- require authenticated tenant context;
- enforce user_id ownership on every repository operation;
- do not log transcript, translation, raw provider JSON, or audio URLs in ordinary logs;
- encrypt MongoDB and object storage at rest and in transit;
- use short-lived signed URLs for audio;
- define retention and deletion rules;
- redact provider errors;
- add audit logging if product requirements require it;
- test cross-user access denial;
- treat current unauthenticated PoC routes as non-production.

Patient data must not be copied into arbitrary metadata or test output.

## Test plan

### Unit tests

Cover:

- valid user/customer/session/turn construction;
- customer and patient types;
- known and unknown speakers;
- two consecutive same-speaker turns;
- unique participant speaker IDs;
- timestamp and confidence validation;
- required ASR/NMT fields;
- legal status transitions;
- malformed Gemini output;
- sequence gaps after simulated failure;
- idempotent retry with same turn_id;
- conflict when a turn_id is reused for a different session.

### Repository tests

Cover:

- ownership mismatch;
- customer/session mismatch;
- concurrent sequence reservation;
- duplicate turn insertion;
- received to transcribed to translated to completed;
- failed/cancelled retention;
- sequence cursor pagination;
- cross-user read denial;
- MongoDB validator rejection;
- unique-index rejection.

Use a fake repository for service tests and an opt-in real MongoDB integration suite for actual validators and indexes.

### Endpoint tests

Verify:

- existing response shapes remain unchanged;
- versioned session creation returns stable IDs;
- one durable turn per logical REST result;
- deferred results are not double-written;
- dual requests produce two deterministic turns;
- WebSocket turn_id is preserved;
- final ASR is persisted;
- NMT failure leaves an inspectable failed turn;
- persistence disabled preserves legacy behavior.

### Existing regression suite

Run at minimum:

~~~text
pytest tests/test_ws_connection_lifecycle.py
pytest tests/test_ws_speaker_lock.py
pytest tests/test_ws_asr_adapter.py
pytest tests/test_legacy_http_adapters.py
pytest tests/test_legacy_http_pipeline.py
pytest tests/test_buffering_api.py
~~~

Run baseline latency tests only with explicit credentials and a dedicated test database. Confirm old reports are unchanged during dual-write.

### Load and failure tests

Measure:

- persistence latency added to REST turns;
- persistence latency added to WebSocket finalization;
- concurrent sessions;
- rejected concurrent turn claims within one session;
- MongoDB outage during ASR/NMT/finalization;
- retry after process restart;
- large-session cursor pagination;
- hot-session write behavior;
- tenant-heavy query behavior.

Do not call the feature production-ready until a MongoDB outage cannot silently lose a turn reported as completed.

## Rollout and migration

### Step 0: protect existing data

- Confirm active database names from deployment settings, not only .env.example.
- Export or snapshot existing varta_test_data, varta_metrics, and legacy log databases.
- Confirm no production consumer depends on collections planned for retirement.
- Do not add drop commands to setup scripts.

### Step 1: create the isolated schema

Run setup_conversation_schema.py against the selected test database.

Verify:

- four collections exist;
- validators are strict;
- named indexes exist;
- synthetic user, customer, session, and two same-speaker turns reconstruct correctly.

### Step 2: validate repository without route wiring

Run unit, fake-repository, and opt-in integration tests. Confirm every child document carries the parent user_id and customer_id.

### Step 3: dual-write the latency harness

Run the baseline with explicit dual-write enabled.

Verify:

- pipeline_events counts are unchanged;
- existing latency reports remain valid;
- each successful event maps to one durable turn;
- same-speaker turns retain separate sequences;
- failures remain visible.

### Step 4: staging application rollout

Enable persistence for one worker first. Validate REST and WebSocket flows. Then repeat with the intended multi-worker deployment and Redis configured.

### Step 5: production-readiness review

Complete authentication, tenant checks, retention, object storage, audit, backup/restore, and load testing.

### Step 6: retirement

Keep old collections read-only until every report and dashboard uses the replacement source. Retire old collections only through an approved, backed-up data-lifecycle change.

## Rollback

1. Set CONVERSATION_PERSISTENCE_ENABLED=false.
2. Stop versioned durable writes.
3. Keep legacy translation and latency metrics available.
4. Preserve new collections for diagnosis.
5. Do not delete the new database during incident response.
6. If code is rolled back, leave the new database intact for later resumption.

## Acceptance criteria

The implementation is complete only when:

- One user can own multiple customers/patients.
- One customer can have multiple sessions.
- One session can have many independent turns.
- Every turn identifies its speaker.
- Two continuous turns by the same speaker remain two records.
- Every completed translation turn contains final ASR and final NMT text.
- Failed/cancelled turns remain inspectable.
- Gemini diarization output validates and maps to the same schema.
- Replay returns exact stored order with cursor pagination.
- Duplicate retries do not create duplicate turns.
- Cross-user reads/writes are rejected.
- Redis remains responsible for transient coordination.
- Existing latency collections and reports remain functional.
- MongoDB outage behavior is explicit and tested.
- No audio bytes or secrets are stored in MongoDB.
- Full regression passes with persistence disabled and opt-in tests enabled.

## Checklist for the implementing LLM

Before editing:

- [ ] Confirm branch is test-latency-tracking.
- [ ] Read current versions of all planned files.
- [ ] Check for overlapping user changes.
- [ ] Confirm database name and backup status.

While implementing:

- [ ] Add Pydantic contracts first.
- [ ] Add repository and fake-repository tests.
- [ ] Add validators and indexes without destructive operations.
- [ ] Add ownership and idempotency checks.
- [ ] Add versioned API.
- [ ] Add REST persistence.
- [ ] Add WebSocket finalization persistence.
- [ ] Add Gemini contract tests without a network call.
- [ ] Add latency-harness dual-write behind a flag.
- [ ] Preserve existing REST/WebSocket behavior.

Before reporting completion:

- [ ] Run syntax, formatting, and type checks available in the repository.
- [ ] Run focused new tests.
- [ ] Run existing adapter, buffering, WebSocket, and legacy HTTP tests.
- [ ] Run git diff --check.
- [ ] Report tests skipped due to credentials, MongoDB, Redis, or provider access.
- [ ] Report exactly which files changed.
- [ ] Confirm no database was dropped.

