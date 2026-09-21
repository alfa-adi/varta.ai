# Varta.ai — Pipeline → SQLite → Frontend Integration Bridge

> **Scope:** This document is internal planning only. It lives in the Antigravity brain,
> not in the project root. Do not copy to `varta.ai/`.
>
> **Sources:**
> - [`frontendschema.md`](file:///c:/Users/admin/OneDrive/Desktop/varta.com(master%20folder)/varta.ai/frontendschema.md) — what the frontend needs (getpoints)
> - [`endpoint_analysis.md`](file:///C:/Users/admin/.gemini/antigravity-ide/brain/4f4e2d75-c1ef-4299-9ed6-5d414bdb6205/endpoint_analysis.md) — what the pipeline produces (open ends)
> - `web/server.py` (branch `test-latency-tracking` @ `38d40c6`) — live code

---

## Overview: The Three-Layer Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Varta.ai Pipeline (exists today)                          │
│  FastAPI + Sarvam ASR/NMT/TTS + Redis + MongoDB (varta_metrics)     │
│  Produces: translation_turns, translation_sessions (MongoDB planned)│
└──────────────────────────────┬──────────────────────────────────────┘
                               │  write path (turn_persisted event /
                               │  POST /api/sessions on session close)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: Clinical SQLite Database (future — this document)         │
│  8 tables: doctors, clinics, patients, clinical_sessions,           │
│  session_vitals, session_tags, appointments, prescriptions          │
│  Keyed by: doctor_id = durable_user_id from Varta JWT               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  read path (getpoints)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Frontend React/RN App (mockData.ts → live API)            │
│  11 getpoints across 5 domains:                                     │
│  Patients, Sessions, Appointments, Doctor Profile, Dashboard Stats  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part A: SQLite Schema

### Design principles
- **SQLite** for local/edge deployments (clinic tablet, offline-first). Migrate to PostgreSQL unchanged — all queries are ANSI SQL.
- **`doctor_id`** is the foreign key anchor for every clinical entity. It equals `durable_user_id` from the Varta JWT, so no new identity system is needed.
- **`varta_session_id`** and **`varta_turn_id`** are preserved in `clinical_sessions` and `session_turns` as foreign references back to MongoDB `varta_conversations`, making the two databases joinable.
- All text enums are stored as `TEXT` with `CHECK` constraints for SQLite compatibility.

---

### Table 1 — `doctors`

The identity anchor. Created once per practitioner.

```sql
CREATE TABLE doctors (
    doctor_id       TEXT PRIMARY KEY,     -- = durable_user_id from Varta JWT (UUID)
    name            TEXT NOT NULL,
    first_name      TEXT NOT NULL,
    qualification   TEXT,                 -- "MBBS", "MD", etc.
    reg_number      TEXT UNIQUE,          -- Medical council registration
    specialisation  TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
```

**Pipeline open end bridged:** `durable_user_id` from `POST /session/create_durable` JWT  
**Frontend getpoint served:** `GET /api/doctor/profile` (→ `DoctorProfile.name`, `.qualification`, etc.)

---

### Table 2 — `clinics`

```sql
CREATE TABLE clinics (
    clinic_id   TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    doctor_id   TEXT NOT NULL REFERENCES doctors(doctor_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    address     TEXT,
    phone       TEXT,
    website     TEXT,
    gstin       TEXT,
    is_primary  INTEGER NOT NULL DEFAULT 1 CHECK (is_primary IN (0,1)),
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_clinics_doctor ON clinics(doctor_id);
```

**Frontend getpoint served:** `GET /api/doctor/profile` → `DoctorProfile.clinic`

---

### Table 3 — `patients`

```sql
CREATE TABLE patients (
    patient_id          TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    doctor_id           TEXT NOT NULL REFERENCES doctors(doctor_id) ON DELETE CASCADE,
    abha_id             TEXT,            -- Ayushman Bharat Health Account ID
    name                TEXT NOT NULL,
    age                 INTEGER,
    gender              TEXT CHECK (gender IN ('male','female','other')),
    blood_group         TEXT,
    phone               TEXT,
    photo_url           TEXT,
    urgency             TEXT NOT NULL DEFAULT 'routine'
                            CHECK (urgency IN ('routine','attention','urgent')),
    living_summary      TEXT,           -- freetext paragraph
    emergency_contact   TEXT,           -- JSON: {name, relation, phone}
    last_visit_date     TEXT,           -- ISO-8601 date, updated by trigger
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_patients_doctor   ON patients(doctor_id);
CREATE INDEX idx_patients_abha     ON patients(abha_id);
CREATE INDEX idx_patients_name_fts ON patients(name COLLATE NOCASE);

-- Trigger: keep last_visit_date in sync whenever a session is inserted
CREATE TRIGGER trg_patient_last_visit
AFTER INSERT ON clinical_sessions
BEGIN
    UPDATE patients
    SET last_visit_date = NEW.session_date,
        updated_at      = datetime('now')
    WHERE patient_id = NEW.patient_id
      AND (last_visit_date IS NULL OR last_visit_date < NEW.session_date);
END;
```

**Pipeline open end bridged:** none (doctor enters patient data manually or via ABHA lookup)  
**Frontend getpoints served:**
- `GET /api/patients?search=&limit=` → `SELECT * FROM patients WHERE doctor_id=? AND name LIKE ?`
- `GET /api/patients/{id}` → single row + JOINs for tags/conditions
- `POST /api/patients` → `INSERT INTO patients ...`

---

### Table 4 — `patient_tags`

Implements `allergies` and `chronicConditions` as typed rows (replaces TagItem arrays).

```sql
CREATE TABLE patient_tags (
    tag_id      TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    patient_id  TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    tag_type    TEXT NOT NULL CHECK (tag_type IN ('allergy','chronic_condition')),
    label       TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_patient_tags_patient ON patient_tags(patient_id, tag_type);
```

**Frontend getpoint served:** `GET /api/patients/{id}` → `.allergies[]`, `.chronicConditions[]`

---

### Table 5 — `clinical_sessions`

The central join table. **One row = one complete doctor-patient encounter.**  
Receives data from **both** the Varta pipeline (via `varta_session_id`) and the doctor's annotation (via `POST /api/sessions`).

```sql
CREATE TABLE clinical_sessions (
    session_id          TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    doctor_id           TEXT NOT NULL REFERENCES doctors(doctor_id) ON DELETE CASCADE,
    patient_id          TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,

    -- Varta pipeline linkage (open ends bridged here)
    varta_session_id    TEXT UNIQUE,     -- = durable_session_id from Varta Redis
    varta_mode          TEXT CHECK (varta_mode IN ('transcribe','translate','none')),

    -- Clinical identity
    session_number      INTEGER NOT NULL, -- auto-increment per patient (see trigger below)
    session_date        TEXT NOT NULL,    -- ISO-8601 date "YYYY-MM-DD"
    session_time        TEXT,             -- "HH:MM"
    duration_minutes    INTEGER,          -- set on session close

    -- Encounter metadata
    session_type        TEXT DEFAULT 'consultation'
                            CHECK (session_type IN ('consultation','follow_up','emergency','teleconsult')),
    chief_complaint     TEXT,
    doctor_notes        TEXT,             -- SOAP / free-form notes added post-session
    follow_up_date      TEXT,             -- ISO-8601 date

    -- Status
    status              TEXT NOT NULL DEFAULT 'open'
                            CHECK (status IN ('open','closed','pending_rx')),

    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_sessions_doctor  ON clinical_sessions(doctor_id, session_date DESC);
CREATE INDEX idx_sessions_patient ON clinical_sessions(patient_id, session_date DESC);
CREATE INDEX idx_sessions_varta   ON clinical_sessions(varta_session_id);

-- Auto-increment session_number per patient
CREATE TRIGGER trg_session_number
BEFORE INSERT ON clinical_sessions
BEGIN
    SELECT RAISE(ABORT, 'doctor_id required') WHERE NEW.doctor_id IS NULL;
    UPDATE clinical_sessions
    SET session_number = (
        SELECT COALESCE(MAX(session_number), 0) + 1
        FROM clinical_sessions
        WHERE patient_id = NEW.patient_id
    )
    WHERE session_id = NEW.session_id;
END;
```

**Pipeline open ends bridged:**

| Varta pipeline field | SQLite column |
|---|---|
| `durable_session_id` (Redis `is_durable`) | `varta_session_id` |
| `durable_user_id` | `doctor_id` |
| `session_mode` inferred from `varta_mode` | `varta_mode` |
| Turn sequence count from `translation_sessions.turn_count` | used to compute `duration_minutes` |

**Frontend getpoints served:**
- `GET /api/patients/{id}/sessions` → `SELECT ... FROM clinical_sessions WHERE patient_id=?`
- `GET /api/sessions/{id}` → full row + JOINs
- `POST /api/sessions` → `INSERT INTO clinical_sessions ...`

---

### Table 6 — `session_turns`

Stores the turn-by-turn transcript assembled from Varta `translation_turns`. This is what becomes the `transcript[]` array in the frontend `Session` model.

```sql
CREATE TABLE session_turns (
    turn_id             TEXT PRIMARY KEY,  -- = translation_turns.turn_id from MongoDB
    session_id          TEXT NOT NULL REFERENCES clinical_sessions(session_id) ON DELETE CASCADE,
    sequence            INTEGER NOT NULL,
    speaker             TEXT NOT NULL CHECK (speaker IN ('doctor','patient','a','b')),
    src_language        TEXT,              -- BCP-47 e.g. "hi-IN"
    tgt_language        TEXT,              -- BCP-47 e.g. "en-IN"
    transcript          TEXT,             -- source language text (from update_asr_sync)
    translation         TEXT,             -- translated text (from complete_turn_sync)
    timestamp_ms        INTEGER,          -- wall-clock ms from turn created_at
    latency_ms          INTEGER,          -- total pipeline latency from timing doc
    status              TEXT NOT NULL DEFAULT 'completed'
                            CHECK (status IN ('received','transcribed','completed','failed','persist_failed'))
);
CREATE INDEX idx_turns_session ON session_turns(session_id, sequence ASC);
```

**Pipeline open ends bridged:**

| MongoDB `translation_turns` field | SQLite column |
|---|---|
| `turn_id` | `turn_id` (PK) |
| `sequence` | `sequence` |
| `speaker_id` | `speaker` |
| `source_language` | `src_language` |
| `translation.tgt_language` | `tgt_language` |
| `transcript` | `transcript` |
| `translation.translated_text` | `translation` |
| `created_at` (epoch ms) | `timestamp_ms` |
| `timing.nmt_latency_ms` | `latency_ms` |
| `status` | `status` |

**Frontend getpoint served:** `GET /api/sessions/{id}` → `Session.transcript[]`  
Each row maps to one `TranscriptLine` object:
```json
{
  "speaker": "doctor",
  "text": "आपको कब से बुखार है?",
  "translation": "Since when do you have fever?",
  "language_code": "hi-IN",
  "timestamp": 1234567890123
}
```

---

### Table 7 — `session_vitals`

```sql
CREATE TABLE session_vitals (
    vital_id    TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    session_id  TEXT NOT NULL REFERENCES clinical_sessions(session_id) ON DELETE CASCADE,
    label       TEXT NOT NULL,   -- "BP", "Temperature", "SpO2", "Pulse", "Weight"
    value       TEXT NOT NULL,   -- "120/80 mmHg", "98.6°F"
    recorded_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_vitals_session ON session_vitals(session_id);
```

**Frontend getpoint served:** `GET /api/sessions/{id}` → `Session.vitals[]`

---

### Table 8 — `session_tags`

Single table for `symptoms`, `medicines`, and `tests` — all are `TagItem[]` in the frontend model.

```sql
CREATE TABLE session_tags (
    tag_id      TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    session_id  TEXT NOT NULL REFERENCES clinical_sessions(session_id) ON DELETE CASCADE,
    tag_type    TEXT NOT NULL
                    CHECK (tag_type IN ('symptom','medicine','test')),
    label       TEXT NOT NULL,
    meta        TEXT            -- JSON for dose/frequency on medicines
);
CREATE INDEX idx_session_tags_session ON session_tags(session_id, tag_type);
```

**Frontend getpoint served:** `GET /api/sessions/{id}` → `Session.symptoms[]`, `.medicines[]`, `.tests[]`

---

### Table 9 — `prescriptions`

```sql
CREATE TABLE prescriptions (
    rx_id           TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    session_id      TEXT NOT NULL UNIQUE REFERENCES clinical_sessions(session_id) ON DELETE CASCADE,
    advice          TEXT,           -- freetext advice lines
    follow_up_date  TEXT,           -- ISO-8601
    medicines_json  TEXT,           -- JSON array: [{name, dose, frequency, duration}]
    tests_json      TEXT,           -- JSON array: [{name, instructions}]
    status          TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','signed','dispensed')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_prescriptions_session ON prescriptions(session_id);
```

**Frontend getpoint served:** `GET /api/sessions/{id}` → `Session.prescription`  
**Dashboard stat:** `pendingRx = COUNT(*) FROM prescriptions WHERE status='pending' AND doctor_id=?`

---

### Table 10 — `appointments`

```sql
CREATE TABLE appointments (
    appointment_id  TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    doctor_id       TEXT NOT NULL REFERENCES doctors(doctor_id) ON DELETE CASCADE,
    patient_id      TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    session_id      TEXT REFERENCES clinical_sessions(session_id),  -- set when session is created
    appt_date       TEXT NOT NULL,    -- "YYYY-MM-DD"
    appt_time       TEXT NOT NULL,    -- "HH:MM"
    duration_min    INTEGER DEFAULT 15,
    appt_type       TEXT DEFAULT 'consultation',
    urgency         TEXT NOT NULL DEFAULT 'routine'
                        CHECK (urgency IN ('routine','attention','urgent')),
    status          TEXT NOT NULL DEFAULT 'scheduled'
                        CHECK (status IN ('scheduled','completed','cancelled','no_show')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_appts_doctor_date ON appointments(doctor_id, appt_date, appt_time);
CREATE INDEX idx_appts_patient     ON appointments(patient_id);
```

**Frontend getpoint served:** `GET /api/appointments?date=YYYY-MM-DD` → `Appointment[]`

---

## Part B: Write Path — Pipeline → SQLite

This section defines exactly how a completed Varta translation session flows into the SQLite database. No existing pipeline code changes; this is a new consumer of the `turn_persisted` WS event and the planned persistence queue.

### Trigger: Session Close

When the doctor ends a recording session (frontend fires `POST /api/sessions`), the write path executes:

```
POST /api/sessions
Body: {
  patient_id:      "pat-uuid",
  varta_session_id: "sess-uuid",   // from Varta /session/create_durable response
  chief_complaint: "Fever since 3 days",
  duration_minutes: 12,
  vitals: [{label: "Temp", value: "102°F"}, ...],
  symptoms: ["fever", "headache"],
  medicines: [...],
  tests: [...],
  prescription: { advice: "...", follow_up_date: "2026-10-01" }
}
```

```
STEP 1  INSERT INTO clinical_sessions (doctor_id, patient_id, varta_session_id, ...)
        → returns session_id

STEP 2  Fetch turn data from MongoDB varta_conversations.translation_turns
        WHERE session_id = varta_session_id AND status = 'completed'
        ORDER BY sequence ASC

STEP 3  INSERT INTO session_turns (turn_id, session_id, sequence, speaker, transcript, ...)
        (bulk insert, one row per translation_turn)

STEP 4  INSERT INTO session_vitals (session_id, label, value, ...)   [from body.vitals]

STEP 5  INSERT INTO session_tags (session_id, tag_type, label)       [from body.symptoms/medicines/tests]

STEP 6  INSERT INTO prescriptions (session_id, advice, follow_up_date, ...)

STEP 7  UPDATE patients SET last_visit_date = ... WHERE patient_id = ...
        (handled by trigger trg_patient_last_visit)

STEP 8  If an appointment existed for today:
        UPDATE appointments SET status='completed', session_id=<new_session_id>
```

### Async background path (alternative — no POST /api/sessions needed)

For fully automated ingestion via the WS `turn_persisted` event:

```
Browser emits turn_persisted → backend queues it
→ On session close (WS disconnect or explicit signal):
  → Fetch all completed turns from MongoDB (by varta_session_id)
  → Auto-create a clinical_sessions row (status='open', no clinical annotation yet)
  → Insert session_turns rows
  → Doctor later opens the session in the UI → adds vitals, tags, prescription
  → UI fires PATCH /api/sessions/{id}  (closes status, adds clinical context)
```

---

## Part C: Getpoint → SQL Query Map

Every endpoint from `frontendschema.md` mapped to its exact SQLite query.

---

### `GET /api/patients?search=&limit=`

```sql
SELECT
    p.patient_id        AS id,
    p.patient_id        AS patientId,
    p.abha_id           AS abhaId,
    p.name,
    p.age,
    p.gender,
    p.blood_group       AS bloodGroup,
    p.phone,
    p.photo_url         AS photo,
    p.urgency,
    p.last_visit_date   AS lastVisitDate,
    p.living_summary    AS livingSummary,
    p.emergency_contact AS emergencyContact
FROM patients p
WHERE p.doctor_id = :doctor_id          -- from JWT
  AND (:search IS NULL OR p.name LIKE '%' || :search || '%')
ORDER BY p.last_visit_date DESC NULLS LAST
LIMIT :limit;
-- allergies and chronicConditions fetched in a second query or LEFT JOIN
```

---

### `GET /api/patients/{id}`

```sql
-- Main row
SELECT * FROM patients WHERE patient_id = :id AND doctor_id = :doctor_id;

-- Tags (two calls or one UNION)
SELECT tag_type, label FROM patient_tags WHERE patient_id = :id;
-- app splits by tag_type into allergies[] and chronicConditions[]
```

---

### `POST /api/patients`

```sql
INSERT INTO patients (doctor_id, abha_id, name, age, gender, blood_group,
                      phone, photo_url, urgency, living_summary, emergency_contact)
VALUES (:doctor_id, :abha_id, :name, :age, :gender, :blood_group,
        :phone, :photo_url, :urgency, :living_summary, :emergency_contact);

-- Then insert tags:
INSERT INTO patient_tags (patient_id, tag_type, label)
SELECT :patient_id, 'allergy',            value FROM json_each(:allergies_json)
UNION ALL
SELECT :patient_id, 'chronic_condition',  value FROM json_each(:conditions_json);
```

---

### `GET /api/patients/{id}/sessions`

```sql
SELECT
    cs.session_id       AS id,
    cs.session_number   AS sessionNumber,
    cs.session_date     AS date,
    strftime('%d %b', cs.session_date) AS shortDate,
    cs.duration_minutes AS duration,
    cs.session_type     AS type,
    cs.varta_mode       AS mode,
    cs.chief_complaint  AS chiefComplaint,
    cs.doctor_notes     AS doctorNotes,
    cs.follow_up_date   AS followUpDate,
    cs.status
FROM clinical_sessions cs
WHERE cs.patient_id = :patient_id
  AND cs.doctor_id  = :doctor_id
ORDER BY cs.session_date DESC, cs.session_time DESC
LIMIT :limit;
```

---

### `GET /api/sessions/{id}` *(the heaviest query — full session detail)*

```sql
-- 1. Session header
SELECT cs.*, p.name AS patient_name
FROM clinical_sessions cs
JOIN patients p ON p.patient_id = cs.patient_id
WHERE cs.session_id = :session_id AND cs.doctor_id = :doctor_id;

-- 2. Transcript lines (from session_turns — sourced from Varta pipeline)
SELECT sequence, speaker, transcript, translation,
       src_language AS language_code, timestamp_ms AS timestamp
FROM session_turns
WHERE session_id = :session_id
ORDER BY sequence ASC;

-- 3. Vitals
SELECT label, value FROM session_vitals WHERE session_id = :session_id;

-- 4. Tags (symptoms / medicines / tests)
SELECT tag_type, label, meta FROM session_tags WHERE session_id = :session_id;

-- 5. Prescription
SELECT advice, follow_up_date, medicines_json, tests_json
FROM prescriptions WHERE session_id = :session_id;
```

The API layer assembles these 5 result sets into a single `Session` JSON object.

---

### `POST /api/sessions`

See **Part B, Write Path** above — executes Steps 1–8 in a single SQLite transaction.

---

### `GET /api/appointments?date=`

```sql
SELECT
    a.appointment_id            AS id,
    a.patient_id                AS patientId,
    p.name                      AS patientName,
    a.appt_date                 AS date,
    a.appt_time                 AS time,
    a.duration_min              AS duration,
    a.appt_type                 AS type,
    a.urgency,
    cs.session_number           AS sessionNumber
FROM appointments a
JOIN patients p ON p.patient_id = a.patient_id
LEFT JOIN clinical_sessions cs ON cs.session_id = a.session_id
WHERE a.doctor_id  = :doctor_id
  AND a.appt_date  = :date          -- or BETWEEN :start AND :end for date range
  AND a.status NOT IN ('cancelled')
ORDER BY a.appt_time ASC;
```

---

### `GET /api/doctor/profile`

```sql
SELECT
    d.doctor_id         AS id,
    d.name,
    d.first_name        AS firstName,
    d.qualification,
    d.reg_number        AS regNumber,
    d.specialisation,
    c.name              AS clinic_name,
    c.address           AS clinic_address,
    c.phone             AS clinic_phone,
    c.website           AS clinic_website,
    c.gstin             AS clinic_gstin
FROM doctors d
LEFT JOIN clinics c ON c.doctor_id = d.doctor_id AND c.is_primary = 1
WHERE d.doctor_id = :doctor_id;
```

The API layer reshapes this flat row into the nested `DoctorProfile.clinic` object.

---

### `GET /api/dashboard/stats`

```sql
SELECT
    -- seenToday: sessions completed today
    (SELECT COUNT(*) FROM clinical_sessions
     WHERE doctor_id = :doctor_id
       AND session_date = date('now')
       AND status = 'closed')                               AS seenToday,

    -- nextTwoHours: appointments in next 2h
    (SELECT COUNT(*) FROM appointments
     WHERE doctor_id = :doctor_id
       AND appt_date = date('now')
       AND appt_time BETWEEN time('now') AND time('now', '+2 hours')
       AND status = 'scheduled')                            AS nextTwoHours,

    -- pendingRx: prescriptions not yet signed
    (SELECT COUNT(*) FROM prescriptions rx
     JOIN clinical_sessions cs ON cs.session_id = rx.session_id
     WHERE cs.doctor_id = :doctor_id
       AND rx.status = 'pending')                           AS pendingRx,

    -- dueThisWeek: follow-ups falling this week
    (SELECT COUNT(*) FROM clinical_sessions
     WHERE doctor_id = :doctor_id
       AND follow_up_date BETWEEN date('now') AND date('now', '+7 days')
       AND status != 'open')                                AS dueThisWeek;
```

---

## Part D: Identity Bridge

The existing Varta pipeline uses a JWT (`AUTH_SECRET`) to create `durable_user_id`. That same value is used as the primary key in `doctors` — **no new auth system needed**.

```
POST /session/create_durable
  Authorization: Bearer <JWT>
  → JWT.sub == doctor_id == doctors.doctor_id  (created on first login)

POST /translate/dual  (durable session)
  Authorization: Bearer <JWT>
  → _authorize_durable_session() checks JWT.sub == session.durable_user_id
  → same check applies to all /api/* endpoints below
```

All `/api/*` endpoints **reuse the same Bearer JWT**:

```python
# Shared FastAPI dependency — add once, apply to all /api/* routes
async def get_doctor(authorization: str | None = Header(None)) -> str:
    from web.auth import _decode
    if not authorization:
        raise HTTPException(401, "Authorization required")
    token = authorization.removeprefix("Bearer ").strip()
    user = _decode(token)
    if user is None:
        raise HTTPException(401, "Invalid or expired token")
    return user.user_id   # == doctor_id in SQLite
```

---

## Part E: Open End → Table Column Cross-Reference

Full traceability from each pipeline open end to the SQLite column that absorbs it.

| Pipeline source | Field | SQLite table.column |
|---|---|---|
| `POST /session/create_durable` JWT | `durable_user_id` | `clinical_sessions.doctor_id`, `patients.doctor_id` |
| `POST /session/create_durable` response | `session_id` | `clinical_sessions.varta_session_id` |
| `translation_sessions.next_sequence` | sequence counter | used to populate `session_turns.sequence` |
| `translation_sessions.turn_count` | turn count | used to estimate `clinical_sessions.duration_minutes` |
| `translation_turns.turn_id` | turn UUID | `session_turns.turn_id` (PK) |
| `translation_turns.speaker_id` | "a" / "b" | `session_turns.speaker` |
| `translation_turns.transcript` | source text | `session_turns.transcript` |
| `translation_turns.translation.translated_text` | translated text | `session_turns.translation` |
| `translation_turns.source_language` | BCP-47 | `session_turns.src_language` |
| `translation_turns.translation.tgt_language` | BCP-47 | `session_turns.tgt_language` |
| `translation_turns.created_at` | epoch ms | `session_turns.timestamp_ms` |
| `translation_turns.timing.nmt_latency_ms` | int ms | `session_turns.latency_ms` |
| `translation_turns.status` | enum string | `session_turns.status` |
| `varta_metrics.request_latency` | latency doc | not stored in SQLite (kept in Mongo for analytics) |
| `log_translation()` (currently dead) | translation log | superseded by `session_turns` — do not re-enable |

---

## Part F: What Is NOT Bridged (Explicit Non-Goals Here)

| Item | Reason |
|---|---|
| Audio bytes (TTS output) | TEXT_ONLY policy — plan §17 #6 |
| `varta_metrics.model_performance` | Mongo-only analytics; not surfaced in frontend |
| `request_latency` / browser timing | Stays in Mongo; frontend never reads it |
| Multi-doctor / multi-clinic SaaS | Out of scope — one SQLite DB per doctor/device |
| ABHA real-time lookup | External API — integration point only, not schema |
| Report file storage | Frontend `reports[]` assumed to be URL references to blob storage, not SQLite rows |

---

## Part G: Migration / Init Script Outline

Run once on first launch (or deploy). Mirrors the Mongo migration pattern.

```sql
-- migrations/sqlite/m001_create_clinical_schema.sql

PRAGMA journal_mode = WAL;       -- safe for concurrent reads (mobile apps)
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS _migrations (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Guard: only apply if not already run
INSERT OR IGNORE INTO _migrations (name) VALUES ('m001_create_clinical_schema');

-- Then: CREATE TABLE IF NOT EXISTS for all 10 tables above
-- Then: CREATE INDEX IF NOT EXISTS for all indexes above
-- Then: CREATE TRIGGER IF NOT EXISTS for all triggers above
```

---

## Summary Table

| Frontend getpoint | SQLite tables touched | Pipeline data injected |
|---|---|---|
| `GET /api/patients` | `patients` | None (entered by doctor) |
| `GET /api/patients/{id}` | `patients`, `patient_tags` | None |
| `POST /api/patients` | `patients`, `patient_tags` | None |
| `GET /api/patients/{id}/sessions` | `clinical_sessions` | `varta_session_id` reference |
| `GET /api/sessions/{id}` | `clinical_sessions`, `session_turns`, `session_vitals`, `session_tags`, `prescriptions` | **All `session_turns` rows** come from MongoDB `translation_turns` |
| `POST /api/sessions` | all 5 above + triggers | Fetches & inserts turns from Mongo |
| `GET /api/appointments` | `appointments`, `patients`, `clinical_sessions` | None |
| `GET /api/doctor/profile` | `doctors`, `clinics` | `doctor_id` = JWT `durable_user_id` |
| `GET /api/dashboard/stats` | `clinical_sessions`, `appointments`, `prescriptions` | `session_date` sourced from Varta session |
