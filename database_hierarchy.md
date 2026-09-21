# Varta.ai — Hierarchical Database Structure

## 1. Full System Data Hierarchy

```mermaid
graph TD
    subgraph IDENTITY["🔐 IDENTITY LAYER"]
        JWT["JWT Token<br/>─────────<br/>sub: doctor_id<br/>roles: []<br/>exp: timestamp"]
    end

    subgraph REDIS["⚡ REDIS — Live Session State (TTL: 2h)"]
        RS["session:{id}<br/>─────────<br/>lang_a: BCP-47<br/>lang_b: BCP-47<br/>pending_transcript_a: text<br/>pending_transcript_b: text<br/>is_durable: bool<br/>durable_user_id: UUID<br/>durable_customer_id: str<br/>durable_session_id: UUID<br/>pending_turn_id_a: UUID<br/>pending_turn_id_b: UUID"]
        RL["live-owner:{id}:{spk}<br/>─────────<br/>owner_uuid: str<br/>TTL: 30s (renewed)"]
        RT["live-turn:{id}<br/>─────────<br/>speaker: a|b<br/>turn_id: UUID<br/>TTL: 120s"]
        RA["asr:lang:{spk}:{id}<br/>─────────<br/>language: BCP-47<br/>TTL: 2h"]
    end

    subgraph MONGO_METRICS["📊 MONGODB — varta_metrics"]
        MRL["request_latency<br/>─────────<br/>session_id: str<br/>endpoint: str<br/>timestamp: datetime<br/>browser: {upload_ms, ...}<br/>server: {total_ms, ...}<br/>asr: {total_ms, tcp_ms, ...}<br/>nmt: {total_ms, tcp_ms, ...}<br/>tts: {total_ms, tcp_ms, ...}<br/>src_language, tgt_language<br/>char_count: int"]
        MMP["model_performance<br/>─────────<br/>session_id: str<br/>model_id: str<br/>model_type: ASR|NMT|TTS<br/>src_language, tgt_language<br/>char_count: int<br/>total_ms, tcp_ms, api_ms"]
    end

    subgraph MONGO_CONV["💾 MONGODB — varta_conversations (planned)"]
        MTS["translation_sessions<br/>─────────<br/>_id: session_id (UUID)<br/>user_id: str<br/>customer_id: str<br/>created_at: datetime<br/>updated_at: datetime<br/>source_language: BCP-47<br/>target_language: BCP-47<br/>next_sequence: int<br/>turn_count: int<br/>status: active|closed"]
        MTT["translation_turns<br/>─────────<br/>_id: ObjectId<br/>turn_id: UUID (unique)<br/>session_id: UUID (FK)<br/>user_id: str<br/>speaker_id: a|b<br/>sequence: int<br/>status: received|transcribed|<br/>  completed|failed|cancelled<br/>audio_fingerprint: str<br/>transcript: str<br/>source_language: BCP-47<br/>translation: {<br/>  translated_text: str<br/>  tgt_language: BCP-47<br/>}<br/>timing: {nmt_latency_ms, ...}<br/>error_message: str<br/>created_at, updated_at"]
    end

    subgraph SQLITE["🏥 SQLITE — Clinical Database (planned)"]
        SD["doctors<br/>─────────<br/>doctor_id: PK (= JWT sub)<br/>name, first_name<br/>qualification, reg_number<br/>specialisation<br/>created_at, updated_at"]
        SC["clinics<br/>─────────<br/>clinic_id: PK<br/>doctor_id: FK → doctors<br/>name, address, phone<br/>website, gstin<br/>is_primary: bool"]
        SP["patients<br/>─────────<br/>patient_id: PK<br/>doctor_id: FK → doctors<br/>abha_id, name, age<br/>gender, blood_group<br/>phone, photo_url<br/>urgency: routine|attention|urgent<br/>emergency_contact: JSON<br/>living_summary<br/>last_visit_date (auto-trigger)"]
        SPT["patient_tags<br/>─────────<br/>tag_id: PK<br/>patient_id: FK → patients<br/>tag_type: allergy |<br/>  chronic_condition<br/>label: str"]
        SCS["clinical_sessions<br/>─────────<br/>session_id: PK<br/>doctor_id: FK → doctors<br/>patient_id: FK → patients<br/>varta_session_id: UNIQUE<br/>  (FK → Mongo sessions)<br/>varta_mode: transcribe|translate<br/>session_number: int<br/>session_date, session_time<br/>duration_minutes<br/>session_type, chief_complaint<br/>doctor_notes, follow_up_date<br/>status: open|pending_rx|closed"]
        SST["session_turns<br/>─────────<br/>turn_id: PK<br/>  (= Mongo turn_id)<br/>session_id: FK → sessions<br/>sequence: int<br/>speaker: doctor|patient|a|b<br/>src_language, tgt_language<br/>transcript, translation<br/>timestamp_ms, latency_ms<br/>status"]
        SSV["session_vitals<br/>─────────<br/>vital_id: PK<br/>session_id: FK → sessions<br/>label: str (BP, Temp...)<br/>value: str"]
        SSTG["session_tags<br/>─────────<br/>tag_id: PK<br/>session_id: FK → sessions<br/>tag_type: symptom |<br/>  medicine | test<br/>label: str<br/>meta: JSON"]
        SSR["session_reports<br/>─────────<br/>report_id: PK<br/>session_id: FK → sessions<br/>title, file_url<br/>mime_type, uploaded_at"]
        SRX["prescriptions<br/>─────────<br/>rx_id: PK<br/>session_id: FK → sessions<br/>  (UNIQUE)<br/>advice: text<br/>follow_up_date<br/>status: pending|signed|<br/>  dispensed"]
        SAP["appointments<br/>─────────<br/>appointment_id: PK<br/>doctor_id: FK → doctors<br/>patient_id: FK → patients<br/>session_id: FK → sessions<br/>  (nullable)<br/>appt_date, appt_time<br/>duration_min, appt_type<br/>urgency, status, notes"]
    end

    %% Identity flows
    JWT -->|"sub = doctor_id"| SD
    JWT -->|"sub = durable_user_id"| RS

    %% Redis to Mongo link
    RS -->|"durable_session_id"| MTS

    %% Mongo internal hierarchy
    MTS -->|"session_id"| MTT

    %% Mongo to SQLite bridge
    MTS -->|"session_id = varta_session_id"| SCS
    MTT -->|"turn_id = turn_id (PK)"| SST

    %% SQLite hierarchy
    SD --> SC
    SD --> SP
    SD --> SAP
    SP --> SPT
    SP --> SCS
    SP --> SAP
    SCS --> SST
    SCS --> SSV
    SCS --> SSTG
    SCS --> SSR
    SCS --> SRX
    SCS -.->|"session_id set on<br/>appointment completion"| SAP

    %% Metrics (standalone)
    RS -.->|"session_id"| MRL
    MRL -.-> MMP

    style IDENTITY fill:#1a1a2e,stroke:#e94560,color:#fff
    style REDIS fill:#0f3460,stroke:#16213e,color:#fff
    style MONGO_METRICS fill:#1a1a2e,stroke:#e94560,color:#fff
    style MONGO_CONV fill:#16213e,stroke:#0f3460,color:#fff
    style SQLITE fill:#0a3d62,stroke:#3c6382,color:#fff
```

---

## 2. Data Flow — Write Path (How Data Moves Through The System)

```mermaid
flowchart TB
    subgraph BROWSER["🌐 Browser / Frontend"]
        MIC["🎤 Microphone<br/>PCM audio"]
        UI["📋 Doctor UI<br/>vitals, notes, Rx"]
    end

    subgraph FASTAPI["⚙️ FastAPI Server"]
        WS["WebSocket<br/>/ws/asr/{id}/{spk}"]
        REST_T["REST<br/>/translate/*"]
        REST_S["REST<br/>/session/create_durable"]
        REST_API["REST<br/>/api/* (clinical)"]
    end

    subgraph SARVAM["🤖 Sarvam AI APIs"]
        ASR["ASR<br/>Speech → Text"]
        NMT["NMT<br/>Hindi → English"]
        TTS["TTS<br/>Text → Speech"]
    end

    subgraph STORAGE["💾 Storage Layer"]
        R[("⚡ Redis<br/>Live state<br/>TTL: 2h")]
        MDB_M[("📊 MongoDB<br/>varta_metrics<br/>Latency data")]
        MDB_C[("💾 MongoDB<br/>varta_conversations<br/>Transcripts")]
        SQL[("🏥 SQLite<br/>Clinical DB<br/>Patients, Rx")]
    end

    MIC -->|"PCM binary frames"| WS
    MIC -->|"audio file upload"| REST_T
    UI -->|"patient data, vitals"| REST_API

    WS --> ASR
    REST_T --> ASR
    ASR -->|"transcript"| NMT
    NMT -->|"translation"| TTS
    TTS -->|"audio chunks"| BROWSER

    REST_S -->|"create session doc"| R
    REST_S -->|"create session doc"| MDB_C
    WS -->|"session snapshot"| R
    REST_T -->|"session snapshot"| R
    WS -->|"turn_persisted event"| MDB_C
    REST_T -->|"persist turn"| MDB_C
    REST_T -->|"log_metrics()"| MDB_M
    WS -->|"log_metrics()"| MDB_M

    REST_API -->|"patients, sessions,<br/>appointments, Rx"| SQL
    MDB_C -->|"fetch completed turns<br/>on session close"| SQL

    style BROWSER fill:#2d3436,stroke:#636e72,color:#fff
    style FASTAPI fill:#0c2461,stroke:#1e3799,color:#fff
    style SARVAM fill:#6c5ce7,stroke:#a29bfe,color:#fff
    style STORAGE fill:#0a3d62,stroke:#3c6382,color:#fff
```

---

## 3. Entity Relationship Diagram — SQLite Clinical Database

```mermaid
erDiagram
    doctors ||--o{ clinics : "has"
    doctors ||--o{ patients : "treats"
    doctors ||--o{ appointments : "schedules"

    patients ||--o{ patient_tags : "has"
    patients ||--o{ clinical_sessions : "visits"
    patients ||--o{ appointments : "attends"

    clinical_sessions ||--o{ session_turns : "contains"
    clinical_sessions ||--o{ session_vitals : "records"
    clinical_sessions ||--o{ session_tags : "tagged with"
    clinical_sessions ||--o{ session_reports : "attaches"
    clinical_sessions ||--|| prescriptions : "produces"
    clinical_sessions ||--o| appointments : "fulfills"

    doctors {
        text doctor_id PK "= JWT sub"
        text name
        text first_name
        text qualification
        text reg_number UK
        text specialisation
    }

    clinics {
        text clinic_id PK
        text doctor_id FK
        text name
        text address
        text phone
        int is_primary
    }

    patients {
        text patient_id PK
        text doctor_id FK
        text abha_id
        text name
        int age
        text gender
        text urgency
        text last_visit_date "auto-updated"
    }

    patient_tags {
        text tag_id PK
        text patient_id FK
        text tag_type "allergy | chronic_condition"
        text label
    }

    clinical_sessions {
        text session_id PK
        text doctor_id FK
        text patient_id FK
        text varta_session_id UK "bridge to MongoDB"
        text varta_mode
        int session_number
        text session_date
        text status "open | pending_rx | closed"
    }

    session_turns {
        text turn_id PK "= MongoDB turn_id"
        text session_id FK
        int sequence
        text speaker "doctor | patient"
        text transcript "source language"
        text translation "target language"
        int timestamp_ms
    }

    session_vitals {
        text vital_id PK
        text session_id FK
        text label "BP, Temp, SpO2"
        text value
    }

    session_tags {
        text tag_id PK
        text session_id FK
        text tag_type "symptom | medicine | test"
        text label
        text meta "JSON dose/freq"
    }

    session_reports {
        text report_id PK
        text session_id FK
        text title
        text file_url "blob storage URL"
        text mime_type
    }

    prescriptions {
        text rx_id PK
        text session_id FK "UNIQUE"
        text advice
        text follow_up_date
        text status "pending | signed"
    }

    appointments {
        text appointment_id PK
        text doctor_id FK
        text patient_id FK
        text session_id FK "nullable"
        text appt_date
        text appt_time
        text urgency
        text status "scheduled | completed"
    }
```

---

## 4. Cross-Database Bridge Points

```mermaid
flowchart LR
    subgraph JWT["🔐 JWT Token"]
        SUB["sub: UUID"]
    end

    subgraph REDIS["⚡ Redis"]
        SESS["session:{id}<br/>durable_user_id<br/>durable_session_id"]
    end

    subgraph MONGO["💾 MongoDB"]
        MSESS["translation_sessions<br/>._id = session_id<br/>.user_id"]
        MTURN["translation_turns<br/>.turn_id<br/>.session_id"]
    end

    subgraph SQLITE["🏥 SQLite"]
        DOC["doctors<br/>.doctor_id"]
        CS["clinical_sessions<br/>.doctor_id<br/>.varta_session_id"]
        ST["session_turns<br/>.turn_id"]
    end

    SUB ==>|"same UUID"| SESS
    SUB ==>|"same UUID"| MSESS
    SUB ==>|"same UUID"| DOC

    SESS -->|"durable_session_id"| MSESS
    MSESS -->|"session_id"| CS
    MTURN -->|"turn_id"| ST
    MSESS -->|"session_id"| MTURN
    DOC -->|"doctor_id"| CS

    style JWT fill:#e94560,stroke:#1a1a2e,color:#fff
    style REDIS fill:#0f3460,stroke:#16213e,color:#fff
    style MONGO fill:#16213e,stroke:#0f3460,color:#fff
    style SQLITE fill:#0a3d62,stroke:#3c6382,color:#fff
```

> **One UUID rules them all:** The `sub` field from the JWT token is the same value used as `durable_user_id` in Redis, `user_id` in MongoDB, and `doctor_id` in SQLite. No ID translation layer needed — the three databases are joinable on this single key.

---

## 5. Reading Guide

| Diagram | What it shows |
|---|---|
| **§1 — Full Hierarchy** | Every table/collection/key across all 3 databases with all their fields, plus the FK arrows between them |
| **§2 — Write Path** | How audio from the microphone flows through Sarvam APIs and into all 3 storage systems |
| **§3 — ER Diagram** | Standard entity-relationship diagram for the SQLite clinical database (the one serving frontend getpoints) |
| **§4 — Bridge Points** | The 3 cross-database join keys that stitch Redis, MongoDB, and SQLite together using one UUID |
