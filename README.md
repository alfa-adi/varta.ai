# 🇮🇳 varta.ai

> **A modular multilingual speech translation engine for Indian languages.**
>
> *ASR → NMT → TTS • Intelligent Model Routing • FastAPI • Sarvam AI • Bhashini*

---

## Overview

varta.ai is a modular speech translation backend designed for real-time multilingual communication across Indian languages.

Unlike traditional translation services that rely on a single AI provider, varta.ai is built around a **provider-agnostic architecture** capable of orchestrating multiple speech and translation models while selecting the most appropriate pipeline for every language pair.

The long-term vision is to build a routing engine capable of supporting all 22 Scheduled Indian Languages by combining the strengths of government (Bhashini/ULCA), academic (AI4Bharat, IIT Madras, IIIT Hyderabad), and commercial (Sarvam AI) models behind a unified interface.

The current implementation uses Sarvam AI as the primary provider while the architecture is intentionally designed to support plug-and-play integration of additional providers without modifying the application logic.

---

# Why varta.ai?

India has hundreds of languages and dialects.

No single AI model provides the highest quality speech recognition, translation, and speech synthesis for every language.

Instead of treating translation as a single API call, varta.ai treats it as an orchestration problem.

For every request the system can independently choose

- Speech Recognition (ASR)
- Machine Translation (NMT)
- Speech Synthesis (TTS)

allowing the best model to be selected for each stage.

---

# Architecture

```
                    Browser
                       │
                       ▼
                FastAPI Backend
                       │
               Translation Router
                       │
          ┌────────────┴────────────┐
          │                         │
     Model Registry           Session Store
          │                         │
          ▼                         ▼
      Provider Adapters        Redis / Memory
          │
          ▼
  ┌───────────┬────────────┬─────────────┐
  │           │            │
 Sarvam    Bhashini     Future Providers
  │
  ▼
ASR → NMT → TTS
```

The project follows a layered architecture where every concern remains independent.

- Registry knows what models exist.
- Router decides which models to use.
- Adapters communicate with providers.
- Pipelines orchestrate execution.
- The web layer only coordinates requests.

---

# Features

## Speech Translation

- Automatic language detection
- Speech-to-text (ASR)
- Machine Translation (NMT)
- Text-to-speech (TTS)
- End-to-end translated audio response

---

## Two Speaker Translation

Supports simultaneous bidirectional conversations.

Instead of asking users to select languages beforehand, the system detects the language spoken by each participant and automatically establishes the translation pair.

```
Speaker A
   ↓
Detected Hindi
   ↓
Stored in Session

Speaker B
   ↓
Detected Tamil
   ↓

Translation Pair

Hindi ⇄ Tamil
```

---

## Provider Agnostic Design

Every AI provider is wrapped behind a common interface.

```
ASRInput
      │
      ▼
 SarvamAdapter

BhashiniAdapter

FutureAdapter

      │
      ▼
ASROutput
```

The routing logic never depends on provider-specific APIs.

---

## Registry Driven Model Selection

The project maintains a central model registry containing

- supported languages
- latency
- quality score
- provider
- streaming support
- pricing

Adding a new AI model should require changing only the registry rather than application code.

---

## Modular Pipelines

Current pipelines include

- Single speaker translation
- Dual speaker translation

The architecture is intentionally extensible for

- streaming translation
- meeting transcription
- batch processing
- future WebSocket pipelines

---

# Technology Stack

### Backend

- Python 3.11+
- FastAPI
- Gunicorn
- Uvicorn

### AI

- Sarvam AI
- Bhashini (planned routing)
- AI4Bharat
- IIT Madras models

### Infrastructure

- MongoDB Atlas
- Upstash Redis
- httpx
- Render

### Testing

- pytest
- Playwright (browser latency benchmarking)

---

# Repository Structure

```
adapter/
    base.py
    sarvam_asr.py
    sarvam_nmt.py
    sarvam_tts.py

pipeline/
    single.py
    dual.py
    types.py

registry/
    models.json

web/
    server.py

tests/

static/

requirements.txt
```

---

# Design Principles

## Separation of Concerns

Each layer has exactly one responsibility.

```
Router
    ↓
Registry
    ↓
Adapter
    ↓
Provider
```

---

## Stateless Backend

Application state does not live inside FastAPI workers.

Conversation state is externalized to Redis allowing

- horizontal scaling
- zero-downtime deployment
- worker restarts without session loss

---

## Data Before Optimization

The project intentionally avoids premature optimization.

Before migrating from REST to WebSockets, the system measures real browser-side latency using Playwright to determine where time is actually being spent.

Measured metrics include

- upload time
- TCP handshake
- server processing
- ASR latency
- NMT latency
- TTS latency
- audio download
- browser decode

Architecture decisions are based on measured bottlenecks rather than assumptions.

---

# Running Locally

Clone the repository

```bash
git clone https://github.com/alfa-adi/varta.ai.git
cd varta.ai
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create environment file

```bash
cp .env.example .env
```

Run

```bash
uvicorn web.server:app --reload
```

---

# Environment Variables

```
SARVAM_API_KEY=

REDIS_URL=

MONGO_URL=

MONGO_DB_NAME=

ALLOWED_ORIGINS=
```

---

# API

| Endpoint | Description |
|-----------|-------------|
| `/health` | Health check |
| `/translate/single` | One-way speech translation |
| `/translate/dual` | Bidirectional translation |
| `/translate/speaker_a` | Speaker A |
| `/translate/speaker_b` | Speaker B |
| `/metrics/browser` | Browser latency reporting |

---

# Current Status

## Completed

- Adapter architecture
- Modular pipelines
- Sarvam AI integration
- Session management
- MongoDB logging
- Browser interface
- Translation APIs

---

## In Progress

- Intelligent model router
- Browser latency benchmarking
- Registry-based routing
- Performance optimization
- WebSocket evaluation

---

# Roadmap

- [x] Modular adapter architecture
- [x] Dual speaker translation
- [x] Provider abstraction
- [x] Browser UI
- [ ] Registry-driven routing
- [ ] Multi-provider execution
- [ ] Browser latency profiler
- [ ] Intelligent routing engine
- [ ] Streaming translation
- [ ] Production WebSocket support

---

# Vision

varta.ai is not intended to become "another translation API."

The long-term objective is to build an extensible multilingual speech infrastructure capable of intelligently combining the best available AI models across India's language ecosystem while remaining independent of any single provider.

---

# License

MIT License
