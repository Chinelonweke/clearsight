# 👁 ClearSight — AI Eye Clinic Triage Assistant

> A production-ready AI triage system for a Nigerian optometrist eye clinic.
> Patients describe their symptoms via voice or text, receive an instant clinical urgency assessment, and get a real appointment booked — all before they leave the chat.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-orange)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Security & Hardening](#security--hardening)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Environment Variables](#environment-variables)
- [Database Setup](#database-setup)
- [Running the App](#running-the-app)
- [Docker Deployment](#docker-deployment)
- [AWS EC2 Deployment](#aws-ec2-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Staff Dashboard](#staff-dashboard)
- [API Reference](#api-reference)
- [Knowledge Base](#knowledge-base)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎙 **Voice Triage** | Real-time voice conversations via Whisper STT + Piper TTS |
| 📷 **Eye Image Analysis** | Camera capture + LLaVA vision model for visual triage |
| 🧠 **AI Urgency Scoring** | LLaMA3 + RAG pipeline scores urgency 1-10 across 3 levels |
| 📅 **Auto Appointment Booking** | Slots booked instantly in NeonDB after triage |
| 📧 **Email Confirmations** | Booking confirmation emails via Gmail SMTP |
| 🏥 **Staff Dashboard** | Doctor/admin login, workload view, Mark Seen button |
| 📊 **Analytics Dashboard** | Sessions, bookings, error rates, urgency breakdown |
| 🔒 **JWT Authentication** | Separate doctor and admin login flows |
| 🧠 **Patient Memory** | mem0 cloud — remembers patients across visits |
| 🔄 **Session Resumption** | Incomplete sessions saved to Redis (30-min TTL) |
| 👤 **Returning Patient Detection** | NeonDB-based detection — never asks for known details again |
| 🛡 **Prompt Injection Protection** | 28-pattern pre-LLM interception layer |
| ⚡ **4-Provider LLM Fallback** | Groq → OpenRouter → Together AI → HuggingFace |
| 📡 **Observability** | Sentry error tracking + BetterUptime monitoring |

---

## 🏗 Architecture

```
Patient Browser
      │
      │ WebSocket (voice/text/image)
      ▼
┌─────────────────────────────────────────────┐
│           FastAPI Application                │
│                                              │
│  WebSocket ──► Injection Check (pre-LLM)    │
│              ──► LLM Service (4-provider)    │
│              ──► STT Service (Whisper)       │
│              ──► TTS Service (Piper)         │
│              ──► Vision (LLaVA)              │
│              ──► Triage Service              │
│              ──► Booking Service             │
│              ──► Email Service               │
│              ──► Memory Service (mem0)       │
│                         │                    │
└─────────────────────────┼────────────────────┘
                          │
        ┌─────────────────┼──────────────────┐
        ▼                 ▼                  ▼
   NeonDB (PG)      Upstash Redis       ChromaDB
   Patients         Sessions (TTL)      RAG Vectors
   Appointments     Incomplete          Eye Conditions
   Doctors          Session Cache       (14 files)
   Slots
        │
        ▼
   mem0 Cloud
   Patient Memory
   Cross-session facts
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI (async) |
| **LLM — Primary** | Groq — LLaMA 3.3 70B Versatile |
| **LLM — Fallback 1** | OpenRouter — LLaMA/Mistral/Qwen/Gemma |
| **LLM — Fallback 2** | Together AI — LLaMA 3.3 70B |
| **LLM — Fallback 3** | HuggingFace — Mistral 7B |
| **STT** | Groq — Whisper Large v3 |
| **Vision** | Groq — LLaVA v1.5 7B |
| **TTS** | Piper TTS (en_US-lessac-medium) |
| **Database** | NeonDB (PostgreSQL + asyncpg) |
| **Sessions** | Upstash Redis |
| **Vector Store** | ChromaDB 0.5.23 |
| **Embeddings** | fastembed (ONNX — replaces sentence-transformers, 84% smaller image) |
| **Patient Memory** | mem0 Cloud |
| **Email** | Gmail SMTP |
| **Auth** | JWT (python-jose) + bcrypt |
| **Logging** | Loguru (structured, color-coded) |
| **Error Tracking** | Sentry (errors + traces + LLM calls) |
| **Uptime Monitoring** | BetterUptime (5-min health checks) |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions → AWS ECR → EC2 |
| **Infrastructure** | AWS EC2 t3.small (eu-north-1) + Nginx + Let's Encrypt |

---

## 🔒 Security & Hardening

ClearSight has undergone production security hardening across four layers.

### 1. Prompt Injection Protection

Every patient message is checked against 28 known injection patterns **before it reaches the LLM**. If a match is found, the message is silently blocked and logged to Sentry — the LLM never sees it.

Patterns covered include: `ignore previous instructions`, `you are now`, `act as`, `jailbreak`, `list all patients`, `reveal patient data`, and 22 more.



### 2. Non-Root Container Execution

All Docker containers run as a non-privileged user (`clearsight`, UID 1001) to prevent host system breakout. The Dockerfile creates and switches to this user before starting the application.

```dockerfile
RUN useradd -m -u 1001 clearsight
USER clearsight
```

### 3. Input Validation

- **Diagnosis interception**: Questions asking for a diagnosis are intercepted before the LLM, with a hard-coded clinical refusal
- **Emergency fast-path**: Chemical eye emergency keywords trigger an immediate response without LLM processing
- **Email validation**: Patient email captured via regex, never trusted as-is from LLM output

### 4. LLM Fallback Chain (Availability Hardening)

A 4-provider automatic fallback chain prevents single-provider outages or rate limits from taking down the service:

```
Groq (primary) → OpenRouter → Together AI → HuggingFace
```

Each provider is tried automatically. Patients experience zero interruption if any upstream provider fails.

### 5. Observability & Alerting

- **Sentry**: Every error, injection attempt, and LLM call is tracked with full context
- **BetterUptime**: Health endpoint monitored every 5 minutes — SMS alert on downtime
- **mem0 quota alerts**: Sentry warning at 20% remaining, critical alert at 10%

---

## 📁 Project Structure

```
clearsight/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── admin.py          # Analytics + appointments endpoints
│   │   │   ├── auth.py           # Login + doctor-login endpoints
│   │   │   ├── booking.py        # Booking API
│   │   │   ├── session.py        # Session management
│   │   │   └── router.py         # Route registration
│   │   └── websocket.py          # Main triage WebSocket endpoint
│   ├── core/
│   │   ├── exceptions.py         # Custom exception classes
│   │   ├── logger.py             # Loguru configuration
│   │   ├── middleware.py         # Request logging middleware
│   │   └── security.py           # JWT helpers
│   ├── dashboard/
│   │   └── templates/
│   │       ├── index.html        # Patient triage UI
│   │       └── staff.html        # Staff dashboard
│   ├── db/
│   │   ├── neon.py               # NeonDB connection + init
│   │   └── redis_client.py       # Upstash Redis connection
│   ├── models/
│   │   ├── booking.py            # Appointment ORM model
│   │   ├── database.py           # SQLAlchemy base + session
│   │   ├── doctor.py             # Doctor + AvailabilitySlot models
│   │   ├── intake.py             # IntakeForm model
│   │   ├── patient.py            # Patient model
│   │   └── session.py            # ConversationSession model
│   ├── rag/
│   │   ├── chroma_client.py      # ChromaDB client + helpers
│   │   ├── chunking.py           # Semantic chunking strategy
│   │   └── ingest.py             # Knowledge base ingestion script
│   ├── services/
│   │   ├── analytics_service.py  # Event tracking
│   │   ├── booking_service.py    # Slot selection + booking engine
│   │   ├── email_service.py      # Gmail SMTP notifications
│   │   ├── intake_service.py     # Intake form auto-fill
│   │   ├── llm_service.py        # 4-provider LLM wrapper with fallback
│   │   ├── memory_service.py     # mem0 patient memory (quota-aware)
│   │   ├── rag_service.py        # ChromaDB retrieval
│   │   ├── session_service.py    # Redis session management
│   │   ├── stt_service.py        # Whisper transcription
│   │   ├── triage_service.py     # Clinical urgency scoring
│   │   ├── tts_service.py        # Piper TTS synthesis
│   │   └── vision_service.py     # LLaVA eye image analysis
│   ├── utils/
│   │   ├── audio.py              # Audio format helpers
│   │   └── image.py              # Image preprocessing
│   ├── config.py                 # Pydantic settings
│   ├── dependencies.py           # FastAPI dependency injection
│   └── main.py                   # App factory + lifespan
├── data/
│   └── knowledge_base/           # 14 eye condition markdown files
│       ├── cataracts.md
│       ├── chemical_injury.md
│       ├── conjunctivitis.md
│       ├── corneal_ulcer.md
│       ├── diabetic_retinopathy.md
│       ├── dry_eye_syndrome.md
│       ├── glaucoma.md
│       ├── macular_degeneration.md
│       ├── onchocerciasis.md
│       ├── pterygium.md
│       ├── refractive_errors.md
│       ├── retinal_detachment.md
│       ├── trachoma.md
│       └── uveitis.md
├── docker/
│   ├── Dockerfile                # Multi-stage production image (non-root)
│   └── docker-compose.yml        # Local dev: app + redis + chromadb
├── .github/
│   └── workflows/
│       ├── ci.yml                # Lint + Docker build check on PR
│       └── deploy.yml            # Build → ECR → EC2 deploy on main merge
├── .env.example                  # Environment variable template
├── pyproject.toml                # Dependencies
└── README.md
```

---

## ✅ Prerequisites

- Python 3.11+
- Docker Desktop
- Git
- A [Groq](https://console.groq.com) API key (free)
- A [NeonDB](https://neon.tech) account (free)
- An [Upstash](https://upstash.com) Redis account (free)
- A [mem0](https://app.mem0.ai) account (free — 1,000 searches/month)
- A [Sentry](https://sentry.io) account (free)
- OpenRouter, Together AI, HuggingFace API keys (all free)
- Piper TTS model file (`en_US-lessac-medium.onnx`)

---

## 🚀 Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/Chinelonweke/clearsight.git
cd clearsight
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Copy and fill the environment file

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 5. Start ChromaDB

```bash
docker run -d \
  --name clearsight_chroma \
  -p 8001:8000 \
  -v clearsight_chroma_data:/chroma/chroma \
  chromadb/chroma:0.5.23
```

### 6. Run database migrations

```bash
python -c "from app.db.neon import init_db; import asyncio; asyncio.run(init_db())"
```

### 7. Ingest the knowledge base

```bash
python -m app.rag.ingest
```

### 8. Start the server

```bash
uvicorn app.main:app --port 8000
```

Visit:
- Patient UI: http://localhost:8000
- Staff Dashboard: http://localhost:8000/staff
- API Docs: http://localhost:8000/docs

---

## 🔐 Environment Variables

```env
# ── Application ──────────────────────────────────────────
SECRET_KEY=your-secret-key-min-32-chars
ADMIN_USERNAME=Nelo
ADMIN_PASSWORD=your-admin-password

# ── LLM Providers (fallback chain) ───────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxx
TOGETHER_API_KEY=tgp_xxxxxxxxxxxxxxxxxxxx
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx

# ── Groq Models ──────────────────────────────────────────
GROQ_LLM_MODEL=llama-3.3-70b-versatile
GROQ_WHISPER_MODEL=whisper-large-v3
GROQ_VISION_MODEL=llava-v1.5-7b

# ── NeonDB ───────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://user:password@host/dbname?ssl=require

# ── Upstash Redis ─────────────────────────────────────────
REDIS_URL=rediss://default:password@host:6379

# ── ChromaDB ─────────────────────────────────────────────
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION_NAME=eye_conditions

# ── Piper TTS ─────────────────────────────────────────────
PIPER_MODEL_PATH=data/tts_models/en_US-lessac-medium.onnx

# ── Patient Memory ────────────────────────────────────────
MEM0_API_KEY=m0-xxxxxxxxxxxxxxxxxxxx

# ── Error Tracking ────────────────────────────────────────
SENTRY_DSN=https://xxxx@sentry.io/xxxx

# ── Email ─────────────────────────────────────────────────
GMAIL_USER=your@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# ── Clinic Information ────────────────────────────────────
CLINIC_NAME=ClearSight Eye Clinic
CLINIC_PHONE=07081848941
CLINIC_ADDRESS=N01, Okitipupa Crescent Phase4, Kubwa, Abuja
CLINIC_OPENING_HOUR=8
CLINIC_CLOSING_HOUR=18
```

---

## 🗄 Database Setup

ClearSight uses **NeonDB** (serverless PostgreSQL). Tables are created automatically on first startup via SQLAlchemy.

**Tables:**
- `patients` — patient records + auth
- `doctors` — doctor profiles + login credentials
- `availability_slots` — 30-minute appointment slots
- `appointments` — booked appointments with triage data
- `intake_forms` — auto-filled patient intake forms
- `conversation_sessions` — session tracking for returning patient detection
- `analytics_events` — session + triage + booking events
- `password_reset_tokens` — password reset flow

---

## 🐳 Docker Deployment

### Run with Docker Compose (local)

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Build production image

```bash
docker build -f docker/Dockerfile -t clearsight:latest .
```

### Run production container

```bash
docker run -d \
  --name clearsight_api \
  --env-file .env \
  -p 8000:8000 \
  --network docker_clearsight_net \
  --restart unless-stopped \
  clearsight:latest
```

> **Note:** The production image uses fastembed (ONNX) instead of sentence-transformers, reducing the Docker image size by ~84% by eliminating the PyTorch dependency.

---

## ☁️ AWS EC2 Deployment

### Current Production Setup

- **Instance:** t3.small (2 vCPU, 2GB RAM)
- **Region:** eu-north-1 (Stockholm)
- **OS:** Ubuntu 22.04 LTS
- **IP:** 13.53.60.140
- **Domain:** clearsightclinic.online
- **SSL:** Let's Encrypt (auto-renews)
- **Reverse Proxy:** Nginx

### Step 1 — Launch EC2 instance

1. Go to AWS Console → EC2 → Launch Instance
2. Choose **Ubuntu 22.04 LTS**
3. Instance type: **t3.small**
4. Create or select a key pair (save the `.pem` file)
5. Security group — open these ports:
   - 22 (SSH)
   - 80 (HTTP — redirects to HTTPS)
   - 443 (HTTPS)
6. Storage: **20GB gp3**

### Step 2 — Connect and install Docker

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 3 — Set up Nginx + SSL

```bash
sudo apt install nginx certbot python3-certbot-nginx -y
sudo certbot --nginx -d yourdomain.com
```

### Step 4 — Deploy the app

```bash
git clone https://github.com/Chinelonweke/clearsight.git
cd clearsight
nano .env  # add your production values

docker run -d \
  --name clearsight_api \
  --env-file .env \
  -p 8000:8000 \
  --network docker_clearsight_net \
  --restart unless-stopped \
  clearsight:latest

docker exec clearsight_api python -m app.rag.ingest
```

---

## ⚙️ CI/CD Pipeline

ClearSight uses two GitHub Actions workflows:

### `ci.yml` — Runs on every push and PR
- Lint check (Ruff)
- Docker build check
- Blocks merge if either fails

### `deploy.yml` — Runs on merge to main only
```
Merge to main
     ↓
Run tests
     ↓
Build Docker image
     ↓
Push to AWS ECR
     ↓
SSH into EC2
     ↓
Pull new image
     ↓
Restart container
     ↓
Health check (/health)
```

### GitHub Secrets Required

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | `eu-north-1` |
| `ECR_REPOSITORY` | `clearsight` |
| `EC2_HOST` | EC2 public IP |
| `EC2_SSH_KEY` | Contents of `.pem` file |
| `EC2_USER` | `ubuntu` |

---

## 🏥 Staff Dashboard

Access at `/staff`. Two login roles:

| Role | Access |
|---|---|
| Doctor | Own patients only, Mark Seen button |
| Admin | All patients, workload view, analytics |

**Features:**
- Today's appointments sorted by slot time
- Urgency badges (Emergency / Urgent / Routine)
- Chief complaint per patient
- Assigned doctor column (admin only)
- Doctor workload comparison bars (admin only)
- Mark Seen button
- Filter by urgency level
- Auto-refresh every 60 seconds

---

## 🧠 Patient Memory System

ClearSight remembers patients across visits using mem0 cloud.

**How it works:**

1. Patient completes triage → transcript sent to mem0
2. mem0 extracts key facts (name, phone, symptoms, urgency, appointment)
3. Facts stored linked to patient UUID
4. Next visit → facts retrieved and injected into LLM system prompt
5. AI greets patient by name, skips known questions

**Returning patient detection** uses NeonDB  as source of truth:



**Session resumption:** If a patient leaves mid-triage, the incomplete session is saved to Redis with a 30-minute TTL. On return, they are offered: *"Would you like to continue where we left off?"*

---

## 📡 API Reference

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/auth/login` | Admin login |
| POST | `/api/v1/auth/doctor-login` | Doctor login |

### WebSocket
| Endpoint | Description |
|---|---|
| `WS /ws/conversation/{session_id}` | Main triage conversation |

### Admin
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/admin/dashboard` | HTML analytics dashboard |
| GET | `/api/v1/admin/metrics` | Raw metrics JSON |
| GET | `/api/v1/admin/appointments/today` | Today's appointments |
| PATCH | `/api/v1/admin/appointments/{id}/seen` | Mark patient seen |
| GET | `/api/v1/admin/rag-stats` | ChromaDB collection stats |

### Health
| Method | Endpoint | Description |
|---|---|---|
| GET/HEAD | `/health` | Service health check (supports uptime monitors) |

Full interactive docs at `/docs` when running locally.

---

## 📚 Knowledge Base

14 eye conditions ingested into ChromaDB with semantic chunking:

| File | Condition |
|---|---|
| `cataracts.md` | Cataracts |
| `chemical_injury.md` | Chemical Eye Injury |
| `conjunctivitis.md` | Conjunctivitis (Pink Eye) |
| `corneal_ulcer.md` | Corneal Ulcer |
| `diabetic_retinopathy.md` | Diabetic Retinopathy |
| `dry_eye_syndrome.md` | Dry Eye Syndrome |
| `glaucoma.md` | Glaucoma |
| `macular_degeneration.md` | Macular Degeneration |
| `onchocerciasis.md` | Onchocerciasis (River Blindness) |
| `pterygium.md` | Pterygium |
| `refractive_errors.md` | Refractive Errors |
| `retinal_detachment.md` | Retinal Detachment |
| `trachoma.md` | Trachoma |
| `uveitis.md` | Uveitis |

To re-ingest after adding new files:

```bash
python -m app.rag.ingest --reset
```

---

## 👤 Author

**Chinelo Nweke** — AI Engineer
Built for the ClearSight Eye Clinic, Kubwa, Abuja, Nigeria.

- App: https://clearsightclinic.online
- Staff: https://clearsightclinic.online/staff
- GitHub: https://github.com/Chinelonweke/clearsight

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.