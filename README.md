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
| 📧 **Email Confirmations** | Booking confirmation emails via Resend |
| 🏥 **Staff Dashboard** | Doctor/admin login, workload view, Mark Seen button |
| 📊 **Analytics Dashboard** | Sessions, bookings, error rates, urgency breakdown |
| 🔒 **JWT Authentication** | Separate doctor and admin login flows |

---

## 🏗 Architecture

```
Patient Browser
      │
      │ WebSocket (voice/text/image)
      ▼
┌─────────────────────────────────────┐
│         FastAPI Application          │
│                                      │
│  WebSocket ──► LLM Service (Groq)   │
│              ──► STT Service (Whisper│
│              ──► TTS Service (Piper) │
│              ──► Vision (LLaVA)      │
│              ──► Triage Service      │
│              ──► Booking Service     │
│              ──► Email Service       │
│                       │              │
└───────────────────────┼──────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   NeonDB (PG)    Upstash Redis    ChromaDB
   Appointments   Sessions         RAG Vectors
   Patients       Chat History     Eye Conditions
   Doctors        Metadata         (14 files)
   Slots
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI (async) |
| **LLM** | Groq — LLaMA 3.3 70B Versatile |
| **STT** | Groq — Whisper Large v3 |
| **Vision** | Groq — LLaVA v1.5 7B |
| **TTS** | Piper TTS (en_US-lessac-medium) |
| **Database** | NeonDB (PostgreSQL + asyncpg) |
| **Sessions** | Upstash Redis |
| **Vector Store** | ChromaDB 0.5.23 |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Email** | Resend SDK |
| **Auth** | JWT (python-jose) + bcrypt |
| **Logging** | Loguru (color-coded) |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions → AWS ECR → EC2 |

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
│   │   └── patient.py            # Patient model
│   ├── rag/
│   │   ├── chroma_client.py      # ChromaDB client + helpers
│   │   ├── chunking.py           # Semantic chunking strategy
│   │   └── ingest.py             # Knowledge base ingestion script
│   ├── services/
│   │   ├── analytics_service.py  # Event tracking
│   │   ├── booking_service.py    # Slot selection + booking engine
│   │   ├── email_service.py      # Resend email notifications
│   │   ├── intake_service.py     # Intake form auto-fill
│   │   ├── llm_service.py        # Groq LLM wrapper
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
│   ├── Dockerfile                # Multi-stage production image
│   └── docker-compose.yml        # Local dev: app + redis + chromadb
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD: test → build → push → deploy
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
- A [Resend](https://resend.com) account (free)
- Piper TTS model file (`en_US-lessac-medium.onnx`)

---

## 🚀 Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/clearsight.git
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
# or if using pyproject.toml:
pip install -e ".[dev]"
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

### 7. Seed doctors and availability slots

```sql
-- Run in NeonDB SQL Editor

```



### 8. Ingest the knowledge base

```bash
python -m app.rag.ingest
```

### 9. Start the server

```bash
uvicorn app.main:app --port 8000
```

Visit:
- Patient UI: http://localhost:8000
- Staff Dashboard: http://localhost:8000/staff
- API Docs: http://localhost:8000/docs
- Analytics: http://localhost:8000/api/v1/admin/dashboard

---

## 🔐 Environment Variables

Create a `.env` file from `.env.example`:

```env
# ── Application ──────────────────────────────────────────
SECRET_KEY=your-secret-key-min-32-chars
ADMIN_USERNAME=Nelo
ADMIN_PASSWORD=your-admin-password

# ── Groq API ─────────────────────────────────────────────
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
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
CHROMA_COLLECTION_NAME=clearsight_eye_conditions

# ── Piper TTS ─────────────────────────────────────────────
PIPER_MODEL_PATH=data/tts_models/en_US-lessac-medium.onnx

# ── Email (Resend) ────────────────────────────────────────
RESEND_API_KEY=re_xxxxxxxxxxxxxxxxxxxx
RESEND_FROM_EMAIL=onboarding@resend.dev

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
- `patients` — patient records
- `doctors` — doctor profiles + login credentials
- `availability_slots` — 30-minute appointment slots
- `appointments` — booked appointments with triage data
- `intake_forms` — auto-filled patient intake forms
- `analytics_events` — session + triage + booking events

---

## 🐳 Docker Deployment

### Run with Docker Compose (local)

```bash
# From project root
docker-compose -f docker/docker-compose.yml up --build
```

This starts three containers:
- `clearsight_api` — FastAPI on port 8000
- `clearsight_chroma` — ChromaDB on port 8001
- `clearsight_redis` — Redis on port 6379 (local only — use Upstash in production)

### Build image only

```bash
docker build -f docker/Dockerfile -t clearsight:latest .
```

---

## ☁️ AWS EC2 Deployment

### Prerequisites
- AWS account with EC2 access
- AWS CLI installed and configured
- Docker Hub or AWS ECR account

### Step 1 — Launch EC2 instance

1. Go to AWS Console → EC2 → Launch Instance
2. Choose **Ubuntu 22.04 LTS**
3. Instance type: **t3.small** (2GB RAM, free tier eligible with credits)
4. Create or select a key pair (save the `.pem` file)
5. Security group — open these ports:
   - 22 (SSH)
   - 8000 (App)
   - 8001 (ChromaDB — internal only in production)
6. Storage: **20GB gp3**
7. Launch

### Step 2 — Connect and install Docker

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Step 3 — Deploy the app

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/clearsight.git
cd clearsight

# Create .env with production values
nano .env

# Start services
docker-compose -f docker/docker-compose.yml up -d

# Ingest knowledge base
docker exec clearsight_api python -m app.rag.ingest
```

### Step 4 — Access your app

```
http://YOUR_EC2_PUBLIC_IP:8000        ← Patient UI
http://YOUR_EC2_PUBLIC_IP:8000/staff  ← Staff Dashboard
```

---

## ⚙️ CI/CD Pipeline

ClearSight uses **GitHub Actions** for automated deployment to AWS EC2.

### How it works

```
git push → GitHub Actions → Build Docker image
                          → Push to AWS ECR
                          → SSH into EC2
                          → Pull new image
                          → Restart containers
                          → Health check
```

### Setup

**1 — Add GitHub Secrets** (Settings → Secrets → Actions):

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | e.g. `clearsight` |
| `EC2_HOST` | Your EC2 public IP |
| `EC2_SSH_KEY` | Contents of your `.pem` file |
| `EC2_USER` | `ubuntu` |

**2 — Create AWS ECR repository:**

```bash
aws ecr create-repository --repository-name clearsight --region us-east-1
```

**3 — Create IAM user with permissions:**
- `AmazonEC2ContainerRegistryFullAccess`
- `AmazonEC2FullAccess`

The workflow file is at `.github/workflows/deploy.yml` and triggers automatically on every push to `main`.

---

## 🏥 Staff Dashboard

Access at `/staff`. Two login roles:

| Role | Username | Password | Access |
|---|---|---|---|
| Doctor | `dr.ijeoma` | `xxxxxxxx` | Own patients only |
| Doctor | `dr.adaeze` | `xxxxxxx` | Own patients only |
| Admin | `Nelo` | (from .env) | All patients + workload view |

**Features:**
- Today's appointments sorted by slot time
- Urgency badges (Emergency / Urgent / Routine)
- Chief complaint per patient
- Assigned doctor column (admin only)
- Doctor workload comparison bars (admin only)
- Mark Seen button (doctor sees only their patients)
- Filter by urgency level
- Auto-refresh every 60 seconds

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
| GET | `/health` | Service health check |

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
App link: https://clearsightclinic.online/
Staff link: https://clearsightclinic.online/staff

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.