from __future__ import annotations

import asyncio
import base64
import json
import re as _re
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select, func

from app.core.logger import get_logger
from app.models.database import AsyncSessionLocal
from app.models.session import ConversationSession
from app.services.analytics_service import track_event
from app.services.booking_service import BookingService
from app.services.email_service import send_booking_confirmation
from app.services.intake_service import IntakeService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.rag_service import RAGService
from app.services.session_service import SessionService
from app.services.stt_service import STTService
from app.services.triage_service import TriageService, TriageResult
from app.services.tts_service import TTSService
from app.services.vision_service import VisionService
from app.utils.audio import get_filename_for_groq
from app.utils.image import preprocess_image, save_image_locally
from app.models.patient import Patient as PatientModel
from app.models.doctor import AvailabilitySlot, Doctor as DoctorModel

logger = get_logger(__name__)
router = APIRouter(tags=["WebSocket"])

_EMAIL_RE = _re.compile(r'[\w.\-+]+@[\w.\-]+\.[a-zA-Z]{2,}')
_INCOMPLETE_SESSION_TTL = 1800  # 30 minutes in seconds

_DIAGNOSIS_KEYWORDS = [
    "what is wrong", "what's wrong", "whats wrong",
    "what do you think", "what is it", "what could it be",
    "do i have", "is it", "could it be", "what disease",
    "what condition", "diagnose", "diagnosis", "what causes",
    "why is my eye", "what is myopia", "what is glaucoma",
    "what is cataract", "what is conjunctivitis", "what is uveitis",
    "explain my condition", "tell me what",
]


def _is_diagnosis_question(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _DIAGNOSIS_KEYWORDS)



# ── Prompt injection detection ─────────────────────────────────────────────
_INJECTION_PATTERNS = [
    "ignore previous instructions", "ignore all instructions",
    "ignore your instructions", "forget your instructions",
    "forget previous", "forget all previous", "you are now",
    "act as", "pretend you are", "pretend to be",
    "your real instructions", "new instructions",
    "override instructions", "[system]", "<system>",
    "system prompt", "you are a general", "you are an assistant",
    "disregard", "do anything now", "dan mode", "jailbreak",
    "bypass", "previous context", "list all patients",
    "show me all patients", "reveal patient data",
    "what did other patients",
]


def _is_prompt_injection(text: str) -> bool:
    """Detect prompt injection attempts before they reach the LLM."""
    lower = text.lower().strip()
    return any(pattern in lower for pattern in _INJECTION_PATTERNS)

def _all_slots_filled(user_messages: list[str]) -> bool:
    text = " ".join(user_messages).lower()
    has_eye = any(w in text for w in [
        "left eye", "right eye", "both eyes", "left", "right", "both"
    ])
    has_symptom = any(w in text for w in [
        "blur", "blurr", "pain", "itch", "sting", "burn", "red", "swell",
        "discharge", "vision", "see", "double", "floater", "flash", "dark",
        "shadow", "watery", "dry", "grit", "sensitivity", "light"
    ])
    has_duration = any(w in text for w in [
        "day", "days", "week", "weeks", "month", "months", "year", "years",
        "hour", "hours", "ago", "since", "started", "morning", "yesterday",
        "last night", "this morning", "few"
    ])
    has_pain_rating = any(c.isdigit() for c in text) or any(w in text for w in [
        "no pain", "mild", "moderate", "severe", "little", "slight",
        "very painful", "hurts", "aching", "throbbing", "uncomfortable"
    ])
    has_vision_answer = any(w in text for w in [
        "vision", "blur", "blurr", "double", "clear", "see", "sight",
        "no change", "no vision", "fine", "normal", "same", "worse", "better",
        "yes", "no", "yeah", "nope", "none"
    ])
    filled = [has_eye, has_symptom, has_duration, has_pain_rating, has_vision_answer]
    logger.debug(
        f"Slot check | eye={has_eye} symptom={has_symptom} "
        f"duration={has_duration} pain={has_pain_rating} vision={has_vision_answer}"
    )
    return all(filled)


def _get_triage_prompt(memory_context: str = "") -> str:
    from app.config import settings
    base = f"""You are ClearSight, a warm, professional AI triage assistant at a Nigerian eye clinic.
You speak clearly and warmly. Your name is ClearSight.

CLINIC INFORMATION (use ONLY these exact details - never invent any):
- Clinic name: {settings.clinic_name}
- Clinic phone: {settings.clinic_phone}
- Clinic address: {settings.clinic_address}
- Opening hours: {settings.clinic_opening_hour}:00 to {settings.clinic_closing_hour}:00

YOUR CONVERSATION GOALS (follow this order strictly):
1. GREETING: Welcome the patient. Ask for their full name, phone number and email address.
2. SYMPTOM COLLECTION: Ask about their eye problem - one question at a time.
   Cover ALL of: which eye, what symptom, how long, pain level (0-10), family history,
   any vision changes, visited eye clinic before.
   Wait for each answer before asking the next question.
3. TRIAGE: After collecting ALL symptom fields (at least 6 patient responses), say:
   "Thank you for sharing that. Let me assess your situation."
   Only output [READY_FOR_TRIAGE] on its own line AFTER the patient has answered ALL questions.
   NEVER output [READY_FOR_TRIAGE] in the same message as a question.
   The vision changes question MUST be asked AND answered before triggering triage.
4. BOOKING: The system will book automatically. When given a slot time, confirm it to the patient.
   Do NOT suggest or mention any time before the system provides one.
   Do NOT change a confirmed booking.
5. CLOSING: Thank the patient and give care instructions appropriate to their urgency level.

CRITICAL RULES - NEVER BREAK THESE:
- NEVER diagnose the patient. Never say "you have X" or "this is X condition".
- If the patient asks what is wrong, what condition they have, or for any diagnosis,
  always say: "I'm not able to diagnose - only a qualified optometrist can do that after
  examining you in person. Your appointment is booked so the doctor will assess you properly."
- NEVER invent clinic information. Only use the CLINIC INFORMATION above.
- Keep each response under 40 words.
- Ask only ONE question per turn.
- Be empathetic - patients may be anxious.
- If patient mentions chemical in eye: "This is an emergency. Please wash your eye with
  water NOW and go to the nearest clinic immediately." Then output: [EMERGENCY_CHEMICAL]

LANGUAGE: English only.

RETURNING PATIENT RULE: If Patient Memory is provided above, you already have their name, phone, and email. Do NOT ask for these again. Skip straight to asking about their current eye concern today."""

    if memory_context:
        base += f"\n\n{memory_context}"
    return base


async def _is_returning_patient_db(patient_id: str) -> bool:
    """
    Check if patient is returning using NeonDB as source of truth.
    Senior dev pattern: ask YOUR OWN database, never an external service.
    """
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(func.count()).where(
                    ConversationSession.patient_id == patient_id
                )
            )
            session_count = result.scalar() or 0
            logger.debug(f"DB session count | patient_id={patient_id} | count={session_count}")
            return session_count >= 1
    except Exception as exc:
        logger.warning(f"DB returning-patient check failed (non-fatal): {exc}")
        return False


async def _save_incomplete_session(patient_id: str, history: list, metadata: dict) -> None:
    """
    Save incomplete session to Redis with 30-minute TTL.
    Called when patient disconnects before completing triage.
    """
    try:
        from app.db.redis_client import get_redis_client
        redis = get_redis_client()
        if not redis:
            return
        key = f"incomplete_session:{patient_id}"
        data = json.dumps({
            "history": history,
            "metadata": metadata,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        })
        await redis.setex(key, _INCOMPLETE_SESSION_TTL, data)
        logger.info(f"Incomplete session saved | patient_id={patient_id} | TTL=30min")
    except Exception as exc:
        logger.warning(f"Failed to save incomplete session (non-fatal): {exc}")


async def _load_incomplete_session(patient_id: str) -> dict | None:
    """
    Load incomplete session from Redis if it exists and is within 30 minutes.
    Returns None if no incomplete session found.
    """
    try:
        from app.db.redis_client import get_redis_client
        redis = get_redis_client()
        if not redis:
            return None
        key = f"incomplete_session:{patient_id}"
        data = await redis.get(key)
        if data:
            parsed = json.loads(data)
            logger.info(f"Incomplete session found | patient_id={patient_id}")
            return parsed
        return None
    except Exception as exc:
        logger.warning(f"Failed to load incomplete session (non-fatal): {exc}")
        return None


async def _delete_incomplete_session(patient_id: str) -> None:
    """Delete incomplete session after patient resumes or declines."""
    try:
        from app.db.redis_client import get_redis_client
        redis = get_redis_client()
        if redis:
            await redis.delete(f"incomplete_session:{patient_id}")
    except Exception as exc:
        logger.warning(f"Failed to delete incomplete session (non-fatal): {exc}")


async def _send(ws: WebSocket, msg: dict) -> None:
    try:
        await ws.send_json(msg)
    except Exception as exc:
        logger.debug(f"WebSocket send failed: {exc}")


async def _send_error(ws: WebSocket, message: str) -> None:
    await _send(ws, {"type": "error", "message": message})


async def _tts(tts_svc: TTSService, text: str) -> str:
    try:
        audio = await tts_svc.synthesize(text)
        return base64.b64encode(audio).decode() if audio else ""
    except Exception as exc:
        logger.warning(f"TTS synthesis failed: {exc}")
        return ""


def _get_session_service():
    try:
        from app.db.redis_client import get_redis_client
        redis = get_redis_client()
        return SessionService(redis)
    except Exception as exc:
        logger.warning(f"Redis unavailable - using in-memory session fallback: {exc}")
        return None


class _InMemorySession:
    def __init__(self):
        self._history: list[dict] = []
        self._meta: dict = {}
        self._symptoms: dict = {}

    async def create_session(self, **kwargs): pass
    async def session_exists(self, sid): return True
    async def close_session(self, sid, outcome=""): pass
    async def set_stage(self, sid, stage): self._meta["stage"] = stage

    async def append_message(self, sid, role, content):
        self._history.append({"role": role, "content": content})

    async def get_history(self, sid, include_timestamps=False):
        return [{"role": m["role"], "content": m["content"]} for m in self._history]

    async def get_history_text(self, sid):
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history
        )

    async def get_metadata(self, sid):
        return {**self._meta, "message_count": len(self._history)}

    async def update_metadata(self, sid, updates):
        self._meta.update(updates)

    async def get_symptoms(self, sid):
        return self._symptoms

    async def store_symptoms(self, sid, symptoms):
        self._symptoms.update(symptoms)


@router.websocket("/ws/conversation/{session_id}")
async def conversation_endpoint(ws: WebSocket, session_id: str, token: str = ""):
    await ws.accept()
    logger.info(f"WebSocket connected | session={session_id}")

    redis_svc = _get_session_service()
    session_svc = redis_svc if redis_svc is not None else _InMemorySession()

    llm_svc = LLMService()
    stt_svc = STTService()
    tts_svc = TTSService()
    vision_svc = VisionService()
    rag_svc = RAGService()
    memory_svc = MemoryService()
    triage_svc = TriageService(llm=llm_svc, rag=rag_svc)

    try:
        if not await session_svc.session_exists(session_id):
            await session_svc.create_session(session_id=session_id)
    except Exception as exc:
        logger.warning(f"Session initialization warning | session={session_id}: {exc}")

    await track_event("session_start", session_id=session_id)

    image_urls: list[str] = []
    triage_triggered = False
    waiting_for_resume_answer = False
    incomplete_session_data = None

    # ── Determine patient identity ─────────────────────────────────────────────
    patient_memory_context = ""
    patient_id_for_memory = None
    is_returning = False

    try:
        init_meta = await session_svc.get_metadata(session_id)
        patient_id_for_memory = init_meta.get("patient_id")

        if not patient_id_for_memory and token:
            try:
                from app.core.security import verify_token
                payload = verify_token(token, expected_type="access")
                if payload.get("role") == "patient":
                    patient_id_for_memory = payload.get("sub")
                    logger.info(f"Patient ID from WS token | patient_id={patient_id_for_memory}")
            except Exception:
                pass

        if patient_id_for_memory:
            # ✅ Step 1: Check for incomplete session (left mid-conversation)
            incomplete_session_data = await _load_incomplete_session(patient_id_for_memory)

            # ✅ Step 2: Check if returning patient via NeonDB (source of truth)
            is_returning = await _is_returning_patient_db(patient_id_for_memory)
            logger.info(
                f"Patient check | patient_id={patient_id_for_memory} "
                f"| is_returning={is_returning} "
                f"| has_incomplete={incomplete_session_data is not None}"
            )

            # ✅ Save session to NeonDB so returning patient check works next time
        if patient_id_for_memory:
            try:
                async with AsyncSessionLocal() as db:
                    import uuid as _uuid
                    new_session = ConversationSession(
                        id=_uuid.UUID(session_id) if len(session_id) == 36 else _uuid.uuid4(),
                        patient_id=_uuid.UUID(patient_id_for_memory),
                        outcome="in_progress",
                    )
                    db.add(new_session)
                    await db.commit()
                    logger.info(f"Session saved to DB | patient_id={patient_id_for_memory}")
            except Exception as exc:
                logger.warning(f"Failed to save session to DB (non-fatal): {exc}")

        if is_returning and not incomplete_session_data:
                # Returning patient with no incomplete session — load mem0 context
                patient_memory_context = await memory_svc.get_patient_context(
                    patient_id_for_memory
                )
                logger.info(f"Returning patient memory loaded | patient_id={patient_id_for_memory}")

    except Exception as exc:
        logger.warning(f"Memory load failed (non-fatal): {exc}")

    try:
        # -- Greeting ----------------------------------------------------------
        if incomplete_session_data:
            # Patient left mid-conversation — ask if they want to resume
            waiting_for_resume_answer = True
            greeting = (
                "Welcome back! It looks like we were in the middle of your eye assessment. "
                "Would you like to continue where we left off, or start a new assessment? "
                "Reply 'continue' or 'new'."
            )
        elif is_returning:
            # Returning patient who completed previous session
            greeting = (
                "Welcome back to ClearSight! I remember you from your previous visit. "
                "I'm here to help assess your eye concern today. "
                "Could you tell me your full name and what's bothering you?"
            )
        else:
            # Brand new patient
            greeting = (
                "Hello! I'm ClearSight, your eye clinic assistant. "
                "I'm here to help assess your eye concern and book you an appointment. "
                "May I have your full name, phone number and email address to get started?"
            )

        await session_svc.append_message(session_id, "assistant", greeting)
        await _send(ws, {
            "type": "response",
            "text": greeting,
            "audio": await _tts(tts_svc, greeting),
        })

        # -- Main loop ----------------------------------------------------------
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_json(), timeout=300)
            except (asyncio.TimeoutError, TimeoutError):
                await _send(ws, {"type": "error", "message": "Session timed out."})
                break

            msg_type = data.get("type")

            if msg_type == "ping":
                await _send(ws, {"type": "pong"})
                continue

            if msg_type == "end_session":
                break

            # -- Handle resume answer ------------------------------------------
            if waiting_for_resume_answer:
                user_input = data.get("text", "").strip().lower() if msg_type == "text" else ""
                await session_svc.append_message(session_id, "user", user_input)

                if any(w in user_input for w in ["continue", "yes", "yeah", "yep", "resume", "carry on"]):
                    # Restore previous history
                    waiting_for_resume_answer = False
                    prev_history = incomplete_session_data.get("history", [])
                    prev_meta = incomplete_session_data.get("metadata", {})

                    for msg in prev_history:
                        await session_svc.append_message(session_id, msg["role"], msg["content"])
                    await session_svc.update_metadata(session_id, prev_meta)
                    await _delete_incomplete_session(patient_id_for_memory)

                    # Find last assistant message to show context
                    last_ai_msg = next(
                        (m["content"] for m in reversed(prev_history) if m["role"] == "assistant"),
                        "Let me remind you where we were."
                    )
                    resume_msg = (
                        f"Great! Let's continue. I was last asking: \"{last_ai_msg}\""
                    )
                    await session_svc.append_message(session_id, "assistant", resume_msg)
                    await _send(ws, {
                        "type": "response",
                        "text": resume_msg,
                        "audio": await _tts(tts_svc, resume_msg),
                    })
                    logger.info(f"Session resumed | patient_id={patient_id_for_memory}")
                else:
                    # Start fresh
                    waiting_for_resume_answer = False
                    await _delete_incomplete_session(patient_id_for_memory)
                    fresh_greeting = (
                        "No problem! Let's start fresh. "
                        "May I have your full name, phone number and email address to get started?"
                    )
                    await session_svc.append_message(session_id, "assistant", fresh_greeting)
                    await _send(ws, {
                        "type": "response",
                        "text": fresh_greeting,
                        "audio": await _tts(tts_svc, fresh_greeting),
                    })
                    logger.info(f"Patient chose fresh start | patient_id={patient_id_for_memory}")
                continue

            # -- Image ----------------------------------------------------------
            if msg_type == "image":
                t_start = time.perf_counter()
                try:
                    image_bytes = base64.b64decode(data.get("image", ""))
                    processed_bytes, processed_mime = preprocess_image(image_bytes)
                    path = save_image_locally(processed_bytes, session_id)
                    image_urls.append(path)

                    symptoms_so_far = await session_svc.get_symptoms(session_id)
                    vision_result = await vision_svc.analyze_eye_image(
                        image_bytes=processed_bytes,
                        session_id=session_id,
                        mime_type=processed_mime,
                        additional_context=str(symptoms_so_far),
                    )
                    await session_svc.update_metadata(
                        session_id, {"vision_observation": vision_result["raw_observation"]}
                    )
                    await _send(ws, {"type": "vision_result", "data": vision_result})

                    vision_text = (
                        f"I can see the image of your eye. "
                        f"{vision_result['visible_features']} "
                        f"I'll include this in my assessment."
                    )
                    await session_svc.append_message(session_id, "assistant", vision_text)
                    await _send(ws, {
                        "type": "response",
                        "text": vision_text,
                        "audio": await _tts(tts_svc, vision_text),
                    })
                    duration_ms = int((time.perf_counter() - t_start) * 1000)
                    await track_event(
                        "vision_analysis", session_id=session_id,
                        duration_ms=duration_ms,
                        metadata={"urgency_flag": vision_result["urgency_flag"]}
                    )
                except Exception as exc:
                    logger.error(f"Image error | session={session_id}: {exc}")
                    await _send_error(
                        ws, "I had trouble with the image. Please describe your symptoms instead."
                    )
                continue

            # -- Audio / Text ---------------------------------------------------
            user_input = ""

            if msg_type == "audio":
                t_stt = time.perf_counter()
                try:
                    audio_bytes = base64.b64decode(data.get("audio", ""))
                    content_type = data.get("content_type", "audio/webm")
                    filename = get_filename_for_groq(audio_bytes, content_type)
                    user_input = await stt_svc.transcribe(
                        audio_bytes=audio_bytes,
                        filename=filename,
                        prompt="Eye clinic patient describing eye symptoms in Nigeria.",
                    )
                    stt_ms = int((time.perf_counter() - t_stt) * 1000)
                    await track_event("stt_complete", session_id=session_id, duration_ms=stt_ms)
                    await _send(ws, {"type": "transcript", "text": user_input})
                except Exception as exc:
                    logger.error(f"STT error | session={session_id}: {exc}")
                    await _send_error(ws, "I couldn't hear that. Please type your message.")
                    continue

            elif msg_type == "text":
                user_input = data.get("text", "").strip()

            if not user_input:
                continue

            await session_svc.append_message(session_id, "user", user_input)

            email_match = _EMAIL_RE.search(user_input)
            if email_match:
                await session_svc.update_metadata(
                    session_id, {"patient_email": email_match.group(0)}
                )
                logger.debug(f"Patient email captured | email={email_match.group(0)}")

            import re as _re2
            name_match = _re2.search(
                r'(?:full\s*name\s*[:\-]?\s*)([A-Za-z]+(?:\s+[A-Za-z]+)+)',
                user_input, _re2.IGNORECASE
            )
            if name_match:
                extracted_name = name_match.group(1).strip().title()
                await session_svc.update_metadata(
                    session_id, {"patient_name": extracted_name}
                )
                logger.debug(f"Patient name captured | name={extracted_name}")

            history = await session_svc.get_history(session_id)
            meta = await session_svc.get_metadata(session_id)
            turn_count = meta.get("message_count", 0)

            if _is_diagnosis_question(user_input):
                meta_stage = meta.get("stage", "")
                if meta_stage in ("done", "booking"):
                    no_diagnosis_msg = (
                        "I'm not able to provide a diagnosis - only a qualified optometrist "
                        "can do that after a proper in-person examination. "
                        "Your appointment is already booked. The doctor will assess and "
                        "explain your condition when you visit."
                    )
                else:
                    no_diagnosis_msg = (
                        "I'm not able to diagnose eye conditions - that requires a qualified "
                        "optometrist examining you in person. "
                        "I'm here to assess the urgency of your symptoms and book you an "
                        "appointment so the right doctor can help you."
                    )
                await session_svc.append_message(session_id, "assistant", no_diagnosis_msg)
                await _send(ws, {
                    "type": "response",
                    "text": no_diagnosis_msg,
                    "audio": await _tts(tts_svc, no_diagnosis_msg),
                })
                continue

            from app.services.triage_service import EMERGENCY_KEYWORDS
            if any(w in user_input.lower() for w in EMERGENCY_KEYWORDS):
                emergency_msg = (
                    "This sounds like a chemical eye emergency! "
                    "Please flush your eye with clean water IMMEDIATELY and continuously. "
                    "Go to the nearest hospital or clinic right now. Do not wait."
                )
                await session_svc.append_message(session_id, "assistant", emergency_msg)
                await _send(ws, {
                    "type": "response",
                    "text": emergency_msg,
                    "audio": await _tts(tts_svc, emergency_msg),
                    "is_emergency": True,
                })
                await track_event(
                    "triage_complete", session_id=session_id,
                    metadata={"urgency_level": "emergency", "urgency_score": 10,
                              "trigger": "chemical_emergency_keyword"}
                )
                continue

            # -- Prompt injection check ----------------------------------------
            if _is_prompt_injection(user_input):
                injection_msg = (
                    "I can only help with eye clinic triage and appointment booking. "
                    "Please describe your eye symptoms so I can assist you."
                )
                logger.warning(
                    f"Prompt injection attempt | session={session_id} "
                    f"| input={user_input[:100]}"
                )
                await session_svc.append_message(session_id, "assistant", injection_msg)
                await _send(ws, {
                    "type": "response",
                    "text": injection_msg,
                    "audio": await _tts(tts_svc, injection_msg),
                })
                continue

            try:
                context_chunks = await rag_svc.retrieve(user_input, top_k=2)
                reply_text = await llm_svc.chat_with_context(
                    system=_get_triage_prompt(memory_context=patient_memory_context if len(history) <= 2 else ""),
                    history=history,
                    context_chunks=context_chunks,
                )
                if not reply_text or not reply_text.strip():
                    reply_text = "I'm sorry, I didn't catch that. Could you please repeat that?"
            except Exception as exc:
                logger.error(f"LLM error | session={session_id}: {exc}")
                reply_text = "I'm having a technical difficulty. Could you please repeat that?"

            user_messages = [m["content"] for m in history if m["role"] == "user"]
            slots_ready = _all_slots_filled(user_messages)

            reply_text = reply_text.replace("READY_FOR_TRIAGE", "[READY_FOR_TRIAGE]")
            if "[READY_FOR_TRIAGE]" in reply_text and not slots_ready:
                logger.debug(f"Triage tag blocked - slots not filled | session={session_id}")
                reply_text = reply_text.replace("[READY_FOR_TRIAGE]", "").strip()
                triage_now = False
            else:
                triage_now = "[READY_FOR_TRIAGE]" in reply_text and slots_ready

            reply_clean = (
                reply_text
                .replace("[READY_FOR_TRIAGE]", "")
                .replace("[EMERGENCY_CHEMICAL]", "")
                .strip()
            )

            if reply_clean:
                await session_svc.append_message(session_id, "assistant", reply_clean)
                await _send(ws, {
                    "type": "response",
                    "text": reply_clean,
                    "audio": await _tts(tts_svc, reply_clean),
                })

            if triage_now and not triage_triggered:
                triage_triggered = True
                await session_svc.set_stage(session_id, "triage")

                thinking_msg = "Analysing your symptoms now, please hold for just a moment..."
                await session_svc.append_message(session_id, "assistant", thinking_msg)
                await _send(ws, {
                    "type": "response",
                    "text": thinking_msg,
                    "audio": await _tts(tts_svc, thinking_msg),
                })

                t_triage = time.perf_counter()
                try:
                    transcript = await session_svc.get_history_text(session_id)
                    session_meta = await session_svc.get_metadata(session_id)
                    vision_obs = session_meta.get("vision_observation")

                    triage_result = await triage_svc.assess(
                        symptoms_transcript=transcript,
                        patient_metadata={"session_id": session_id},
                        vision_observation=vision_obs,
                    )

                    triage_ms = int((time.perf_counter() - t_triage) * 1000)
                    await track_event(
                        "triage_complete", session_id=session_id,
                        duration_ms=triage_ms,
                        metadata=triage_result.to_dict()
                    )
                    await _send(ws, {"type": "triage_result", "data": triage_result.to_dict()})
                    await session_svc.set_stage(session_id, "booking")

                    appointment_id = None
                    slot_time = None
                    doctor_name = "Our Optometrist"
                    patient_obj_id = None

                    async with AsyncSessionLocal() as db:
                        booking_svc = BookingService(db)
                        patient_id = session_meta.get("patient_id")
                        patient = None

                        if patient and patient.email:
                            if not session_meta.get("patient_email"):
                               await session_svc.update_metadata(
                                   session_id, {"patient_email": patient.email}
                               )
                               session_meta["patient_email"] = patient.email

                        if not patient:
                            patient = PatientModel(
                                full_name=session_meta.get("patient_name", "Walk-in Patient"),
                                phone=session_meta.get("patient_phone"),
                            )
                            db.add(patient)
                            await db.flush()

                        patient_obj_id = str(patient.id)

                        appointment = await booking_svc.book_best_slot(
                            patient=patient,
                            triage=triage_result,
                        )

                        if appointment and appointment.slot_id:
                            slot_result = await db.execute(
                                select(AvailabilitySlot).where(
                                    AvailabilitySlot.id == appointment.slot_id
                                )
                            )
                            loaded_slot = slot_result.scalar_one_or_none()
                            if loaded_slot:
                                slot_time = loaded_slot.slot_start.strftime(
                                    "%A %d %B %Y at %I:%M %p"
                                )

                            if appointment.doctor_id:
                                dr_result = await db.execute(
                                    select(DoctorModel).where(
                                        DoctorModel.id == appointment.doctor_id
                                    )
                                )
                                dr = dr_result.scalar_one_or_none()
                                if dr:
                                    doctor_name = dr.full_name

                            appointment_id = str(appointment.id)

                        await db.commit()

                    if slot_time:
                        patient_email = session_meta.get("patient_email", "")
                        asyncio.create_task(send_booking_confirmation(
                            patient_email=patient_email,
                            patient_name=session_meta.get("patient_name", "Patient"),
                            slot_time=slot_time,
                            urgency_level=triage_result.urgency_level,
                            urgency_score=triage_result.urgency_score,
                            chief_complaint=triage_result.chief_complaint,
                            patient_instruction=triage_result.patient_instruction,
                            doctor_name=doctor_name,
                        ))

                    if slot_time:
                        if triage_result.is_routine:
                            booking_msg = (
                                f"Assessment complete - urgency is ROUTINE "
                                f"({triage_result.urgency_score}/10). "
                                f"I've provisionally booked you for {slot_time} "
                                f"with {doctor_name}. "
                                f"Call us to reschedule if needed. "
                                f"{triage_result.patient_instruction}"
                            )
                        else:
                            booking_msg = (
                                f"Assessment complete - urgency is "
                                f"{triage_result.urgency_level.upper()}. "
                                f"I've booked your appointment for {slot_time} "
                                f"with {doctor_name}. "
                                f"{triage_result.patient_instruction}"
                            )

                        patient_email = session_meta.get("patient_email", "")
                        if patient_email:
                            booking_msg += (
                                f" A confirmation email has been sent to {patient_email}."
                            )

                        await track_event(
                            "booking_created", session_id=session_id,
                            metadata={
                                "appointment_id": appointment_id,
                                "urgency_level": triage_result.urgency_level,
                                "slot_time": slot_time,
                            }
                        )
                        await _send(ws, {
                            "type": "booking_result",
                            "data": {
                                "appointment_id": appointment_id,
                                "slot_time": slot_time,
                                "urgency_level": triage_result.urgency_level,
                                "urgency_score": triage_result.urgency_score,
                                "patient_instruction": triage_result.patient_instruction,
                            }
                        })
                        logger.success(
                            f"Booking confirmed | session={session_id} "
                            f"| appt={appointment_id} | slot={slot_time} "
                            f"| doctor={doctor_name}"
                        )
                        await session_svc.set_stage(session_id, "done")

                    else:
                        from app.config import settings as _s
                        booking_msg = (
                            f"Your urgency is {triage_result.urgency_level.upper()}. "
                            f"No slots are currently available online. "
                            f"Please call us directly on {_s.clinic_phone} to book."
                        )
                        await track_event("booking_failed", session_id=session_id)
                        logger.warning(f"No slots available | session={session_id}")

                    await session_svc.append_message(session_id, "assistant", booking_msg)
                    await _send(ws, {
                        "type": "response",
                        "text": booking_msg,
                        "audio": await _tts(tts_svc, booking_msg),
                    })

                    try:
                        async with AsyncSessionLocal() as db2:
                            intake_svc = IntakeService(llm=llm_svc, db=db2)
                            transcript_full = await session_svc.get_history_text(session_id)
                            await intake_svc.create_intake_form(
                                session_id=session_id,
                                patient_id=patient_obj_id,
                                transcript=transcript_full,
                                image_urls=image_urls,
                            )
                            await db2.commit()
                        await track_event("intake_complete", session_id=session_id)
                    except Exception as intake_exc:
                        logger.warning(f"Intake form error (non-fatal): {intake_exc}")

                    if patient_obj_id:
                        try:
                            transcript_full = await session_svc.get_history_text(session_id)
                            await memory_svc.save_session_memories(
                                patient_id=patient_id_for_memory or patient_obj_id,
                                patient_name=session_meta.get("patient_name", "Patient"),
                                transcript=transcript_full,
                                triage_result=triage_result,
                                session_metadata=session_meta,
                            )
                            logger.info(f"Session memories saved | patient_id={patient_obj_id}")
                        except Exception as mem_exc:
                            logger.warning(f"Memory save failed (non-fatal): {mem_exc}")

                except Exception as exc:
                    logger.error(f"Triage/booking error | session={session_id}: {exc}")
                    await _send_error(
                        ws,
                        "I had a technical issue completing the assessment. "
                        "Please call the clinic directly."
                    )

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected | session={session_id}")

    except Exception as exc:
        logger.exception(f"Unexpected WebSocket error | session={session_id}: {exc}")

    finally:
        # ── Save incomplete session if triage was never completed ──────────────
        if patient_id_for_memory and not triage_triggered:
            try:
                history = await session_svc.get_history(session_id)
                meta = await session_svc.get_metadata(session_id)
                # Only save if patient actually said something (more than just the greeting)
                user_msg_count = sum(1 for m in history if m["role"] == "user")
                if user_msg_count > 0:
                    await _save_incomplete_session(patient_id_for_memory, history, meta)
                    # Save partial memories to mem0 so returning patient context is preserved
                    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
                    await memory_svc.save_session_memories(
                        patient_id=patient_id_for_memory,
                        patient_name=meta.get("patient_name", "Patient"),
                        transcript=transcript,
                        is_partial=True,
                    )
                    logger.info(
                        f"Incomplete session saved on disconnect | "
                        f"patient_id={patient_id_for_memory} | messages={user_msg_count}"
                    )
            except Exception as exc:
                logger.warning(f"Failed to save incomplete session on disconnect: {exc}")

        try:
            await session_svc.close_session(session_id, outcome="completed")
        except Exception as exc:
            logger.debug(f"Session close failed (non-fatal): {exc}")
        await track_event("session_end", session_id=session_id)
        await _send(ws, {"type": "session_closed", "outcome": "completed"})
        logger.info(f"WebSocket session finalised | session={session_id}")
