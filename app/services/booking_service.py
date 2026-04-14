from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BookingNotFoundError,
    DoctorNotFoundError,
    SlotNotAvailableError,
)
from app.core.logger import get_logger
from app.models.booking import Appointment
from app.models.doctor import AvailabilitySlot, Doctor
from app.models.patient import Patient
from app.services.triage_service import TriageResult

logger = get_logger(__name__)


def _now_utc() -> datetime:
    """Current UTC time, always timezone-aware."""
    return datetime.now(timezone.utc)


class BookingService:

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_available_slots(
        self,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        doctor_id: UUID | None = None,
        limit: int = 20,
    ) -> list[AvailabilitySlot]:
        now = _now_utc()
        # Always start strictly AFTER current time to never book past slots
        from_dt = max(from_dt or now, now)
        to_dt = to_dt or (now + timedelta(days=7))

        query = (
            select(AvailabilitySlot)
            .where(
                and_(
                    AvailabilitySlot.is_booked == False,  # noqa: E712
                    AvailabilitySlot.slot_start > now,    # strictly future
                    AvailabilitySlot.slot_start >= from_dt,
                    AvailabilitySlot.slot_start <= to_dt,
                )
            )
            .order_by(AvailabilitySlot.slot_start)
            .limit(limit)
        )

        if doctor_id:
            query = query.where(AvailabilitySlot.doctor_id == doctor_id)

        result = await self._db.execute(query)
        slots = result.scalars().all()

        logger.debug(
            f"Slot query | from={from_dt.isoformat()[:16]} "
            f"| to={to_dt.isoformat()[:16]} | found={len(slots)}"
        )
        return list(slots)

    async def get_slots_for_urgency(
        self, triage: TriageResult
    ) -> list[AvailabilitySlot]:
        """
        EMERGENCY → next 4 hours
        URGENT    → rest of today (after NOW), or first thing tomorrow if after 17:00
        ROUTINE   → returns empty list intentionally — caller should ask patient
        """
        now = _now_utc()

        if triage.is_emergency:
            to_dt = now + timedelta(hours=4)
            logger.info("Slot search: EMERGENCY — next 4 hours")
            return await self.get_available_slots(from_dt=now, to_dt=to_dt)

        elif triage.is_urgent:
            # End of today at 18:00 UTC
            today_close = now.replace(hour=17, minute=0, second=0, microsecond=0)
            if today_close <= now:
                # After closing — look at tomorrow
                tomorrow = now + timedelta(days=1)
                from_dt = tomorrow.replace(hour=8, minute=0, second=0, microsecond=0)
                to_dt = tomorrow.replace(hour=17, minute=0, second=0, microsecond=0)
                logger.info("Slot search: URGENT — tomorrow (after closing hours)")
            else:
                from_dt = now
                to_dt = today_close
                logger.info("Slot search: URGENT — rest of today")
            return await self.get_available_slots(from_dt=from_dt, to_dt=to_dt)

        else:
            # ROUTINE — do NOT auto-book. Return empty so caller asks patient.
            logger.info("Slot search: ROUTINE — deferring to patient choice")
            return []

    async def get_slots_for_preferred_time(
        self,
        preferred_dt: datetime,
        window_hours: int = 2,
    ) -> list[AvailabilitySlot]:
        """
        Find slots within ±window_hours of the patient's preferred time.
        Used for routine bookings after patient states their preference.
        """
        from_dt = preferred_dt - timedelta(hours=window_hours)
        to_dt = preferred_dt + timedelta(hours=window_hours)
        # Never look in the past
        from_dt = max(from_dt, _now_utc())
        return await self.get_available_slots(from_dt=from_dt, to_dt=to_dt)

    async def book_appointment(
        self,
        patient: Patient,
        slot: AvailabilitySlot,
        triage: TriageResult,
        notes: str | None = None,
    ) -> Appointment:
        logger.info(
            f"Booking | patient={patient.id} | slot={slot.id} "
            f"| urgency={triage.urgency_level} "
            f"| slot_time={slot.slot_start.isoformat()[:16]}"
        )

        # Double-check slot is still in the future
        if slot.slot_start <= _now_utc():
            logger.warning(f"Slot {slot.id} is in the past — skipping.")
            raise SlotNotAvailableError(slot_id=str(slot.id))

        # Atomic lock — prevent race condition
        stmt = (
            update(AvailabilitySlot)
            .where(
                and_(
                    AvailabilitySlot.id == slot.id,
                    AvailabilitySlot.is_booked == False,  # noqa: E712
                )
            )
            .values(is_booked=True)
            .returning(AvailabilitySlot.id)
        )
        result = await self._db.execute(stmt)
        updated_id = result.scalar_one_or_none()

        if updated_id is None:
            logger.warning(f"Slot {slot.id} was already booked.")
            raise SlotNotAvailableError(slot_id=str(slot.id))

        appointment = Appointment(
            patient_id=patient.id,
            doctor_id=slot.doctor_id,
            slot_id=slot.id,
            urgency_level=triage.urgency_level,
            urgency_score=triage.urgency_score,
            chief_complaint=triage.chief_complaint,
            notes=notes or triage.triage_reasoning,
            status="scheduled",
            metadata_={
                "suspected_conditions": triage.suspected_conditions,
                "red_flags": triage.red_flags_detected,
                "recommended_timeframe": triage.recommended_timeframe,
            },
        )
        self._db.add(appointment)
        await self._db.flush()

        logger.success(
            f"Appointment created | id={appointment.id} | "
            f"slot={slot.slot_start.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"urgency={triage.urgency_level}"
        )
        return appointment

    async def book_best_slot(
        self,
        patient: Patient,
        triage: TriageResult,
        notes: str | None = None,
    ) -> Appointment | None:
        """
        For EMERGENCY and URGENT: auto-book the earliest future slot.
        For ROUTINE: returns None — the caller should ask the patient for
        their preferred time and call book_appointment() directly.
        """
        slots = await self.get_slots_for_urgency(triage)

        if not slots and (triage.is_emergency or triage.is_urgent):
            logger.warning(
                f"No slots in urgency window for {triage.urgency_level}. "
                f"Falling back to next 14-day window."
            )
            slots = await self.get_available_slots(
                to_dt=_now_utc() + timedelta(days=14)
            )

        if not slots:
            if triage.is_emergency or triage.is_urgent:
                logger.error("No future slots available.")
            return None

        best_slot = slots[0]
        try:
            return await self.book_appointment(patient, best_slot, triage, notes)
        except SlotNotAvailableError:
            # Slot was just taken — try next one
            if len(slots) > 1:
                return await self.book_appointment(patient, slots[1], triage, notes)
            return None

    async def cancel_appointment(self, appointment_id: UUID) -> Appointment:
        result = await self._db.execute(
            select(Appointment).where(Appointment.id == appointment_id)
        )
        appt = result.scalar_one_or_none()
        if not appt:
            raise BookingNotFoundError(booking_id=appointment_id)

        appt.status = "cancelled"

        if appt.slot_id:
            await self._db.execute(
                update(AvailabilitySlot)
                .where(AvailabilitySlot.id == appt.slot_id)
                .values(is_booked=False)
            )

        logger.info(f"Appointment cancelled | id={appointment_id}")
        return appt

    async def mark_appointment_seen(self, appointment_id: UUID) -> Appointment:
        """Mark an appointment as completed (patient was seen)."""
        result = await self._db.execute(
            select(Appointment).where(Appointment.id == appointment_id)
        )
        appt = result.scalar_one_or_none()
        if not appt:
            raise BookingNotFoundError(booking_id=appointment_id)

        appt.status = "completed"
        logger.info(f"Appointment marked seen | id={appointment_id}")
        return appt

    async def seed_slots(
        self,
        doctor_id: UUID,
        days_ahead: int = 14,
        slot_duration_minutes: int = 30,
        start_hour: int = 8,
        end_hour: int = 18,
    ) -> int:
        result = await self._db.execute(select(Doctor).where(Doctor.id == doctor_id))
        doctor = result.scalar_one_or_none()
        if not doctor:
            raise DoctorNotFoundError(doctor_id=doctor_id)

        now = _now_utc().replace(minute=0, second=0, microsecond=0)
        count = 0

        for day_offset in range(days_ahead):
            day = now + timedelta(days=day_offset + 1)
            if day.weekday() in (5, 6):
                continue

            current = day.replace(hour=start_hour, minute=0)
            end = day.replace(hour=end_hour, minute=0)

            while current < end:
                slot_end = current + timedelta(minutes=slot_duration_minutes)
                slot = AvailabilitySlot(
                    doctor_id=doctor_id,
                    slot_start=current,
                    slot_end=slot_end,
                    is_booked=False,
                )
                self._db.add(slot)
                current = slot_end
                count += 1

        await self._db.flush()
        logger.success(f"Slots seeded | doctor={doctor_id} | slots={count}")
        return count