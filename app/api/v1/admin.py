from __future__ import annotations
"""
app/api/v1/admin.py
─────────────────────────────────────────────────────────
Admin and staff API endpoints.
"""

import asyncio
import uuid
from datetime import date, datetime, time, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import and_, cast, func, select, String, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from app.dependencies import require_admin, db_session
from app.models.booking import Appointment
from app.models.doctor import AvailabilitySlot, Doctor
from app.models.intake import IntakeForm
from app.models.patient import Patient
from app.rag.chroma_client import get_collection_stats
from app.services.analytics_service import get_dashboard_stats
from app.services.booking_service import BookingService

logger = get_logger(__name__)
router = APIRouter()


# ── Metrics & RAG ─────────────────────────────────────────────────────────────

@router.get("/metrics")
async def get_metrics(_: dict = Depends(require_admin)):
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None, get_dashboard_stats)
    return stats


@router.get("/rag-stats")
async def rag_stats(_: dict = Depends(require_admin)):
    stats = await get_collection_stats()
    return stats


# ── Enhanced Analytics ────────────────────────────────────────────────────────

@router.get("/analytics")
async def get_analytics(
    current_user=Depends(require_admin),
    db: AsyncSession = Depends(db_session),
):
    """
    Comprehensive analytics for clinic dashboard:
    - Top eye conditions
    - Gender distribution
    - Age range breakdown
    - Urgency trends (7 days)
    - Doctor workload
    - Peak hours
    - Slot utilization
    """
    analytics = {}

    # ── 1. Top eye conditions from chief complaints ────────────────────────────
    try:
        conditions_result = await db.execute(
            select(
                Appointment.chief_complaint,
                func.count(Appointment.id).label("count")
            )
            .where(Appointment.chief_complaint.isnot(None))
            .group_by(Appointment.chief_complaint)
            .order_by(func.count(Appointment.id).desc())
            .limit(8)
        )
        top_conditions = [
            {"condition": row.chief_complaint, "count": row.count}
            for row in conditions_result.all()
        ]
        analytics["top_conditions"] = top_conditions
    except Exception as e:
        logger.warning(f"Top conditions query failed: {e}")
        analytics["top_conditions"] = []

    # ── 2. Urgency distribution ────────────────────────────────────────────────
    try:
        urgency_result = await db.execute(
            select(
                Appointment.urgency_level,
                func.count(Appointment.id).label("count")
            )
            .where(Appointment.urgency_level.isnot(None))
            .group_by(Appointment.urgency_level)
        )
        urgency_dist = {
            row.urgency_level: row.count
            for row in urgency_result.all()
        }
        analytics["urgency_distribution"] = {
            "emergency": urgency_dist.get("emergency", 0),
            "urgent": urgency_dist.get("urgent", 0),
            "routine": urgency_dist.get("routine", 0),
        }
    except Exception as e:
        logger.warning(f"Urgency distribution query failed: {e}")
        analytics["urgency_distribution"] = {"emergency": 0, "urgent": 0, "routine": 0}

    # ── 3. Urgency trend over last 7 days ─────────────────────────────────────
    try:
        seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
        trend_result = await db.execute(
            select(
                func.date(Appointment.created_at).label("appt_date"),
                Appointment.urgency_level,
                func.count(Appointment.id).label("count")
            )
            .where(
                and_(
                    Appointment.created_at >= seven_days_ago,
                    Appointment.urgency_level.isnot(None)
                )
            )
            .group_by(func.date(Appointment.created_at), Appointment.urgency_level)
            .order_by(func.date(Appointment.created_at))
        )
        trend_rows = trend_result.all()
        trend_data = {}
        for row in trend_rows:
            d = str(row.appt_date)
            if d not in trend_data:
                trend_data[d] = {"emergency": 0, "urgent": 0, "routine": 0}
            trend_data[d][row.urgency_level] = row.count
        analytics["urgency_trend"] = trend_data
    except Exception as e:
        logger.warning(f"Urgency trend query failed: {e}")
        analytics["urgency_trend"] = {}

    # ── 4. Doctor workload ────────────────────────────────────────────────────
    try:
        workload_result = await db.execute(
            select(
                Doctor.full_name,
                func.count(Appointment.id).label("total"),
                func.sum(
                    func.cast(Appointment.status == "completed", Integer := None)
                ).label("completed")
            )
            .join(Appointment, Doctor.id == Appointment.doctor_id, isouter=True)
            .where(Doctor.is_active == True)
            .group_by(Doctor.id, Doctor.full_name)
        )
        workload = []
        for row in workload_result.all():
            workload.append({
                "doctor": row.full_name,
                "total": row.total or 0,
                "completed": row.completed or 0,
            })
        analytics["doctor_workload"] = workload
    except Exception as e:
        logger.warning(f"Doctor workload query failed: {e}")
        # Fallback simpler query
        try:
            workload_result2 = await db.execute(
                select(
                    Doctor.full_name,
                    func.count(Appointment.id).label("total")
                )
                .join(Appointment, Doctor.id == Appointment.doctor_id, isouter=True)
                .where(Doctor.is_active == True)
                .group_by(Doctor.id, Doctor.full_name)
            )
            analytics["doctor_workload"] = [
                {"doctor": row.full_name, "total": row.total or 0, "completed": 0}
                for row in workload_result2.all()
            ]
        except Exception:
            analytics["doctor_workload"] = []

    # ── 5. Peak booking hours ─────────────────────────────────────────────────
    try:
        hours_result = await db.execute(
            text("""
                SELECT 
                    EXTRACT(HOUR FROM s.slot_start AT TIME ZONE 'Africa/Lagos') as hour,
                    COUNT(*) as count
                FROM appointments a
                JOIN availability_slots s ON a.slot_id = s.id
                GROUP BY hour
                ORDER BY hour
            """)
        )
        peak_hours = {
            int(row.hour): row.count
            for row in hours_result.all()
        }
        analytics["peak_hours"] = peak_hours
    except Exception as e:
        logger.warning(f"Peak hours query failed: {e}")
        analytics["peak_hours"] = {}

    # ── 6. Slot utilization rate ──────────────────────────────────────────────
    try:
        slot_result = await db.execute(
            select(
                func.count(AvailabilitySlot.id).label("total"),
                func.sum(
                    func.cast(AvailabilitySlot.is_booked == True, type_=None)
                ).label("booked")
            )
        )
        slot_row = slot_result.one()
        total_slots = slot_row.total or 0
        booked_slots = slot_row.booked or 0
        analytics["slot_utilization"] = {
            "total": total_slots,
            "booked": booked_slots,
            "available": total_slots - booked_slots,
            "rate": round((booked_slots / total_slots * 100), 1) if total_slots > 0 else 0,
        }
    except Exception as e:
        logger.warning(f"Slot utilization query failed: {e}")
        analytics["slot_utilization"] = {"total": 0, "booked": 0, "available": 0, "rate": 0}

    # ── 7. Total patients & appointments ──────────────────────────────────────
    try:
        patient_count = await db.execute(select(func.count(Patient.id)))
        appt_count = await db.execute(select(func.count(Appointment.id)))
        today_count = await db.execute(
            select(func.count(Appointment.id))
            .join(AvailabilitySlot, Appointment.slot_id == AvailabilitySlot.id)
            .where(
                func.date(AvailabilitySlot.slot_start) == date.today()
            )
        )
        analytics["totals"] = {
            "patients": patient_count.scalar() or 0,
            "appointments": appt_count.scalar() or 0,
            "today": today_count.scalar() or 0,
        }
    except Exception as e:
        logger.warning(f"Totals query failed: {e}")
        analytics["totals"] = {"patients": 0, "appointments": 0, "today": 0}

    # ── 8. Average urgency score ──────────────────────────────────────────────
    try:
        avg_result = await db.execute(
            select(func.avg(Appointment.urgency_score))
            .where(Appointment.urgency_score.isnot(None))
        )
        avg_score = avg_result.scalar()
        analytics["avg_urgency_score"] = round(float(avg_score), 1) if avg_score else 0
    except Exception as e:
        analytics["avg_urgency_score"] = 0

    # ── 9. Eye affected distribution ─────────────────────────────────────────
    try:
        eye_result = await db.execute(
            select(
                IntakeForm.eye_affected,
                func.count(IntakeForm.id).label("count")
            )
            .where(IntakeForm.eye_affected.isnot(None))
            .group_by(IntakeForm.eye_affected)
        )
        analytics["eye_affected"] = {
            row.eye_affected: row.count
            for row in eye_result.all()
        }
    except Exception as e:
        analytics["eye_affected"] = {}

    # ── 10. Completion rate ───────────────────────────────────────────────────
    try:
        completed_result = await db.execute(
            select(func.count(Appointment.id))
            .where(Appointment.status == "completed")
        )
        total_appts = analytics["totals"]["appointments"]
        completed = completed_result.scalar() or 0
        analytics["completion_rate"] = round(
            (completed / total_appts * 100), 1
        ) if total_appts > 0 else 0
    except Exception as e:
        analytics["completion_rate"] = 0

    return analytics


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(_: dict = Depends(require_admin)):
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None, get_dashboard_stats)
    urgency = stats.get("urgency_breakdown", {})
    hourly = stats.get("hourly_sessions", {})
    recent = stats.get("recent_events", [])

    recent_rows = "".join([
        f"""<tr>
          <td>{e.get('created_at','')[:19]}</td>
          <td><code>{e.get('event_type','')}</code></td>
          <td>{e.get('session_id','')[:8] if e.get('session_id') else '—'}</td>
          <td class="{'text-success' if e.get('success') else 'text-danger'}">
            {'✔' if e.get('success') else '✗'}
          </td>
        </tr>"""
        for e in recent
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ClearSight — Analytics</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;padding:24px}}
    h1{{font-size:22px;font-weight:600;color:#63b3ed;margin-bottom:4px}}
    .subtitle{{color:#718096;font-size:13px;margin-bottom:28px}}
    .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}}
    .card{{background:#1a1d27;border:1px solid #2d3748;border-radius:10px;padding:20px}}
    .card .value{{font-size:32px;font-weight:700;color:#68d391;line-height:1}}
    .card .label{{font-size:12px;color:#718096;margin-top:6px;text-transform:uppercase;letter-spacing:.5px}}
    .card.warn .value{{color:#f6ad55}}
    .card.danger .value{{color:#fc8181}}
    .charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px}}
    .chart-card{{background:#1a1d27;border:1px solid #2d3748;border-radius:10px;padding:20px}}
    .chart-card h3{{font-size:14px;font-weight:500;color:#a0aec0;margin-bottom:16px}}
    canvas{{max-height:200px}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    th{{text-align:left;padding:8px 12px;color:#718096;font-weight:500;border-bottom:1px solid #2d3748;text-transform:uppercase;font-size:11px}}
    td{{padding:8px 12px;border-bottom:1px solid #1e2533;color:#cbd5e0}}
    code{{background:#2d3748;padding:2px 6px;border-radius:4px;font-size:11px;color:#68d391}}
    .text-success{{color:#68d391}}.text-danger{{color:#fc8181}}
    @media(max-width:640px){{.charts{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
  <h1>ClearSight Analytics</h1>
  <p class="subtitle">Generated {stats.get('generated_at','')} &nbsp;·&nbsp;
    <a href="/api/v1/admin/metrics" style="color:#63b3ed">JSON</a></p>
  <div class="grid">
    <div class="card"><div class="value">{stats['total_sessions']}</div><div class="label">Total Sessions</div></div>
    <div class="card"><div class="value">{stats['sessions_today']}</div><div class="label">Sessions Today</div></div>
    <div class="card"><div class="value">{stats['bookings_today']}</div><div class="label">Bookings Today</div></div>
    <div class="card"><div class="value">{stats['bookings_total']}</div><div class="label">Total Bookings</div></div>
    <div class="card warn"><div class="value">{stats['avg_triage_ms']}ms</div><div class="label">Avg Triage Time</div></div>
    <div class="card danger"><div class="value">{urgency.get('emergency', 0)}</div><div class="label">Emergencies</div></div>
  </div>
  <div class="charts">
    <div class="chart-card"><h3>Urgency Breakdown</h3><canvas id="urgencyChart"></canvas></div>
    <div class="chart-card"><h3>Sessions by Hour</h3><canvas id="hourlyChart"></canvas></div>
  </div>
  <div class="chart-card" style="margin-bottom:28px">
    <h3>Recent Events</h3>
    <table>
      <thead><tr><th>Time</th><th>Event</th><th>Session</th><th>Status</th></tr></thead>
      <tbody>{recent_rows}</tbody>
    </table>
  </div>
  <script>
    Chart.defaults.color='#718096';Chart.defaults.borderColor='#2d3748';
    new Chart(document.getElementById('urgencyChart'),{{type:'doughnut',data:{{labels:['Emergency','Urgent','Routine'],datasets:[{{data:[{urgency.get('emergency',0)},{urgency.get('urgent',0)},{urgency.get('routine',0)}],backgroundColor:['#fc8181','#f6ad55','#68d391'],borderWidth:0}}]}},options:{{plugins:{{legend:{{position:'bottom'}}}},cutout:'65%'}}}});
    new Chart(document.getElementById('hourlyChart'),{{type:'bar',data:{{labels:{list(hourly.keys())}.map(h=>h+':00'),datasets:[{{label:'Sessions',data:{list(hourly.values())},backgroundColor:'#63b3ed88',borderColor:'#63b3ed',borderWidth:1,borderRadius:4}}]}},options:{{scales:{{y:{{beginAtZero:true}},x:{{grid:{{display:false}}}}}},plugins:{{legend:{{display:false}}}}}}}});
  </script>
</body></html>"""
    return HTMLResponse(content=html)


# ── Today's Appointments ───────────────────────────────────────────────────────

@router.get("/appointments/today")
async def get_today_appointments(
    current_user=Depends(require_admin),
    db: AsyncSession = Depends(db_session),
):
    today = date.today()
    today_start = datetime.combine(today, time.min, tzinfo=timezone.utc)
    today_end = datetime.combine(today, time.max, tzinfo=timezone.utc)

    query = (
        select(Appointment, AvailabilitySlot, Patient, Doctor)
        .join(AvailabilitySlot, Appointment.slot_id == AvailabilitySlot.id)
        .join(Patient, Appointment.patient_id == Patient.id)
        .outerjoin(Doctor, Appointment.doctor_id == Doctor.id)
        .where(
            and_(
                AvailabilitySlot.slot_start >= today_start,
                AvailabilitySlot.slot_start <= today_end,
            )
        )
        .order_by(AvailabilitySlot.slot_start)
    )

    if current_user.get("role") == "doctor":
        doctor_id = current_user.get("doctor_id")
        if doctor_id:
            query = query.where(Appointment.doctor_id == uuid.UUID(doctor_id))

    result = await db.execute(query)
    rows = result.all()

    appointments = []
    for appt, slot, patient, doctor in rows:
        appointments.append({
            "appointment_id": str(appt.id),
            "patient_name": patient.full_name or "Unknown Patient",
            "patient_phone": patient.phone or "—",
            "patient_email": patient.email or "—",
            "doctor_name": doctor.full_name if doctor else "Unassigned",
            "doctor_id": str(appt.doctor_id) if appt.doctor_id else None,
            "urgency_level": appt.urgency_level or "routine",
            "urgency_score": appt.urgency_score or 1,
            "chief_complaint": appt.chief_complaint or "—",
            "slot_time": slot.slot_start.isoformat(),
            "status": appt.status or "scheduled",
        })

    return {
        "appointments": appointments,
        "total": len(appointments),
        "viewer_role": current_user.get("role", "admin"),
        "viewer_doctor_id": current_user.get("doctor_id"),
    }


@router.patch("/appointments/{appointment_id}/seen")
async def mark_appointment_seen(
    appointment_id: str,
    current_user=Depends(require_admin),
    db: AsyncSession = Depends(db_session),
):
    svc = BookingService(db)
    appt = await svc.mark_appointment_seen(uuid.UUID(appointment_id))
    await db.commit()
    return {"status": "completed", "appointment_id": str(appt.id)}