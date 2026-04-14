from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import get_logger
from app.dependencies import require_admin, db_session
from app.rag.chroma_client import get_collection_stats
from app.services.analytics_service import get_dashboard_stats

logger = get_logger(__name__)
router = APIRouter()


@router.get("/metrics")
async def get_metrics(_: dict = Depends(require_admin)):
    loop = asyncio.get_event_loop()
    stats = await loop.run_in_executor(None, get_dashboard_stats)
    return stats


@router.get("/rag-stats")
async def rag_stats(_: dict = Depends(require_admin)):
    stats = await get_collection_stats()
    return stats


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
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>ClearSight — Analytics Dashboard</title>
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
    th{{text-align:left;padding:8px 12px;color:#718096;font-weight:500;
        border-bottom:1px solid #2d3748;text-transform:uppercase;font-size:11px}}
    td{{padding:8px 12px;border-bottom:1px solid #1e2533;color:#cbd5e0}}
    code{{background:#2d3748;padding:2px 6px;border-radius:4px;font-size:11px;color:#68d391}}
    .text-success{{color:#68d391}}.text-danger{{color:#fc8181}}
    .refresh{{font-size:12px;color:#718096;margin-top:24px;text-align:right}}
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
    <div class="card warn"><div class="value">{stats['avg_stt_ms']}ms</div><div class="label">Avg STT Time</div></div>
    <div class="card {'danger' if stats['error_rate_today'] > 10 else 'warn' if stats['error_rate_today'] > 5 else ''}">
      <div class="value">{stats['error_rate_today']}%</div><div class="label">Error Rate Today</div>
    </div>
    <div class="card danger"><div class="value">{urgency.get('emergency', 0)}</div><div class="label">Emergencies</div></div>
  </div>
  <div class="charts">
    <div class="chart-card"><h3>Urgency Breakdown (All Time)</h3><canvas id="urgencyChart"></canvas></div>
    <div class="chart-card"><h3>Sessions by Hour (Today)</h3><canvas id="hourlyChart"></canvas></div>
  </div>
  <div class="chart-card" style="margin-bottom:28px">
    <h3>Recent Events</h3>
    <table>
      <thead><tr><th>Time</th><th>Event</th><th>Session</th><th>Status</th></tr></thead>
      <tbody>{recent_rows}</tbody>
    </table>
  </div>
  <p class="refresh"><a href="/api/v1/admin/dashboard" style="color:#718096">↺ Refresh</a></p>
  <script>
    Chart.defaults.color = '#718096';
    Chart.defaults.borderColor = '#2d3748';
    new Chart(document.getElementById('urgencyChart'), {{
      type: 'doughnut',
      data: {{
        labels: ['Emergency', 'Urgent', 'Routine'],
        datasets: [{{
          data: [{urgency.get('emergency', 0)},{urgency.get('urgent', 0)},{urgency.get('routine', 0)}],
          backgroundColor: ['#fc8181','#f6ad55','#68d391'],
          borderWidth: 0,
        }}]
      }},
      options: {{ plugins: {{ legend: {{ position: 'bottom' }} }}, cutout: '65%' }}
    }});
    new Chart(document.getElementById('hourlyChart'), {{
      type: 'bar',
      data: {{
        labels: {list(hourly.keys())}.map(h => h + ':00'),
        datasets: [{{
          label: 'Sessions',
          data: {list(hourly.values())},
          backgroundColor: '#63b3ed88',
          borderColor: '#63b3ed',
          borderWidth: 1,
          borderRadius: 4,
        }}]
      }},
      options: {{
        scales: {{ y: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }}, x: {{ grid: {{ display: false }} }} }},
        plugins: {{ legend: {{ display: false }} }}
      }}
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/appointments/today")
async def get_today_appointments(
    current_user=Depends(require_admin),
    db: AsyncSession = Depends(db_session),
):
    from datetime import date, timezone, time
    from sqlalchemy import and_
    from app.models.booking import Appointment
    from app.models.doctor import AvailabilitySlot, Doctor
    from app.models.patient import Patient

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

    # If logged in as a doctor (not admin), only show their patients
    if current_user.get("role") == "doctor":
        doctor_id = current_user.get("doctor_id")
        if doctor_id:
            import uuid as _uuid
            query = query.where(Appointment.doctor_id == _uuid.UUID(doctor_id))

    result = await db.execute(query)
    rows = result.all()

    appointments = []
    for appt, slot, patient, doctor in rows:
        appointments.append({
            "appointment_id": str(appt.id),
            "patient_name": patient.full_name,
            "patient_phone": patient.phone or "—",
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
    from app.services.booking_service import BookingService
    svc = BookingService(db)
    appt = await svc.mark_appointment_seen(uuid.UUID(appointment_id))
    await db.commit()
    return {"status": "completed", "appointment_id": str(appt.id)}