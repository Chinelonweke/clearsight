from __future__ import annotations
"""
app/services/email_service.py
─────────────────────────────────────────────────────────
Patient email notification service using Gmail SMTP.
Sends booking confirmations and password reset emails.
Non-fatal — email failure never crashes the booking flow.
"""

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.core.logger import get_logger

logger = get_logger(__name__)


async def _send_email(
    to_email: str,
    subject: str,
    body: str,
    html: str | None = None,
) -> bool:
    """
    Generic email sender via Gmail SMTP.
    Used for password reset emails and other transactional emails.
    Returns True if sent successfully, False otherwise.
    Never raises — email failure is non-fatal.
    """
    if not to_email or "@" not in to_email:
        logger.debug("No valid email address — skipping email.")
        return False

    try:
        from app.config import settings

        gmail_user = settings.gmail_user
        gmail_password = settings.gmail_app_password

        if not gmail_user or not gmail_password:
            logger.warning("Gmail credentials not configured — email notifications disabled.")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{settings.clinic_name} <{gmail_user}>"
        msg["To"] = to_email
        msg["Reply-To"] = gmail_user

        # Attach plain text version
        msg.attach(MIMEText(body, "plain"))

        # Attach HTML version if provided
        if html:
            msg.attach(MIMEText(html, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, to_email, msg.as_string())

        logger.success(f"Email sent via Gmail | to={to_email} | subject={subject}")
        return True

    except Exception as exc:
        logger.warning(f"Email send failed (non-fatal) | to={to_email}: {exc}")
        return False


def _build_confirmation_html(
    patient_name: str,
    slot_time: str,
    urgency_level: str,
    urgency_score: int,
    chief_complaint: str,
    patient_instruction: str,
    doctor_name: str,
    clinic_name: str,
    clinic_address: str,
    clinic_phone: str,
) -> str:
    """Build the HTML email body for booking confirmations."""

    urgency_color = {
        "emergency": "#dc2626",
        "urgent": "#d97706",
        "routine": "#16a34a",
    }.get(urgency_level.lower(), "#16a34a")

    urgency_bg = {
        "emergency": "#fee2e2",
        "urgent": "#fef3c7",
        "routine": "#dcfce7",
    }.get(urgency_level.lower(), "#dcfce7")

    urgency_label = urgency_level.upper()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>Appointment Confirmation — {clinic_name}</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f5;font-family:system-ui,-apple-system,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f5;padding:32px 16px;">
    <tr><td align="center">
      <table width="100%" style="max-width:560px;background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">

        <!-- Header -->
        <tr>
          <td style="background:#0a2e2a;padding:28px 32px;">
            <table width="100%"><tr>
              <td>
                <div style="display:inline-flex;align-items:center;gap:10px;">
                  <div style="width:36px;height:36px;background:#22c5b0;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;">👁</div>
                  <div>
                    <div style="color:#ffffff;font-size:18px;font-weight:700;">{clinic_name}</div>
                    <div style="color:#22c5b0;font-size:11px;letter-spacing:0.5px;text-transform:uppercase;">Appointment Confirmation</div>
                  </div>
                </div>
              </td>
            </tr></table>
          </td>
        </tr>

        <!-- Greeting -->
        <tr>
          <td style="padding:32px 32px 0;">
            <p style="margin:0;font-size:16px;color:#0d2420;">
              Dear <strong>{patient_name}</strong>,
            </p>
            <p style="margin:12px 0 0;font-size:14px;color:#4b7a72;line-height:1.6;">
              Your eye assessment is complete and your appointment has been confirmed.
              Please find the details below.
            </p>
          </td>
        </tr>

        <!-- Appointment Card -->
        <tr>
          <td style="padding:24px 32px;">
            <table width="100%" style="background:#eafaf8;border-radius:12px;padding:20px;border:1.5px solid #d0f5f0;">
              <tr>
                <td>
                  <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#6b9490;margin-bottom:6px;">
                    📅 Appointment Date & Time
                  </div>
                  <div style="font-size:22px;font-weight:700;color:#0a2e2a;">
                    {slot_time}
                  </div>
                  <div style="margin-top:12px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#6b9490;">
                    👨‍⚕️ Doctor
                  </div>
                  <div style="font-size:15px;font-weight:600;color:#115c52;margin-top:4px;">
                    {doctor_name}
                  </div>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Urgency Badge -->
        <tr>
          <td style="padding:0 32px 24px;">
            <table width="100%">
              <tr>
                <td style="padding-right:12px;width:50%;">
                  <div style="background:{urgency_bg};border-radius:10px;padding:14px;text-align:center;">
                    <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:{urgency_color};letter-spacing:0.5px;">Urgency Level</div>
                    <div style="font-size:18px;font-weight:700;color:{urgency_color};margin-top:4px;">{urgency_label} — {urgency_score}/10</div>
                  </div>
                </td>
                <td style="width:50%;">
                  <div style="background:#f5f0e8;border-radius:10px;padding:14px;">
                    <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#6b9490;letter-spacing:0.5px;">Chief Complaint</div>
                    <div style="font-size:13px;color:#0d2420;margin-top:4px;line-height:1.4;">{chief_complaint}</div>
                  </div>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Instructions -->
        <tr>
          <td style="padding:0 32px 24px;">
            <table width="100%" style="background:#fef3c7;border-radius:10px;padding:16px;border-left:4px solid #d97706;">
              <tr>
                <td>
                  <div style="font-size:12px;font-weight:700;color:#92400e;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
                    ⚠️ Important Instructions
                  </div>
                  <div style="font-size:13px;color:#78350f;line-height:1.6;">
                    {patient_instruction}
                  </div>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- What to bring -->
        <tr>
          <td style="padding:0 32px 24px;">
            <div style="font-size:13px;font-weight:700;color:#0d2420;margin-bottom:10px;">📋 Please bring to your appointment:</div>
            <table>
              <tr><td style="padding:3px 0;font-size:13px;color:#4b7a72;">✅ &nbsp;Valid ID (National ID, Driver's License)</td></tr>
              <tr><td style="padding:3px 0;font-size:13px;color:#4b7a72;">✅ &nbsp;Previous eye prescriptions or glasses (if any)</td></tr>
              <tr><td style="padding:3px 0;font-size:13px;color:#4b7a72;">✅ &nbsp;List of current medications</td></tr>
              <tr><td style="padding:3px 0;font-size:13px;color:#4b7a72;">✅ &nbsp;Arrive 10–15 minutes early for paperwork</td></tr>
            </table>
          </td>
        </tr>

        <!-- Clinic Info -->
        <tr>
          <td style="padding:0 32px 32px;">
            <table width="100%" style="background:#f5f0e8;border-radius:10px;padding:16px;">
              <tr>
                <td>
                  <div style="font-size:12px;font-weight:700;color:#6b9490;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">
                    📍 Clinic Location
                  </div>
                  <div style="font-size:13px;color:#0d2420;margin-bottom:4px;font-weight:600;">{clinic_name}</div>
                  <div style="font-size:13px;color:#4b7a72;">{clinic_address}</div>
                  <div style="font-size:13px;color:#4b7a72;margin-top:4px;">📞 {clinic_phone}</div>
                  <div style="margin-top:10px;font-size:12px;color:#6b9490;">
                    To reschedule, please call us at least 4 hours in advance.
                  </div>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#0a2e2a;padding:20px 32px;text-align:center;">
            <p style="margin:0;font-size:12px;color:#6b9490;">
              This email was sent automatically by the ClearSight AI Triage System.<br>
              © {clinic_name} · {clinic_address}
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


async def send_booking_confirmation(
    patient_email: str,
    patient_name: str,
    slot_time: str,
    urgency_level: str,
    urgency_score: int,
    chief_complaint: str,
    patient_instruction: str,
    doctor_name: str = "Our Optometrist",
) -> bool:
    """
    Send a booking confirmation email via Gmail SMTP.
    Returns True if sent successfully, False otherwise.
    Never raises — email failure is non-fatal.
    """
    if not patient_email or "@" not in patient_email:
        logger.debug("No valid patient email — skipping confirmation email.")
        return False

    try:
        from app.config import settings

        gmail_user = settings.gmail_user
        gmail_password = settings.gmail_app_password

        if not gmail_user or not gmail_password:
            logger.warning("Gmail credentials not configured — email notifications disabled.")
            return False

        html = _build_confirmation_html(
            patient_name=patient_name,
            slot_time=slot_time,
            urgency_level=urgency_level,
            urgency_score=urgency_score,
            chief_complaint=chief_complaint,
            patient_instruction=patient_instruction,
            doctor_name=doctor_name,
            clinic_name=settings.clinic_name,
            clinic_address=settings.clinic_address,
            clinic_phone=settings.clinic_phone,
        )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Appointment Confirmed — {slot_time} | {settings.clinic_name}"
        msg["From"] = f"{settings.clinic_name} <{gmail_user}>"
        msg["To"] = patient_email
        msg["Reply-To"] = gmail_user

        msg.attach(MIMEText(html, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, patient_email, msg.as_string())

        logger.success(f"Confirmation email sent via Gmail | to={patient_email}")
        return True

    except Exception as exc:
        logger.warning(f"Email send failed (non-fatal) | to={patient_email}: {exc}")
        return False