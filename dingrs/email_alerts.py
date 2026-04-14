"""
Update #36: Multi-canal alerts — Email via SMTP (gratuit).
SMS (Twilio) necesită cont plătit — skip.
Telegram deja implementat în telegram_alerts.py.
"""
import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)

# Configurare din environment variables (nu în cod!)
SMTP_HOST     = os.environ.get("ALADIN_SMTP_HOST",  "smtp.gmail.com")
SMTP_PORT     = int(os.environ.get("ALADIN_SMTP_PORT", "587"))
SMTP_USER     = os.environ.get("ALADIN_SMTP_USER",  "")
SMTP_PASS     = os.environ.get("ALADIN_SMTP_PASS",  "")
EMAIL_TO      = os.environ.get("ALADIN_EMAIL_TO",   "marioyear@yahoo.com")


def send_email_alert(
    subject: str,
    body:    str,
    html:    bool = True,
) -> bool:
    """
    Trimite alertă pe email via SMTP.
    Setează variabilele de mediu înainte de utilizare:
      ALADIN_SMTP_USER=your@gmail.com
      ALADIN_SMTP_PASS=your_app_password
      ALADIN_EMAIL_TO=recipient@email.com
    """
    if not SMTP_USER or not SMTP_PASS:
        logger.debug("Email alerts: credențiale SMTP lipsă (ALADIN_SMTP_USER/ALADIN_SMTP_PASS)")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ALADIN] {subject}"
        msg["From"]    = SMTP_USER
        msg["To"]      = EMAIL_TO

        if html:
            html_body = f"""
            <html><body style='background:#0d1220;color:#c8d0e8;font-family:sans-serif;padding:20px;'>
            <h2 style='color:#a78bfa;'>⚛️ ALADIN SIGNAL</h2>
            <pre style='background:#07090f;padding:15px;border-radius:8px;color:#e0e8ff;'>{body}</pre>
            <p style='color:#4a5a8a;font-size:12px;'>Aladin Quantum-ICT Engine | {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </body></html>"""
            msg.attach(MIMEText(html_body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, EMAIL_TO, msg.as_string())

        logger.info(f"   📧 Email alert trimis: {subject}")
        return True
    except Exception as e:
        logger.warning(f"Email alert error: {e}")
        return False


def send_diamond_alert_email(signal_data: dict) -> bool:
    """Trimite alertă email pentru semnal DIAMOND."""
    score     = signal_data.get("score", 0)
    verdict   = signal_data.get("verdict", "")
    direction = signal_data.get("trade_direction", "")
    regime    = signal_data.get("regime", "")
    kz        = signal_data.get("killzone", "")
    ts        = signal_data.get("timestamp", "")

    subject = f"💎 DIAMOND SIGNAL — {direction} @ {score:.1f}%"
    body    = f"""
💎 DIAMOND SIGNAL DETECTAT

Timestamp:  {ts}
Direcție:   {direction}
Score:      {score:.1f}%
Verdict:    {verdict}
Regim:      {regime}
Killzone:   {kz}

Conviction: DIAMOND (>80%)
""".strip()
    return send_email_alert(subject, body)
