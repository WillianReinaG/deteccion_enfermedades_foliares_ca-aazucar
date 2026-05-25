from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

from app.config.settings import (
    ALERT_CONFIDENCE_MIN,
    ALERT_EMAIL,
    ALERT_FROM,
    SMTP_APP_PASSWORD,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
)
from app.services.report_builder import build_daily_report_html, build_daily_report_text


def is_disease(pred: Dict[str, Any]) -> bool:
    cls = str(pred.get("class_name", "")).strip().lower()
    conf = float(pred.get("confidence", 0))
    return cls != "healthy" and cls != "" and conf >= ALERT_CONFIDENCE_MIN


def alerts_configured() -> bool:
    return bool(SMTP_USER and SMTP_APP_PASSWORD and ALERT_EMAIL)


def _send_email(subject: str, html_body: str, text_body: str) -> bool:
    if not alerts_configured():
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = ALERT_FROM or SMTP_USER
        msg["To"] = ALERT_EMAIL
        msg.attach(MIMEText(text_body, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_APP_PASSWORD)
            server.sendmail(ALERT_FROM or SMTP_USER, [ALERT_EMAIL], msg.as_string())
        return True
    except Exception:
        return False


def send_immediate_alert(pred: Dict[str, Any], session_id: Optional[str] = None) -> bool:
    if not is_disease(pred):
        return False
    cls = pred.get("class_name", "")
    conf = float(pred.get("confidence", 0))
    subject = f"[SugarCane] Alerta: {cls} detectada ({conf:.0%})"
    html = (
        f"<h2>Alerta de enfermedad foliar</h2>"
        f"<p><b>Clase:</b> {cls}</p>"
        f"<p><b>Confianza:</b> {conf:.2%}</p>"
        f"<p><b>Sesión:</b> {session_id or 'N/A'}</p>"
        f"<p><b>Modelo:</b> {pred.get('model_id', '')}</p>"
        "<p>Validar en campo con agrónomo. No es diagnóstico definitivo.</p>"
    )
    text = (
        f"Alerta SugarCane\nClase: {cls}\nConfianza: {conf:.2%}\n"
        f"Sesión: {session_id or 'N/A'}\nValidar en campo."
    )
    return _send_email(subject, html, text)


def send_daily_report(rows: list, period_hours: int = 24) -> bool:
    subject = f"[SugarCane] Informe diario — últimas {period_hours}h"
    html = build_daily_report_html(rows, period_hours)
    text = build_daily_report_text(rows, period_hours)
    return _send_email(subject, html, text)
