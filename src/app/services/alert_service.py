from __future__ import annotations

from typing import Any, Dict, Optional

from app.config.settings import (
    ALERT_CONFIDENCE_MIN,
    ALERT_EMAIL,
    ALERT_FROM,
    SENDGRID_API_KEY,
)
from app.services.report_builder import build_daily_report_html, build_daily_report_text


def is_disease(pred: Dict[str, Any]) -> bool:
    cls = str(pred.get("class_name", "")).strip().lower()
    conf = float(pred.get("confidence", 0))
    return cls != "healthy" and cls != "" and conf >= ALERT_CONFIDENCE_MIN


def alerts_configured() -> bool:
    return bool(SENDGRID_API_KEY and ALERT_EMAIL)


def _send_email(subject: str, html_body: str, text_body: str) -> bool:
    if not alerts_configured():
        return False
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=ALERT_FROM,
            to_emails=ALERT_EMAIL,
            subject=subject,
            plain_text_content=text_body,
            html_content=html_body,
        )
        client = SendGridAPIClient(SENDGRID_API_KEY)
        response = client.send(message)
        return 200 <= response.status_code < 300
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
