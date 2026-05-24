"""Cloud Function: informe diario desde BigQuery + SendGrid."""
from __future__ import annotations

import os

import functions_framework
from google.cloud import bigquery
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


def _load_rows(hours: int = 24) -> list:
    project = os.environ["GCP_PROJECT_ID"]
    dataset = os.getenv("BQ_DATASET", "sugarcane")
    table = os.getenv("BQ_TABLE", "predictions")
    client = bigquery.Client(project=project)
    sql = f"""
    SELECT predicted_at, class_name, confidence, session_id, demo_mode, model_id, source
    FROM `{project}.{dataset}.{table}`
    WHERE predicted_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {int(hours)} HOUR)
    ORDER BY predicted_at DESC
    """
    return [dict(r) for r in client.query(sql).result()]


def _build_html(rows: list, hours: int) -> str:
    if not rows:
        return f"<h2>Informe SugarCane — últimas {hours}h</h2><p>Sin clasificaciones.</p>"
    from collections import Counter

    counts = Counter(r.get("class_name", "?") for r in rows)
    total = len(rows)
    lines = [
        f"<h2>Informe SugarCane — últimas {hours}h</h2>",
        f"<p>Total: {total}</p><ul>",
    ]
    for cls, n in counts.most_common():
        lines.append(f"<li>{cls}: {n}</li>")
    lines.append("</ul>")
    return "\n".join(lines)


def _send(subject: str, html: str) -> bool:
    key = os.environ["SENDGRID_API_KEY"]
    to_email = os.getenv("ALERT_EMAIL", "bebesowi@gmail.com")
    from_email = os.getenv("ALERT_FROM", "noreply@sugarcane.local")
    msg = Mail(from_email=from_email, to_emails=to_email, subject=subject, html_content=html)
    resp = SendGridAPIClient(key).send(msg)
    return 200 <= resp.status_code < 300


@functions_framework.http
def daily_report_http(request):
    hours = 24
    if request and request.args.get("hours"):
        hours = int(request.args.get("hours"))
    rows = _load_rows(hours)
    html = _build_html(rows, hours)
    ok = _send(f"[SugarCane] Informe diario — últimas {hours}h", html)
    return (f"OK: {len(rows)} filas", 200) if ok else ("Send failed", 500)
