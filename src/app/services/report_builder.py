from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List


def build_daily_report_html(rows: List[Dict[str, Any]], period_hours: int = 24) -> str:
    if not rows:
        return (
            f"<h2>Informe SugarCane — últimas {period_hours}h</h2>"
            "<p>No se registraron clasificaciones en el periodo.</p>"
        )

    counts = Counter(r.get("class_name", "Unknown") for r in rows)
    total = len(rows)
    disease_rows = [r for r in rows if str(r.get("class_name", "")).lower() != "healthy"]
    disease_n = len(disease_rows)

    lines = [
        f"<h2>Informe general SugarCane — últimas {period_hours}h</h2>",
        f"<p>Generado: {datetime.now(timezone.utc).isoformat()}</p>",
        f"<p><b>Total clasificaciones:</b> {total}</p>",
        f"<p><b>Posibles enfermedades detectadas:</b> {disease_n}</p>",
        "<h3>Conteo por clase</h3><ul>",
    ]
    for cls, n in counts.most_common():
        pct = (n / total) * 100
        lines.append(f"<li>{cls}: {n} ({pct:.1f}%)</li>")
    lines.append("</ul>")

    if disease_rows:
        lines.append("<h3>Últimas alertas de enfermedad</h3><ul>")
        for r in disease_rows[-10:]:
            lines.append(
                f"<li>{r.get('predicted_at', '')} — {r.get('class_name')} "
                f"({float(r.get('confidence', 0)):.1%})</li>"
            )
        lines.append("</ul>")

    lines.append(
        "<p><i>Clasificación de apoyo por IA; validar en campo con agrónomo.</i></p>"
    )
    return "\n".join(lines)


def build_daily_report_text(rows: List[Dict[str, Any]], period_hours: int = 24) -> str:
    html = build_daily_report_html(rows, period_hours)
    return (
        html.replace("<h2>", "\n== ")
        .replace("</h2>", " ==\n")
        .replace("<h3>", "\n-- ")
        .replace("</h3>", " --\n")
        .replace("<p>", "")
        .replace("</p>", "\n")
        .replace("<ul>", "")
        .replace("</ul>", "")
        .replace("<li>", "  • ")
        .replace("</li>", "\n")
        .replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
    )
