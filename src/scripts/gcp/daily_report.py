#!/usr/bin/env python3
"""Genera y envía el informe diario de clasificaciones (últimas 24 h)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.services.alert_service import alerts_configured, send_daily_report
from app.services.prediction_logger import load_predictions_last_hours


def main() -> int:
    parser = argparse.ArgumentParser(description="Informe diario SugarCane")
    parser.add_argument("--hours", type=int, default=24, help="Ventana en horas (default: 24)")
    args = parser.parse_args()

    if not alerts_configured():
        print("ERROR: Configure SMTP_USER, SMTP_APP_PASSWORD y ALERT_EMAIL en .env", file=sys.stderr)
        return 1

    rows = load_predictions_last_hours(args.hours)
    ok = send_daily_report(rows, period_hours=args.hours)
    if ok:
        print(f"Informe enviado ({len(rows)} clasificaciones, últimas {args.hours}h)")
        return 0
    print("ERROR: no se pudo enviar el informe", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
