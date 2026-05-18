"""
Monitoreo de drift / calidad y recomendación de retraining (heurística inicial).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
THRESHOLD = float(os.getenv("RETRAIN_CONFIDENCE_THRESHOLD", "0.55"))


def main() -> int:
    monitoring_dir = PROJECT_ROOT / "data" / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    validation = monitoring_dir / "validation_report.json"

    demo_mode = True
    avg_confidence = None
    if validation.exists():
        data = json.loads(validation.read_text(encoding="utf-8"))
        demo_mode = bool(data.get("demo_mode", True))

    # Heurística: sin modelo real o umbral bajo → sugerir retraining
    should_retrain = demo_mode or (
        avg_confidence is not None and avg_confidence < THRESHOLD
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demo_mode": demo_mode,
        "avg_confidence": avg_confidence,
        "retrain_recommended": should_retrain,
        "threshold": THRESHOLD,
    }
    path = monitoring_dir / "monitor_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as fh:
            fh.write(f"should_retrain={'true' if should_retrain else 'false'}\n")

    return 1 if should_retrain else 0


if __name__ == "__main__":
    raise SystemExit(main())
