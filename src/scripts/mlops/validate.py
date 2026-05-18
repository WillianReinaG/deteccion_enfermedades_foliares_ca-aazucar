"""
Validación post-entrenamiento: comprueba artefactos mínimos y carga del predictor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.classifier.predictor import SugarCanePredictor  # noqa: E402


def main() -> int:
    artifacts = PROJECT_ROOT / "artifacts"
    models = PROJECT_ROOT / "models"
    leaderboard = artifacts / "leaderboard_final.csv"
    metadata = models / "model_metadata.json"

    if not leaderboard.exists() and not metadata.exists():
        print("[validate] Sin leaderboard ni metadata — modo demo permitido en runtime.")
    else:
        print("[validate] Metadatos encontrados.")

    predictor = SugarCanePredictor()
    report = {
        "model_id": predictor.model_id,
        "framework": predictor.framework,
        "demo_mode": predictor.demo_mode,
        "checkpoint": str(predictor.ckpt_path) if predictor.ckpt_path else None,
        "class_names": predictor.class_names,
    }
    out_dir = PROJECT_ROOT / "data" / "monitoring"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[validate] Informe guardado en {report_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
