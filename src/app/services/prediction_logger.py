from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config.settings import (
    BASE_DIR,
    GCP_PROJECT_ID,
    BQ_DATASET,
    BQ_TABLE,
)


PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
LOCAL_LOG = PREDICTIONS_DIR / "predictions.jsonl"


def _row_from_prediction(pred: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "class_name": str(pred.get("class_name", "")),
        "confidence": float(pred.get("confidence", 0)),
        "session_id": session_id or "",
        "demo_mode": bool(pred.get("demo_mode", False)),
        "model_id": str(pred.get("model_id", "")),
        "source": "streamlit",
    }


def _append_local(row: Dict[str, Any]) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with LOCAL_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _insert_bigquery(row: Dict[str, Any]) -> bool:
    if not GCP_PROJECT_ID:
        return False
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=GCP_PROJECT_ID)
        table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        errors = client.insert_rows_json(table_id, [row])
        return not errors
    except Exception:
        return False


def log_prediction(pred: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
    row = _row_from_prediction(pred, session_id)
    _append_local(row)
    if GCP_PROJECT_ID:
        _insert_bigquery(row)
    return row


def load_predictions_last_hours(hours: int = 24) -> List[Dict[str, Any]]:
    if GCP_PROJECT_ID:
        bq_rows = _query_bigquery_last_hours(hours)
        if bq_rows is not None:
            return bq_rows
    return _load_local_last_hours(hours)


def _load_local_last_hours(hours: int) -> List[Dict[str, Any]]:
    if not LOCAL_LOG.exists():
        return []
    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    rows: List[Dict[str, Any]] = []
    for line in LOCAL_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ts = datetime.fromisoformat(row["predicted_at"].replace("Z", "+00:00"))
            if ts.timestamp() >= cutoff:
                rows.append(row)
        except Exception:
            continue
    return rows


def _query_bigquery_last_hours(hours: int) -> Optional[List[Dict[str, Any]]]:
    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=GCP_PROJECT_ID)
        sql = f"""
        SELECT predicted_at, class_name, confidence, session_id, demo_mode, model_id, source
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE predicted_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {int(hours)} HOUR)
        ORDER BY predicted_at DESC
        """
        return [dict(r) for r in client.query(sql).result()]
    except Exception:
        return None
