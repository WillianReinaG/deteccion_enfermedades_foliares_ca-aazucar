import json
from unittest.mock import patch

import app.services.prediction_logger as prediction_logger


def test_log_prediction_local_jsonl(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(prediction_logger, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(prediction_logger, "LOCAL_LOG", log_file)
    monkeypatch.setattr(prediction_logger, "GCP_PROJECT_ID", "")

    pred = {"class_name": "Rust", "confidence": 0.75, "demo_mode": False, "model_id": "test"}
    row = prediction_logger.log_prediction(pred, session_id="abc")

    assert row["class_name"] == "Rust"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["session_id"] == "abc"
    assert saved["source"] == "streamlit"


def test_load_predictions_last_hours_local(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(prediction_logger, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(prediction_logger, "LOCAL_LOG", log_file)
    monkeypatch.setattr(prediction_logger, "GCP_PROJECT_ID", "")

    pred = {"class_name": "Yellow", "confidence": 0.6, "demo_mode": True, "model_id": "m"}
    prediction_logger.log_prediction(pred)

    rows = prediction_logger.load_predictions_last_hours(24)
    assert len(rows) == 1
    assert rows[0]["class_name"] == "Yellow"


@patch.object(prediction_logger, "GCP_PROJECT_ID", "my-project")
def test_log_prediction_bigquery_fallback_to_local(tmp_path, monkeypatch):
    log_file = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(prediction_logger, "PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr(prediction_logger, "LOCAL_LOG", log_file)

    with patch.object(prediction_logger, "_insert_bigquery", return_value=False):
        prediction_logger.log_prediction({"class_name": "Healthy", "confidence": 0.9})

    assert log_file.exists()
