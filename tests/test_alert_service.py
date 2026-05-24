from unittest.mock import MagicMock, patch

import app.services.alert_service as alert_service


def test_is_disease_healthy_false():
    assert alert_service.is_disease({"class_name": "Healthy", "confidence": 0.99}) is False


def test_is_disease_low_confidence_false():
    assert alert_service.is_disease({"class_name": "Rust", "confidence": 0.3}) is False


def test_is_disease_true():
    assert alert_service.is_disease({"class_name": "Rust", "confidence": 0.8}) is True


def test_alerts_configured():
    with patch.object(alert_service, "SENDGRID_API_KEY", "key"), patch.object(
        alert_service, "ALERT_EMAIL", "test@example.com"
    ):
        assert alert_service.alerts_configured() is True


@patch.object(alert_service, "SENDGRID_API_KEY", "test-key")
@patch.object(alert_service, "ALERT_EMAIL", "dest@example.com")
@patch.object(alert_service, "ALERT_FROM", "from@example.com")
def test_send_immediate_alert_mock_sendgrid():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 202
    mock_client.send.return_value = mock_response

    with patch("sendgrid.SendGridAPIClient", return_value=mock_client):
        pred = {"class_name": "Mosaic", "confidence": 0.91, "model_id": "demo"}
        ok = alert_service.send_immediate_alert(pred, session_id="sess-1")
        assert ok is True
        mock_client.send.assert_called_once()


@patch.object(alert_service, "SENDGRID_API_KEY", "")
def test_send_immediate_alert_not_configured():
    pred = {"class_name": "Mosaic", "confidence": 0.91}
    assert alert_service.send_immediate_alert(pred) is False
