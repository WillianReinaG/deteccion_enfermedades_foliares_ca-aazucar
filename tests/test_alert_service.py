from unittest.mock import MagicMock, patch

import app.services.alert_service as alert_service


def test_is_disease_healthy_false():
    assert alert_service.is_disease({"class_name": "Healthy", "confidence": 0.99}) is False


def test_is_disease_low_confidence_false():
    assert alert_service.is_disease({"class_name": "Rust", "confidence": 0.3}) is False


def test_is_disease_true():
    assert alert_service.is_disease({"class_name": "Rust", "confidence": 0.8}) is True


def test_alerts_configured():
    with patch.object(alert_service, "SMTP_USER", "user@gmail.com"), patch.object(
        alert_service, "SMTP_APP_PASSWORD", "app-pass"
    ), patch.object(alert_service, "ALERT_EMAIL", "test@example.com"):
        assert alert_service.alerts_configured() is True


@patch.object(alert_service, "SMTP_USER", "from@gmail.com")
@patch.object(alert_service, "SMTP_APP_PASSWORD", "app-pass")
@patch.object(alert_service, "ALERT_EMAIL", "dest@example.com")
@patch.object(alert_service, "ALERT_FROM", "from@gmail.com")
@patch.object(alert_service, "SMTP_HOST", "smtp.gmail.com")
@patch.object(alert_service, "SMTP_PORT", 587)
def test_send_immediate_alert_mock_smtp():
    mock_server = MagicMock()
    with patch("smtplib.SMTP") as mock_smtp:
        mock_smtp.return_value.__enter__.return_value = mock_server
        pred = {"class_name": "Mosaic", "confidence": 0.91, "model_id": "demo"}
        ok = alert_service.send_immediate_alert(pred, session_id="sess-1")
        assert ok is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("from@gmail.com", "app-pass")
        mock_server.sendmail.assert_called_once()


@patch.object(alert_service, "SMTP_APP_PASSWORD", "")
def test_send_immediate_alert_not_configured():
    pred = {"class_name": "Mosaic", "confidence": 0.91}
    assert alert_service.send_immediate_alert(pred) is False
