"""HTTP smoke tests for main routes."""

from __future__ import annotations


def test_index_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.content_type is not None and "html" in response.content_type
    assert b"DOCTYPE" in response.data or b"html" in response.data.lower()
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "X-Request-ID" in response.headers


def test_health_json_when_models_stubbed(client, health_mocks):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["status"] == "ready"
    assert "models_loaded" in data
    assert data["models_loaded"]["grounding_dino"] is True
    assert data["models_loaded"]["sam_predictor"] is True
    assert "device" in data
    assert "torch_available" in data


def test_api_config_exposes_frontend_limits(client):
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.get_json()
    assert data["max_image_bytes"] > 0
    assert "png" in data["allowed_extensions"]
    assert data["max_prompt_chars"] > 0


def test_livez_returns_alive(client):
    response = client.get("/livez")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "alive"


def test_readyz_returns_503_when_models_missing(client, monkeypatch):
    import app as flask_app

    monkeypatch.setattr(flask_app, "grounding_dino", None)
    monkeypatch.setattr(flask_app, "sam_predictor", None)
    response = client.get("/readyz")
    assert response.status_code == 503
    data = response.get_json()
    assert data["models_loaded"]["grounding_dino"] is False
