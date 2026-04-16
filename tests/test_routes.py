"""HTTP smoke tests for main routes."""

from __future__ import annotations


def test_index_returns_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.content_type is not None and "html" in response.content_type
    assert b"DOCTYPE" in response.data or b"html" in response.data.lower()


def test_health_json_when_models_stubbed(client, health_mocks):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["models_loaded"]["grounding_dino"] is True
    assert data["models_loaded"]["sam_predictor"] is True
    assert "device" in data
    assert "torch_available" in data
