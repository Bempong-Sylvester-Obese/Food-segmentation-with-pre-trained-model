"""Pytest fixtures: put webapp on path and expose Flask test client."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = ROOT / "webapp"
if str(WEBAPP) not in sys.path:
    sys.path.insert(0, str(WEBAPP))


@pytest.fixture
def app():
    import app as flask_app

    flask_app.app.config.update(
        TESTING=True,
        SEGMENT_API_KEY="",
        RATELIMIT_ENABLED=False,
        SEGMENT_RATE_LIMIT="5/minute",
        MAX_IMAGE_PIXELS=flask_app.MAX_IMAGE_PIXELS,
    )
    flask_app._RATE_LIMIT_BUCKETS.clear()
    return flask_app.app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def tiny_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def segment_mocks(monkeypatch):
    """Avoid loading GroundingDINO/MobileSAM during POST /segment validation tests."""
    import app as flask_app

    monkeypatch.setattr(flask_app, "load_models", lambda: True)
    monkeypatch.setattr(flask_app, "grounding_dino", MagicMock())
    monkeypatch.setattr(flask_app, "sam_predictor", MagicMock())
    monkeypatch.setattr(flask_app, "device", "cpu")


@pytest.fixture
def health_mocks(monkeypatch):
    """Fast /health without loading real checkpoints."""
    import app as flask_app

    mock = MagicMock()
    monkeypatch.setattr(flask_app, "load_models", lambda: True)
    monkeypatch.setattr(flask_app, "grounding_dino", mock)
    monkeypatch.setattr(flask_app, "sam_predictor", mock)
    monkeypatch.setattr(flask_app, "device", "cpu")
