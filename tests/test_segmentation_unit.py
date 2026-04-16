"""Unit-level checks for run_segmentation validation (no checkpoints)."""

from __future__ import annotations

from unittest.mock import MagicMock


def test_run_segmentation_rejects_empty_bytes(monkeypatch):
    import app as m

    monkeypatch.setattr(m, "load_models", lambda: True)
    monkeypatch.setattr(m, "grounding_dino", MagicMock())
    monkeypatch.setattr(m, "sam_predictor", MagicMock())
    o, r = m.run_segmentation(b"", "x")
    assert o is None and r is None


def test_run_segmentation_rejects_empty_prompt(monkeypatch):
    import app as m

    monkeypatch.setattr(m, "load_models", lambda: True)
    monkeypatch.setattr(m, "grounding_dino", MagicMock())
    monkeypatch.setattr(m, "sam_predictor", MagicMock())
    o, r = m.run_segmentation(b"not-used", "  ")
    assert o is None and r is None


def test_run_segmentation_rejects_oversize(monkeypatch):
    import app as m

    monkeypatch.setattr(m, "load_models", lambda: True)
    monkeypatch.setattr(m, "grounding_dino", MagicMock())
    monkeypatch.setattr(m, "sam_predictor", MagicMock())
    big = bytes(11 * 1024 * 1024)
    o, r = m.run_segmentation(big, "food")
    assert o is None and r is None


def test_run_segmentation_rejects_invalid_image(monkeypatch):
    import app as m

    monkeypatch.setattr(m, "load_models", lambda: True)
    monkeypatch.setattr(m, "grounding_dino", MagicMock())
    monkeypatch.setattr(m, "sam_predictor", MagicMock())
    o, r = m.run_segmentation(b"not an image", "food")
    assert o is None and r is None
