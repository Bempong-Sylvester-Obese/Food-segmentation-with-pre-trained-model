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


def test_run_segmentation_cleans_up_on_failure(monkeypatch):
    """CUDA + GC cleanup must run even when validation short-circuits."""
    import app as m

    monkeypatch.setattr(m, "load_models", lambda: True)
    monkeypatch.setattr(m, "grounding_dino", MagicMock())
    monkeypatch.setattr(m, "sam_predictor", MagicMock())

    if m.torch is None:
        # Can't exercise the CUDA branch without PyTorch; still exercise gc.
        pass
    else:
        monkeypatch.setattr(m.torch.cuda, "is_available", lambda: True)
        cuda_spy = MagicMock()
        monkeypatch.setattr(m.torch.cuda, "empty_cache", cuda_spy)

    gc_spy = MagicMock()
    monkeypatch.setattr(m.gc, "collect", gc_spy)

    o, r = m.run_segmentation(b"", "food")
    assert o is None and r is None
    gc_spy.assert_called()
    if m.torch is not None:
        cuda_spy.assert_called()
