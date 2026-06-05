"""Unit-level checks for run_segmentation validation (no checkpoints)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
from PIL import Image


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


def test_run_segmentation_detailed_combines_multiple_masks(monkeypatch):
    import app as m

    if m.torch is None:
        return

    buf = BytesIO()
    Image.new("RGB", (4, 4), color=(255, 255, 255)).save(buf, format="PNG")

    detections = MagicMock()
    detections.xyxy = np.array([[0, 0, 2, 2], [2, 2, 4, 4]], dtype=np.float32)
    detections.confidence = np.array([0.9, 0.8], dtype=np.float32)

    grounding_dino = MagicMock()
    grounding_dino.predict_with_caption.return_value = (detections, ["rice", "plantain"])

    mask_one = m.torch.zeros((4, 4), dtype=m.torch.bool)
    mask_two = m.torch.zeros((4, 4), dtype=m.torch.bool)
    mask_one[0, 0] = True
    mask_two[3, 3] = True
    masks = m.torch.stack([mask_one, mask_two]).unsqueeze(1)

    sam_predictor = MagicMock()
    sam_predictor.transform.apply_boxes_torch.side_effect = lambda boxes, _shape: boxes
    sam_predictor.predict_torch.return_value = (masks, None, None)

    monkeypatch.setattr(m, "grounding_dino", grounding_dino)
    monkeypatch.setattr(m, "sam_predictor", sam_predictor)
    monkeypatch.setattr(m, "device", "cpu")

    result = m.run_segmentation_detailed(buf.getvalue(), "rice, plantain")
    assert result.success is True
    assert result.metadata["detections"]["count"] == 2
    assert result.metadata["detections"]["phrases"] == ["rice", "plantain"]
    sam_predictor.predict_torch.assert_called_once()
