"""Optional full-stack segment test (requires weights + RUN_MODEL_INTEGRATION=1)."""

from __future__ import annotations

import os
from io import BytesIO

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_INTEGRATION") != "1",
    reason="Set RUN_MODEL_INTEGRATION=1 to run (loads GroundingDINO + MobileSAM).",
)
def test_segment_real_pipeline(client, tiny_png_bytes):
    """End-to-end POST /segment; may be slow and needs checkpoints on disk or download."""
    response = client.post(
        "/segment",
        data={
            "image_file": (BytesIO(tiny_png_bytes), "tiny.png"),
            "prompt": "food",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "success" in data
    # Tiny random-colored square may not detect "food"; accept either outcome.
    assert data["success"] in (True, False)
