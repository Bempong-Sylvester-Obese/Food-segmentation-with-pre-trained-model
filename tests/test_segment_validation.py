"""POST /segment validation and error JSON (models mocked)."""

from __future__ import annotations
from io import BytesIO

def test_segment_missing_image_file(client, segment_mocks):
    response = client.post(
        "/segment",
        data={"prompt": "apple"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False
    assert "image" in data["error"].lower() or "file" in data["error"].lower()


def test_segment_empty_filename(client, segment_mocks, tiny_png_bytes):
    response = client.post(
        "/segment",
        data={
            "image_file": (BytesIO(tiny_png_bytes), ""),
            "prompt": "food",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False


def test_segment_empty_prompt(client, segment_mocks, tiny_png_bytes):
    response = client.post(
        "/segment",
        data={
            "image_file": (BytesIO(tiny_png_bytes), "x.png"),
            "prompt": "",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False
    assert "prompt" in data["error"].lower()


def test_segment_bad_extension(client, segment_mocks, tiny_png_bytes):
    response = client.post(
        "/segment",
        data={
            "image_file": (BytesIO(tiny_png_bytes), "x.exe"),
            "prompt": "food",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False
    assert "valid" in data["error"].lower() or "upload" in data["error"].lower()


def test_segment_empty_file_body(client, segment_mocks):
    response = client.post(
        "/segment",
        data={
            "image_file": (BytesIO(b""), "empty.png"),
            "prompt": "food",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False


def test_segment_success_when_run_segmentation_mocked(client, segment_mocks, tiny_png_bytes, monkeypatch):
    import app as m
    import base64

    fake_b64 = base64.b64encode(b"fake").decode("ascii")

    def fake_run_segmentation(image_bytes, prompt):
        return fake_b64, fake_b64

    monkeypatch.setattr(m, "run_segmentation", fake_run_segmentation)
    response = client.post(
        "/segment",
        data={
            "image_file": (BytesIO(tiny_png_bytes), "plate.png"),
            "prompt": "jollof rice",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["original_image"] == fake_b64
    assert data["result_image"] == fake_b64
