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
    error = data["error"].lower()
    assert "valid image file" in error or "filename" in error


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
    error = data["error"].lower()
    assert "no image data" in error or "image data" in error


def test_segment_success_when_run_segmentation_mocked(client, segment_mocks, tiny_png_bytes, monkeypatch):
    import base64

    import app as m

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


def test_segment_returns_error_when_models_unavailable(client, tiny_png_bytes, monkeypatch):
    import app as m

    monkeypatch.setattr(m, "load_models", lambda: False)
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
    assert data["success"] is False
    assert "models are not available" in data["error"].lower()


def test_segment_rejects_non_image_bytes(client, segment_mocks):
    response = client.post(
        "/segment",
        data={
            "image_file": (BytesIO(b"this is not an image"), "fake.png"),
            "prompt": "food",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False
    assert "valid image" in data["error"].lower()


def test_segment_rejects_oversized_upload(client, segment_mocks):
    import app as m

    oversized = b"\x89PNG\r\n\x1a\n" + bytes(m.MAX_IMAGE_BYTES + 1)
    response = client.post(
        "/segment",
        data={
            "image_file": (BytesIO(oversized), "big.png"),
            "prompt": "food",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    data = response.get_json()
    assert data["success"] is False
    assert "10" in data["error"] or "limit" in data["error"].lower()


def test_api_segment_uses_standard_status_codes(client, tiny_png_bytes, monkeypatch):
    import app as m

    monkeypatch.setattr(m, "load_models", lambda: False)
    response = client.post(
        "/api/v1/segment",
        data={
            "image_file": (BytesIO(tiny_png_bytes), "plate.png"),
            "prompt": "jollof rice",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    data = response.get_json()
    assert data["success"] is False
    assert data["code"] == "models_not_loaded"


def test_segment_requires_api_key_when_configured(client, tiny_png_bytes, monkeypatch):
    import app as m

    client.application.config["SEGMENT_API_KEY"] = "secret"
    monkeypatch.setattr(m, "load_models", lambda: True)
    response = client.post(
        "/api/v1/segment",
        data={
            "image_file": (BytesIO(tiny_png_bytes), "plate.png"),
            "prompt": "jollof rice",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 401
    data = response.get_json()
    assert data["code"] == "unauthorized"

    client.application.config["SEGMENT_API_KEY"] = ""


def test_segment_rate_limiter_when_enabled(client, segment_mocks, tiny_png_bytes, monkeypatch):
    import app as m

    monkeypatch.setattr(
        m,
        "run_segmentation_detailed",
        lambda _image_bytes, _prompt: m.SegmentationResult(
            success=True,
            original_image="orig",
            result_image="result",
        ),
    )
    client.application.config["RATELIMIT_ENABLED"] = True
    client.application.config["SEGMENT_RATE_LIMIT"] = "1/minute"
    m._RATE_LIMIT_BUCKETS.clear()

    for expected_status in (200, 200):
        response = client.post(
            "/segment",
            data={
                "image_file": (BytesIO(tiny_png_bytes), "plate.png"),
                "prompt": "jollof rice",
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == expected_status

    data = response.get_json()
    assert data["success"] is False
    assert data["code"] == "rate_limit_exceeded"
    client.application.config["RATELIMIT_ENABLED"] = False


def test_segment_rejects_image_over_pixel_limit(client, segment_mocks, tiny_png_bytes):
    client.application.config["MAX_IMAGE_PIXELS"] = 4
    response = client.post(
        "/api/v1/segment",
        data={
            "image_file": (BytesIO(tiny_png_bytes), "plate.png"),
            "prompt": "jollof rice",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    data = response.get_json()
    assert data["success"] is False
    assert "too large" in data["error"].lower()
    client.application.config["MAX_IMAGE_PIXELS"] = 2048 * 2048
