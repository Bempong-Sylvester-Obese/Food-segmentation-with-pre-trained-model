"""Flask entrypoint for the food segmentation webapp."""

from __future__ import annotations

import base64
import gc
import logging
import os
import sys as _sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

from flask import Flask, g, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from . import model_loader
except ImportError:  # pragma: no cover - supports `python webapp/app.py`
    import model_loader

cv2: Any | None = None
np: Any | None = None
torch = None
TORCH_AVAILABLE = False
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIM = 2048
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(MAX_IMAGE_DIM * MAX_IMAGE_DIM)))
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "200"))
SEGMENT_RATE_LIMIT = os.environ.get("SEGMENT_RATE_LIMIT", "5/minute")
SEGMENT_API_KEY = os.environ.get("SEGMENT_API_KEY", "")
SECURITY_HEADERS_ENABLED = os.environ.get("SECURITY_HEADERS_ENABLED", "1").lower() not in {"0", "false", "no"}
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)
FRONTEND_DIST = Path(__file__).resolve().parent / "static" / "frontend"


def _get_cv2() -> Any:
    global cv2
    if cv2 is None:
        import cv2 as cv2_module

        cv2 = cv2_module
    return cv2


def _get_np() -> Any:
    global np
    if np is None:
        import numpy as np_module

        np = np_module
    return np


# Magic-byte signatures for the formats listed in ALLOWED_EXTENSIONS. Used as a
# cheap content sniff so a renamed non-image cannot slip past the extension
# check alone.
_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
)


def _sniff_image_format(data: bytes) -> str | None:
    for sig, kind in _MAGIC_SIGNATURES:
        if data.startswith(sig):
            return kind
    return None


class SegmentFailure(str, Enum):
    EMPTY_IMAGE = "empty_image"
    EMPTY_PROMPT = "empty_prompt"
    PROMPT_TOO_LONG = "prompt_too_long"
    IMAGE_TOO_LARGE = "image_too_large"
    INVALID_IMAGE = "invalid_image"
    IMAGE_DIMENSIONS_TOO_LARGE = "image_dimensions_too_large"
    TORCH_UNAVAILABLE = "torch_unavailable"
    MODELS_NOT_LOADED = "models_not_loaded"
    DETECTION_FAILED = "detection_failed"
    NO_DETECTIONS = "no_detections"
    SAM_FAILED = "sam_failed"
    POST_PROCESSING_FAILED = "post_processing_failed"
    UNEXPECTED_ERROR = "unexpected_error"


@dataclass(slots=True)
class SegmentationResult:
    success: bool
    original_image: str | None = None
    result_image: str | None = None
    reason: SegmentFailure | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


_INFERENCE_LOCK = threading.Lock()
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


app = Flask(__name__)
# Reject oversize uploads at the Werkzeug layer, before the whole body is
# buffered into memory. `run_segmentation` still checks the same limit as a
# defence in depth for callers that bypass Flask (e.g. direct function use).
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
app.config["SEGMENT_API_KEY"] = SEGMENT_API_KEY
app.config["SEGMENT_RATE_LIMIT"] = SEGMENT_RATE_LIMIT
app.config["MAX_IMAGE_PIXELS"] = MAX_IMAGE_PIXELS
app.config["MAX_PROMPT_CHARS"] = MAX_PROMPT_CHARS
app.config["SECURITY_HEADERS_ENABLED"] = SECURITY_HEADERS_ENABLED


def _json_error(message: str, status_code: int = 200, code: str | None = None):
    payload = {"success": False, "error": message}
    if code:
        payload["code"] = code
    return jsonify(payload), status_code


def _parse_rate_limit(limit: str) -> tuple[int, int]:
    try:
        count, period = limit.split("/", 1)
        seconds = {"second": 1, "minute": 60, "hour": 3600}.get(period.strip().lower())
        if seconds is None:
            return 5, 60
        return max(1, int(count)), seconds
    except Exception:
        return 5, 60


def _rate_limit_key() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.remote_addr or "unknown"


def _check_segment_rate_limit() -> tuple[bool, int]:
    if app.config.get("TESTING") and not app.config.get("RATELIMIT_ENABLED", False):
        return True, 0

    limit = str(app.config.get("SEGMENT_RATE_LIMIT", SEGMENT_RATE_LIMIT))
    max_requests, window_seconds = _parse_rate_limit(limit)
    now = time.time()
    key = _rate_limit_key()

    with _RATE_LIMIT_LOCK:
        bucket = _RATE_LIMIT_BUCKETS[key]
        while bucket and bucket[0] <= now - window_seconds:
            bucket.popleft()
        if len(bucket) >= max_requests:
            retry_after = max(1, int(window_seconds - (now - bucket[0])))
            return False, retry_after
        bucket.append(now)
    return True, 0


def _require_segment_auth() -> tuple[bool, str | None]:
    expected_key = app.config.get("SEGMENT_API_KEY") or ""
    if not expected_key:
        return True, None

    supplied_key = request.headers.get("X-API-Key")
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        supplied_key = auth_header.removeprefix("Bearer ").strip()

    if supplied_key == expected_key:
        return True, None
    return False, "A valid API key is required."


def validate_image_payload(image_bytes: bytes) -> str | None:
    """Verify content type and dimensions before expensive OpenCV decoding."""
    if _sniff_image_format(image_bytes) is None:
        return "File does not look like a valid image."

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                return "Image dimensions are invalid."
            if width * height > int(app.config.get("MAX_IMAGE_PIXELS", MAX_IMAGE_PIXELS)):
                return "Image dimensions are too large."
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        return "File does not look like a valid image."

    return None


def validate_prompt(prompt: str) -> str | None:
    if not prompt:
        return "Please provide a prompt."
    if len(prompt) > int(app.config.get("MAX_PROMPT_CHARS", MAX_PROMPT_CHARS)):
        return f"Prompt must be {app.config.get('MAX_PROMPT_CHARS', MAX_PROMPT_CHARS)} characters or fewer."
    return None


@app.before_request
def _start_request_timer():
    g.request_id = request.headers.get("X-Request-ID", str(uuid4()))
    g.request_started_at = time.perf_counter()


@app.after_request
def _finish_request(response):
    duration_ms = (time.perf_counter() - g.get("request_started_at", time.perf_counter())) * 1000
    response.headers["X-Request-ID"] = g.get("request_id", "")
    if app.config.get("SECURITY_HEADERS_ENABLED", True):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        )
    logger.info(
        "request_complete method=%s path=%s status=%s duration_ms=%.2f request_id=%s",
        request.method,
        request.path,
        response.status_code,
        duration_ms,
        g.get("request_id", ""),
    )
    return response


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_e: RequestEntityTooLarge):
    return _json_error("Image exceeds 10 MB limit.", 413, "image_too_large")


# Module-level handles. Kept here (not inside model_loader) so tests can
# monkeypatch them directly on the `app` module.
grounding_dino = None
sam_predictor = None
device = None


def load_models() -> bool:
    """Initialize both models and mirror their handles onto this module.

    Safe to call repeatedly; `model_loader.initialize()` is itself idempotent.
    Returns True only when both models loaded successfully.
    """
    global TORCH_AVAILABLE, device, grounding_dino, sam_predictor, torch
    try:
        status = model_loader.initialize()
    except Exception as e:
        logger.exception("Failed to initialise models: %s", e)
        return False

    torch = model_loader.get_torch()
    TORCH_AVAILABLE = torch is not None
    grounding_dino = model_loader.grounding_dino_model
    sam_predictor = model_loader.sam_predictor
    try:
        device = model_loader.get_device_lazy()
    except Exception as e:
        logger.warning("Failed to resolve device: %s", e)
        device = "cpu"

    return all(status.values())


def _model_status_payload() -> dict:
    loader_status = model_loader.get_status() if hasattr(model_loader, "get_status") else {}
    models_loaded = {
        "grounding_dino": grounding_dino is not None,
        "sam_predictor": sam_predictor is not None,
    }
    state = "ready" if all(models_loaded.values()) else loader_status.get("state", "not_loaded")
    return {
        "status": state,
        "models_loaded": models_loaded,
        "device": str(device),
        "sam_predictor_type": type(sam_predictor).__name__ if sam_predictor else None,
        "torch_available": TORCH_AVAILABLE or model_loader.torch is not None,
        "loader": loader_status,
    }


def is_allowed_file(filename: str | None) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_segmentation_detailed(image_bytes: bytes, prompt: str) -> SegmentationResult:
    """Run GroundingDINO + MobileSAM over `image_bytes` guided by `prompt`.

    The entire body is wrapped in ``try/finally`` so CUDA caches and Python GC
    are always released, even on validation shortcuts or inference errors.
    """
    start_time = time.time()
    global TORCH_AVAILABLE, torch

    def failure(reason: SegmentFailure, error: str) -> SegmentationResult:
        return SegmentationResult(
            success=False,
            reason=reason,
            error=error,
            metadata={"duration_ms": round((time.time() - start_time) * 1000, 2)},
        )

    try:
        if not image_bytes:
            logger.info("run_segmentation: empty image bytes")
            return failure(SegmentFailure.EMPTY_IMAGE, "No image data provided.")
        if not prompt or not prompt.strip():
            logger.info("run_segmentation: empty prompt")
            return failure(SegmentFailure.EMPTY_PROMPT, "Please provide a prompt.")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            logger.info("run_segmentation: image exceeds max size")
            return failure(SegmentFailure.IMAGE_TOO_LARGE, "Image exceeds 10 MB limit.")
        torch = model_loader.get_torch()
        TORCH_AVAILABLE = torch is not None
        if not TORCH_AVAILABLE or torch is None:
            logger.error("run_segmentation: PyTorch not available")
            return failure(SegmentFailure.TORCH_UNAVAILABLE, "PyTorch is not available.")
        _get_cv2()
        _get_np()

        if grounding_dino is None or sam_predictor is None:
            logger.error("run_segmentation: models not loaded")
            return failure(SegmentFailure.MODELS_NOT_LOADED, "Models are not loaded.")

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning("run_segmentation: cv2.imdecode failed: %s", e)
            return failure(SegmentFailure.INVALID_IMAGE, "File does not look like a valid image.")

        if source_image is None:
            logger.warning("run_segmentation: invalid image format")
            return failure(SegmentFailure.INVALID_IMAGE, "File does not look like a valid image.")

        height, width = source_image.shape[:2]
        if height == 0 or width == 0:
            logger.warning("run_segmentation: invalid image dimensions")
            return failure(SegmentFailure.INVALID_IMAGE, "Image dimensions are invalid.")

        if max(height, width) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            source_image = cv2.resize(source_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = source_image.shape[:2]
            logger.info("Resized image to: %sx%s", width, height)

        logger.info("Processing image: %sx%s", width, height)

        try:
            detections, phrases = grounding_dino.predict_with_caption(
                image=source_image,
                caption=prompt,
                box_threshold=0.35,
                text_threshold=0.25,
            )
        except Exception as e:
            logger.exception("run_segmentation: GroundingDINO inference failed: %s", e)
            return failure(SegmentFailure.DETECTION_FAILED, "Object detection failed.")

        if detections is None or len(detections.xyxy) == 0:
            logger.info("No objects detected for prompt")
            return failure(SegmentFailure.NO_DETECTIONS, "No matching objects were detected.")

        logger.info("Detected %s objects", len(detections.xyxy))

        try:
            current_device = device or model_loader.get_device_lazy()
            sam_predictor.set_image(source_image)

            input_boxes = torch.tensor(detections.xyxy, device=current_device)
            if input_boxes.dim() == 1:
                input_boxes = input_boxes.unsqueeze(0)
        except Exception as e:
            logger.exception("run_segmentation: SAM setup failed: %s", e)
            return failure(SegmentFailure.SAM_FAILED, "Segmentation setup failed.")

        binary_mask = None

        # Preferred path: segment every detection in one forward pass so
        # multi-object prompts ("rice, plantain") produce a combined mask.
        try:
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, source_image.shape[:2])
            masks, _scores, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            if masks is not None and len(masks) > 0:
                mask_np = masks.squeeze(1).detach().cpu().numpy()
                combined = (mask_np > 0).any(axis=0)
                binary_mask = combined.astype(np.uint8) * 255
                logger.info("Mask generated via predict_torch for %s boxes, shape=%s", len(masks), binary_mask.shape)
        except Exception as e:
            logger.warning("predict_torch SAM failed: %s", e)

        # Fallback for older SAM/MobileSAM builds without `predict_torch`: use
        # the centre-point prompt on the first detection only.
        if binary_mask is None:
            try:
                logger.info("Falling back to centre-point prompt")
                box = detections.xyxy[0]
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)

                masks, _scores, _ = sam_predictor.predict(
                    point_coords=np.array([[center_x, center_y]]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                if masks is not None and len(masks) > 0:
                    binary_mask = (masks[0] > 0).astype(np.uint8) * 255
                    logger.info("Mask generated via point prompt, shape=%s", binary_mask.shape)
            except Exception as e:
                logger.warning("Point-prompt SAM also failed: %s", e)

        if binary_mask is None:
            logger.warning("run_segmentation: SAM could not generate a mask")
            return failure(SegmentFailure.SAM_FAILED, "Segmentation mask generation failed.")

        try:
            result_image = source_image.copy()
            overlay = np.zeros_like(source_image)
            overlay[binary_mask > 0] = [0, 255, 0]
            result_image = cv2.addWeighted(result_image, 1, overlay, 0.3, 0)

            for raw_box in detections.xyxy:
                x1, y1, x2, y2 = map(int, raw_box)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = prompt
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(result_image, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
                cv2.putText(
                    result_image,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            _, orig_buf = cv2.imencode(".png", source_image)
            _, result_buf = cv2.imencode(".png", result_image)
            original_b64 = base64.b64encode(orig_buf.tobytes()).decode("utf-8")
            result_b64 = base64.b64encode(result_buf.tobytes()).decode("utf-8")
        except Exception as e:
            logger.exception("run_segmentation: post-processing failed: %s", e)
            return failure(SegmentFailure.POST_PROCESSING_FAILED, "Segmentation post-processing failed.")

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info("Segmentation completed in %.2fms", duration_ms)
        return SegmentationResult(
            success=True,
            original_image=original_b64,
            result_image=result_b64,
            metadata={
                "duration_ms": duration_ms,
                "image": {"width": width, "height": height},
                "detections": {
                    "count": int(len(detections.xyxy)),
                    "boxes": np.asarray(detections.xyxy).tolist(),
                    "confidence": np.asarray(detections.confidence).tolist()
                    if getattr(detections, "confidence", None) is not None
                    else [],
                    "phrases": phrases if isinstance(phrases, list) else [],
                },
                "device": str(device or model_loader.get_device_lazy()),
            },
        )
    except Exception as e:
        logger.exception("run_segmentation failed: %s", e)
        return failure(SegmentFailure.UNEXPECTED_ERROR, "An unexpected segmentation error occurred.")
    finally:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def run_segmentation(image_bytes: bytes, prompt: str):
    """Compatibility wrapper returning the historical ``(original, result)`` tuple."""
    result = run_segmentation_detailed(image_bytes, prompt)
    return result.original_image, result.result_image


@app.route("/")
def index():
    if (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(FRONTEND_DIST, "index.html")
    return render_template("index.html")


@app.route("/health")
def health_check():
    return jsonify(_model_status_payload())


@app.route("/livez")
def livez():
    return jsonify({"status": "alive", "request_id": g.get("request_id")})


@app.route("/readyz")
def readyz():
    payload = _model_status_payload()
    ready = payload["models_loaded"]["grounding_dino"] and payload["models_loaded"]["sam_predictor"]
    return jsonify(payload), 200 if ready else 503


@app.route("/api/v1/models/status")
def api_model_status():
    return jsonify(_model_status_payload())


@app.route("/api/v1/config")
def api_config():
    return jsonify(
        {
            "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
            "max_image_bytes": MAX_IMAGE_BYTES,
            "max_image_dim": MAX_IMAGE_DIM,
            "max_image_pixels": int(app.config.get("MAX_IMAGE_PIXELS", MAX_IMAGE_PIXELS)),
            "max_prompt_chars": int(app.config.get("MAX_PROMPT_CHARS", MAX_PROMPT_CHARS)),
            "rate_limit": app.config.get("SEGMENT_RATE_LIMIT", SEGMENT_RATE_LIMIT),
            "api_key_required": bool(app.config.get("SEGMENT_API_KEY")),
        }
    )


@app.route("/api/v1/health")
def api_health():
    return livez()


def _status_for_reason(reason: SegmentFailure | None) -> int:
    return {
        SegmentFailure.EMPTY_IMAGE: 400,
        SegmentFailure.EMPTY_PROMPT: 400,
        SegmentFailure.PROMPT_TOO_LONG: 400,
        SegmentFailure.IMAGE_TOO_LARGE: 413,
        SegmentFailure.INVALID_IMAGE: 400,
        SegmentFailure.IMAGE_DIMENSIONS_TOO_LARGE: 413,
        SegmentFailure.TORCH_UNAVAILABLE: 503,
        SegmentFailure.MODELS_NOT_LOADED: 503,
        SegmentFailure.NO_DETECTIONS: 422,
        SegmentFailure.DETECTION_FAILED: 500,
        SegmentFailure.SAM_FAILED: 500,
        SegmentFailure.POST_PROCESSING_FAILED: 500,
        SegmentFailure.UNEXPECTED_ERROR: 500,
    }.get(reason, 500)


def _segment_response(*, standard_status: bool):
    rate_allowed, retry_after = _check_segment_rate_limit()
    if not rate_allowed:
        response = _json_error("Rate limit exceeded. Try again later.", 429, "rate_limit_exceeded")
        response[0].headers["Retry-After"] = str(retry_after)
        if standard_status:
            return response
        return _json_error("Rate limit exceeded. Try again later.", 200, "rate_limit_exceeded")

    auth_allowed, auth_error = _require_segment_auth()
    if not auth_allowed:
        return _json_error(auth_error or "Unauthorized.", 401 if standard_status else 200, "unauthorized")

    if not load_models():
        return _json_error(
            "Models are not available. Check /readyz for details.",
            503 if standard_status else 200,
            SegmentFailure.MODELS_NOT_LOADED.value,
        )

    if "image_file" not in request.files:
        return _json_error("No image file provided.", 400 if standard_status else 200, SegmentFailure.EMPTY_IMAGE.value)

    image_file = request.files["image_file"]
    prompt = request.form.get("prompt", "").strip()

    if not image_file or not image_file.filename:
        return _json_error(
            "Please select a valid image file.",
            400 if standard_status else 200,
            SegmentFailure.EMPTY_IMAGE.value,
        )

    if not is_allowed_file(image_file.filename):
        return _json_error(
            "Upload a valid image file (PNG, JPG, JPEG, GIF, BMP).",
            400 if standard_status else 200,
            SegmentFailure.INVALID_IMAGE.value,
        )

    prompt_error = validate_prompt(prompt)
    if prompt_error:
        code = SegmentFailure.EMPTY_PROMPT.value if not prompt else SegmentFailure.PROMPT_TOO_LONG.value
        return _json_error(prompt_error, 400 if standard_status else 200, code)

    image_bytes = image_file.read()
    if not image_bytes:
        return _json_error("No image data provided.", 400 if standard_status else 200, SegmentFailure.EMPTY_IMAGE.value)

    validation_error = validate_image_payload(image_bytes)
    if validation_error:
        status_code = 413 if "too large" in validation_error else 400
        return _json_error(
            validation_error,
            status_code if standard_status else 200,
            SegmentFailure.INVALID_IMAGE.value,
        )

    try:
        with _INFERENCE_LOCK:
            result = run_segmentation_detailed(image_bytes, prompt)
    except Exception as e:
        logger.exception("Unexpected error in /segment: %s", e)
        return _json_error("An unexpected server error occurred.", 500, SegmentFailure.UNEXPECTED_ERROR.value)

    if not result.success:
        message = result.error or f"Could not produce a segmentation for '{prompt}'. Try another image or prompt."
        status_code = _status_for_reason(result.reason) if standard_status else 200
        payload = {
            "success": False,
            "error": message,
            "code": result.reason.value if result.reason else "segmentation_failed",
            "metadata": result.metadata,
        }
        return jsonify(payload), status_code

    return jsonify(
        {
            "success": True,
            "original_image": result.original_image,
            "result_image": result.result_image,
            "metadata": result.metadata,
        }
    )


@app.route("/segment", methods=["POST"])
def segment():
    return _segment_response(standard_status=False)


@app.route("/api/v1/segment", methods=["POST"])
def api_segment():
    return _segment_response(standard_status=True)


def _should_eager_load() -> bool:
    """Eager-load only outside pytest so test runs stay fast and hermetic."""
    if os.environ.get("SKIP_STARTUP_LOAD"):
        return False
    if "pytest" in _sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return True


# Eager load at import time so `gunicorn --preload` warms the workers once,
# rather than stalling the first HTTP request.
if _should_eager_load():
    try:
        load_models()
    except Exception as e:
        logger.exception("Startup model load failed; will retry on first request: %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=False, host="0.0.0.0", port=port)
