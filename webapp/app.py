"""Flask entrypoint for the food segmentation webapp."""

from __future__ import annotations

import base64
import gc
import os
import sys as _sys
import time
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None

import model_loader

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_DIM = 2048

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


app = Flask(__name__)
# Reject oversize uploads at the Werkzeug layer, before the whole body is
# buffered into memory. `run_segmentation` still checks the same limit as a
# defence in depth for callers that bypass Flask (e.g. direct function use).
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_e: RequestEntityTooLarge):
    return (
        jsonify({"success": False, "error": "Image exceeds 10 MB limit."}),
        413,
    )


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
    global grounding_dino, sam_predictor, device
    try:
        status = model_loader.initialize()
    except Exception as e:
        print(f"Failed to initialise models: {e}")
        traceback.print_exc()
        return False

    grounding_dino = model_loader.grounding_dino_model
    sam_predictor = model_loader.sam_predictor
    try:
        device = model_loader.get_device_lazy()
    except Exception as e:
        print(f"Failed to resolve device: {e}")
        device = "cpu"

    return all(status.values())


def is_allowed_file(filename: str | None) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_segmentation(image_bytes: bytes, prompt: str):
    """Run GroundingDINO + MobileSAM over `image_bytes` guided by `prompt`.

    Returns ``(original_b64, result_b64)`` on success. Returns ``(None, None)``
    for any validation or inference failure; callers decide how to surface that.

    The entire body is wrapped in ``try/finally`` so CUDA caches and Python GC
    are always released, even on validation shortcuts or inference errors.
    """
    start_time = time.time()

    try:
        if not image_bytes:
            print("run_segmentation: empty image bytes")
            return None, None
        if not prompt or not prompt.strip():
            print("run_segmentation: empty prompt")
            return None, None
        if len(image_bytes) > MAX_IMAGE_BYTES:
            print("run_segmentation: image exceeds max size")
            return None, None
        if not TORCH_AVAILABLE or torch is None:
            print("run_segmentation: PyTorch not available")
            return None, None

        if grounding_dino is None or sam_predictor is None:
            print("run_segmentation: models not loaded")
            return None, None

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"run_segmentation: cv2.imdecode failed: {e}")
            return None, None

        if source_image is None:
            print("run_segmentation: invalid image format")
            return None, None

        height, width = source_image.shape[:2]
        if height == 0 or width == 0:
            print("run_segmentation: invalid image dimensions")
            return None, None

        if max(height, width) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            source_image = cv2.resize(source_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = source_image.shape[:2]
            print(f"Resized image to: {width}x{height}")

        print(f"Processing image: {width}x{height}")

        try:
            detections, _phrases = grounding_dino.predict_with_caption(
                image=source_image,
                caption=prompt,
                box_threshold=0.35,
                text_threshold=0.25,
            )
        except Exception as e:
            print(f"run_segmentation: GroundingDINO inference failed: {e}")
            traceback.print_exc()
            return None, None

        if detections is None or len(detections.xyxy) == 0:
            print(f"No objects detected for prompt: '{prompt}'")
            return None, None

        print(f"Detected {len(detections.xyxy)} objects, confidence: {detections.confidence}")

        try:
            current_device = device or model_loader.get_device_lazy()
            sam_predictor.set_image(source_image)

            input_boxes = torch.tensor(detections.xyxy, device=current_device)
            if input_boxes.dim() == 1:
                input_boxes = input_boxes.unsqueeze(0)
        except Exception as e:
            print(f"run_segmentation: SAM setup failed: {e}")
            traceback.print_exc()
            return None, None

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
                print(f"Mask generated via predict_torch for {len(masks)} boxes, shape: {binary_mask.shape}")
        except Exception as e:
            print(f"predict_torch SAM failed: {e}")

        # Fallback for older SAM/MobileSAM builds without `predict_torch`: use
        # the centre-point prompt on the first detection only.
        if binary_mask is None:
            try:
                print("Falling back to centre-point prompt ...")
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
                    print(f"Mask generated via point prompt, shape: {binary_mask.shape}")
            except Exception as e:
                print(f"Point-prompt SAM also failed: {e}")

        if binary_mask is None:
            print("run_segmentation: SAM could not generate a mask")
            return None, None

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
            print(f"run_segmentation: post-processing failed: {e}")
            traceback.print_exc()
            return None, None

        print(f"Segmentation completed in {time.time() - start_time:.2f}s")
        return original_b64, result_b64
    except Exception as e:
        print(f"run_segmentation failed: {e}")
        traceback.print_exc()
        return None, None
    finally:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health_check():
    try:
        load_models()
        return jsonify(
            {
                "status": "healthy",
                "models_loaded": {
                    "grounding_dino": grounding_dino is not None,
                    "sam_predictor": sam_predictor is not None,
                },
                "device": str(device),
                "sam_predictor_type": type(sam_predictor).__name__ if sam_predictor else None,
                "torch_available": TORCH_AVAILABLE,
            }
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "torch_available": TORCH_AVAILABLE,
                }
            ),
            500,
        )


@app.route("/segment", methods=["POST"])
def segment():
    if not load_models():
        return jsonify(
            {
                "success": False,
                "error": "Models are not available. Check /health for details.",
            }
        )

    if "image_file" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."})

    image_file = request.files["image_file"]
    prompt = request.form.get("prompt", "").strip()

    if not image_file or not image_file.filename:
        return jsonify({"success": False, "error": "Please select a valid image file."})

    if not is_allowed_file(image_file.filename):
        return jsonify(
            {
                "success": False,
                "error": "Upload a valid image file (PNG, JPG, JPEG, GIF, BMP).",
            }
        )

    if not prompt:
        return jsonify({"success": False, "error": "Please provide a prompt."})

    image_bytes = image_file.read()
    if not image_bytes:
        return jsonify({"success": False, "error": "No image data provided."})

    if _sniff_image_format(image_bytes) is None:
        return jsonify(
            {
                "success": False,
                "error": "File does not look like a valid image.",
            }
        )

    try:
        original_b64, result_b64 = run_segmentation(image_bytes, prompt)
    except Exception as e:
        print(f"Unexpected error in /segment: {e}")
        traceback.print_exc()
        return (
            jsonify({"success": False, "error": "An unexpected server error occurred."}),
            500,
        )

    if original_b64 is None:
        # run_segmentation returns (None, None) for many distinct failure modes
        # (decode failure, GroundingDINO crash, SAM crash, post-processing, as
        # well as "no detections"). Avoid the misleading "No '<prompt>'
        # detected" phrasing and keep the message applicable to all of them.
        return jsonify(
            {
                "success": False,
                "error": (f"Could not produce a segmentation for '{prompt}'. Try another image or prompt."),
            }
        )

    return jsonify(
        {
            "success": True,
            "original_image": original_b64,
            "result_image": result_b64,
        }
    )


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
        print(f"Startup model load failed (will retry on first request): {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=False, host="0.0.0.0", port=port)
