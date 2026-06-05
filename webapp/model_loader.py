"""Model import / download / load orchestration for GroundingDINO + MobileSAM."""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
import threading
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

ABS_PROJECT_DIR = Path(__file__).parent.parent.absolute()
torch: Any | None = None
_TORCH_IMPORT_ERROR: Exception | None = None
_TORCH_LOCK = threading.Lock()


def get_torch() -> Any | None:
    """Import PyTorch lazily so Flask/test import does not stall on startup."""
    global torch, _TORCH_IMPORT_ERROR
    with _TORCH_LOCK:
        if torch is not None:
            return torch
        if _TORCH_IMPORT_ERROR is not None:
            return None
        try:
            torch = importlib.import_module("torch")
        except Exception as e:
            _TORCH_IMPORT_ERROR = e
            logger.warning("PyTorch import failed: %s", e)
            return None
        return torch


def torch_available() -> bool:
    return get_torch() is not None


class ModelState(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


_state = ModelState.NOT_LOADED
_last_error: str | None = None
_last_loaded_at: float | None = None
_last_attempt_at: float | None = None
INITIALIZE_RETRY_SECONDS = int(os.environ.get("INITIALIZE_RETRY_SECONDS", "60"))


def _set_state(state: ModelState, error: str | None = None) -> None:
    global _state, _last_error
    _state = state
    _last_error = error


def get_status() -> dict[str, Any]:
    return {
        "state": _state.value,
        "last_error": _last_error,
        "last_loaded_at": _last_loaded_at,
        "last_attempt_at": _last_attempt_at,
        "models_loaded": {
            "grounding_dino": grounding_dino_model is not None,
            "mobile_sam": sam_predictor is not None,
        },
    }


# ------------------------------------------------------------------ #
# Device                                                               #
# ------------------------------------------------------------------ #
def get_device() -> Any:
    torch_module = get_torch()
    if torch_module is None:
        raise RuntimeError("PyTorch is not available")
    try:
        return torch_module.device("cuda") if torch_module.cuda.is_available() else torch_module.device("cpu")
    except Exception as e:
        logger.warning("Could not initialize CUDA, falling back to CPU: %s", e)
        return torch_module.device("cpu")


_DEVICE: Optional[Any] = None
_DEVICE_LOCK = threading.Lock()


def get_device_lazy() -> Any:
    global _DEVICE
    with _DEVICE_LOCK:
        if _DEVICE is None:
            _DEVICE = get_device()
            logger.info("Initialized device: %s", _DEVICE)
    return _DEVICE


# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #
GROUNDING_DINO_DIR = ABS_PROJECT_DIR / "webapp" / "GroundingDINO"
MOBILE_SAM_DIR = ABS_PROJECT_DIR / "webapp" / "MobileSAM"

try:
    GROUNDING_DINO_DIR.mkdir(parents=True, exist_ok=True)
    MOBILE_SAM_DIR.mkdir(parents=True, exist_ok=True)
    (MOBILE_SAM_DIR / "weights").mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.warning("Could not create directories: %s", e)


# ------------------------------------------------------------------ #
# Model variable singletons                                            #
# ------------------------------------------------------------------ #
GroundingDINO: Optional[Any] = None
sam_model_registry: Optional[dict] = None
SamPredictor: Optional[Any] = None

grounding_dino_model = None
sam_predictor = None

_initialized = False
_init_lock = threading.Lock()


# ------------------------------------------------------------------ #
# Utils                                                                #
# ------------------------------------------------------------------ #
def safe_import(
    module_name: str,
    from_list: Optional[list] = None,
) -> Any:
    try:
        if from_list:
            module = __import__(module_name, fromlist=from_list)
            if len(from_list) == 1:
                return getattr(module, from_list[0])
            return tuple(getattr(module, item) for item in from_list)
        return __import__(module_name)
    except ImportError as e:
        logger.warning("Import failed for %s: %s", module_name, e)
    except Exception as e:
        logger.warning("Unexpected error importing %s: %s", module_name, e)
    return None


def add_to_path_if_exists(directory: Path) -> bool:
    if directory.exists() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
        logger.info("Added %s to sys.path", directory)
        return True
    return False


# ------------------------------------------------------------------ #
# Imports                                                              #
# ------------------------------------------------------------------ #
def import_grounding_dino() -> bool:
    global GroundingDINO
    logger.info("Attempting to import GroundingDINO")

    GroundingDINO = safe_import("groundingdino.util.inference", ["Model"])
    if GroundingDINO:
        logger.info("Successfully imported GroundingDINO (standard method)")
        return True

    for pkg in ["groundingdino", "GroundingDINO", "grounding_dino"]:
        try:
            if safe_import(pkg):
                result = safe_import(f"{pkg}.util.inference", ["Model"])
                if result:
                    GroundingDINO = result
                    logger.info("Successfully imported GroundingDINO (package: %s)", pkg)
                    return True
        except Exception:
            continue

    if GROUNDING_DINO_DIR.exists():
        for path in [GROUNDING_DINO_DIR, GROUNDING_DINO_DIR / "groundingdino"]:
            if add_to_path_if_exists(path):
                result = safe_import("groundingdino.util.inference", ["Model"])
                if result:
                    GroundingDINO = result
                    logger.info("Successfully imported GroundingDINO from %s", path)
                    return True

    logger.error("All GroundingDINO import methods failed")
    _print_grounding_dino_help()
    return False


def import_mobile_sam() -> bool:
    global sam_model_registry, SamPredictor
    logger.info("Attempting to import MobileSAM")

    for pkg in ["mobile_sam", "segment_anything"]:
        result = safe_import(pkg, ["sam_model_registry", "SamPredictor"])
        if result and len(result) == 2:
            sam_model_registry, SamPredictor = result
            logger.info("Successfully imported MobileSAM (package: %s)", pkg)
            return True

    if MOBILE_SAM_DIR.exists():
        add_to_path_if_exists(MOBILE_SAM_DIR)
        result = safe_import("mobile_sam", ["sam_model_registry", "SamPredictor"])
        if result and len(result) == 2:
            sam_model_registry, SamPredictor = result
            logger.info("Successfully imported MobileSAM from local directory")
            return True

    logger.error("All MobileSAM import methods failed")
    _print_mobile_sam_help()
    return False


# ------------------------------------------------------------------ #
# Install helpers                                                      #
# ------------------------------------------------------------------ #
def _print_grounding_dino_help():
    logger.error(
        "GROUNDINGDINO INSTALLATION REQUIRED: pip install groundingdino-py "
        "or pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'"
    )


def _print_mobile_sam_help():
    logger.error(
        "MOBILESAM INSTALLATION REQUIRED: pip install mobile-sam "
        "or pip install 'git+https://github.com/ChaoningZhang/MobileSAM.git'"
    )


# ------------------------------------------------------------------ #
# File downloads (timeout + progress)                                  #
# ------------------------------------------------------------------ #
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120
ALLOW_MODEL_DOWNLOADS = os.environ.get("ALLOW_MODEL_DOWNLOADS", "1").lower() not in {"0", "false", "no"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_valid(
    path: Path,
    description: str,
    expected_sha256: str | None = None,
    min_bytes: int = 1,
) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False

    size = path.stat().st_size
    if size < min_bytes:
        logger.error("%s at %s is too small (%s bytes, expected at least %s)", description, path, size, min_bytes)
        return False

    if expected_sha256:
        actual_sha256 = _sha256_file(path)
        if actual_sha256.lower() != expected_sha256.lower():
            logger.error("%s hash mismatch at %s", description, path)
            return False

    return True


def download_file_robust(
    url: str,
    destination: Path,
    description: str,
    max_retries: int = 3,
    expected_sha256: str | None = None,
    min_bytes: int = 1,
) -> bool:
    if _checkpoint_valid(destination, description, expected_sha256, min_bytes):
        logger.info("%s already exists at %s", description, destination)
        return True
    if destination.exists():
        logger.warning("%s exists but failed validation; it will be replaced", description)

    if not ALLOW_MODEL_DOWNLOADS:
        logger.error("%s missing or invalid and ALLOW_MODEL_DOWNLOADS is disabled", description)
        return False

    logger.info("Downloading %s", description)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")

    import requests

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            received = 0

            with open(tmp_destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        received += len(chunk)
                        if total:
                            pct = received / total * 100
                            logger.info("%s download progress: %.1f%%", description, pct)

            if not _checkpoint_valid(tmp_destination, description, expected_sha256, min_bytes):
                tmp_destination.unlink(missing_ok=True)
                logger.error("%s downloaded but failed validation", description)
                return False

            os.replace(tmp_destination, destination)
            logger.info("%s downloaded successfully (%s bytes)", description, f"{received:,}")
            return True

        except requests.exceptions.ConnectTimeout:
            logger.warning("Attempt %s/%s: connection timed out for %s", attempt, max_retries, url)
        except requests.exceptions.ReadTimeout:
            logger.warning("Attempt %s/%s: read timed out for %s", attempt, max_retries, url)
        except requests.exceptions.RequestException as e:
            logger.warning("Attempt %s/%s: download failed - %s", attempt, max_retries, e)
        except Exception as e:
            logger.warning("Attempt %s/%s: unexpected error - %s", attempt, max_retries, e)
        finally:
            tmp_destination.unlink(missing_ok=True)

        if attempt < max_retries:
            import time

            wait = 2**attempt
            logger.info("Retrying in %ss", wait)
            time.sleep(wait)

    logger.error("All %s download attempts failed for %s", max_retries, description)
    return False


def find_config_file() -> Optional[Path]:
    for root, _dirs, files in os.walk(GROUNDING_DINO_DIR):
        if "GroundingDINO_SwinT_OGC.py" in files:
            return Path(root) / "GroundingDINO_SwinT_OGC.py"
    return None


def setup_grounding_dino_files() -> tuple[bool, Optional[Path]]:
    logger.info("Setting up GroundingDINO files")
    checkpoint_url = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
    checkpoint_path = GROUNDING_DINO_DIR / "groundingdino_swint_ogc.pth"

    checkpoint_ready = download_file_robust(
        checkpoint_url,
        checkpoint_path,
        "GroundingDINO checkpoint",
        expected_sha256=os.environ.get("GROUNDING_DINO_SHA256"),
        min_bytes=int(os.environ.get("GROUNDING_DINO_MIN_BYTES", str(100 * 1024 * 1024))),
    )
    config_path = find_config_file()

    if not config_path:
        logger.error("Config file not found. Please install the GroundingDINO repo correctly.")
        return False, None

    return checkpoint_ready, config_path


def setup_mobile_sam_files() -> bool:
    logger.info("Setting up MobileSAM files")
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    checkpoint_path = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
    return download_file_robust(
        url,
        checkpoint_path,
        "MobileSAM checkpoint",
        expected_sha256=os.environ.get("MOBILE_SAM_SHA256"),
        min_bytes=int(os.environ.get("MOBILE_SAM_MIN_BYTES", str(10 * 1024 * 1024))),
    )


# ------------------------------------------------------------------ #
# Model loading                                                        #
# ------------------------------------------------------------------ #
def load_grounding_dino_model(config_path: Path, checkpoint_path: Path) -> Optional[Any]:
    if not GroundingDINO:
        logger.error("GroundingDINO class not available")
        return None
    try:
        logger.info("Loading GroundingDINO model")
        device = get_device_lazy()
        model = GroundingDINO(str(config_path), str(checkpoint_path), device)
        logger.info("GroundingDINO loaded successfully")
        return model
    except Exception as e:
        logger.exception("Error loading GroundingDINO: %s", e)
        return None


def load_mobile_sam_model(checkpoint_path: Path, sam_type: str = "vit_t") -> Optional[Any]:
    if not sam_model_registry or not SamPredictor:
        logger.error("MobileSAM components not available")
        return None
    try:
        device = get_device_lazy()
        sam = sam_model_registry[sam_type](checkpoint=str(checkpoint_path))
        sam.to(device)
        return SamPredictor(sam)
    except Exception as e:
        logger.exception("Error loading MobileSAM: %s", e)
        return None


# ------------------------------------------------------------------ #
# Public initializer (call explicitly - NOT on import)          #
# ------------------------------------------------------------------ #
def initialize(force: bool = False) -> dict[str, bool]:
    """Load both models.

    Safe to call multiple times; subsequent calls are no-ops unless ``force=True``.
    Returns a status dict, e.g. ``{'grounding_dino': True, 'mobile_sam': False}``.
    """
    global _initialized, _last_attempt_at, _last_loaded_at, grounding_dino_model, sam_predictor

    with _init_lock:
        if _initialized and not force:
            return {
                "grounding_dino": grounding_dino_model is not None,
                "mobile_sam": sam_predictor is not None,
            }
        if _state == ModelState.FAILED and _last_attempt_at is not None and not force:
            import time

            if time.time() - _last_attempt_at < INITIALIZE_RETRY_SECONDS:
                return {
                    "grounding_dino": grounding_dino_model is not None,
                    "mobile_sam": sam_predictor is not None,
                }

        import time

        _last_attempt_at = time.time()
        _set_state(ModelState.LOADING)
        logger.info("Starting model loading process")

        logger.info("Importing model libraries")
        gd_available = import_grounding_dino()
        sam_available = import_mobile_sam()

        logger.info("Setting up model files")
        gd_files_ready, gd_config_path = False, None
        sam_files_ready = False

        if gd_available:
            gd_files_ready, gd_config_path = setup_grounding_dino_files()
        if sam_available:
            sam_files_ready = setup_mobile_sam_files()

        logger.info("Loading models")
        if gd_available and gd_files_ready and gd_config_path:
            cp = GROUNDING_DINO_DIR / "groundingdino_swint_ogc.pth"
            grounding_dino_model = load_grounding_dino_model(gd_config_path, cp)

        if sam_available and sam_files_ready:
            cp = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
            sam_predictor = load_mobile_sam_model(cp)

        status = {
            "grounding_dino": grounding_dino_model is not None,
            "mobile_sam": sam_predictor is not None,
        }
        if all(status.values()):
            _initialized = True
            _last_loaded_at = time.time()
            _set_state(ModelState.READY)
        else:
            _initialized = False
            _set_state(ModelState.FAILED, f"Model loading incomplete: {status}")

        logger.info("Model loading summary")
        for name, ok in status.items():
            logger.info("%s: %s", name, "Loaded" if ok else "Failed")

        return status


if __name__ == "__main__":
    initialize()
