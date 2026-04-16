"""Model import / download / load orchestration for GroundingDINO + MobileSAM."""

from __future__ import annotations

import os
import sys
import threading
import traceback
import warnings
from pathlib import Path
from typing import Any, Optional

import torch

warnings.filterwarnings("ignore", category=UserWarning)

ABS_PROJECT_DIR = Path(__file__).parent.parent.absolute()


# ------------------------------------------------------------------ #
# Device                                                               #
# ------------------------------------------------------------------ #
def get_device() -> torch.device:
    try:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    except Exception as e:
        print(f"Warning: Could not initialize CUDA, falling back to CPU: {e}")
        return torch.device("cpu")


_DEVICE: Optional[torch.device] = None
_DEVICE_LOCK = threading.Lock()


def get_device_lazy() -> torch.device:
    global _DEVICE
    with _DEVICE_LOCK:
        if _DEVICE is None:
            _DEVICE = get_device()
            print(f"Initialized device: {_DEVICE}")
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
    print(f"Warning: Could not create directories: {e}")


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
    as_name: Optional[str] = None,
) -> Any:
    try:
        if from_list:
            module = __import__(module_name, fromlist=from_list)
            if len(from_list) == 1:
                return getattr(module, from_list[0])
            return tuple(getattr(module, item) for item in from_list)
        return __import__(module_name)
    except ImportError as e:
        print(f"Import failed for {module_name}: {e}")
    except Exception as e:
        print(f"Unexpected error importing {module_name}: {e}")
    return None


def add_to_path_if_exists(directory: Path) -> bool:
    if directory.exists() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
        print(f"Added {directory} to sys.path")
        return True
    return False


# ------------------------------------------------------------------ #
# Imports                                                              #
# ------------------------------------------------------------------ #
def import_grounding_dino() -> bool:
    global GroundingDINO
    print("Attempting to import GroundingDINO...")

    GroundingDINO = safe_import("groundingdino.util.inference", ["Model"])
    if GroundingDINO:
        print("Successfully imported GroundingDINO (standard method)")
        return True

    for pkg in ["groundingdino", "GroundingDINO", "grounding_dino"]:
        try:
            if safe_import(pkg):
                result = safe_import(f"{pkg}.util.inference", ["Model"])
                if result:
                    GroundingDINO = result
                    print(f"Successfully imported GroundingDINO (package: {pkg})")
                    return True
        except Exception:
            continue

    if GROUNDING_DINO_DIR.exists():
        for path in [GROUNDING_DINO_DIR, GROUNDING_DINO_DIR / "groundingdino"]:
            if add_to_path_if_exists(path):
                result = safe_import("groundingdino.util.inference", ["Model"])
                if result:
                    GroundingDINO = result
                    print(f"Successfully imported GroundingDINO from {path}")
                    return True

    print("All GroundingDINO import methods failed")
    _print_grounding_dino_help()
    return False


def import_mobile_sam() -> bool:
    global sam_model_registry, SamPredictor
    print("Attempting to import MobileSAM...")

    for pkg in ["mobile_sam", "segment_anything"]:
        result = safe_import(pkg, ["sam_model_registry", "SamPredictor"])
        if result and len(result) == 2:
            sam_model_registry, SamPredictor = result
            print(f"Successfully imported MobileSAM (package: {pkg})")
            return True

    if MOBILE_SAM_DIR.exists():
        add_to_path_if_exists(MOBILE_SAM_DIR)
        result = safe_import("mobile_sam", ["sam_model_registry", "SamPredictor"])
        if result and len(result) == 2:
            sam_model_registry, SamPredictor = result
            print("Successfully imported MobileSAM from local directory")
            return True

    print("All MobileSAM import methods failed")
    _print_mobile_sam_help()
    return False


# ------------------------------------------------------------------ #
# Install helpers                                                      #
# ------------------------------------------------------------------ #
def _print_grounding_dino_help():
    print("\nGROUNDINGDINO INSTALLATION REQUIRED:")
    print("  pip install groundingdino-py")
    print("  or")
    print("  pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'")


def _print_mobile_sam_help():
    print("\nMOBILESAM INSTALLATION REQUIRED:")
    print("  pip install mobile-sam")
    print("  or")
    print("  pip install 'git+https://github.com/ChaoningZhang/MobileSAM.git'")


# ------------------------------------------------------------------ #
# File downloads (timeout + progress)                                  #
# ------------------------------------------------------------------ #
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 120


def download_file_robust(
    url: str,
    destination: Path,
    description: str,
    max_retries: int = 3,
) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"{description} already exists at {destination}")
        return True

    print(f"Downloading {description} ...")
    destination.parent.mkdir(parents=True, exist_ok=True)

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

            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        received += len(chunk)
                        if total:
                            pct = received / total * 100
                            print(f"\r  {description}: {pct:.1f}%", end="", flush=True)

            print()
            print(f"{description} downloaded successfully ({received:,} bytes)")
            return True

        except requests.exceptions.ConnectTimeout:
            print(f"\nAttempt {attempt}/{max_retries}: connection timed out for {url}")
        except requests.exceptions.ReadTimeout:
            print(f"\nAttempt {attempt}/{max_retries}: read timed out for {url}")
        except requests.exceptions.RequestException as e:
            print(f"\nAttempt {attempt}/{max_retries}: download failed - {e}")
        except Exception as e:
            print(f"\nAttempt {attempt}/{max_retries}: unexpected error - {e}")

        if attempt < max_retries:
            import time

            wait = 2**attempt
            print(f"Retrying in {wait}s ...")
            time.sleep(wait)

    print(f"All {max_retries} download attempts failed for {description}")
    return False


def find_config_file() -> Optional[Path]:
    for root, _dirs, files in os.walk(GROUNDING_DINO_DIR):
        if "GroundingDINO_SwinT_OGC.py" in files:
            return Path(root) / "GroundingDINO_SwinT_OGC.py"
    return None


def setup_grounding_dino_files() -> tuple[bool, Optional[Path]]:
    print("Setting up GroundingDINO files...")
    checkpoint_url = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth"
    checkpoint_path = GROUNDING_DINO_DIR / "groundingdino_swint_ogc.pth"

    checkpoint_ready = download_file_robust(checkpoint_url, checkpoint_path, "GroundingDINO checkpoint")
    config_path = find_config_file()

    if not config_path:
        print("Config file not found. Please install the GroundingDINO repo correctly.")
        return False, None

    return checkpoint_ready, config_path


def setup_mobile_sam_files() -> bool:
    print("Setting up MobileSAM files...")
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    checkpoint_path = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
    return download_file_robust(url, checkpoint_path, "MobileSAM checkpoint")


# ------------------------------------------------------------------ #
# Model loading                                                        #
# ------------------------------------------------------------------ #
def load_grounding_dino_model(config_path: Path, checkpoint_path: Path) -> Optional[Any]:
    if not GroundingDINO:
        print("GroundingDINO class not available")
        return None
    try:
        print("Loading GroundingDINO model...")
        device = get_device_lazy()
        model = GroundingDINO(str(config_path), str(checkpoint_path), device)
        print("GroundingDINO loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading GroundingDINO: {e}")
        traceback.print_exc()
        return None


def load_mobile_sam_model(checkpoint_path: Path, sam_type: str = "vit_t") -> Optional[Any]:
    if not sam_model_registry or not SamPredictor:
        print("MobileSAM components not available")
        return None
    try:
        device = get_device_lazy()
        sam = sam_model_registry[sam_type](checkpoint=str(checkpoint_path))
        sam.to(device)
        return SamPredictor(sam)
    except Exception as e:
        print(f"Error loading MobileSAM: {e}")
        return None


# ------------------------------------------------------------------ #
# Public initializer (call explicitly - NOT on import)          #
# ------------------------------------------------------------------ #
def initialize(force: bool = False) -> dict[str, bool]:
    """Load both models.

    Safe to call multiple times; subsequent calls are no-ops unless ``force=True``.
    Returns a status dict, e.g. ``{'grounding_dino': True, 'mobile_sam': False}``.
    """
    global grounding_dino_model, sam_predictor, _initialized

    with _init_lock:
        if _initialized and not force:
            return {
                "grounding_dino": grounding_dino_model is not None,
                "mobile_sam": sam_predictor is not None,
            }

        print("\n=== Starting model loading process ===")

        print("\n--- Importing model libraries ---")
        gd_available = import_grounding_dino()
        sam_available = import_mobile_sam()

        print("\n--- Setting up model files ---")
        gd_files_ready, gd_config_path = False, None
        sam_files_ready = False

        if gd_available:
            gd_files_ready, gd_config_path = setup_grounding_dino_files()
        if sam_available:
            sam_files_ready = setup_mobile_sam_files()

        print("\n--- Loading models ---")
        if gd_available and gd_files_ready and gd_config_path:
            cp = GROUNDING_DINO_DIR / "groundingdino_swint_ogc.pth"
            grounding_dino_model = load_grounding_dino_model(gd_config_path, cp)

        if sam_available and sam_files_ready:
            cp = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
            sam_predictor = load_mobile_sam_model(cp)

        _initialized = True

        status = {
            "grounding_dino": grounding_dino_model is not None,
            "mobile_sam": sam_predictor is not None,
        }
        print("\n=== Model loading summary ===")
        for name, ok in status.items():
            print(f"  {name}: {'Loaded' if ok else 'Failed'}")

        return status


if __name__ == "__main__":
    initialize()
