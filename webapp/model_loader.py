import supervision as sv
import sys
import os
import torch
import urllib.request
from typing import Optional, Union, Any
from pathlib import Path
import warnings
import traceback
warnings.filterwarnings('ignore')

ABS_PROJECT_DIR = Path(__file__).parent.parent.absolute()

def get_device():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
    except Exception as e:
        print(f"Warning: Could not initialize CUDA device, falling back to CPU: {e}")
        return torch.device("cpu")

DEVICE = None

GROUNDING_DINO_DIR = ABS_PROJECT_DIR / "webapp" / "GroundingDINO"
MOBILE_SAM_DIR = ABS_PROJECT_DIR / "webapp" / "MobileSAM"

# Directories
try:
    GROUNDING_DINO_DIR.mkdir(parents=True, exist_ok=True)
    MOBILE_SAM_DIR.mkdir(parents=True, exist_ok=True)
    (MOBILE_SAM_DIR / "weights").mkdir(parents=True, exist_ok=True)
    print("Directories created successfully")
except Exception as e:
    print(f"Warning: Could not create directories: {e}")

# Initialize model variables
GroundingDINO: Optional[type] = None
sam_model_registry: Optional[dict] = None
SamPredictor: Optional[type] = None


# ---------------- UTILS ----------------
def safe_import(module_name: str, from_list: list = None, as_name: str = None):
    try:
        if from_list:
            module = __import__(module_name, fromlist=from_list)
            if len(from_list) == 1:
                return getattr(module, from_list[0])
            else:
                return tuple(getattr(module, item) for item in from_list)
        else:
            return __import__(module_name)
    except ImportError as e:
        print(f"Import failed for {module_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error importing {module_name}: {e}")
        return None

def add_to_path_if_exists(directory: Path) -> bool:
    if directory.exists() and str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
        print(f"Added {directory} to sys.path")
        return True
    return False


# ---------------- IMPORTS ----------------
def import_grounding_dino() -> bool:
    global GroundingDINO
    
    print("Attempting to import GroundingDINO...")
    try:
        GroundingDINO = safe_import('groundingdino.util.inference', ['Model'])
        if GroundingDINO:
            print("Successfully imported GroundingDINO (standard method)")
            return True
    except Exception:
        pass
    
    package_variations = ['groundingdino', 'GroundingDINO', 'grounding_dino']
    for pkg in package_variations:
        try:
            module = safe_import(pkg)
            if module:
                inference_module = safe_import(f'{pkg}.util.inference', ['Model'])
                if inference_module:
                    GroundingDINO = inference_module
                    print(f"Successfully imported GroundingDINO (package: {pkg})")
                    return True
        except Exception:
            continue
    
    if GROUNDING_DINO_DIR.exists():
        add_to_path_if_exists(GROUNDING_DINO_DIR)
        for path in [GROUNDING_DINO_DIR, GROUNDING_DINO_DIR / "groundingdino"]:
            if add_to_path_if_exists(path):
                try:
                    GroundingDINO = safe_import('groundingdino.util.inference', ['Model'])
                    if GroundingDINO:
                        print(f"Successfully imported GroundingDINO from {path}")
                        return True
                except Exception:
                    continue
    
    print("All GroundingDINO import methods failed")
    print_grounding_dino_installation_help()
    return False

def import_mobile_sam() -> bool:
    global sam_model_registry, SamPredictor
    
    print("Attempting to import MobileSAM...")
    try:
        result = safe_import('mobile_sam', ['sam_model_registry', 'SamPredictor'])
        if result and len(result) == 2:
            sam_model_registry, SamPredictor = result
            print("Successfully imported MobileSAM")
            return True
    except Exception:
        pass
    
    try:
        result = safe_import('segment_anything', ['sam_model_registry', 'SamPredictor'])
        if result and len(result) == 2:
            sam_model_registry, SamPredictor = result
            print("Successfully imported SAM as fallback")
            return True
    except Exception:
        pass
    
    if MOBILE_SAM_DIR.exists():
        add_to_path_if_exists(MOBILE_SAM_DIR)
        try:
            result = safe_import('mobile_sam', ['sam_model_registry', 'SamPredictor'])
            if result and len(result) == 2:
                sam_model_registry, SamPredictor = result
                print("Successfully imported MobileSAM from local directory")
                return True
        except Exception:
            pass
    
    print("All MobileSAM import methods failed")
    print_mobile_sam_installation_help()
    return False


# ---------------- INSTALL HELPERS ----------------
def print_grounding_dino_installation_help():
    print("\nGROUNDINGDINO INSTALLATION REQUIRED:")
    print("pip install groundingdino-py")
    print("or")
    print("pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'")

def print_mobile_sam_installation_help():
    print("\nMOBILESAM INSTALLATION REQUIRED:")
    print("pip install mobile-sam")
    print("or")
    print("pip install 'git+https://github.com/ChaoningZhang/MobileSAM.git'")


# ---------------- FILE SETUP ----------------
def download_file_robust(url: str, destination: Path, description: str, max_retries: int = 3) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"{description} already exists at {destination}")
        return True
    
    print(f"Downloading {description}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
        print(f"{description} downloaded successfully")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def find_config_file() -> Optional[Path]:
    for root, dirs, files in os.walk(GROUNDING_DINO_DIR):
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
        print("Config file not found. Please install GroundingDINO repo correctly.")
        return False, None
    
    return checkpoint_ready, config_path

def setup_mobile_sam_files() -> bool:
    print("Setting up MobileSAM files...")
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    checkpoint_path = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
    return download_file_robust(url, checkpoint_path, "MobileSAM checkpoint")


# ---------------- LOADING ----------------
def load_grounding_dino_model(config_path: Path, checkpoint_path: Path) -> Optional[Any]:
    if not GroundingDINO:
        print("GroundingDINO class not available")
        return None
    try:
        print("Loading GroundingDINO model...")
        device = get_device()
        print(f"Using device: {device}")
        model = GroundingDINO(str(config_path), str(checkpoint_path), device)
        print("GroundingDINO loaded successfully")
        return model
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None

def load_mobile_sam_model(checkpoint_path: Path, sam_type: str = "vit_t") -> Optional[Any]:
    if not sam_model_registry or not SamPredictor:
        print("MobileSAM components not available")
        return None
    try:
        device = get_device()
        print(f"Using device: {device}")
        sam = sam_model_registry[sam_type](checkpoint=str(checkpoint_path))
        sam.to(device)
        return SamPredictor(sam)
    except Exception as e:
        print(f"Error: {e}")
        return None


# ---------------- MAIN ----------------
print("\n=== Starting model loading process ===")

print("\n--- Importing model libraries ---")
grounding_dino_available = import_grounding_dino()
mobile_sam_available = import_mobile_sam()

print("\n--- Setting up model files ---")
grounding_dino_files_ready, grounding_dino_config_path = (False, None)
mobile_sam_files_ready = False

if grounding_dino_available:
    grounding_dino_files_ready, grounding_dino_config_path = setup_grounding_dino_files()
if mobile_sam_available:
    mobile_sam_files_ready = setup_mobile_sam_files()

print("\n--- Loading models ---")
grounding_dino_model = None
sam_predictor = None

if grounding_dino_available and grounding_dino_files_ready and grounding_dino_config_path:
    checkpoint_path = GROUNDING_DINO_DIR / "groundingdino_swint_ogc.pth"
    grounding_dino_model = load_grounding_dino_model(grounding_dino_config_path, checkpoint_path)

if mobile_sam_available and mobile_sam_files_ready:
    checkpoint_path = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
    sam_predictor = load_mobile_sam_model(checkpoint_path)

print("\n=== Model loading summary ===")
print(f"GroundingDINO: {'Loaded' if grounding_dino_model else 'Failed'}")
print(f"MobileSAM: {'Loaded' if sam_predictor else 'Failed'}")

grounding_dino = grounding_dino_model

def get_device_lazy():
    global DEVICE
    if DEVICE is None:
        DEVICE = get_device()
        print(f"Initialized device: {DEVICE}")
    return DEVICE

device = get_device_lazy

print("\n=== Model loader completed ===")