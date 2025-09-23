import supervision as sv
import sys
import os
import torch
import urllib.request
from typing import Optional, Union, Any
from pathlib import Path
import warnings
import traceback
import importlib.util
warnings.filterwarnings('ignore')

ABS_PROJECT_DIR = Path(__file__).parent.parent.absolute()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define paths
GROUNDING_DINO_DIR = ABS_PROJECT_DIR / "webapp" / "GroundingDINO"
MOBILE_SAM_DIR = ABS_PROJECT_DIR / "webapp" / "MobileSAM"

# Create directories safely
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

def import_grounding_dino() -> bool:
    global GroundingDINO
    
    print("Attempting to import GroundingDINO...")
    
    # Method 1: Standard pip install import
    try:
        GroundingDINO = safe_import('groundingdino.util.inference', ['Model'])
        if GroundingDINO:
            print("Successfully imported GroundingDINO (standard method)")
            return True
    except Exception as e:
        print(f"Standard import failed: {e}")
    
    # Method 2: Try with groundingdino-py package
    try:
        groundingdino = safe_import('groundingdino')
        if groundingdino:
            GroundingDINO = safe_import('groundingdino.util.inference', ['Model'])
            if GroundingDINO:
                print("Successfully imported GroundingDINO (groundingdino-py package)")
                return True
    except Exception as e:
        print(f"groundingdino-py import failed: {e}")
    
    # Method 3: Try different package variations
    package_variations = [
        'groundingdino',
        'GroundingDINO', 
        'grounding_dino'
    ]
    
    for pkg in package_variations:
        try:
            # Try direct package import first
            module = safe_import(pkg)
            if module:
                # Try to get the Model class
                inference_module = safe_import(f'{pkg}.util.inference', ['Model'])
                if inference_module:
                    GroundingDINO = inference_module
                    print(f"Successfully imported GroundingDINO (package: {pkg})")
                    return True
        except Exception as e:
            continue
    
    # Method 4: Try local directory import
    if GROUNDING_DINO_DIR.exists():
        print(f"Trying local directory import from {GROUNDING_DINO_DIR}")
        
        # Add main directory
        add_to_path_if_exists(GROUNDING_DINO_DIR)
        
        # Try different subdirectory structures
        possible_paths = [
            GROUNDING_DINO_DIR,
            GROUNDING_DINO_DIR / "groundingdino",
            GROUNDING_DINO_DIR / "GroundingDINO" / "groundingdino"
        ]
        
        for path in possible_paths:
            if add_to_path_if_exists(path):
                try:
                    GroundingDINO = safe_import('groundingdino.util.inference', ['Model'])
                    if GroundingDINO:
                        print(f"Successfully imported GroundingDINO from {path}")
                        return True
                    
                    # Try alternative import structure
                    GroundingDINO = safe_import('util.inference', ['Model'])
                    if GroundingDINO:
                        print(f"Successfully imported GroundingDINO (alt structure) from {path}")
                        return True
                        
                except Exception as e:
                    continue
    
    print("All GroundingDINO import methods failed")
    print_grounding_dino_installation_help()
    return False

def import_mobile_sam() -> bool:
    """Try to import MobileSAM with fallback methods"""
    global sam_model_registry, SamPredictor
    
    print("Attempting to import MobileSAM...")
    
    # Method 1: Standard MobileSAM import
    try:
        result = safe_import('mobile_sam', ['sam_model_registry', 'SamPredictor'])
        if result and len(result) == 2:
            sam_model_registry, SamPredictor = result
            print("Successfully imported MobileSAM")
            return True
    except Exception as e:
        print(f"MobileSAM import failed: {e}")
    
    # Method 2: Fallback to regular SAM
    try:
        result = safe_import('segment_anything', ['sam_model_registry', 'SamPredictor'])
        if result and len(result) == 2:
            sam_model_registry, SamPredictor = result
            print("Successfully imported SAM as fallback")
            return True
    except Exception as e:
        print(f"SAM fallback import failed: {e}")
    
    # Method 3: Try local directory import
    if MOBILE_SAM_DIR.exists():
        add_to_path_if_exists(MOBILE_SAM_DIR)
        try:
            result = safe_import('mobile_sam', ['sam_model_registry', 'SamPredictor'])
            if result and len(result) == 2:
                sam_model_registry, SamPredictor = result
                print("Successfully imported MobileSAM from local directory")
                return True
        except Exception as e:
            print(f"Local MobileSAM import failed: {e}")
    
    print("All MobileSAM import methods failed")
    print_mobile_sam_installation_help()
    return False

def print_grounding_dino_installation_help():
    """Print installation instructions for GroundingDINO"""
    print("\nGROUNDINGDINO INSTALLATION REQUIRED:")
    print("Please run ONE of these commands:")
    print("")
    print("Option 1 (Recommended):")
    print("  pip install groundingdino-py")
    print("")
    print("Option 2:")
    print("  pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'")
    print("")
    print("Option 3 (If above fail, try with dependencies):")
    print("  pip install torch torchvision")
    print("  pip install transformers addict yapf timm")
    print("  pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'")
    print("")

def print_mobile_sam_installation_help():
    """Print installation instructions for MobileSAM"""
    print("\nMOBILESAM INSTALLATION REQUIRED:")
    print("Please run ONE of these commands:")
    print("")
    print("Option 1:")
    print("  pip install mobile-sam")
    print("")
    print("Option 2:")
    print("  pip install 'git+https://github.com/ChaoningZhang/MobileSAM.git'")
    print("")
    print("Option 3 (Regular SAM as fallback):")
    print("  pip install segment-anything")
    print("")

def download_file_robust(url: str, destination: Path, description: str, max_retries: int = 3) -> bool:
    """Download a file with progress indication and retry logic"""
    if destination.exists():
        print(f"{description} already exists at {destination}")
        return True
    
    print(f"Downloading {description}...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100.0 / total_size, 100.0)
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, str(destination), show_progress)
            print(f"\nSuccessfully downloaded {description}")
            return True
            
        except Exception as e:
            print(f"\nAttempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                if destination.exists():
                    destination.unlink()  # Remove partial download
            else:
                print(f"Failed to download {description} after {max_retries} attempts")
                return False
    
    return False

def find_config_file() -> Optional[Path]:
    """Find GroundingDINO config file in various possible locations"""
    possible_config_paths = [
        GROUNDING_DINO_DIR / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
        GROUNDING_DINO_DIR / "configs" / "GroundingDINO_SwinT_OGC.py",
        GROUNDING_DINO_DIR / "config" / "GroundingDINO_SwinT_OGC.py",
        GROUNDING_DINO_DIR / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
    ]
    
    for config_path in possible_config_paths:
        if config_path.exists():
            print(f"Found config file at: {config_path}")
            return config_path
    
    # Try to find config file recursively
    if GROUNDING_DINO_DIR.exists():
        for root, dirs, files in os.walk(GROUNDING_DINO_DIR):
            if "GroundingDINO_SwinT_OGC.py" in files:
                config_path = Path(root) / "GroundingDINO_SwinT_OGC.py"
                print(f"Found config file at: {config_path}")
                return config_path
    
    return None

def setup_grounding_dino_files() -> tuple[bool, Optional[Path]]:
    """Setup GroundingDINO files (download checkpoint, find config)"""
    print("Setting up GroundingDINO files...")
    
    # Download checkpoint
    checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    checkpoint_path = GROUNDING_DINO_DIR / "groundingdino_swint_ogc.pth"
    
    checkpoint_success = download_file_robust(checkpoint_url, checkpoint_path, "GroundingDINO checkpoint")
    
    # Find config file
    config_path = find_config_file()
    
    if not config_path:
        print("Config file not found. This usually means GroundingDINO repository is not properly installed.")
        print("Please install GroundingDINO using one of the methods above.")
        return False, None
    
    return checkpoint_success, config_path

def setup_mobile_sam_files() -> bool:
    """Setup MobileSAM files (download checkpoint)"""
    print("Setting up MobileSAM files...")
    
    mobile_sam_url = "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt?raw=true"
    checkpoint_path = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
    
    return download_file_robust(mobile_sam_url, checkpoint_path, "MobileSAM checkpoint")

def load_grounding_dino_model(config_path: Path, checkpoint_path: Path) -> Optional[Any]:
    """Load GroundingDINO model safely"""
    if not GroundingDINO:
        print("GroundingDINO class not available")
        return None
    
    try:
        print("Loading GroundingDINO model...")
        print(f"Config: {config_path}")
        print(f"Checkpoint: {checkpoint_path}")
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        model = GroundingDINO(str(config_path), str(checkpoint_path), DEVICE)
        print("GroundingDINO model loaded successfully")
        return model
        
    except Exception as e:
        print(f"Error loading GroundingDINO model: {e}")
        traceback.print_exc()
        return None

def load_mobile_sam_model(checkpoint_path: Path, sam_type: str = "vit_t") -> Optional[Any]:
    """Load MobileSAM model safely"""
    if not sam_model_registry or not SamPredictor:
        print("MobileSAM components not available")
        return None
    
    try:
        print("Loading MobileSAM model...")
        print(f"SAM type: {sam_type}")
        print(f"Checkpoint: {checkpoint_path}")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Check if sam_type is available
        if sam_type not in sam_model_registry:
            available_types = list(sam_model_registry.keys())
            print(f"SAM type '{sam_type}' not available. Available types: {available_types}")
            if available_types:
                sam_type = available_types[0]
                print(f"Using {sam_type} instead")
            else:
                raise ValueError("No SAM model types available")
        
        sam = sam_model_registry[sam_type](checkpoint=str(checkpoint_path))
        sam.to(DEVICE)
        predictor = SamPredictor(sam)
        
        print("MobileSAM model loaded successfully")
        print(f"Model device: {next(sam.parameters()).device}")
        return predictor
        
    except Exception as e:
        print(f"Error loading MobileSAM model: {e}")
        traceback.print_exc()
        return None

# Main execution
print("\n=== Starting model loading process ===")

# Step 1: Import models
print("\n--- Importing model libraries ---")
grounding_dino_available = import_grounding_dino()
mobile_sam_available = import_mobile_sam()

# Step 2: Setup files
print("\n--- Setting up model files ---")
grounding_dino_files_ready = False
grounding_dino_config_path = None
mobile_sam_files_ready = False

if grounding_dino_available:
    grounding_dino_files_ready, grounding_dino_config_path = setup_grounding_dino_files()

if mobile_sam_available:
    mobile_sam_files_ready = setup_mobile_sam_files()

# Step 3: Load models
print("\n--- Loading models ---")
grounding_dino_model = None
sam_predictor = None

if grounding_dino_available and grounding_dino_files_ready and grounding_dino_config_path:
    checkpoint_path = GROUNDING_DINO_DIR / "groundingdino_swint_ogc.pth"
    grounding_dino_model = load_grounding_dino_model(grounding_dino_config_path, checkpoint_path)

if mobile_sam_available and mobile_sam_files_ready:
    checkpoint_path = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
    sam_predictor = load_mobile_sam_model(checkpoint_path)

# Step 4: Summary
print("\n=== Model loading summary ===")
print(f"GroundingDINO: {'Loaded' if grounding_dino_model is not None else 'Failed'}")
print(f"MobileSAM: {'Loaded' if sam_predictor is not None else 'Failed'}")

if grounding_dino_model is None:
    print("\nTo fix GroundingDINO:")
    if not grounding_dino_available:
        print("  - Install the package (see instructions above)")
    elif not grounding_dino_files_ready:
        print("  - Files are missing or corrupted")

if sam_predictor is None:
    print("\nTo fix MobileSAM:")
    if not mobile_sam_available:
        print("  - Install the package (see instructions above)")
    elif not mobile_sam_files_ready:
        print("  - Files are missing or corrupted")

# Export variables for use in other modules
grounding_dino = grounding_dino_model
device = DEVICE

print("\n=== Model loader completed ===")