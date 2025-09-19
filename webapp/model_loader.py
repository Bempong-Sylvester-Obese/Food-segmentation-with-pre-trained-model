import supervision as sv
import sys
import os
import torch
import urllib.request
from typing import Optional, Union
from pathlib import Path

ABS_PROJECT_DIR = Path(__file__).parent.parent.absolute()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Define paths
GROUNDING_DINO_DIR = ABS_PROJECT_DIR / "webapp" / "GroundingDINO"
MOBILE_SAM_DIR = ABS_PROJECT_DIR / "webapp" / "MobileSAM"

for directory in [GROUNDING_DINO_DIR, MOBILE_SAM_DIR]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
        print(f"Added {directory} to sys.path")

GroundingDINO: Optional[type] = None
try:
    from groundingdino.util.inference import Model as GroundingDINO
    print("Successfully imported GroundingDINO")
except ImportError as e:
    print(f"Error importing GroundingDINO model: {e}")
    print("Please ensure GroundingDINO is properly installed")
    # Try alternative import method
    try:
        sys.path.insert(0, str(GROUNDING_DINO_DIR / "groundingdino"))
        from util.inference import Model as GroundingDINO
        print("Successfully imported GroundingDINO using alternative method")
    except ImportError as e2:
        print(f"Alternative import also failed: {e2}")
        print("Please ensure GroundingDINO is properly installed")

sam_model_registry: Optional[dict] = None
SamPredictor: Optional[type] = None

try:
    from mobile_sam import sam_model_registry, SamPredictor
    print("Successfully imported MobileSAM")
except ImportError as e:
    print(f"Error importing MobileSAM: {e}")
    print("Please ensure MobileSAM is properly installed")

GROUNDING_DINO_CONFIG_PATH = GROUNDING_DINO_DIR / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = GROUNDING_DINO_DIR / "groundingdino_swint_ogc.pth"
MOBILE_SAM_CHECKPOINT_PATH = MOBILE_SAM_DIR / "weights" / "mobile_sam.pt"
SAM_TYPE = "vit_t"

def download_groundingdino_checkpoint() -> bool:
    if not GROUNDING_DINO_CHECKPOINT_PATH.exists():
        print("GroundingDINO checkpoint not found. Downloading...")
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        try:
            urllib.request.urlretrieve(url, str(GROUNDING_DINO_CHECKPOINT_PATH))
            print(f"Successfully downloaded GroundingDINO checkpoint to {GROUNDING_DINO_CHECKPOINT_PATH}")
            return True
        except Exception as e:
            print(f"Error downloading GroundingDINO checkpoint: {e}")
            return False
    return True

if not GROUNDING_DINO_CONFIG_PATH.exists():
    print(f"Warning: GroundingDINO config file not found at {GROUNDING_DINO_CONFIG_PATH}")
    
if not GROUNDING_DINO_CHECKPOINT_PATH.exists():
    print(f"GroundingDINO checkpoint file not found at {GROUNDING_DINO_CHECKPOINT_PATH}")
    if not download_groundingdino_checkpoint():
        print("Failed to download GroundingDINO checkpoint")
    
if not MOBILE_SAM_CHECKPOINT_PATH.exists():
    print(f"Warning: MobileSAM checkpoint file not found at {MOBILE_SAM_CHECKPOINT_PATH}")

# Initialize models
grounding_dino_model: Optional[GroundingDINO] = None
sam_predictor: Optional[SamPredictor] = None

# Load GroundingDINO model
if GroundingDINO is not None:
    try:
        print("Loading GroundingDINO model...")
        grounding_dino_model = GroundingDINO(
            str(GROUNDING_DINO_CONFIG_PATH), 
            str(GROUNDING_DINO_CHECKPOINT_PATH), 
            DEVICE
        )
        print("GroundingDINO model loaded successfully")
    except Exception as e:
        print(f"Error loading GroundingDINO model: {e}")
        print("Continuing without GroundingDINO model...")
else:
    print("GroundingDINO not available - skipping model loading")

# Load MobileSAM model
if sam_model_registry is not None and SamPredictor is not None:
    try:
        print("Loading MobileSAM model...")
        print(f"Using SAM type: {SAM_TYPE}")
        print(f"Checkpoint path: {MOBILE_SAM_CHECKPOINT_PATH}")
        print(f"Checkpoint exists: {MOBILE_SAM_CHECKPOINT_PATH.exists()}")
        
        sam = sam_model_registry[SAM_TYPE](checkpoint=str(MOBILE_SAM_CHECKPOINT_PATH))
        sam.to(DEVICE)
        sam_predictor = SamPredictor(sam)
        print("MobileSAM model loaded successfully")
        print(f"Model device: {next(sam.parameters()).device}")
    except Exception as e:
        print(f"Error loading MobileSAM model: {e}")
        print("Continuing without MobileSAM model...")
        import traceback
        traceback.print_exc()
else:
    print("MobileSAM not available - skipping model loading")
print("\nModel loading completed")

grounding_dino = grounding_dino_model
device = DEVICE
