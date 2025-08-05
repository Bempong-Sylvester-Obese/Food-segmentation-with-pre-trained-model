import supervision as sv
import sys
import os
import torch
import urllib.request

# Define the absolute project directory
ABS_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Add GroundingDINO directory to sys.path
grounding_dino_dir = os.path.join(ABS_PROJECT_DIR, "webapp", "GroundingDINO")
if grounding_dino_dir not in sys.path:
    sys.path.append(grounding_dino_dir)
    print(f"Added {grounding_dino_dir} to sys.path")

# Add MobileSAM directory to sys.path
mobile_sam_dir = os.path.join(ABS_PROJECT_DIR, "webapp", "MobileSAM")
if mobile_sam_dir not in sys.path:
    sys.path.append(mobile_sam_dir)
    print(f"Added {mobile_sam_dir} to sys.path")

# Import GroundingDINO with error handling
try:
    # Import directly from the local directory
    sys.path.insert(0, grounding_dino_dir)
    from groundingdino.util.inference import Model as GroundingDINO
    print("Successfully imported GroundingDINO")
except ModuleNotFoundError as e:
    print(f"Error importing GroundingDINO model: {e}")
    print("Please ensure GroundingDINO is properly installed")
    sys.exit("Could not import GroundingDINO model.")

# Import MobileSAM with error handling
try:
    # Import directly from the local directory
    sys.path.insert(0, mobile_sam_dir)
    from mobile_sam import sam_model_registry, SamPredictor
    print("Successfully imported MobileSAM")
except ModuleNotFoundError as e:
    print(f"Error importing MobileSAM: {e}")
    print("Please ensure MobileSAM is properly installed")
    sys.exit("Could not import MobileSAM.")

# GroundingDINO configuration
GROUNDING_DINO_CONFIG_PATH = os.path.join(grounding_dino_dir, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(grounding_dino_dir, "groundingdino_swint_ogc.pth")

# MobileSAM configuration
MOBILE_SAM_CHECKPOINT_PATH = os.path.join(mobile_sam_dir, "weights", "mobile_sam.pt")
SAM_TYPE = "vit_t"

# Function to download GroundingDINO checkpoint if missing
def download_groundingdino_checkpoint():
    if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
        print("GroundingDINO checkpoint not found. Downloading...")
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        try:
            urllib.request.urlretrieve(url, GROUNDING_DINO_CHECKPOINT_PATH)
            print(f"Successfully downloaded GroundingDINO checkpoint to {GROUNDING_DINO_CHECKPOINT_PATH}")
        except Exception as e:
            print(f"Error downloading GroundingDINO checkpoint: {e}")
            return False
    return True

# Check if model files exist and download if necessary
if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
    print(f"Warning: GroundingDINO config file not found at {GROUNDING_DINO_CONFIG_PATH}")
    
if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
    print(f"GroundingDINO checkpoint file not found at {GROUNDING_DINO_CHECKPOINT_PATH}")
    if not download_groundingdino_checkpoint():
        print("Failed to download GroundingDINO checkpoint")
    
if not os.path.exists(MOBILE_SAM_CHECKPOINT_PATH):
    print(f"Warning: MobileSAM checkpoint file not found at {MOBILE_SAM_CHECKPOINT_PATH}")

# Initialize models with error handling
grounding_dino_model = None
sam_predictor = None

try:
    print("Loading GroundingDINO model..")
    grounding_dino_model = GroundingDINO(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, DEVICE)
    print("GroundingDINO model loaded successfully")
except Exception as e:
    print(f"Error loading GroundingDINO model: {e}")
    print("Continuing without GroundingDINO model...")

try:
    print("Loading MobileSAM model..")
    sam = sam_model_registry[SAM_TYPE](checkpoint=MOBILE_SAM_CHECKPOINT_PATH)
    sam.to(DEVICE)
    sam_predictor = SamPredictor(sam)
    print("MobileSAM model loaded successfully")
except Exception as e:
    print(f"Error loading MobileSAM model: {e}")
    print("Continuing without MobileSAM model...")

print("\nModel loading completed")

# Export the models and device for use in other modules
grounding_dino = grounding_dino_model
device = DEVICE