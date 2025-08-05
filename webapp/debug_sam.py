import sys
import os
import numpy as np
import torch
import cv2

# Add the project directories to sys.path
ABS_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grounding_dino_dir = os.path.join(ABS_PROJECT_DIR, "webapp", "GroundingDINO")
mobile_sam_dir = os.path.join(ABS_PROJECT_DIR, "webapp", "MobileSAM")

sys.path.append(grounding_dino_dir)
sys.path.append(mobile_sam_dir)

def test_mobile_sam():
    print("=== MobileSAM Debug Test ===")
    
    # Test 1: Import
    try:
        from mobile_sam import sam_model_registry, SamPredictor # ignore this error
        print("Successfully imported MobileSAM")
    except Exception as e:
        print(f"Failed to import MobileSAM: {e}")
        return False
    
    # Test 2: Check model registry
    print(f"Available model types: {list(sam_model_registry.keys())}")
    
    # Test 3: Model loading
    MOBILE_SAM_CHECKPOINT_PATH = os.path.join(mobile_sam_dir, "weights", "mobile_sam.pt")
    SAM_TYPE = "vit_t"
    
    print(f"Checkpoint path: {MOBILE_SAM_CHECKPOINT_PATH}")
    print(f"Checkpoint exists: {os.path.exists(MOBILE_SAM_CHECKPOINT_PATH)}")
    print(f"SAM type: {SAM_TYPE}")
    
    try:
        sam = sam_model_registry[SAM_TYPE](checkpoint=MOBILE_SAM_CHECKPOINT_PATH)
        print("Successfully loaded MobileSAM model")
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam.to(device)
        print(f"Model moved to device: {device}")
        
        # Create predictor
        sam_predictor = SamPredictor(sam)
        print("Successfully created SamPredictor")
        
    except Exception as e:
        print(f"Failed to load MobileSAM model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Simple prediction
    try:
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White square
        
        print(f"Test image shape: {test_image.shape}")
        
        # Set the image
        sam_predictor.set_image(test_image)
        print("Successfully set image")
        
        # Try box prediction
        test_box = np.array([30, 30, 70, 70])
        print(f"Test box: {test_box}")
        
        masks, scores, logits = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=test_box,
            multimask_output=False,
        )
        
        print(f"Prediction successful!")
        print(f"Masks shape: {masks.shape}")
        print(f"Scores: {scores}")
        print(f"Logits shape: {logits.shape}")
        
        # Test point prediction
        center_point = np.array([[50, 50]])
        point_labels = np.array([1])
        
        masks2, scores2, logits2 = sam_predictor.predict(
            point_coords=center_point,
            point_labels=point_labels,
            multimask_output=False,
        )
        
        print(f"Point prediction successful!")
        print(f"Masks shape: {masks2.shape}")
        print(f"Scores: {scores2}")
        
        return True
        
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mobile_sam()
    if success:
        print("\n All tests passed! MobileSAM is working correctly.")
    else:
        print("\n Tests failed. Check the errors above.") 