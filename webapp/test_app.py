#!/usr/bin/env python3
"""
Test script to verify the food segmentation app functionality
"""

import os
import sys
import numpy as np
import cv2

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from model_loader import grounding_dino, sam_predictor, device
        print("✓ Model imports successful")
        return True
    except Exception as e:
        print(f"✗ Model imports failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be created"""
    try:
        from app import app
        print("✓ Flask app creation successful")
        return True
    except Exception as e:
        print(f"✗ Flask app creation failed: {e}")
        return False

def test_segmentation_function():
    """Test the segmentation function with a dummy image"""
    try:
        from app import run_segmentation
        
        # Create a dummy image (100x100 white image)
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        _, buffer = cv2.imencode('.jpg', dummy_image)
        image_bytes = buffer.tobytes()
        
        # Test with a simple prompt
        original_path, result_path = run_segmentation(image_bytes, "the object")
        
        print("✓ Segmentation function test completed")
        return True
    except Exception as e:
        print(f"✗ Segmentation function test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Food Segmentation App...")
    print("=" * 40)
    
    tests = [
        ("Import Tests", test_imports),
        ("Flask App Tests", test_flask_app),
        ("Segmentation Function Tests", test_segmentation_function)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} failed")
    
    print("\n" + "=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The app is ready to use.")
        print("\nTo run the app:")
        print("  cd webapp")
        print("  python app.py")
        print("  Then open http://localhost:5000 in your browser")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 