from flask import Flask, request, render_template_string, send_from_directory
import cv2
import numpy as np
import torch
import uuid
import os

# Import the pre-loaded models from model_loader
from model_loader import grounding_dino, sam_predictor, device

# Setup Flask App 
app = Flask(__name__)

# Create a directory to store uploaded and generated images
os.makedirs("static/images", exist_ok=True)

# Main Inference Function
def run_segmentation(image_bytes: bytes, prompt: str):
    try:
        # Validate inputs
        if not image_bytes:
            raise ValueError("No image data provided")
        
        if not prompt or not prompt.strip():
            raise ValueError("No prompt provided")
        
        # Convert image bytes to an OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if source_image is None:
            raise ValueError("Invalid image format. Please upload a valid image file.")
        
        # Check image dimensions
        height, width = source_image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Invalid image dimensions")
        
        # Detect with GroundingDINO
        detections, _ = grounding_dino.predict_with_caption(
            image=source_image,
            caption=prompt,
            box_threshold=0.35,
            text_threshold=0.25
        )

        if len(detections.xyxy) == 0:
            return None, None # No object detected

        # Segment with MobileSAM
        sam_predictor.set_image(source_image)
        input_boxes = torch.tensor(detections.xyxy, device=device)

        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=input_boxes,
            multimask_output=False,
        )
        # Create a binary mask
        final_mask = masks[0].cpu().numpy().squeeze()
        binary_mask = (final_mask > 0).astype(np.uint8) * 255

        # Apply the mask to the original image to get the segmented object
        segmented_image = cv2.bitwise_and(source_image, source_image, mask=binary_mask)

        # Make the background transparent
        b_channel, g_channel, r_channel = cv2.split(segmented_image)
        alpha_channel = np.where(binary_mask == 255, 255, 0).astype(b_channel.dtype)
        img_bgra = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        # Generate unique filenames to avoid conflicts
        unique_id = uuid.uuid4()
        original_filename = f"static/Images/{unique_id}_original.png"
        result_filename = f"static/Generated_Images/{unique_id}_result.png"

        # Ensure the directory exists
        os.makedirs("static/images", exist_ok=True)

        # Save images
        cv2.imwrite(original_filename, source_image)
        cv2.imwrite(result_filename, img_bgra)
        
        # Verify files were created
        if not os.path.exists(original_filename) or not os.path.exists(result_filename):
            raise ValueError("Failed to save processed images")

        return original_filename, result_filename
        
    except Exception as e:
        print(f"Error in run_segmentation: {str(e)}")
        return None, None

# HTML for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prompt-Guided Food Segmentation</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input[type="file"], input[type="text"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            box-sizing: border-box;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .image-container {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .image-box {
            flex: 1;
            min-width: 300px;
            text-align: center;
        }
        .image-box img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .error {
            color: #dc3545;
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 5px;
        }
        .loading {
            text-align: center;
            margin-top: 20px;
            color: #666;
            padding: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Food Segmentation with AI</h1>
        
        <form id="upload-form" enctype="multipart/form-data">
            <div class="form-group">
                <label for="image_file">Select an image to upload:</label>
                <input type="file" name="image_file" id="image_file" accept="image/*" required>
            </div>
            
            <div class="form-group">
                <label for="prompt">Enter a food prompt (e.g., 'the burger', 'pizza', 'salad'):</label>
                <input type="text" name="prompt" id="prompt" placeholder="e.g., the burger" required>
            </div>
            
            <button type="submit" id="submit-btn">Segment Food</button>
        </form>
        
        <div id="loading" class="loading" style="display: none;">
            <div class="spinner"></div>
            Processing... Please wait.
        </div>
        
        <div id="error" class="error" style="display: none;"></div>
        
        <div id="results" class="results">
            <h2>Results</h2>
            <div class="image-container">
                <div class="image-box">
                    <h3>Original Image</h3>
                    <img id="original-image" src="" alt="Original">
                </div>
                <div class="image-box">
                    <h3>Segmented Object</h3>
                    <img id="result-image" src="" alt="Segmented">
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('upload-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const results = document.getElementById('results');
            const submitBtn = document.getElementById('submit-btn');
            
            // Validate inputs
            const imageFile = document.getElementById('image_file').files[0];
            const prompt = document.getElementById('prompt').value.trim();
            
            if (!imageFile) {
                showError('Please select an image file.');
                return;
            }
            
            if (!prompt) {
                showError('Please enter a prompt.');
                return;
            }
            
            // Show loading
            loading.style.display = 'block';
            error.style.display = 'none';
            results.style.display = 'none';
            submitBtn.disabled = true;
            
            try {
                const response = await fetch('/segment', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                if (result.success) {
                    // Add timestamp to prevent caching
                    const timestamp = new Date().getTime();
                    document.getElementById('original-image').src = result.original_path + '?t=' + timestamp;
                    document.getElementById('result-image').src = result.result_path + '?t=' + timestamp;
                    results.style.display = 'block';
                } else {
                    showError(result.error || 'An error occurred during processing.');
                }
            } catch (err) {
                console.error('Error:', err);
                showError('An error occurred while processing the image. Please try again.');
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });
        
        function showError(message) {
            const error = document.getElementById('error');
            error.textContent = message;
            error.style.display = 'block';
        }
        
        // Preview image before upload
        document.getElementById('image_file').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    // You could add a preview here if needed
                };
                reader.readAsDataURL(file);
            }
        });
    </script>
</body>
</html>
"""

# --- Web App Routes ---

@app.route('/')
def index():
    """The main page with the upload form."""
    return HTML_TEMPLATE

@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'models_loaded': {
            'grounding_dino': grounding_dino is not None,
            'sam_predictor': sam_predictor is not None
        }
    }

@app.route('/segment', methods=['POST'])
def segment():
    try:
        # Check if models are loaded
        if grounding_dino is None:
            return {'success': False, 'error': 'GroundingDINO model is not loaded. Please check the server logs.'}
        
        if sam_predictor is None:
            return {'success': False, 'error': 'MobileSAM model is not loaded. Please check the server logs.'}
        
        if 'image_file' not in request.files:
            return {'success': False, 'error': 'No image file provided.'}
        
        image_file = request.files['image_file']
        prompt = request.form.get('prompt', '').strip()
        
        if not image_file or image_file.filename == '':
            return {'success': False, 'error': 'Please select a valid image file.'}
        
        if not prompt:
            return {'success': False, 'error': 'Please provide a prompt describing the food item.'}
        
        # Check file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in image_file.filename or \
           image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return {'success': False, 'error': 'Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP).'}
        
        # Read image bytes
        image_bytes = image_file.read()
        
        if len(image_bytes) == 0:
            return {'success': False, 'error': 'The uploaded file is empty.'}
        
        # Run the model
        original_path, result_path = run_segmentation(image_bytes, prompt)
        
        if original_path is None or result_path is None:
            return {'success': False, 'error': 'Could not detect the specified object. Please try a different prompt or image.'}
        
        return {
            'success': True,
            'original_path': f'/{original_path}',
            'result_path': f'/{result_path}'
        }
        
    except Exception as e:
        print(f"Error in segment route: {str(e)}")
        return {'success': False, 'error': f'An error occurred during processing: {str(e)}'}

# Route to serve static files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Route to serve images
@app.route('/static/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('static/images', filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)