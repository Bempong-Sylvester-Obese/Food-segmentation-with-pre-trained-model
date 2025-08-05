from flask import Flask, request, render_template_string, send_from_directory
import cv2
import numpy as np
import torch
import uuid
import os

# Import the pre-loaded models from model_loader
from model_loader import grounding_dino, sam_predictor, device

# --- Setup Flask App ---
app = Flask(__name__)

# Create a directory to store uploaded and generated images
os.makedirs("static/images", exist_ok=True)

# --- Main Inference Function ---
def run_segmentation(image_bytes: bytes, prompt: str):
    # Convert image bytes to an OpenCV image
    nparr = np.frombuffer(image_bytes, np.uint8)
    source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- 1. Detect with GroundingDINO ---
    detections, _ = grounding_dino.predict_with_caption(
        image=source_image,
        caption=prompt,
        box_threshold=0.35,
        text_threshold=0.25
    )

    if len(detections.xyxy) == 0:
        return None, None # No object detected

    # --- 2. Segment with MobileSAM ---
    sam_predictor.set_image(source_image)
    input_boxes = torch.tensor(detections.xyxy, device=device)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=input_boxes,
        multimask_output=False,
    )

    # --- 3. Process and Save the Mask ---
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
    original_filename = f"static/images/{unique_id}_original.png"
    result_filename = f"static/images/{unique_id}_result.png"

    cv2.imwrite(original_filename, source_image)
    cv2.imwrite(result_filename, img_bgra)

    return original_filename, result_filename

# HTML template for the web interface
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
        }
        button:hover {
            background-color: #0056b3;
        }
        .results {
            margin-top: 30px;
            display: none;
        }
        .image-container {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        .image-box {
            flex: 1;
            text-align: center;
        }
        .image-box img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .error {
            color: red;
            text-align: center;
            margin-top: 20px;
        }
        .loading {
            text-align: center;
            margin-top: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload an Image and a Prompt</h1>
        
        <form id="upload-form" enctype="multipart/form-data">
            <div class="form-group">
                <label for="image_file">Select an image to upload:</label>
                <input type="file" name="image_file" id="image_file" accept="image/*" required>
            </div>
            
            <div class="form-group">
                <label for="prompt">Enter a food prompt (e.g., 'the burger'):</label>
                <input type="text" name="prompt" id="prompt" required>
            </div>
            
            <button type="submit">Segment Food</button>
        </form>
        
        <div id="loading" class="loading" style="display: none;">
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
            
            // Show loading
            loading.style.display = 'block';
            error.style.display = 'none';
            results.style.display = 'none';
            
            try {
                const response = await fetch('/segment', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('original-image').src = result.original_path;
                    document.getElementById('result-image').src = result.result_path;
                    results.style.display = 'block';
                } else {
                    error.textContent = result.error;
                    error.style.display = 'block';
                }
            } catch (err) {
                error.textContent = 'An error occurred while processing the image.';
                error.style.display = 'block';
            } finally {
                loading.style.display = 'none';
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

@app.route('/segment', methods=['POST'])
def segment():
    """Handles the form submission and returns the segmented image."""
    try:
        if 'image_file' not in request.files:
            return {'success': False, 'error': 'No image file provided.'}
        
        image_file = request.files['image_file']
        prompt = request.form.get('prompt', '')
        
        if not image_file or not prompt:
            return {'success': False, 'error': 'Please provide both an image and a prompt.'}
        
        # Read image bytes
        image_bytes = image_file.read()
        
        # Run the model
        original_path, result_path = run_segmentation(image_bytes, prompt)
        
        if not result_path:
            return {'success': False, 'error': 'Could not detect the object. Please try a different prompt.'}
        
        return {
            'success': True,
            'original_path': f'/{original_path}',
            'result_path': f'/{result_path}'
        }
        
    except Exception as e:
        return {'success': False, 'error': f'An error occurred: {str(e)}'}

# --- Route to serve static files ---
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)