from flask import Flask, request, render_template_string, send_from_directory
import cv2
import numpy as np
import uuid
import os
import traceback
import gc
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    torch = None

grounding_dino = None
sam_predictor = None
device = None
models_loaded = False

def load_models():
    global grounding_dino, sam_predictor, device, models_loaded
    
    if models_loaded and grounding_dino is not None and sam_predictor is not None:
        return True
        
    print("Loading models for first time...")
    start_time = time.time()
    
    try:
        from model_loader import grounding_dino as gd, sam_predictor as sp, device as dev
        grounding_dino = gd
        sam_predictor = sp
        device = dev
        models_loaded = True
        
        load_time = time.time() - start_time
        print(f"Models loaded successfully in {load_time:.2f} seconds")
        
        # Clear cache to free memory
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return False

app = Flask(__name__)

os.makedirs("static/images", exist_ok=True)
os.makedirs("static/GeneratedImages", exist_ok=True)

# Load models on startup
print("Initializing application...")

# Main Inference Function
def run_segmentation(image_bytes: bytes, prompt: str):
    start_time = time.time()
    
    try:
        if not image_bytes:
            raise ValueError("No image data provided")
        
        if not prompt or not prompt.strip():
            raise ValueError("No prompt provided")
        
        # Check file size limit (10MB for Cloud Run)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_bytes) > max_size:
            raise ValueError(f"Image file too large. Maximum size is {max_size // (1024*1024)}MB.")
            
        if not TORCH_AVAILABLE or torch is None:
            raise ValueError("PyTorch is not available. Please install PyTorch.")
        
        # Load models if not already loaded
        if not load_models():
            raise ValueError("Failed to load models. Check the server logs.")
        
        if grounding_dino is None:
            raise ValueError("GroundingDINO model is not loaded. Check the server logs.")
        
        if sam_predictor is None:
            raise ValueError("MobileSAM model is not loaded. Check the server logs.")
        
        # Convert image bytes to an OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        source_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if source_image is None:
            raise ValueError("Invalid image format. Please upload a valid image file.")
        
        # Check image dimensions and resize if too large for Cloud Run
        height, width = source_image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Invalid image dimensions")
        
        # Resize image if too large (memory optimization for Cloud Run)
        max_dim = 2048  # Maximum dimension
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            source_image = cv2.resize(source_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            height, width = source_image.shape[:2]
            print(f"Resized image to: {width}x{height}")
            
        print(f"Processing image with dimensions: {width}x{height}")
        
        # Detect with GroundingDINO
        detections, phrases = grounding_dino.predict_with_caption(
            image=source_image,
            caption=prompt,
            box_threshold=0.35,
            text_threshold=0.25
        )

        # Check object detection
        if detections is None or len(detections.xyxy) == 0:
            print(f"No objects detected for prompt: '{prompt}'")
            return None, None # No object detected

        print(f"Detected {len(detections.xyxy)} objects with confidence scores: {detections.confidence}")

        # Segment with MobileSAM
        sam_predictor.set_image(source_image)
        
        # Convert detections to the correct format for SAM
        current_device = device() if callable(device) else device
        if TORCH_AVAILABLE and torch is not None:
            input_boxes = torch.tensor(detections.xyxy, device=current_device)
        else:
            raise ValueError("PyTorch is not available for tensor operations")
        
        # Ensure boxes are in the correct format [x1, y1, x2, y2]
        if input_boxes.dim() == 1:
            input_boxes = input_boxes.unsqueeze(0)
        
        print(f"Input boxes shape: {input_boxes.shape}")
        print(f"Input boxes: {input_boxes}")

        try:
            masks, scores, logits = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes[0].cpu().numpy(),  # Use first box
                multimask_output=False,
            )
            
            if masks is None or len(masks) == 0:
                print("No masks generated by MobileSAM")
                return None, None
                
            # Binary mask
            final_mask = masks[0]  # Use first mask
            binary_mask = (final_mask > 0).astype(np.uint8) * 255
            
            print(f"Successfully generated mask with shape: {binary_mask.shape}")
            print(f"Mask values range: {final_mask.min()} to {final_mask.max()}")
            
        except Exception as e:
            print(f"Error in MobileSAM prediction: {str(e)}")
            # Alternative approach with point prompts
            try:
                print("Trying alternative approach with point prompts...")
                box = detections.xyxy[0]
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                
                masks, scores, logits = sam_predictor.predict(
                    point_coords=np.array([[center_x, center_y]]),
                    point_labels=np.array([1]),  # 1 for foreground point
                    multimask_output=False,
                )
                
                if masks is not None and len(masks) > 0:
                    final_mask = masks[0]
                    binary_mask = (final_mask > 0).astype(np.uint8) * 255
                    print(f"Successfully generated mask with point prompts, shape: {binary_mask.shape}")
                else:
                    print("No masks generated with point prompts")
                    return None, None
                    
            except Exception as e2:
                print(f"Error in alternative MobileSAM prediction: {str(e2)}")
                return None, None

        result_image = source_image.copy()
        
        # Overlay for the segmentation mask
        overlay = np.zeros_like(source_image)
        overlay[binary_mask > 0] = [0, 255, 0]  # Green overlay for segmentation
        
        # Blend the overlay with the original image
        alpha = 0.3  # Transparency factor
        result_image = cv2.addWeighted(result_image, 1, overlay, alpha, 0)
        
        # Draw bounding boxes
        for box in detections.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{prompt}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_image, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (0, 0, 255), -1)
            cv2.putText(result_image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Generate unique filenames to avoid conflicts
        unique_id = uuid.uuid4()
        original_filename = f"static/images/{unique_id}_original.png"
        result_filename = f"static/GeneratedImages/{unique_id}_result.png"

        # Ensure the directories exist
        os.makedirs("static/images", exist_ok=True)
        os.makedirs("static/GeneratedImages", exist_ok=True)

        # Save images
        cv2.imwrite(original_filename, source_image)
        cv2.imwrite(result_filename, result_image)
        
        # Verify files
        if not os.path.exists(original_filename) or not os.path.exists(result_filename):
            raise ValueError("Failed to save processed images")

        # Memory cleanup for Cloud Run
        total_time = time.time() - start_time
        print(f"Segmentation completed in {total_time:.2f} seconds")
        
        # Clear memory
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return original_filename, result_filename
        
    except Exception as e:
        print(f"Error in run_segmentation: {str(e)}")
        # Memory cleanup
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return None, None

# HTML for web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Food Segmentation With GroundingDINO and MobileSAM</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f8fafc;
            min-height: 100vh;
            color: #334155;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 50px;
            color: #1e293b;
        }

        .header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2rem;
            color: #64748b;
            font-weight: 400;
        }

        .main-card {
            background: white;
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
            margin-bottom: 30px;
        }

        .form-section {
            margin-bottom: 40px;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .section-title i {
            color: #475569;
        }

        .form-group {
            margin-bottom: 25px;
        }

        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #374151;
            font-size: 0.95rem;
        }

        .file-upload-area {
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            background: #f9fafb;
        }

        .file-upload-area:hover {
            border-color: #6b7280;
            background: #f3f4f6;
        }

        .file-upload-area.dragover {
            border-color: #4b5563;
            background: #f3f4f6;
        }

        .file-upload-icon {
            font-size: 3rem;
            color: #9ca3af;
            margin-bottom: 15px;
        }

        .file-upload-text {
            color: #6b7280;
            font-size: 1.1rem;
            margin-bottom: 10px;
        }

        .file-upload-hint {
            color: #9ca3af;
            font-size: 0.9rem;
        }

        .file-input {
            display: none;
        }

        .text-input {
            width: 100%;
            padding: 16px 20px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background: white;
        }

        .text-input:focus {
            outline: none;
            border-color: #4b5563;
            box-shadow: 0 0 0 3px rgba(75, 85, 99, 0.1);
        }

        .text-input::placeholder {
            color: #9ca3af;
        }

        .submit-btn {
            background: #374151;
            color: white;
            padding: 16px 32px;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            position: relative;
            overflow: hidden;
        }

        .submit-btn:hover {
            background: #4b5563;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .submit-btn i {
            margin-right: 8px;
        }

        .loading-container {
            text-align: center;
            padding: 40px 20px;
            display: none;
        }

        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid #f3f4f6;
            border-top: 4px solid #4b5563;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            color: #374151;
            font-size: 1.1rem;
            font-weight: 500;
        }

        .error-container {
            background: #fef2f2;
            color: #dc2626;
            padding: 16px 20px;
            border-radius: 8px;
            margin: 20px 0;
            display: none;
            border: 1px solid #fecaca;
        }

        .results-container {
            margin-top: 40px;
            display: none;
        }

        .results-header {
            text-align: center;
            margin-bottom: 30px;
        }

        .results-title {
            font-size: 2rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 10px;
        }

        .results-subtitle {
            color: #64748b;
            font-size: 1.1rem;
        }

        .image-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }

        .image-card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e5e7eb;
            transition: transform 0.3s ease;
        }

        .image-card:hover {
            transform: translateY(-2px);
        }

        .image-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 15px;
            text-align: center;
        }

        .image-wrapper {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e5e7eb;
        }

        .image-wrapper img {
            width: 100%;
            height: auto;
            display: block;
            transition: transform 0.3s ease;
        }

        .image-wrapper:hover img {
            transform: scale(1.02);
        }

        .success-animation {
            animation: fadeInUp 0.6s ease-out;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .file-preview {
            margin-top: 15px;
            display: none;
        }

        .file-preview img {
            max-width: 200px;
            max-height: 150px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        @media (max-width: 768px) {
            .container {
                padding: 20px 15px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .main-card {
                padding: 25px 20px;
            }

            .image-grid {
                grid-template-columns: 1fr;
                gap: 20px;
            }

            .submit-btn {
                padding: 14px 24px;
                font-size: 1rem;
            }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Food Segmentation With GroundingDINO and MobileSAM</h1>
            <p>Upload an image and describe the food item to get precise segmentation result</p>
        </div>

        <div class="main-card">
            <form id="upload-form" enctype="multipart/form-data">
                <div class="form-section">
                    <div class="section-title">
                        Upload Image
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Select an image to upload</label>
                        <div class="file-upload-area" id="file-upload-area">
                            <div class="file-upload-icon">
                            </div>
                            <div class="file-upload-text">Click to upload or drag and drop</div>
                            <div class="file-upload-hint">Supports: JPG, PNG, GIF, BMP (Max 10MB)</div>
                            <input type="file" name="image_file" id="image_file" accept="image/*" class="file-input" required>
                        </div>
                        <div class="file-preview" id="file-preview"></div>
                    </div>
                </div>

                <div class="form-section">
                    <div class="section-title">
                        Describe the Food
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Enter a food prompt</label>
                        <input type="text" name="prompt" id="prompt" class="text-input" 
                               placeholder="e.g., Waakye, Popcorn, Mango, Sliced Yam, Boiled Egg" required>
                    </div>
                </div>

                <button type="submit" id="submit-btn" class="submit-btn">
                    Segment Food
                </button>
            </form>

            <div id="loading" class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">Processing your image...</div>
            </div>

            <div id="error" class="error-container"></div>

            <div id="results" class="results-container">
                <div class="results-header">
                    <div class="results-title">Segmentation Results</div>
                    <div class="results-subtitle">Your food item has been successfully segmented</div>
                </div>
                
                <div class="image-grid">
                    <div class="image-card">
                        <div class="image-title">Original Image</div>
                        <div class="image-wrapper">
                            <img id="original-image" src="" alt="Original">
                        </div>
                    </div>
                    <div class="image-card">
                        <div class="image-title">Segmentation Result</div>
                        <div class="image-wrapper">
                            <img id="result-image" src="" alt="Segmented">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // File upload handling
        const fileUploadArea = document.getElementById('file-upload-area');
        const fileInput = document.getElementById('image_file');
        const filePreview = document.getElementById('file-preview');

        fileUploadArea.addEventListener('click', () => fileInput.click());
        
        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('dragover');
        });

        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('dragover');
        });

        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    filePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                    filePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        }

        // Form submission
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
                showError('Please enter a prompt describing the food item.');
                return;
            }
            
            // Show loading
            loading.style.display = 'block';
            error.style.display = 'none';
            results.style.display = 'none';
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
            
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
                    
                    // Show results with animation
                    results.style.display = 'block';
                    results.classList.add('success-animation');
                    
                    // Scroll to results
                    results.scrollIntoView({ behavior: 'smooth' });
                } else {
                    showError(result.error || 'An error occurred during processing.');
                }
            } catch (err) {
                console.error('Error:', err);
                showError('An error occurred while processing the image. Please try again.');
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fas fa-magic"></i> Segment Food';
            }
        });
        
        function showError(message) {
            const error = document.getElementById('error');
            error.textContent = message;
            error.style.display = 'block';
            error.scrollIntoView({ behavior: 'smooth' });
        }

        // Add some interactive effects
        document.addEventListener('DOMContentLoaded', function() {
            // Add pulse animation to submit button on page load
            const submitBtn = document.getElementById('submit-btn');
            submitBtn.classList.add('pulse');
            
            setTimeout(() => {
                submitBtn.classList.remove('pulse');
            }, 2000);
        });
    </script>
</body>
</html>
"""

# Web App Routes

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/health')
def health_check():
    try:
        # Try to load models if not already loaded
        if grounding_dino is None or sam_predictor is None:
            try:
                load_models()
            except Exception as e:
                return {
                    'status': 'unhealthy',
                    'error': f'Failed to load models: {str(e)}',
                    'torch_available': TORCH_AVAILABLE
                }
        
        current_device = device() if callable(device) else device
        return {
            'status': 'healthy',
            'models_loaded': {
                'grounding_dino': grounding_dino is not None,
                'sam_predictor': sam_predictor is not None
            },
            'device': str(current_device),
            'sam_predictor_type': type(sam_predictor).__name__ if sam_predictor else None,
            'torch_available': TORCH_AVAILABLE
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'torch_available': TORCH_AVAILABLE
        }

@app.route('/segment', methods=['POST'])
def segment():
    try:
        try:
            load_models()
        except Exception as e:
            return {'success': False, 'error': f'Failed to load models: {str(e)}'}
        
        if grounding_dino is None:
            return {'success': False, 'error': 'GroundingDINO model is not loaded. Check the server logs.'}
        
        if sam_predictor is None:
            return {'success': False, 'error': 'MobileSAM model is not loaded. Check the server logs.'}
        
        if 'image_file' not in request.files:
            return {'success': False, 'error': 'No image file provided.'}
        
        image_file = request.files['image_file']
        prompt = request.form.get('prompt', '').strip()
        
        if not image_file or image_file.filename == '':
            return {'success': False, 'error': 'Please select a valid image file.'}
        
        if not prompt:
            return {'success': False, 'error': 'Please provide a prompt describing the food item.'}
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if '.' not in image_file.filename or \
           image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return {'success': False, 'error': 'Upload a valid image file (PNG, JPG, JPEG, GIF, BMP).'}
        
        # Read image
        image_bytes = image_file.read()
        
        if len(image_bytes) == 0:
            return {'success': False, 'error': 'The uploaded file is empty.'}
        
        # Run models
        original_path, result_path = run_segmentation(image_bytes, prompt)
        
        if original_path is None or result_path is None:
            return {'success': False, 'error': 'Could not detect the specified object. Try a different prompt or image. Make sure your prompt clearly describes the food item you want to segment (e.g., "the boiled Egg", "Red Tomato Stew", "Green Lettuce", "Sliced Watermelon").'}
        
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)