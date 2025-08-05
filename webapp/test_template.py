#!/usr/bin/env python3
"""
Simple test script to verify the HTML template works correctly.
This script tests the template rendering without requiring the ML models.
"""

from flask import Flask, render_template_string
import os

# Create a minimal Flask app for testing
app = Flask(__name__)

# HTML template from app.py (simplified for testing)
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

@app.route('/')
def index():
    """Test the template rendering."""
    return HTML_TEMPLATE

@app.route('/segment', methods=['POST'])
def segment():
    """Mock segment endpoint for testing."""
    return {
        'success': False, 
        'error': 'This is a test endpoint. The actual segmentation requires the ML models to be loaded.'
    }

if __name__ == "__main__":
    print("Starting test server...")
    print("Visit http://localhost:5001 to test the template")
    app.run(debug=True, host='0.0.0.0', port=5001) 