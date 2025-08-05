# Food Segmentation Using GroundingDINO and MobileSAM

<img width="1489" height="402" alt="Unknown" src="https://github.com/user-attachments/assets/ac699880-2d8c-4ba0-8614-248f586e8bca" />
<img width="1489" height="402" alt="Unknown-2" src="https://github.com/user-attachments/assets/921cdd81-8236-40d3-a31e-2c7f9a8891ab" />

A comprehensive project for prompt-guided food segmentation using state-of-the-art pre-trained models. This project combines GroundingDINO for object detection and MobileSAM for precise segmentation, providing both a Jupyter notebook for experimentation and a Flask web application for easy deployment.

## 🚀 Features

- **Prompt-guided segmentation**: Upload an image and provide a text prompt to segment specific food items
- **Real-time processing**: Fast inference using pre-trained models
- **Multiple interfaces**: 
  - Jupyter notebook for experimentation and analysis
  - Flask web application for easy deployment
- **User-friendly interface**: Clean, responsive web interface
- **Transparent background**: Segmented objects are saved with transparent backgrounds
- **Comprehensive testing**: Built-in test suite to verify functionality

## 📁 Project Structure

```
Food-segmentation-with-pre-trained-model/
├── Food_Segmentation.ipynb     # Jupyter notebook for experimentation
├── webapp/                     # Flask web application
│   ├── app.py                  # Main Flask application
│   ├── model_loader.py         # Model loading and initialization
│   ├── test_app.py             # Test script
│   ├── requirements.txt        # Python dependencies
│   ├── static/                 # Static files (generated images)
│   │   └── images/            # Uploaded and processed images
│   ├── GroundingDINO/         # GroundingDINO model files
│   └── MobileSAM/             # MobileSAM model files
├── images/                     # Sample food images for testing
├── Results/                    # Segmentation results and analysis
│   ├── accurateresults/       # Successful segmentation results
│   ├── inaccuracies/          # Failed segmentation cases
│   └── result.json            # Detailed results data
└── readme.md                  # This file
```

## 🛠️ Prerequisites

Make sure you have the following installed:
- Python 3.7+
- PyTorch
- OpenCV
- Flask
- Jupyter Notebook
- Other dependencies listed in `webapp/requirements.txt`

## 📦 Installation

### Option 1: Using the Jupyter Notebook (Recommended for Development)

1. Clone this repository:
```bash
git clone <repository-url>
cd Food-segmentation-with-pre-trained-model
```

2. Install dependencies:
```bash
pip install -r webapp/requirements.txt
```

3. Open `Food_Segmentation.ipynb` in Jupyter Notebook:
```bash
jupyter notebook Food_Segmentation.ipynb
```

4. Run the notebook cells to set up the environment and models automatically.

### Option 2: Using the Web Application

1. Navigate to the webapp directory:
```bash
cd webapp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the model files are in the correct locations:
   - GroundingDINO checkpoint: `GroundingDINO/groundingdino_swint_ogc.pth`
   - MobileSAM checkpoint: `MobileSAM/weights/mobile_sam.pt`

## 🚀 Usage

### Web Application

1. Start the web application:
```bash
cd webapp
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image and enter a prompt describing the food item you want to segment (e.g., "the burger", "the pizza", "the apple")

4. Click "Segment Food" to process the image

5. View the results showing both the original image and the segmented object

### Jupyter Notebook

1. Open `Food_Segmentation.ipynb` in Jupyter Notebook
2. Run the cells sequentially to:
   - Set up the environment
   - Download and load models
   - Perform segmentation on sample images
   - Analyze results

## 🔧 API Endpoints (Web Application)

- `GET /`: Main web interface
- `POST /segment`: Process image segmentation (expects multipart form data with `image_file` and `prompt`)
- `GET /static/<filename>`: Serve static files (images)

## 🤖 Model Information

- **GroundingDINO**: Used for object detection based on text prompts
  - Repository: https://github.com/IDEA-Research/GroundingDINO
  - Detects objects in images based on natural language descriptions
- **MobileSAM**: Used for precise segmentation of detected objects
  - Repository: https://github.com/ChaoningZhang/MobileSAM
  - Lightweight version of SAM (Segment Anything Model) for mobile deployment
- Both models run on CPU by default (GPU support available if CUDA is installed)

## 🧪 Testing

### Web Application Testing

Run the test script to verify everything is working:
```bash
cd webapp
python test_app.py
```

### Manual Testing

1. Use the sample images in the `images/` directory
2. Try different prompts to test segmentation accuracy
3. Check the `Results/` directory for example outputs

## 📊 Results

The project includes:
- **Sample Results**: Check `Results/accurateresults/` for successful segmentations
- **Analysis**: Review `Results/inaccuracies/` for cases where segmentation failed
- **Data**: Detailed results in `Results/result.json`

## 🔍 Troubleshooting

1. **Import errors**: Make sure all dependencies are installed
2. **Model loading errors**: Check that model checkpoint files exist in the correct locations
3. **CUDA errors**: The app defaults to CPU mode. For GPU acceleration, ensure CUDA is properly installed
4. **Memory issues**: Large images may require more RAM. Consider resizing images if needed

## 📝 Dependencies

Key dependencies include:
- `torch>=1.9.0`
- `torchvision>=0.10.0`
- `supervision>=0.3.0`
- `opencv-python>=4.5.0`
- `numpy>=1.21.0`
- `flask>=2.0.0`
- `transformers>=4.20.0`
- `ultralytics>=8.0.0`

For a complete list, see `webapp/requirements.txt`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project uses pre-trained models from:
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- MobileSAM: https://github.com/ChaoningZhang/MobileSAM

Please refer to their respective licenses for model usage terms.


**Note**: This project is designed for experimental purposes. The models are pre-trained and may not work perfectly on all types of food images. For production use, consider fine-tuning the models on your specific dataset.
