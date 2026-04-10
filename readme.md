# Food Segmentation Using GroundingDINO and MobileSAM

<img width="1489" height="402" alt="Unknown" src="https://github.com/user-attachments/assets/ac699880-2d8c-4ba0-8614-248f586e8bca" />
<img width="1489" height="402" alt="Unknown-2" src="https://github.com/user-attachments/assets/921cdd81-8236-40d3-a31e-2c7f9a8891ab" />

A comprehensive project for prompt-guided food segmentation using state-of-the-art pre-trained models. This project combines GroundingDINO for object detection and MobileSAM for precise segmentation, providing both a Google Colab Notebook for experimentation and a Flask web application for easy deployment.

## **IMPORTANT: Research First!**

Before diving into the implementation, we strongly recommend reading through the research papers in the `research/` folder to understand the theoretical foundations and capabilities of the pre-trained models used in this project:

- **GroundingDINO Research**: Understanding prompt-guided object detection
- **Guided Diffusion Model for Adversarial Purification**: Advanced model techniques
- **Image Segmentation Using Text and Image Prompts**: Core segmentation concepts

These papers provide valuable insights into how the models work, their limitations, and best practices for optimal results.

## Features

- **Prompt-guided segmentation**: Upload an image and provide a text prompt to segment specific food items
- **Real-time processing**: Fast inference using pre-trained models
- **Multiple interfaces**: 
  - Google Colab notebook for experimentation and analysis
  - Flask web application for easy deployment
- **User-friendly interface**: Clean, responsive web interface
- **Transparent background**: Segmented objects are saved with transparent backgrounds
- **Docker**: `Dockerfile` at the repo root for containerized runs (see below)
- **Multiple model support**: Includes both MobileSAM and MobileSAMv2 for enhanced performance
- **Automatic model download**: Models are automatically downloaded if not present
- **Error handling**: Robust error handling and validation for various edge cases
- **Health check endpoint**: Built-in health monitoring for production deployment

## Project Structure

```
Food-segmentation-with-pre-trained-model/
├── requirements.txt            # Python dependencies (Docker + local install)
├── Food_Segmentation.ipynb     # Google Colab notebook for experimentation
├── webapp/                     # Flask web application
│   ├── app.py                  # Main Flask application with segmentation logic
│   ├── model_loader.py         # Model loading and initialization with auto-download
│   ├── static/                 # Static files (generated images)
│   │   ├── images/            # Uploaded and processed images
│   │   └── GeneratedImages/   # Segmentation results with transparent backgrounds
│   ├── GroundingDINO/         # GroundingDINO model files
│   └── MobileSAM/             # MobileSAM and MobileSAMv2 model files
│       ├── MobileSAMv2/       # Enhanced MobileSAMv2 implementation
│       └── weights/           # Model weights
├── images/                     # Sample food images for testing (40+ images)
├── Results/                    # Segmentation results and analysis
│   ├── accurateresults/       # Successful segmentation results
│   ├── inaccuracies/          # Failed segmentation cases
│   └── result.json            # Detailed results data (7,000+ lines)
└── readme.md                  # This file
```

## Prerequisites

Make sure you have the following installed:
- Python 3.7+
- PyTorch
- OpenCV
- Flask
- Google Colab (for notebook experimentation)
- Other dependencies listed in `requirements.txt` (repository root)

## Installation

### Option 1: Using Google Colab (Recommended for Development)

1. Open the Jupyter notebook in Google Colab:
   - Click the "Open in Colab" button in the notebook
   - Or manually upload `Food_Segmentation.ipynb` to Google Colab

2. The notebook will automatically:
   - Set up the environment
   - Clone the required model repositories
   - Install all dependencies
   - Download model weights
   - Load the models for experimentation

3. Run the cells sequentially to perform food segmentation experiments

### Option 2: Using the Web Application

1. From the repository root, install dependencies:
```bash
pip install -r requirements.txt
```

2. Install **GroundingDINO** as a package so `import groundingdino` works (the repo already includes the sources under `webapp/GroundingDINO`). Use `--no-build-isolation` so the build sees your already-installed PyTorch (the upstream `setup.py` is not compatible with PEP 517 isolated builds):
```bash
pip install -e webapp/GroundingDINO --no-build-isolation
```
   This step compiles small native extensions when possible. For **GPU**, install a CUDA-enabled PyTorch build first so the compile step matches your toolkit. For **CPU-only**, ensure you have a C++ compiler available (e.g. Xcode CLI tools on macOS, `build-essential` on Debian/Ubuntu); if the extension build fails, check the [upstream GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) issues for your platform.

3. Optional but recommended: install **MobileSAM** from the vendored copy so imports are consistent:
```bash
pip install -e webapp/MobileSAM
```

4. The application will automatically download model files if they don't exist:
   - GroundingDINO checkpoint: `webapp/GroundingDINO/groundingdino_swint_ogc.pth`
   - MobileSAM checkpoint: `webapp/MobileSAM/weights/mobile_sam.pt`

### Option 3: Docker

From the repository root (where the `Dockerfile` lives):

```bash
docker build -t food-segmentation .
docker run --rm -p 8080:8080 food-segmentation
```

The app listens on port `8080` (see `Dockerfile` `ENV PORT=8080`). Open `http://127.0.0.1:8080` and use the `/health` endpoint for readiness checks.

### Troubleshooting (local run)

- **Use this repo’s venv** from the project root: `source .venv/bin/activate`, then `cd webapp` and `python app.py`. Mixing Conda’s `python` with a partially-updated `.venv` can produce errors like `No module named 'werkzeug.datastructures.range'`; fix with `pip install -r ../requirements.txt` (or recreate `.venv`).
- **GroundingDINO and `transformers`**: this project pins **`transformers` 4.x** (not 5.x). Version 5 removed BERT helpers that GroundingDINO still expects (`get_head_mask`), which shows up as `BertModel` attribute errors during model load. Keep dependencies in sync via `requirements.txt`.
- **Port already in use**: local `python app.py` defaults to port **5001**. If that fails too, run `PORT=8765 python app.py` (any free port). Docker still uses **8080** inside the container via `ENV PORT=8080`.

## Usage

### Web Application

1. Activate the project venv from the repo root, then start the app (default port **5001** for local runs; override if that is busy):
```bash
source .venv/bin/activate
cd webapp
python app.py
# or e.g. PORT=8765 python app.py
```

2. Open the app in your browser, e.g. `http://127.0.0.1:5001` (or whatever port Flask prints). If you see “address already in use”, pick another: `PORT=8765 python app.py`.

3. Upload an image and enter a prompt describing the food item you want to segment (e.g., "Banku", "Jollof Rice", "Tomato Stew")

4. Click "Segment Food" to process the image

5. View the results showing both the original image and the segmented object

### Google Colab Notebook

1. Open `Food_Segmentation.ipynb` in Google Colab
2. Run the cells sequentially to:
   - Set up the environment (automatic in Colab)
   - Clone and install model repositories
   - Download and load models
   - Perform segmentation on sample images
   - Analyze results

## API Endpoints (Web Application)

- `GET /`: Main web interface
- `POST /segment`: Process image segmentation (expects multipart form data with `image_file` and `prompt`)
- `GET /health`: Health check endpoint for monitoring
- `GET /static/<filename>`: Serve static files (images)
- `GET /static/images/<filename>`: Serve processed images

## Model Information

- **GroundingDINO**: Used for object detection based on text prompts
  - Repository: https://github.com/IDEA-Research/GroundingDINO
  - Detects objects in images based on natural language descriptions
  - Automatically downloaded if not present
- **MobileSAM**: Used for precise segmentation of detected objects
  - Repository: https://github.com/ChaoningZhang/MobileSAM
  - Lightweight version of SAM (Segment Anything Model) for mobile deployment
- **MobileSAMv2**: Enhanced version with object-aware prompt sampling
  - Available in the MobileSAM directory
  - Faster segmentation with improved accuracy
- Both models run on CPU by default (GPU support available if CUDA is installed)

## Testing

### Automated tests (pytest)

From the repository root, install dev dependencies and run the fast suite (integration tests that load real models are excluded by default):

```bash
pip install -r requirements-dev.txt
pytest
```

The first pytest run may take a while with little output while **PyTorch** and **OpenCV** load; that is expected on some setups.

- **Integration / full pipeline** (slow; needs checkpoints and may download weights):  
  `RUN_MODEL_INTEGRATION=1 pytest -m integration`
- **`git status` slow locally?** Ensure you are not scanning a huge untracked tree: the root [`.gitignore`](.gitignore) ignores `.venv/`, caches, and generated static paths. Vendored `webapp/GroundingDINO` and `webapp/MobileSAM` are plain source trees and do **not** need a nested `.git` folder for the app to work; update them by replacing the directory or using a pinned VCS dependency if you prefer not to vendor.

### Manual Testing

1. Use the sample images in the `images/` directory (40+ food images available)
2. Try different prompts to test segmentation accuracy
3. Check the `Results/` directory for example outputs

## Results

The project includes comprehensive testing results:
- **Sample Results**: Check `Results/accurateresults/` for successful segmentations
- **Analysis**: Review `Results/inaccuracies/` for cases where segmentation failed
- **Data**: Detailed results in `Results/result.json` (7,000+ lines of analysis data)
- **Generated Images**: Processed images in `webapp/static/GeneratedImages/`

### Performance Metrics
- Successfully tested on 40+ food images
- Supports various food types: burgers, pizza, fruits, vegetables, etc.
- Real-time processing with automatic error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project uses pre-trained models from:
- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- MobileSAM: https://github.com/ChaoningZhang/MobileSAM

Please refer to their respective licenses for model usage terms.

## Project Status

 **In Progress:**
- Performance optimization for large images
- Additional model fine-tuning options
- Food Nutritional Content Analysis

**Note**: This project is designed for experimental purposes. The models are pre-trained and may not work perfectly on all types of food images. For production use, consider fine-tuning the models on your specific dataset.
