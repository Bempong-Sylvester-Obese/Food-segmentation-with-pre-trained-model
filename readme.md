# Food Segmentation Using GroundingDINO and MobileSAM

<img width="1489" height="402" alt="Unknown" src="https://github.com/user-attachments/assets/ac699880-2d8c-4ba0-8614-248f586e8bca" />
<img width="1489" height="402" alt="Unknown-2" src="https://github.com/user-attachments/assets/921cdd81-8236-40d3-a31e-2c7f9a8891ab" />

A project for prompt-guided food segmentation using pre-trained models. [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) performs text-conditioned object detection and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) produces precise masks. The repository ships a Flask web application (`webapp/`) and a Google Colab notebook (`Food_Segmentation.ipynb`) for experimentation.

## Research first

Before running the code, skim the papers in `Research/` to understand what the models are (and are not) good at:

- **GroundingDINO** — prompt-guided object detection
- **Guided Diffusion Model for Adversarial Purification**
- **Image Segmentation Using Text and Image Prompts**

## Features

- Prompt-guided segmentation: upload an image, describe the target (e.g. `jollof rice`), get an overlay mask.
- Flask web UI served from a Jinja template (`webapp/templates/index.html`).
- JSON API: `POST /api/v1/segment` returns base64-encoded original + result images plus metadata; `/segment` remains for the browser UI.
- Health endpoints: `/livez` for liveness, `/readyz` for model readiness, and `/health` for compatibility/status.
- Security controls: optional API key enforcement, rate limiting, security headers, upload byte limits, image dimension validation, and checkpoint size/hash verification hooks.
- Automatic checkpoint download on first run in development (with connect/read timeouts and exponential backoff retries) into `webapp/GroundingDINO/` and `webapp/MobileSAM/weights/`. Production Docker disables runtime downloads by default; mount verified checkpoints or bake them into your deployment image.
- Gunicorn-ready: `load_models()` runs at import time under `--preload`, so the first request is not stuck waiting for PyTorch.
- Docker: `Dockerfile` at the repo root runs on Python 3.11 and serves on port 8080.
- CI: ruff lint + format checks and pytest on every push/PR (see `.github/workflows/ci.yml`).

## Project structure

```text
Food-segmentation-with-pre-trained-model/
├── .github/workflows/ci.yml       # Ruff + pytest CI
├── Dockerfile                     # python:3.11-slim base, gunicorn on :8080
├── requirements.txt               # Runtime deps (torch, flask, groundingdino deps, ...)
├── requirements-dev.txt           # pytest, pytest-cov, ruff (pulls in requirements.txt)
├── ruff.toml                      # target-version=py311, select=E/F/W/I, line-length=120
├── pyproject.toml                 # [tool.pytest.ini_options] only
├── cursor.md                      # Agent / Cursor orientation notes
├── Food_Segmentation.ipynb        # Google Colab workflow
├── webapp/
│   ├── app.py                     # Flask app: /, /health, /segment, run_segmentation()
│   ├── model_loader.py            # Imports + downloads + loads GroundingDINO & MobileSAM
│   ├── templates/index.html       # Upload UI (rendered via render_template)
│   ├── GroundingDINO/             # Vendored GroundingDINO (editable install target)
│   └── MobileSAM/                 # Vendored MobileSAM source
├── tests/
│   ├── conftest.py                # Flask client + model mocks
│   ├── test_routes.py             # /, /health
│   ├── test_segment_validation.py # /segment input validation + success path (mocked)
│   ├── test_segmentation_unit.py  # run_segmentation() unit checks
│   └── test_segment_integration.py# Full pipeline (opt-in via RUN_MODEL_INTEGRATION=1)
├── Food images/                   # Sample food images for manual testing
├── Research/                      # Reference papers
└── Results/                       # Example outputs + analysis
```

## Prerequisites

- **Python 3.11** (the Dockerfile, CI, and `ruff.toml` target `py311`; older Python 3.9 fails at import time on modern type-annotation syntax, and Python 3.14 does not yet have compatible `torch` wheels).
- A C++ toolchain if you want GroundingDINO's native CUDA ops (Xcode CLI tools on macOS, `build-essential` on Debian/Ubuntu). CPU-only builds work without the native extensions.
- Deps listed in `requirements.txt` (PyTorch, OpenCV headless, Flask, transformers <5, timm, supervision, etc.).

## Installation

### Option 1: Local Flask app

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

pip install -e webapp/GroundingDINO --no-build-isolation
pip install -e webapp/MobileSAM --no-build-isolation
```

`--no-build-isolation` is required for GroundingDINO so the build sees your already-installed PyTorch; the upstream `setup.py` is not compatible with PEP 517 isolated builds.

Checkpoints are downloaded automatically on first load into:

- `webapp/GroundingDINO/groundingdino_swint_ogc.pth`
- `webapp/MobileSAM/weights/mobile_sam.pt`

### Option 2: Docker

```bash
docker build -t food-segmentation .
docker run --rm -p 8080:8080 food-segmentation
```

The container runs `gunicorn --preload webapp.app:app` on port **8080**. `--preload` plus the import-time `load_models()` means models are loaded once at boot rather than on the first HTTP request. Health check: `GET /health`.

### Option 3: Google Colab

Open `Food_Segmentation.ipynb` in Google Colab and run the cells top to bottom. The notebook clones the upstream GroundingDINO/MobileSAM repos, installs deps, downloads weights, and runs segmentation on sample images.

### Troubleshooting

- **Python 3.14 `.venv` hangs on `import torch`** — torch does not ship wheels for 3.14 yet. Recreate the venv on Python 3.11: `python3.11 -m venv .venv && pip install -r requirements.txt`.
- **`transformers` must stay `<5`**. GroundingDINO relies on BERT helpers removed in transformers v5; `requirements.txt` pins `transformers>=4.40.0,<5.0.0`.
- **Port already in use** — local `python webapp/app.py` defaults to port **5001**; override with `PORT=8765 python webapp/app.py`. Docker uses `ENV PORT=8080`.
- **`git status` slow** — `.gitignore` excludes `.venv/`, caches, and checkpoint files. Vendored `webapp/GroundingDINO` and `webapp/MobileSAM` are plain source trees and do not need a nested `.git` for the app to work.

## Running the web app

```bash
source .venv/bin/activate
python webapp/app.py           # serves on http://127.0.0.1:5001
# or
PORT=8765 python webapp/app.py
```

Open the printed URL, choose an image, type a prompt (e.g. `jollof rice`, `plantain`, `banku`), and submit. The UI renders the original and segmented images side-by-side and surfaces API errors inline.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | HTML upload UI (renders `webapp/templates/index.html`). |
| `POST` | `/segment` | Run segmentation. `multipart/form-data` with fields `image_file` (PNG/JPG/JPEG/GIF/BMP, ≤10 MB) and `prompt` (non-empty string). |
| `GET`  | `/health` | JSON health report. |

### `POST /segment` and `POST /api/v1/segment` responses

`/segment` keeps the browser-compatible contract where validation/inference problems return HTTP `200` with `success: false`. `/api/v1/segment` uses standard HTTP status codes (`400`, `401`, `413`, `429`, `422`, `503`, `500`) plus a machine-readable `code`.

Success:

```json
{
  "success": true,
  "original_image": "<base64-encoded PNG>",
  "result_image": "<base64-encoded PNG with mask overlay + bounding boxes>",
  "metadata": {
    "duration_ms": 1234.56,
    "detections": {
      "count": 2,
      "boxes": [[0, 0, 10, 10]],
      "confidence": [0.91],
      "phrases": ["jollof rice"]
    }
  }
}
```

Failure (e.g. missing field, unsupported extension, model not loaded, no detection):

```json
{ "success": false, "code": "empty_prompt", "error": "Please provide a prompt." }
```

Truly unexpected errors return HTTP `500` with `{"success": false, "error": "An unexpected server error occurred."}`.

### Health and readiness

```json
{
  "status": "ready",
  "models_loaded": { "grounding_dino": true, "sam_predictor": true },
  "device": "cpu",
  "sam_predictor_type": "SamPredictor",
  "torch_available": true
}
```

## Models

- **GroundingDINO** (`webapp/GroundingDINO/`) — text-prompted object detection. Config: `GroundingDINO_SwinT_OGC.py`. Checkpoint: `groundingdino_swint_ogc.pth` (auto-downloaded).
- **MobileSAM** (`webapp/MobileSAM/`) — lightweight SAM variant used for mask generation. Checkpoint: `weights/mobile_sam.pt` (auto-downloaded).
Device selection is lazy (`model_loader.get_device_lazy()`): CUDA when available, otherwise CPU.

## Testing

Install dev dependencies and run the fast suite (integration tests that load real checkpoints are excluded by default via `addopts = -m 'not integration'` in `pyproject.toml`):

```bash
pip install -r requirements-dev.txt
pytest
```

Expected: the fast suite passes with one integration test deselected. The first run can take a while while PyTorch and OpenCV import.

Full integration pipeline (slow, loads real models):

```bash
RUN_MODEL_INTEGRATION=1 pytest -m integration
```

### Linting and formatting

CI enforces both (`.github/workflows/ci.yml`):

```bash
ruff check webapp/ tests/
ruff format --check webapp/ tests/
```

Configuration lives in `ruff.toml` (`target-version = "py311"`, `select = ["E", "F", "W", "I"]`, `line-length = 120`, with `webapp/GroundingDINO` and `webapp/MobileSAM` excluded).

## Results

- `Results/accurateresults/` — examples of successful segmentations.
- `Results/inaccuracies/` — cases where detection or masking failed.
- `Results/result.json` — aggregated run data.

## Contributing

1. Fork and create a feature branch.
2. Make your changes and keep `ruff check` + `ruff format --check` clean.
3. Add or update tests in `tests/`.
4. Open a pull request.

## License

This project uses pre-trained models from:

- GroundingDINO: https://github.com/IDEA-Research/GroundingDINO
- MobileSAM: https://github.com/ChaoningZhang/MobileSAM

Please refer to each upstream license for model usage terms.

## Project status

**In progress:**

- Performance optimization for large images
- Optional artifact storage for generated outputs
- Optional model-selection support for additional segmentation backends

**Note:** this project is experimental. The models are pre-trained and may not generalize to every food image. For production use, consider fine-tuning on your target dataset.
