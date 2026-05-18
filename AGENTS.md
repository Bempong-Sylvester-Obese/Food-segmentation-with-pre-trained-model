# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a **Flask web app** for prompt-guided food image segmentation using GroundingDINO + MobileSAM. Single service, no database, no external APIs at runtime.

### Python version

Python **3.11** is required. The system default may be 3.12+; use `python3.11` explicitly or activate the venv at `/workspace/.venv`.

### Vendored model directories

`webapp/GroundingDINO/` and `webapp/MobileSAM/` must contain the upstream source trees (cloned from GitHub). They are installed as editable packages (`pip install -e ... --no-build-isolation`). If these directories are empty, clone them:

```bash
git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git webapp/GroundingDINO
git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM.git webapp/MobileSAM
rm -rf webapp/GroundingDINO/.git webapp/MobileSAM/.git
```

### Running the app

```bash
source .venv/bin/activate
python webapp/app.py  # serves on http://127.0.0.1:5001
```

On first run, model checkpoints (~700MB GroundingDINO + ~10MB MobileSAM) are auto-downloaded. Startup takes 30–60s on CPU while models load. The server is ready when you see `Running on http://127.0.0.1:5001`.

### Lint / Test / Build

See `readme.md` for full details. Quick reference:

- **Lint:** `ruff check webapp/ tests/ && ruff format --check webapp/ tests/`
- **Tests (fast, mocked):** `pytest` — expects 16 passed, 1 deselected
- **Tests (integration, real models):** `RUN_MODEL_INTEGRATION=1 pytest -m integration`

### Gotchas

- `transformers` must stay `<5` — GroundingDINO uses APIs removed in v5.
- The first `import torch` / `import cv2` can appear to hang for 10–20s; this is normal cold-start behavior, not a stuck process.
- Tests mock the models entirely; no checkpoints needed to run `pytest`.
- The app sets `SKIP_STARTUP_LOAD` or detects pytest to avoid loading models during test runs.
- Port override: `PORT=8080 python webapp/app.py`.
