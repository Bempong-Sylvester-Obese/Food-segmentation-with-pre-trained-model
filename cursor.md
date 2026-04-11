# Food segmentation — Cursor / agent orientation

This repository implements **prompt-guided food segmentation** with **GroundingDINO** (text-conditioned detection) and **MobileSAM** (segmentation). The main product surface is a **Flask** app under `webapp/`; there is also a **Colab** notebook for experimentation.

## Directory map

| Path | Role |
|------|------|
| [`requirements.txt`](requirements.txt) | Production/runtime Python deps (Docker + local). |
| [`requirements-dev.txt`](requirements-dev.txt) | Dev tools: `pytest`, `ruff`, optional `pytest-cov`. |
| [`Dockerfile`](Dockerfile) | Container: installs deps, editable installs for vendored models, gunicorn on port `8080`. |
| [`Food_Segmentation.ipynb`](Food_Segmentation.ipynb) | Colab workflow (clones upstream repos into `/content/…` for standalone runs). |
| [`webapp/app.py`](webapp/app.py) | Flask app: `/`, `/health`, `/segment`, inline HTML UI, `run_segmentation()`. |
| [`webapp/model_loader.py`](webapp/model_loader.py) | Loads GroundingDINO + MobileSAM; paths under `webapp/GroundingDINO` and `webapp/MobileSAM`; auto-download checkpoints when missing. |
| [`webapp/GroundingDINO/`](webapp/GroundingDINO/) | Vendored GroundingDINO sources (editable install target). |
| [`webapp/MobileSAM/`](webapp/MobileSAM/) | Vendored MobileSAM sources (editable install target). |
| [`images/`](images/) | Sample images for manual testing. |
| [`Results/`](Results/) | Example outputs / analysis (`result.json`, etc.). |
| [`tests/`](tests/) | Pytest suite (fast tests mock models; integration tests opt-in). |

## How to run

**Local (from repo root):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e webapp/GroundingDINO --no-build-isolation
pip install -e webapp/MobileSAM --no-build-isolation
cd webapp && python app.py
```

Default port **5001** (override with `PORT=8080`). Docker uses **8080** inside the container.

**Tests:**

```bash
pip install -r requirements-dev.txt
pytest
```

The first run can pause with **no output for a long time** while **PyTorch** and **OpenCV** import; that is normal on some machines, not a stuck process. Integration tests that need real weights: `RUN_MODEL_INTEGRATION=1 pytest -m integration`.

## Constraints and pitfalls

- **`transformers` must stay &lt; 5** (see README). GroundingDINO relies on APIs removed in v5.
- **Editable installs** for vendored code: use `--no-build-isolation` for GroundingDINO so the build sees your installed PyTorch.
- **Weights**: `.pth` / `.pt` are gitignored; first run may download into `webapp/GroundingDINO/` and `webapp/MobileSAM/weights/` per `model_loader.py`.

## Git / repo hygiene

- **Root [`.gitignore`](.gitignore)** ignores `.venv/`, caches, static uploads, and common checkpoint extensions so `git status` stays fast.
- **Vendored model trees** are normal directories; they do **not** need a nested `.git` for Python or inference—only source + checkpoints matter.
- If you only care about tracked changes: `git status -uno` (ignore untracked).

## Upstream updates

To refresh vendored GroundingDINO or MobileSAM, replace the tree from upstream or use a pinned VCS install in `requirements.txt`—there is no `git pull` inside vendored folders in this layout.
