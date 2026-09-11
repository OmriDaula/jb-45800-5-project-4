# Image Models — Cat vs Dog · Detection · Caption

A browser app whose **core deliverable** is a Cat vs Dog CNN **I trained from scratch**,
plus two HuggingFace models as optional extensions — all PyTorch, one FastAPI server,
nginx + Docker Compose.

**PyTorch only — no TensorFlow anywhere** (including my Cat/Dog model, rebuilt in PyTorch for this project).

For a full walkthrough of every component and design choice, see
**[ARCHITECTURE.md](./ARCHITECTURE.md)**.

## Quick start (what the lecturer runs)

```bash
git clone <this-repo-url>
cd jb-45800-5-project-4
docker compose up --build
```

Then open **http://localhost:8080**

- First build downloads HuggingFace DETR + BLIP weights **into the image** (several minutes, once).
- Containers start **offline** — no Hub traffic at runtime.
- Wait until the backend is healthy (models load on CPU, ~1–2 minutes), then use the UI.
- The default tab is **Cat vs Dog** (my trained model).

Stop with `Ctrl+C`, or if started detached: `docker compose down`.

## What it does

| Tab | Role | Model | Endpoint |
|-----|------|--------|----------|
| **Cat vs Dog** | **My trained model (core)** | From-scratch CNN (`catdog.pt`) | `POST /api/classify` |
| Object Detection | Pretrained extension | `facebook/detr-resnet-50` | `POST /api/detect` |
| Image Caption | Pretrained extension | `Salesforce/blip-image-captioning-base` | `POST /api/caption` |

**One-line summary:** *My model classifies; DETR localizes (where); BLIP describes (what).*

## Architecture

```
  Browser  (http://localhost:8080)  — default tab: Cat vs Dog
      |
      |  static HTML/CSS/JS
      |  /api/*  JSON + image upload
      v
  +-----------+       +---------------------------+
  |  nginx    | ----> |  FastAPI (uvicorn)        |
  |  frontend | /api  |  loads 3 models at start  |
  +-----------+       +-------------+-------------+
                                    |
              +---------------------+---------------------+
              |                     |                     |
              v                     v                     v
        CatDog CNN            DETR (HF)             BLIP (HF)
        model/catdog.pt       model_cache/          model_cache/
        (committed in git)    (baked at build)      (baked at build)
```

- **nginx** serves the SPA and reverse-proxies `/api/` → `backend:8000/`.
- **FastAPI** loads every model **once** at startup (`local_files_only=True`).
- **No dataset** in this repo. **No training at runtime** — inference only.

## The models

### 1. Cat vs Dog — my own model (the core deliverable)

A convolutional neural network I **built and trained from scratch in PyTorch**
(no pretrained weights, no transfer learning).

- **Architecture:** input 128×128 RGB → rescaling → 4 Conv+MaxPool blocks
  (16→32→64→128) → Flatten → Dense(64) → Dropout(0.3) → single logit
  (sigmoid at inference). **~621k parameters.**
- **Data:** small, imbalanced cats/dogs set (**275 train / 70 val**). The trained
  weights (`backend/model/catdog.pt`) **are committed** to this repo, as the
  assignment requires. The **dataset itself is NOT committed**.
- **Honest performance:** ~**68–73%** validation accuracy — it detects dogs
  reliably but struggles with cats, because it learned from only **95 cat**
  training images. That is a known **data-ceiling** limitation, documented in full.
- Full training investigation (7 experiments, git branches, calibration) scored
  **100/100**:
  **https://github.com/OmriDaula/jb-45800-5-mission-4**

### 2. Extensions — pretrained models from HuggingFace

Beyond the required single model, I added **two pretrained PyTorch models** to
demonstrate additional computer-vision tasks. These are **off-the-shelf from
HuggingFace**, downloaded at **docker build time** and baked into the image
(not trained by me; not committed — each weight file exceeds GitHub’s 100 MB limit).

- **DETR (`facebook/detr-resnet-50`)** — object detection: finds **where**
  objects are, draws bounding boxes with confidence scores (UI keeps score > 0.7).
- **BLIP (`Salesforce/blip-image-captioning-base`)** — image captioning:
  describes **what** is happening in the image in a sentence.

**My model classifies; DETR localizes (where); BLIP describes (what).**

## Why PyTorch only

The whole project is **PyTorch — no TensorFlow**. That includes Cat vs Dog:
the original mission-4 model was Keras; for this project I **rebuilt and
retrained the same architecture in PyTorch** so DETR, BLIP, and my CNN share
one stack, one Docker image, and one mental model.

## Model weights & GitHub’s 100 MB limit

| File | Size | In git? | How the lecturer gets it |
|------|------|---------|---------------------------|
| `backend/model/catdog.pt` | ~2.4 MB | **Yes** (my trained model) | clone |
| DETR `model.safetensors` | ~**159 MB** | **No** (>100 MB) | `docker compose build` downloads once |
| BLIP `model.safetensors` | ~**944 MB** | **No** (>100 MB) | `docker compose build` downloads once |

Pushing DETR/BLIP with normal git **will fail** (GitHub hard limit 100 MB per file).
We deliberately **do not** commit them and **do not** require git-lfs.
Build-time download keeps: `clone → docker compose up --build → browser` with zero manual steps,
and still **no download on every container start**.

## Project layout

```
project/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── app.py              # FastAPI: /health /detect /caption /classify
│   ├── catdog_model.py     # shared CNN definition
│   ├── model/catdog.pt     # committed (core deliverable)
│   ├── scripts/            # train + one-time HF download helpers
│   └── requirements.txt
└── frontend/
    ├── Dockerfile          # nginx + /api proxy
    ├── nginx.conf
    ├── index.html
    ├── styles.css
    └── app.js
```

## Local development (optional, without Docker)

```bash
# Terminal 1 — API (Python 3.11 venv)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/scripts/download_detr.py   # once
python backend/scripts/download_blip.py   # once
uvicorn backend.app:app --host 127.0.0.1 --port 8000

# Terminal 2 — static UI
cd frontend && python3.11 -m http.server 5500
# open http://127.0.0.1:5500  (meta api-base points at :8000)
```

## Screenshot placeholders

Add your browser screenshots here (URL bar showing `localhost:8080`):

1. **Cat vs Dog** — prediction + confidence bar (default tab)  
2. **Object Detection** — boxes on a real photo  
3. **Image Caption** — generated sentence  

---

Course project · JB-45800-5 · inference-only · Docker Compose · PyTorch
