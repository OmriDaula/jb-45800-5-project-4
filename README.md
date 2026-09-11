# Image Models — Object Detection · Captioning · Cat vs Dog

A browser app that runs **three PyTorch models** behind one FastAPI server,
served with nginx via Docker Compose. Upload an image, switch tabs, get results.

**PyTorch only — no TensorFlow anywhere.**

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

Stop with `Ctrl+C`, or if started detached: `docker compose down`.

## What it does

| Tab | Model | Endpoint |
|-----|--------|----------|
| Object Detection | `facebook/detr-resnet-50` | `POST /api/detect` |
| Image Caption | `Salesforce/blip-image-captioning-base` | `POST /api/caption` |
| Cat vs Dog | My from-scratch CNN (`catdog.pt`) | `POST /api/classify` |

## Architecture

```
  Browser  (http://localhost:8080)
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
        DETR (local)          BLIP (local)         CatDog CNN
        model_cache/          model_cache/         model/catdog.pt
        (baked at build)      (baked at build)     (committed in git)
```

- **nginx** serves the SPA and reverse-proxies `/api/` → `backend:8000/`.
- **FastAPI** loads every model **once** at startup (`local_files_only=True`).
- **No dataset** in this repo. **No training at runtime** — inference only.

## The three models

### 1. Object Detection — DETR

- HuggingFace: [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50)
- Returns boxes `[x, y, w, h]` + labels + scores; UI draws detections with score > 0.7.
- Weights are fetched during **Docker image build**, then served offline.

### 2. Image Caption — BLIP

- HuggingFace: [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
- Returns a short English sentence describing the image.
- Same build-time fetch / runtime-offline pattern as DETR.

### 3. Cat vs Dog — my CNN (mission 4)

A small CNN I trained **from scratch** (no transfer learning) on **275 images**:
four conv blocks (16→32→64→128), dense head, dropout, class imbalance via
`pos_weight`, checkpointed on lowest unweighted validation loss.

- Weights in git: `backend/model/catdog.pt` (~2.4 MB, 621,857 parameters).
- Honest limits: it reliably leans toward dogs and struggles with cats —
  expected on a tiny dataset.
- Full calibration story (experiments, metrics, scored **100/100**):
  **https://github.com/OmriDaula/jb-45800-5-mission-4**

## Model weights & GitHub’s 100 MB limit

| File | Size | In git? | How the lecturer gets it |
|------|------|---------|---------------------------|
| `backend/model/catdog.pt` | ~2.4 MB | **Yes** (required) | clone |
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
│   ├── model/catdog.pt     # committed
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

1. **Object Detection** — boxes on a real photo  
2. **Image Caption** — generated sentence  
3. **Cat vs Dog** — prediction + confidence bar  

---

Course project · JB-45800-5 · inference-only · Docker Compose · PyTorch
