# Architecture — Image Models (Project 4)

This document explains **every major piece** of the project so you can defend
design choices in a review. It complements `README.md` (how to run) with
**why** the system is shaped this way.

**Stack rule:** PyTorch only. No TensorFlow / Keras anywhere in inference or training
(including Cat vs Dog, which was rebuilt in PyTorch for this project).

**Narrative rule (matches the UI + README):**  
**My trained Cat vs Dog model is the core deliverable.** DETR and BLIP are
**pretrained HuggingFace extensions** that demonstrate *where* and *what*.
One-line summary: *My model classifies; DETR localizes (where); BLIP describes (what).*

---

## 1. What the product is

A **single web app**. Opening `http://localhost:8080` lands on **Cat vs Dog** first
(my model). Two more tabs add off-the-shelf vision tasks.

| Order (UI) | Role | Model | Who trained it | Job |
|------------|------|--------|----------------|-----|
| **1st (default)** | **Core deliverable** | `CatDogCNN` → `catdog.pt` | **Me** (mission 4 → PyTorch) | Binary classify + confidence |
| 2nd | Pretrained extension | `facebook/detr-resnet-50` | Meta / HuggingFace | Draw labeled boxes (score > 0.7) |
| 3rd | Pretrained extension | `Salesforce/blip-image-captioning-base` | Salesforce / HuggingFace | One English sentence |

The lecturer’s success path is intentionally boring and automatic:

```text
git clone → docker compose up --build → open http://localhost:8080
         → default tab = Cat vs Dog (my model)
```

No manual weight downloads, no venv, no dataset in this repo, no training at runtime.

---

## 2. Big-picture diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Browser  (http://localhost:8080)                                        │
│  plain HTML + CSS + JS  (no React / Vue)                                 │
│  tabs (default first): Cat vs Dog | Detection | Caption                  │
│  badges: "my model"  vs  "pretrained"                                    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
                    static files │  POST /api/classify|detect|caption
                                │  multipart image upload
                                ▼
┌───────────────────────────────┴──────────────────────────────────────────┐
│  frontend container — nginx :80  (published as host :8080)               │
│                                                                          │
│  location /      → serve index.html, styles.css, app.js                  │
│  location /api/  → proxy_pass http://backend:8000/   (strip /api prefix) │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  Docker network (internal)
                                ▼
┌───────────────────────────────┴──────────────────────────────────────────┐
│  backend container — uvicorn FastAPI :8000  (NOT published to host)      │
│                                                                          │
│  lifespan startup (ONCE):                                                │
│    load CNN   from /app/model/catdog.pt            ← MY weights (git)    │
│    load DETR  from /app/model_cache/...            ← HF (baked at build) │
│    load BLIP  from /app/model_cache/...            ← HF (baked at build) │
│                                                                          │
│  GET  /health     → 200 only when all three models are ready             │
│  POST /classify   → JSON prediction / confidence / p_dog  (core)         │
│  POST /detect     → JSON boxes                         (extension)       │
│  POST /caption    → JSON caption                       (extension)       │
│                                                                          │
│  ENV: HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1  (no Hub at runtime)      │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why nginx in front?**  
The browser talks to **one origin** (`localhost:8080`). Static UI and API share
that origin via `/api`, so there is no CORS pain and no need to expose the
Python port to the lecturer’s machine.

**Why models load once at startup?**  
DETR and BLIP are large. Loading them per request would make every click
unusable on CPU. FastAPI’s `lifespan` loads them into a process-global `state`
dict; endpoints only run `forward()`.

**Why Cat vs Dog is first in the UI?**  
Assignment focus = the model **I** trained. The default tab and the “my model”
badge make that obvious the moment the lecturer opens the page — DETR/BLIP are
clearly labeled as pretrained extensions.

**Health / readiness:**  
`GET /health` returns **200 when CatDog (core) is ready**. DETR/BLIP failures
are listed in the JSON (`models` / `errors`) but do **not** block Compose from
starting the frontend. `/detect` and `/caption` still return **503** if their
own model failed; `/classify` works whenever CatDog loaded.

**Course mapping:** the Python Web module teaches Flask and REST principles.
This project uses **FastAPI** as a framework-level extension while applying the
same request/response, routing, validation, HTTP status-code and REST concepts.

---

## 3. Repository layout (what each file is for)

```
jb-45800-5-project-4/
├── README.md                 # How to run; Cat vs Dog first, then HF extensions
├── ARCHITECTURE.md           # This file — deep explanation
├── .gitignore                # venv, caches; NOT catdog.pt; YES ignore HF caches
├── docker-compose.yml        # Two services, healthcheck, depends_on
│
├── backend/
│   ├── Dockerfile            # python3.11-slim; build-time HF download; offline run
│   ├── .dockerignore         # keep build context small (no host model_cache)
│   ├── requirements.txt      # Pinned torch / transformers / FastAPI (no TF)
│   ├── app.py                # FastAPI server — all HTTP endpoints
│   ├── catdog_model.py       # Shared CNN class (train + serve) — CORE
│   ├── model/
│   │   └── catdog.pt         # MY trained weights — committed (~2.4 MB) — CORE
│   ├── model_cache/          # HF dumps (gitignored; rebuilt in Docker) — extensions
│   ├── samples/demo.jpg      # Smoke-test image
│   └── scripts/
│       ├── train_catdog_torch.py  # Train MY model (reads mission-4 data/)
│       ├── download_detr.py       # Build-time / local fetch → model_cache/
│       ├── download_blip.py       # Same for BLIP
│       └── run_detr_local.py      # Offline DETR smoke test
│
└── frontend/
    ├── Dockerfile            # nginx alpine; rewrite api-base → /api
    ├── nginx.conf            # static + /api reverse proxy + long timeouts
    ├── index.html            # Tabs: classify first; badges my model / pretrained
    ├── styles.css            # Visual design + tab badges
    └── app.js                # Default mode = classify; fetch; canvas; confidence bar
```

### What is intentionally NOT here

| Missing on purpose | Why |
|--------------------|-----|
| Dataset (`data/`) | Lives in mission-4. Assignment: dataset must not be in this git repo. |
| TensorFlow / `model.keras` | Project is PyTorch-only after lecturer feedback. |
| Training at container start | Inference only; training is a separate script you run locally. |
| DETR/BLIP `.safetensors` in git | Each file > 100 MB → GitHub rejects the push. Fetched at **image build**. |

---

## 4. The models in depth

### 4.1 Cat vs Dog — my from-scratch CNN (CORE DELIVERABLE)

This is the model the assignment centers on: **I** designed the architecture,
trained the weights, and committed `catdog.pt`.

Port of the mission-4 winning architecture from Keras → **pure PyTorch**
(no pretrained backbone, no transfer learning).

```
128×128 RGB
    │
    ▼  x / 255.0          ← rescaling INSIDE the model (like Keras Rescaling)
4× (Conv2d → ReLU → MaxPool2d)   channels 16 → 32 → 64 → 128
    │  spatial: 128 → 64 → 32 → 16 → 8
    ▼
Flatten (8×8×128 = 8192)
Linear 8192 → 64 → ReLU → Dropout(0.3)
Linear 64 → 1             ← raw LOGIT (no sigmoid in forward)
```

**Parameter count:** **621,857** — identical topology to the Keras twin
(`padding=1` ≡ Keras `padding="same"`).

**Why logits + `BCEWithLogitsLoss`?**  
Numerically stable training. At inference we apply `torch.sigmoid` to get
`P(dog)`. Threshold: `P(dog) ≥ 0.5` → `"dog"`, else `"cat"` with
`confidence = 1 - P(dog)`.

**Training methodology** (`backend/scripts/train_catdog_torch.py`):

| Knob | Value | Why |
|------|-------|-----|
| Seed | 42 | Reproducible runs |
| Data | `../jb-45800-5-mission-4/data/` | Dataset stays out of this repo |
| Split | 275 train / 70 val | Same as mission 4 |
| Augmentation | Flip + Rotation ±36° (≈ Keras `RandomRotation(0.1)`) | Train only |
| Class imbalance | `pos_weight = n_cat / n_dog` on **train loss only** | Dogs are majority |
| Val loss | **Unweighted** BCE | Matches Keras: `class_weight` never touches val |
| Checkpoint | Lowest **unweighted** val_loss → `catdog.pt` | Avoid picking a lucky accuracy spike |
| Epochs | 30, Adam 1e-3 | Same budget as mission-4 synthesis |

**Honest limits (shown in the UI under the Cat vs Dog tab):**  
~68–73% validation accuracy. Detects dogs more reliably than cats — only
**95 cat** training images (data ceiling). Full investigation (scored 100/100):  
https://github.com/OmriDaula/jb-45800-5-mission-4

**Preprocessing contract (easy to get wrong):**  
Pixels enter as **0..255 floats**. The model divides by 255.  
Do **not** also use torchvision `ToTensor()` (that would scale twice → near-black images → nonsense).

Shared code: `backend/catdog_model.py` is imported by **both** the training
script and `app.py`, so train and serve cannot drift apart.

**API:** `POST /classify` → `{ prediction, confidence, p_dog }`

---

### 4.2 Extensions — pretrained HuggingFace models

Beyond the required single model, the app includes **two off-the-shelf PyTorch
models** from HuggingFace. I did **not** train them. They are downloaded at
**Docker image build** time, baked into the image, and loaded offline at runtime.

#### DETR — object detection (*where*)

- **Source:** HuggingFace `facebook/detr-resnet-50`
- **Libraries:** `transformers` (`DetrImageProcessor`, `DetrForObjectDetection`) + PyTorch
- **Input:** any RGB photo  
- **Output (API):**  
  `{ detections: [{ label, score, box: [x,y,w,h] }], image_width, image_height }`  
  Only detections with **score > 0.7** (server-side `threshold=0.7`).

**Box format:** top-left `(x, y)` plus width/height — easier for canvas drawing
than corner pairs. Coordinates are in the **original image pixel space**; the
frontend scales them to the displayed size using `image_width` / `image_height`.

**Critical offline detail:**  
Newer `transformers` + `timm` try to download a ResNet backbone
(`timm/resnet50.a1_in1k`) when constructing DETR. In Docker we set
`HF_HUB_OFFLINE=1`, so that would crash startup.

**Fix we use at load time:**

```python
DetrForObjectDetection.from_pretrained(
    DETR_DIR,
    local_files_only=True,
    use_pretrained_backbone=False,  # backbone weights already in model.safetensors
)
```

Defense sentence: *“We disable timm’s pretrained backbone download because our
saved DETR checkpoint already contains the trained backbone; runtime must stay offline.”*

#### BLIP — image captioning (*what*)

- **Source:** HuggingFace `Salesforce/blip-image-captioning-base`
- **Libraries:** `BlipProcessor` + `BlipForConditionalGeneration`
- **Input:** RGB image (no text prompt — unconditional caption)
- **Output:** `{ caption: "..." }`

Flow: processor → `model.generate(...)` → `processor.decode(...)`.

Same offline rule: weights live under `model_cache/` inside the image;
`local_files_only=True` at load.

---

## 5. Backend (`app.py`) — request lifecycle

### 5.1 Startup (`lifespan`)

1. `_load_detr()` / `_load_blip()` / `_load_catdog()` each run once.
2. Success → objects stored in `state[...]`.
3. Failure → recorded in `state["errors"]`; other models can still load.
4. Endpoints call `require_model(...)` → **503** JSON if that model is missing.

### 5.2 `GET /health`

Returns **200 when the core CatDog model is loaded**; otherwise **503**.

DETR / BLIP status is included in the JSON (`models`, optional `errors`) but
does **not** decide the HTTP status. Compose `depends_on: service_healthy`
therefore unblocks the frontend as soon as **Cat vs Dog** is usable.

Extension endpoints still return **503** individually when their model is missing.

### 5.3 Shared upload helper `read_rgb_image`

- Accepts JPEG / PNG / WebP (by MIME or file extension).
- Empty file → 400.
- Undecodable bytes → 400 with a clear JSON `detail`.
- Always converts to RGB (3 channels) for the models.

### 5.4 Endpoint summary

| Method | Path | Role | Model | Response shape |
|--------|------|------|--------|----------------|
| GET | `/health` | ops | — | `{ status, models: {detr, blip, catdog} }` |
| POST | `/classify` | **core** | CatDog | `{ prediction, confidence, p_dog }` |
| POST | `/detect` | extension | DETR | `{ detections[], image_width, image_height }` |
| POST | `/caption` | extension | BLIP | `{ caption }` |

CORS is open (`allow_origins=["*"]`) for local Phase-2 testing when the UI is
served from another port. In Docker, same-origin `/api` makes CORS irrelevant.

---

## 6. Frontend — how the browser works

**No framework.** Three files only: `index.html`, `styles.css`, `app.js`.

### 6.1 Tab order and framing (presentation, not model code)

| Tab | Badge | Default? |
|-----|--------|----------|
| **Cat vs Dog** | `my model` | **Yes** (`setMode("classify")` on load) |
| Object Detection | `pretrained` | No |
| Image Caption | `pretrained` | No |

Hints under the tabs spell out the same story (“My trained model…” vs
“Pretrained extension…”). The Cat vs Dog pane keeps the honest note + mission-4 link.

### 6.2 API base URL

```html
<meta name="api-base" content="http://127.0.0.1:8000" />
```

- **Local Phase 2:** meta points at uvicorn directly.
- **Docker:** frontend Dockerfile `sed`s it to `content="/api"` so the browser
  calls the nginx proxy on the same host.

`app.js` resolves: `?api=` query override → meta tag → fallback localhost.

### 6.3 Shared upload

One dropzone / file picker for all tabs. Preview via `URL.createObjectURL`.
Clear resets file + results.

### 6.4 Detection canvas (coordinate scaling)

Server boxes are in **original** pixels `(image_width × image_height)`.
The `<img>` is displayed smaller on screen.

```text
scaleX = displayWidth  / image_width
scaleY = displayHeight / image_height
draw box at (x * scaleX, y * scaleY, w * scaleX, h * scaleY)
```

Canvas bitmap uses `devicePixelRatio` for sharp lines on Retina screens.
Labels show `class` + score percent. Client also filters `score > 0.7`
(belt-and-suspenders; server already filters).

### 6.5 Caption & classify UI

- Classify (default): prediction text + marker on a cat←→dog bar at `P(dog)`,
  honest note, mission-4 link.
- Caption: large blockquote with the sentence; subtitle “Pretrained extension”.
- Detection list + canvas; subtitle “Pretrained extension”.
- Loading spinner + status text while CPU inference runs (seconds).
- Errors shown in a red alert (network down, 4xx/5xx, etc.).

---

## 7. Docker — build vs run (the part reviewers ask about)

### 7.1 Why `catdog.pt` is in git but DETR/BLIP are not

| File | ≈ Size | GitHub limit | Role |
|------|--------|--------------|------|
| `catdog.pt` | 2.4 MB | OK → **committed** | **My trained model** (assignment requires the `.pt`) |
| DETR `model.safetensors` | 159 MB | **> 100 MB** | Public HF extension |
| BLIP `model.safetensors` | 944 MB | **> 100 MB** | Public HF extension |

**Decision: build-time HuggingFace download for extensions, not git-lfs.**

| Approach | Pros | Cons |
|----------|------|------|
| git-lfs | Weights in “git” | Lecturer needs LFS; quota/setup can break |
| **Download in `docker build`** | `clone → compose up --build` just works | First build needs network + time |
| Download on every `docker start` | — | Forbidden by design (slow, flaky, offline-unfriendly) |

Defense sentence: *“`catdog.pt` is the model I trained — it is in git as required.
DETR and BLIP are public HuggingFace checkpoints I added as extensions; they are
fetched once while building the image and never again at container start.”*

### 7.2 Backend Dockerfile layers (cache-friendly)

```text
1. python:3.11-slim + libgomp
2. pip install torch (CPU wheel) + requirements   ← cached unless requirements change
3. RUN download_detr.py && download_blip.py       ← HF extensions; cached unless scripts change
4. COPY model/catdog.pt                           ← MY core weights from git
5. COPY app.py catdog_model.py                    ← changes often; rebuild stays fast
6. ENV HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1   ← runtime offline
```

`.dockerignore` excludes host `model_cache/` so the build context stays small;
the image gets HF weights from step 3, not from your laptop’s local cache.

### 7.3 Frontend Dockerfile

1. `nginx:1.27-alpine`
2. Copy `nginx.conf` + static assets (tabs already ordered: classify first)
3. `sed` api-base → `/api`

### 7.4 `docker-compose.yml`

- **backend:** `expose: 8000` only (internal). Healthcheck hits `/health`.
- **frontend:** `ports: "8080:80"`. `depends_on: backend (service_healthy)`.
- Long nginx `proxy_read_timeout` (300s) because CPU inference is slow.

### 7.5 nginx proxy math

```nginx
location /api/ {
    proxy_pass http://backend:8000/;   # trailing slash strips /api
}
```

| Browser requests | Backend receives |
|------------------|------------------|
| `POST /api/classify` | `POST /classify` |
| `POST /api/detect` | `POST /detect` |
| `GET /api/health` | `GET /health` |

---

## 8. End-to-end path of one click (Cat vs Dog — default)

```
1. User opens localhost:8080 → Cat vs Dog tab already active
2. Selects image, clicks "Run classification"
3. app.js → POST multipart to /api/classify
4. nginx → backend:8000/classify
5. FastAPI reads bytes → PIL RGB → resize 128×128 → float CHW 0..255
6. CatDogCNN forward (/255 inside) → logit → sigmoid → P(dog)
7. JSON { prediction, confidence, p_dog }
8. app.js shows label + confidence bar + honest note
```

Detection / caption follow the same upload path until step 6; they call
`/detect` or `/caption` and render canvas boxes or a caption sentence.

---

## 9. Local development vs Docker

| | Local (Phase 0–2) | Docker (Phase 3 — submission) |
|--|-------------------|-------------------------------|
| Backend | `uvicorn` on `:8000` | container, internal `:8000` |
| Frontend | `python -m http.server 5500` | nginx on host `:8080` |
| Default tab | Cat vs Dog | Cat vs Dog |
| HF weights | `model_cache/` on disk (gitignored) | baked into image at build |
| API URL | `http://127.0.0.1:8000` | `/api` via nginx |
| Training | `train_catdog_torch.py` + mission-4 `data/` | not used |

---

## 10. Design decisions worth defending orally

1. **Cat vs Dog first in UI/docs** — matches the assignment’s “your model” focus; HF models are framed as extensions.
2. **One FastAPI process, three models** — simpler than three microservices; shared upload validation; one healthcheck.
3. **Load once at startup** — CPU memory is acceptable; latency per request stays tolerable.
4. **PyTorch-only** — lecturer feedback; one stack for my CNN + DETR + BLIP (Keras twin rebuilt in Torch).
5. **`/255` inside CatDogCNN** — same contract as Keras `Rescaling`; predict cannot forget to scale.
6. **Unweighted val loss for checkpoints** — mirrors Keras `class_weight` behavior; fixed the bad “60% val” checkpoint.
7. **Score threshold 0.7** — reduces noisy DETR boxes in the UI.
8. **Build-time HF download** — respects GitHub 100 MB limit; lecturer one-command flow; my `.pt` still in git.
9. **Plain HTML/JS** — no frontend build step; nginx only serves files.
10. **Honest UI copy for Cat/Dog** — small data → limited generalization; link to the 100/100 investigation.

---

## 11. How to verify you understand the system

Ask yourself (and be able to answer):

- Why does `/health` return 200 if only CatDog loaded (and BLIP failed)?
- Why is `catdog.pt` in git while DETR/BLIP weights are not?
- Where do DETR/BLIP weights come from on the lecturer’s laptop?
- Why is `use_pretrained_backbone=False` required offline?
- Why isn’t `ToTensor()` used for Cat/Dog?
- Why does the backend enforce a 25 MB upload limit even though nginx also does?
- How does a box at `(40, 70, 135, 47)` on a 640×480 image get drawn on a 320×240 preview?
- What is committed in git vs fetched at build vs never stored here (dataset)?

If you can answer those, you can defend every line.

---

## 12. Related documents

- `README.md` — clone / run; Cat vs Dog first; screenshot checklist
- Mission 4 (training story): https://github.com/OmriDaula/jb-45800-5-mission-4
