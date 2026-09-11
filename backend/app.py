"""
Phase 1 — FastAPI backend (PyTorch only, three models).

Models load ONCE at startup (lifespan), never per request:
  - DETR   → POST /detect
  - BLIP   → POST /caption
  - CatDog → POST /classify

Plus GET /health for compose healthchecks later.

Run (from project root, venv active):
    uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000

Or from backend/:
    cd backend && uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    DetrForObjectDetection,
    DetrImageProcessor,
)

# Make `import catdog_model` work whether uvicorn is started from repo root
# (`backend.app:app`) or from inside backend/ (`app:app`).
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from catdog_model import IMG_SIZE, load_catdog_weights  # noqa: E402

# ----------------------------------------------------------------------------
# Paths (weights are local — no HuggingFace calls at runtime)
# ----------------------------------------------------------------------------
DETR_DIR = BACKEND_DIR / "model_cache" / "facebook-detr-resnet-50"
BLIP_DIR = BACKEND_DIR / "model_cache" / "salesforce-blip-image-captioning-base"
CATDOG_WEIGHTS = BACKEND_DIR / "model" / "catdog.pt"

SCORE_THRESHOLD = 0.7  # same cutoff as Phase 0 script / frontend boxes
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

DEVICE = torch.device("cpu")

# Filled in lifespan; endpoints refuse with 503 if a model failed to load.
state: dict[str, Any] = {
    "detr_processor": None,
    "detr_model": None,
    "blip_processor": None,
    "blip_model": None,
    "catdog_model": None,
    "errors": {},
}


# ----------------------------------------------------------------------------
# Startup / shutdown
# ----------------------------------------------------------------------------
def _load_detr() -> None:
    if not DETR_DIR.is_dir():
        raise FileNotFoundError(f"DETR cache missing: {DETR_DIR}")
    processor = DetrImageProcessor.from_pretrained(DETR_DIR, local_files_only=True)
    # use_pretrained_backbone=False: do NOT let timm download resnet50 from the Hub.
    # The trained backbone weights already live in our local model.safetensors.
    model = DetrForObjectDetection.from_pretrained(
        DETR_DIR,
        local_files_only=True,
        use_pretrained_backbone=False,
    )
    model.to(DEVICE)
    model.eval()
    state["detr_processor"] = processor
    state["detr_model"] = model


def _load_blip() -> None:
    if not BLIP_DIR.is_dir():
        raise FileNotFoundError(f"BLIP cache missing: {BLIP_DIR}")
    processor = BlipProcessor.from_pretrained(BLIP_DIR, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_DIR, local_files_only=True)
    model.to(DEVICE)
    model.eval()
    state["blip_processor"] = processor
    state["blip_model"] = model


def _load_catdog() -> None:
    state["catdog_model"] = load_catdog_weights(CATDOG_WEIGHTS, device=DEVICE)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load every model once. Partial failure is recorded; healthy endpoints still work."""
    loaders = {
        "detr": _load_detr,
        "blip": _load_blip,
        "catdog": _load_catdog,
    }
    for name, loader in loaders.items():
        try:
            print(f"[startup] loading {name}...")
            loader()
            print(f"[startup] {name} ready")
        except Exception as exc:  # keep other models usable
            state["errors"][name] = str(exc)
            print(f"[startup] {name} FAILED: {exc}")
    yield
    state.clear()


app = FastAPI(
    title="Image Models API",
    description="DETR detection · BLIP captioning · Cat vs Dog classification (PyTorch)",
    version="1.0.0",
    lifespan=lifespan,
)

# Browser frontend (Phase 2) will call this API from another origin/port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
async def read_rgb_image(upload: UploadFile) -> Image.Image:
    """Validate upload + decode to RGB PIL image. Raises HTTPException on bad input."""
    content_type = (upload.content_type or "").lower()
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()

    # Accept by MIME or by extension (some clients send application/octet-stream).
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unsupported_file_type",
                    "message": f"Expected JPEG/PNG/WebP, got content_type={content_type!r}",
                },
            )

    data = await upload.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "message": "Uploaded file is empty"},
        )

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_image", "message": "Could not decode image bytes"},
        ) from exc

    return image


def require_model(key: str, human_name: str, err_key: str) -> Any:
    """Return a loaded model/processor or 503 with a clear JSON body."""
    obj = state.get(key)
    if obj is None:
        reason = state.get("errors", {}).get(err_key, "model not loaded")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_unavailable",
                "message": f"{human_name} is not available: {reason}",
            },
        )
    return obj


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    """Compose-friendly healthcheck: 200 only when every model is loaded."""
    models = {
        "detr": state.get("detr_model") is not None,
        "blip": state.get("blip_model") is not None,
        "catdog": state.get("catdog_model") is not None,
    }
    ok = all(models.values())
    body: dict[str, Any] = {
        "status": "ok" if ok else "starting",
        "models": models,
    }
    if state.get("errors"):
        body["errors"] = state["errors"]
    # 503 while models load / if any failed — docker compose waits on this
    return JSONResponse(content=body, status_code=200 if ok else 503)


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Object detection (facebook/detr-resnet-50).

    Returns boxes as [x, y, w, h] in original image pixel space, score > 0.7 only.
    """
    processor = require_model("detr_processor", "DETR", "detr")
    model = require_model("detr_model", "DETR", "detr")
    image = await read_rgb_image(file)
    width, height = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[height, width]]),
        threshold=SCORE_THRESHOLD,
    )[0]

    detections = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        x_min, y_min, x_max, y_max = box.tolist()
        detections.append(
            {
                "label": model.config.id2label[label_id.item()],
                "score": float(score),
                "box": [
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                ],
            }
        )

    return {
        "detections": detections,
        "image_width": width,
        "image_height": height,
    }


@app.post("/caption")
async def caption(file: UploadFile = File(...)) -> dict[str, str]:
    """Image captioning (Salesforce/blip-image-captioning-base)."""
    processor = require_model("blip_processor", "BLIP", "blip")
    model = require_model("blip_model", "BLIP", "blip")
    image = await read_rgb_image(file)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        # Unconditional caption (no text prompt) — BLIP generates a sentence.
        out_ids = model.generate(**inputs, max_new_tokens=30)

    text = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return {"caption": text}


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Cat vs Dog (our PyTorch CNN, weights in model/catdog.pt).

    Preprocessing: resize to IMG_SIZE, keep pixels in 0..255 — /255 is inside
    the model forward(), matching training. P(dog) >= 0.5 → "dog".
    """
    model = require_model("catdog_model", "CatDog", "catdog")
    image = await read_rgb_image(file)

    # Match training: Resize → float CHW 0..255 (no ToTensor /255).
    resized = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(resized, dtype=np.float32)  # HWC 0..255
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        p_dog = float(torch.sigmoid(logit).item())

    if p_dog >= 0.5:
        prediction = "dog"
        confidence = p_dog
    else:
        prediction = "cat"
        confidence = 1.0 - p_dog

    return {
        "prediction": prediction,
        "confidence": confidence,
        "p_dog": p_dog,
    }
