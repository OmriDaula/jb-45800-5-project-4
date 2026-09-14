"""
FastAPI backend — three PyTorch image models.

Models load ONCE at startup (lifespan), never per request:
  - CatDog (core) → POST /classify
  - DETR (extension) → POST /detect
  - BLIP (extension) → POST /caption

GET /health is 200 when the core CatDog model is ready (Compose depends on this).
Extension failures are reported in JSON but do not block the UI.

Run (from project root, venv active):
    uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFile, UnidentifiedImageError

# Make `import catdog_model` work from repo root or backend/.
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from catdog_model import IMG_SIZE, load_catdog_weights  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DETR_DIR = BACKEND_DIR / "model_cache" / "facebook-detr-resnet-50"
BLIP_DIR = BACKEND_DIR / "model_cache" / "salesforce-blip-image-captioning-base"
CATDOG_WEIGHTS = BACKEND_DIR / "model" / "catdog.pt"

SCORE_THRESHOLD = 0.7  # single source of truth for DETR filtering
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB — matches nginx client_max_body_size
MAX_IMAGE_PIXELS = 25_000_000  # ~5000×5000 — decompression-bomb guard
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

DEVICE = torch.device("cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("image_models")

# Filled in lifespan; endpoints refuse with 503 if a model failed to load.
state: dict[str, Any] = {
    "detr_processor": None,
    "detr_model": None,
    "blip_processor": None,
    "blip_model": None,
    "catdog_model": None,
    "errors": {},
}

# Refuse truncated/partial images silently turning into garbage.
ImageFile.LOAD_TRUNCATED_IMAGES = False
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# ----------------------------------------------------------------------------
# Startup / shutdown
# ----------------------------------------------------------------------------
def _load_detr() -> None:
    if not DETR_DIR.is_dir():
        raise FileNotFoundError("DETR cache directory missing")
    from transformers import DetrForObjectDetection, DetrImageProcessor

    processor = DetrImageProcessor.from_pretrained(DETR_DIR, local_files_only=True)
    # use_pretrained_backbone=False: do NOT let timm download resnet50 from the Hub.
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
        raise FileNotFoundError("BLIP cache directory missing")
    from transformers import BlipForConditionalGeneration, BlipProcessor

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
    """Load every model once. Partial failure is recorded; core can still serve."""
    logger.info("Application startup — loading models")
    loaders = {
        "catdog": _load_catdog,
        "detr": _load_detr,
        "blip": _load_blip,
    }
    for name, loader in loaders.items():
        try:
            logger.info("Loading %s…", name)
            loader()
            logger.info("%s ready", name)
        except Exception:
            # Log full detail server-side; expose only a short safe message to clients.
            logger.exception("%s failed to load", name)
            state["errors"][name] = f"{name} failed to load"
    yield
    logger.info("Application shutdown")
    state.clear()


app = FastAPI(
    title="Image Models API",
    description="Cat vs Dog (core) · DETR · BLIP — PyTorch only",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def _http_error(status: int, error: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": error, "message": message})


async def read_rgb_image(upload: UploadFile) -> Image.Image:
    """
    Validate upload and decode to RGB.

    Backend enforces limits independently of nginx:
      - empty → 400
      - > MAX_UPLOAD_BYTES → 413
      - format must be JPEG/PNG/WebP (verified by Pillow, not client MIME alone)
      - pixel count capped (decompression bomb guard)
    """
    # Early reject when the client sent Content-Length.
    content_length = upload.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise _http_error(
                    413,
                    "file_too_large",
                    f"Image exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
                )
        except ValueError:
            pass

    data = await upload.read()
    if not data:
        raise _http_error(400, "empty_file", "Uploaded file is empty")

    if len(data) > MAX_UPLOAD_BYTES:
        raise _http_error(
            413,
            "file_too_large",
            f"Image exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
        )

    try:
        with Image.open(io.BytesIO(data)) as raw:
            fmt = (raw.format or "").upper()
            if fmt not in ALLOWED_PIL_FORMATS:
                raise _http_error(
                    400,
                    "unsupported_file_type",
                    "Expected JPEG, PNG, or WebP image",
                )
            # Force full decode now (catches truncated files).
            raw.load()
            width, height = raw.size
            if width <= 0 or height <= 0:
                raise _http_error(400, "invalid_image", "Image has invalid dimensions")
            if width * height > MAX_IMAGE_PIXELS:
                raise _http_error(
                    400,
                    "image_too_large",
                    "Image pixel count exceeds the allowed maximum",
                )
            image = raw.convert("RGB")
    except HTTPException:
        raise
    except Image.DecompressionBombError as exc:
        raise _http_error(
            400,
            "image_too_large",
            "Image pixel count exceeds the allowed maximum",
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise _http_error(
            400,
            "invalid_image",
            "Could not decode image bytes",
        ) from exc

    return image


def require_model(key: str, human_name: str, err_key: str) -> Any:
    """Return a loaded model/processor or 503 with a clear JSON body."""
    obj = state.get(key)
    if obj is None:
        reason = state.get("errors", {}).get(err_key, "model not loaded")
        raise _http_error(
            503,
            "model_unavailable",
            f"{human_name} is not available: {reason}",
        )
    return obj


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Compose healthcheck: 200 when the CORE CatDog model is ready.

    DETR/BLIP are extensions — their failures appear in JSON but do not
    keep the frontend from starting. /detect and /caption still return 503
    individually when their model is missing.
    """
    models = {
        "catdog": state.get("catdog_model") is not None,
        "detr": state.get("detr_model") is not None,
        "blip": state.get("blip_model") is not None,
    }
    core_ready = models["catdog"]
    body: dict[str, Any] = {
        "status": "ok" if core_ready else "unavailable",
        "core_ready": core_ready,
        "models": models,
    }
    if state.get("errors"):
        body["errors"] = dict(state["errors"])
    return JSONResponse(content=body, status_code=200 if core_ready else 503)


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict[str, Any]:
    """Object detection (pretrained DETR). Boxes already filtered by SCORE_THRESHOLD."""
    processor = require_model("detr_processor", "DETR", "detr")
    model = require_model("detr_model", "DETR", "detr")
    image = await read_rgb_image(file)
    width, height = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
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
        "threshold": SCORE_THRESHOLD,
    }


@app.post("/caption")
async def caption(file: UploadFile = File(...)) -> dict[str, str]:
    """Image captioning (pretrained BLIP)."""
    processor = require_model("blip_processor", "BLIP", "blip")
    model = require_model("blip_model", "BLIP", "blip")
    image = await read_rgb_image(file)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=30)

    text = processor.decode(out_ids[0], skip_special_tokens=True).strip()
    return {"caption": text}


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Cat vs Dog (core deliverable).

    Preprocessing: resize to IMG_SIZE, pixels stay 0..255 — /255 is inside
    CatDogCNN.forward(), matching training. P(dog) >= 0.5 → "dog".
    """
    model = require_model("catdog_model", "CatDog", "catdog")
    image = await read_rgb_image(file)

    resized = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(resized, dtype=np.float32)  # HWC 0..255
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
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
