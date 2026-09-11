#!/usr/bin/env python3
"""
Phase 0 — standalone DETR smoke test (no server, no Docker).

What this does, line by line of intent:
1. Load facebook/detr-resnet-50 ONLY from the local folder we already downloaded.
2. Run object detection on one sample image.
3. Print label / score / bounding box for every detection above a score threshold.

Usage (from repo root, with the Phase-0 venv active):
    source .venv/bin/activate
    python backend/scripts/run_detr_local.py
    # or with your own image:
    python backend/scripts/run_detr_local.py path/to/photo.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor

# Repo layout:
#   project/
#     backend/
#       model_cache/facebook-detr-resnet-50/   ← weights live here (offline)
#       samples/demo.jpg
#       scripts/run_detr_local.py              ← this file
BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / "model_cache" / "facebook-detr-resnet-50"
DEFAULT_IMAGE = BACKEND_DIR / "samples" / "demo.jpg"

# Same threshold we will use later in the browser UI.
SCORE_THRESHOLD = 0.7


def load_image(path: Path) -> Image.Image:
    """Open an image and force RGB (DETR expects 3 channels)."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def run_detection(image: Image.Image, score_threshold: float = SCORE_THRESHOLD):
    """
    Load processor + model from the LOCAL cache (local_files_only=True),
    then return detections as plain Python dicts.
    """
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Local DETR cache missing at {MODEL_DIR}. "
            "Run the download step first (see Phase 0 notes)."
        )

    # local_files_only=True → refuse to hit HuggingFace Hub. Proves offline inference.
    processor = DetrImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model = DetrForObjectDetection.from_pretrained(
        MODEL_DIR, local_files_only=True, use_pretrained_backbone=False
    )
    model.eval()  # inference mode: disables dropout etc.

    # Processor resizes/normalizes the image the way DETR was trained.
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():  # no gradients → faster, less memory
        outputs = model(**inputs)

    # DETR returns boxes in a normalized format; post_process_object_detection
    # converts them to absolute [x_min, y_min, x_max, y_max] in pixel space.
    width, height = image.size
    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([[height, width]]),
        threshold=score_threshold,
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
                # [x, y, w, h] — top-left + size; matches what Phase 1 API will return
                "box": [
                    float(x_min),
                    float(y_min),
                    float(x_max - x_min),
                    float(y_max - y_min),
                ],
            }
        )
    return detections, width, height


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    print(f"Model dir : {MODEL_DIR}")
    print(f"Image     : {image_path}")
    print(f"Threshold : {SCORE_THRESHOLD}")
    print("-" * 60)

    image = load_image(image_path)
    detections, width, height = run_detection(image)

    print(f"Image size: {width} x {height}")
    print(f"Detections above {SCORE_THRESHOLD}: {len(detections)}")
    print("-" * 60)

    if not detections:
        print("No objects above the score threshold.")
        return 0

    for i, det in enumerate(detections, start=1):
        x, y, w, h = det["box"]
        print(
            f"{i:2d}. {det['label']:<20} "
            f"score={det['score']:.3f}  "
            f"box=[x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
