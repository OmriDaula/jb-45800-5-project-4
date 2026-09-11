#!/usr/bin/env python3
"""
One-time download of facebook/detr-resnet-50 into backend/model_cache/.

Why this exists:
- Inference must NOT download from HuggingFace every container start / every run.
- We save processor + weights once into a local folder the app loads with
  local_files_only=True.

Run once (venv active, from repo root):
    python backend/scripts/download_detr.py
"""

from pathlib import Path

from transformers import DetrForObjectDetection, DetrImageProcessor

MODEL_ID = "facebook/detr-resnet-50"
OUT_DIR = Path(__file__).resolve().parents[1] / "model_cache" / "facebook-detr-resnet-50"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID}")
    print(f"Saving to   {OUT_DIR}")

    processor = DetrImageProcessor.from_pretrained(MODEL_ID)
    # From Hub (build / one-time local setup). Runtime loads with
    # use_pretrained_backbone=False so timm never re-fetches ResNet.
    model = DetrForObjectDetection.from_pretrained(MODEL_ID)
    processor.save_pretrained(OUT_DIR)
    model.save_pretrained(OUT_DIR)

    print("Done. Files:")
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            mb = path.stat().st_size / 1e6
            print(f"  {path.relative_to(OUT_DIR)}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
