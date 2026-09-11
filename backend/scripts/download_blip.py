#!/usr/bin/env python3
"""
One-time download of Salesforce/blip-image-captioning-base into model_cache/.

Same pattern as download_detr.py — weights are baked into the image later;
containers must NOT hit HuggingFace at startup.

Usage (venv active, from repo root):
    python backend/scripts/download_blip.py
"""

from pathlib import Path

from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_ID = "Salesforce/blip-image-captioning-base"
OUT_DIR = Path(__file__).resolve().parents[1] / "model_cache" / "salesforce-blip-image-captioning-base"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID}")
    print(f"Saving to   {OUT_DIR}")

    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    processor.save_pretrained(OUT_DIR)
    model.save_pretrained(OUT_DIR)

    print("Done. Files:")
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(OUT_DIR)}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
