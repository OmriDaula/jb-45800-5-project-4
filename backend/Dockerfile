# Backend — Python 3.11, PyTorch CPU.
# Layer order for cache: system → pip deps → HF weights (build-time) → our .pt → app code.
#
# HuggingFace DETR + BLIP are downloaded ONCE here (image build), then the
# runtime image is offline (HF_HUB_OFFLINE=1). No download on container start.
# Our catdog.pt is copied from git (small, committed).

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 1: Python deps (changes rarely) ---
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
         torch==2.2.2 torchvision==0.17.2 \
         --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/req-rest.txt \
    && pip install --no-cache-dir -r /tmp/req-rest.txt \
    && rm /tmp/req-rest.txt

# --- Layer 2: HuggingFace weights — fetch at BUILD time only ---
# Needs network during `docker compose build`. Result is baked into the image.
COPY scripts/download_detr.py scripts/download_blip.py ./scripts/
RUN python scripts/download_detr.py \
    && python scripts/download_blip.py \
    && rm -rf /root/.cache/huggingface /root/.cache/torch

# --- Layer 3: our Cat vs Dog weights (committed in git, ≈ 2.4 MB) ---
COPY model/ ./model/

# --- Layer 4: application code (changes often → fast rebuilds) ---
COPY catdog_model.py app.py ./

# Runtime: refuse Hub access. Weights already on disk inside the image.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
