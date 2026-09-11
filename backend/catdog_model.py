"""
Cat vs Dog CNN — shared PyTorch definition.

Used by:
  - backend/scripts/train_catdog_torch.py   (training)
  - FastAPI /classify                      (Phase 1 inference)

Architecture mirrors mission-4's Keras winner:
  128x128 RGB → /255 → 4 conv blocks (16→32→64→128) → Flatten
  → Linear(8192→64) → ReLU → Dropout(0.3) → Linear(64→1) logits

Why logits (no sigmoid in forward):
  Training uses BCEWithLogitsLoss (numerically stable).
  At inference we apply torch.sigmoid on the logit to get P(dog).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

IMG_SIZE = 128  # height = width; single source of truth for train + serve
CLASS_NAMES = ("cat", "dog")  # index 0 = cat, index 1 = dog (positive class)


class CatDogCNN(nn.Module):
    """From-scratch CNN — every weight learned on the mission-4 275-image set."""

    def __init__(self, dropout_p: float = 0.3) -> None:
        super().__init__()

        # Four conv blocks: channels 16 → 32 → 64 → 128.
        # padding=1 on a 3x3 kernel = Keras padding="same" (spatial size unchanged
        # by the conv; MaxPool2d(2) then halves H and W).
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128 → 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 → 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 → 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 → 8
        )

        # After 4 pools: 128 / 2^4 = 8 → feature map is 128 x 8 x 8 = 8192.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1),  # raw logit; sigmoid applied only at inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: float tensor (N, 3, H, W) with pixel values in 0..255
           (same contract as Keras: Rescaling lived inside the model).
        returns: logits shaped (N, 1)
        """
        x = x / 255.0
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_catdog_weights(weights_path: Path | str, device: torch.device | None = None) -> CatDogCNN:
    """Build the architecture and load a state_dict saved by the training script."""
    device = device or torch.device("cpu")
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cat/Dog weights not found: {path}")

    model = CatDogCNN()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
