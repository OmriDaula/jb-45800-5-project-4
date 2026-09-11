#!/usr/bin/env python3
"""
Phase 0.5 — train the Cat vs Dog CNN in PyTorch (no TensorFlow).

Ports the mission-4 winning methodology to PyTorch:
  - seed 42
  - 128x128 RGB, rescaling 1/255 inside the model
  - RandomHorizontalFlip + RandomRotation(~10% of a turn ≈ ±36°)
  - class imbalance: BCEWithLogitsLoss(pos_weight=...) on TRAIN only
  - validation loss is UNWEIGHTED (matches Keras: class_weight ignores val)
  - 30 epochs, Adam 1e-3
  - checkpoint on LOWEST unweighted val_loss → backend/model/catdog.pt

Dataset stays OUT of this repo. Default path points at the sibling mission-4
checkout; override with --data-dir.

Usage (from project root, Phase-0 venv active):
    source .venv/bin/activate
    python backend/scripts/train_catdog_torch.py

    # or if mission-4 lives elsewhere:
    python backend/scripts/train_catdog_torch.py --data-dir /path/to/jb-45800-5-mission-4/data
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Allow `import catdog_model` when running as a script from repo root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from catdog_model import (  # noqa: E402
    CLASS_NAMES,
    IMG_SIZE,
    CatDogCNN,
    count_parameters,
)

# ----------------------------------------------------------------------------
# Configuration (mirrors mission-4 train.py knobs)
# ----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
# Keras RandomRotation(0.1) = ±10% of a full turn = ±36 degrees.
ROTATION_DEGREES = 36.0

DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "jb-45800-5-mission-4" / "data"
DEFAULT_OUT = BACKEND_DIR / "model" / "catdog.pt"

EXPERIMENT_NAME = "pytorch-port"
LINE = "=" * 64


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch so a rerun is as close as possible.

    Exact bit-equality with the Keras run is NOT expected (different framework,
    different random streams) — that is fine per the Phase 0.5 success bar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class ToFloatTensor255:
    """PIL → float CHW tensor in 0..255 (NO /255 — the model does that)."""

    def __call__(self, img) -> torch.Tensor:
        # np.asarray keeps uint8 0..255; cast to float32 without scaling.
        arr = np.asarray(img, dtype=np.float32)  # HWC
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW


def build_transforms(train: bool) -> transforms.Compose:
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        # Same idea as Keras RandomFlip("horizontal") + RandomRotation(0.1).
        # Active only on the training loader — val sees upright originals.
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=ROTATION_DEGREES),
            ]
        )
    ops.append(ToFloatTensor255())
    return transforms.Compose(ops)


def load_imagefolder(split_dir: Path, train: bool) -> datasets.ImageFolder:
    if not split_dir.is_dir():
        raise SystemExit(
            f"Missing dataset folder: {split_dir}\n"
            "Pass --data-dir pointing at mission-4's data/ "
            "(with train/cat, train/dog, val/cat, val/dog)."
        )
    ds = datasets.ImageFolder(str(split_dir), transform=build_transforms(train=train))
    # Defend the label contract: ImageFolder sorts alphabetically → cat=0, dog=1.
    if tuple(ds.classes) != CLASS_NAMES:
        raise SystemExit(
            f"Expected classes {CLASS_NAMES}, got {tuple(ds.classes)} in {split_dir}"
        )
    return ds


def count_labels(dataset: datasets.ImageFolder) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for _, label in dataset.samples:
        counts[CLASS_NAMES[label]] += 1
    return counts


def compute_pos_weight(train_counts: dict[str, int]) -> torch.Tensor:
    """
    BCEWithLogitsLoss pos_weight for the positive class (dog).

    Dogs are the majority. pos_weight = n_neg / n_pos = n_cat / n_dog
    down-weights dog errors relative to cat errors — same relative effect as
    mission-4's inverse-frequency class_weight (cats cost ~1.9× dogs).
    """
    n_cat = train_counts["cat"]
    n_dog = train_counts["dog"]
    return torch.tensor([n_cat / n_dog], dtype=torch.float32)


def keras_style_class_weights(train_counts: dict[str, int]) -> dict[int, float]:
    """Same formula as mission-4 — printed for the report, not fed to the loss."""
    total = sum(train_counts.values())
    return {
        i: total / (len(CLASS_NAMES) * train_counts[name])
        for i, name in enumerate(CLASS_NAMES)
    }


def majority_class_baseline(counts: dict[str, int]) -> tuple[str, float]:
    majority = max(counts, key=counts.get)
    return majority, counts[majority] / sum(counts.values())


# ----------------------------------------------------------------------------
# Train / eval loops
# ----------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """One pass over a loader. optimizer=None → eval (no grad, no dropout/aug side effects)."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        # BCEWithLogitsLoss wants float targets shaped like the logits (N, 1).
        targets = labels.float().unsqueeze(1).to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, targets)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        # Mean loss over the batch, weighted by batch size for a true epoch mean.
        batch_n = images.size(0)
        total_loss += loss.item() * batch_n
        preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
        correct += (preds.cpu() == labels).sum().item()
        total += batch_n

    return total_loss / total, correct / total


# ----------------------------------------------------------------------------
# Console report (same style as mission-4)
# ----------------------------------------------------------------------------
def print_header(train_counts: dict[str, int], val_counts: dict[str, int]) -> None:
    print(f"\n{LINE}\n  CAT vs DOG  -  training ({EXPERIMENT_NAME})\n{LINE}")
    print(
        f"  images     train {sum(train_counts.values()):>3}   "
        f"(cat {train_counts['cat']}, dog {train_counts['dog']})"
    )
    print(
        f"             val   {sum(val_counts.values()):>3}   "
        f"(cat {val_counts['cat']}, dog {val_counts['dog']})"
    )
    print(
        f"  input      {IMG_SIZE}x{IMG_SIZE} RGB   batch {BATCH_SIZE}   "
        f"epochs {EPOCHS}   seed {SEED}"
    )


def print_summary(
    history: dict[str, list[float]],
    baseline_class: str,
    baseline_acc: float,
    elapsed: float,
    out_path: Path,
) -> None:
    val_loss = history["val_loss"]
    val_acc = history["val_acc"]
    train_acc = history["train_acc"]

    saved = int(np.argmin(val_loss)) + 1
    peak = int(np.argmax(val_acc)) + 1

    print(f"\n{LINE}\n  RESULTS ({EXPERIMENT_NAME})\n{LINE}")
    print(
        f"  majority-class baseline        {baseline_acc:6.2%}"
        f'   (always answer "{baseline_class}")'
    )
    print()
    print(
        f"  saved model -> epoch {saved}"
        f"   (lowest unweighted val_loss; this is what /classify will load)"
    )
    print(f"      val_loss                   {val_loss[saved - 1]:.4f}")
    print(f"      val_accuracy               {val_acc[saved - 1]:6.2%}")
    print(f"      train_accuracy             {train_acc[saved - 1]:6.2%}")
    print(
        f"      lift over baseline         "
        f"{(val_acc[saved - 1] - baseline_acc) * 100:+6.2f} pts"
    )
    print()
    print(
        f"  for reference, the highest val_accuracy of the run was "
        f"{val_acc[peak - 1]:.2%} at epoch {peak}"
    )
    print(
        f"  (val_loss {val_loss[peak - 1]:.4f} there"
        f"{', so it was not saved' if peak != saved else ''})"
    )
    print(LINE)
    print(
        f"  trained in {elapsed:.1f}s on CPU   |   "
        f"epoch {saved} saved -> {out_path}"
    )
    print("  next step: Phase 1 FastAPI /classify\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Cat vs Dog CNN (PyTorch port of mission-4)")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"mission-4 data/ folder (default: {DEFAULT_DATA_DIR})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"where to write state_dict (default: {DEFAULT_OUT})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seeds(SEED)
    device = torch.device("cpu")

    data_dir = args.data_dir.resolve()
    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_ds = load_imagefolder(data_dir / "train", train=True)
    val_ds = load_imagefolder(data_dir / "val", train=False)

    train_counts = count_labels(train_ds)
    val_counts = count_labels(val_ds)
    baseline_class, baseline_acc = majority_class_baseline(val_counts)

    # Generators keep shuffle order tied to SEED across workers (we use 0 workers).
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print_header(train_counts, val_counts)

    model = CatDogCNN().to(device)
    n_params = count_parameters(model)
    print(f"  model      {n_params:,} trainable parameters, trained from scratch")
    # Mission-4 Keras model reported 621,857 — same topology should match.
    if n_params != 621_857:
        print(f"  warning    expected 621,857 params (Keras twin); got {n_params}")

    pos_weight = compute_pos_weight(train_counts).to(device)
    class_weights = keras_style_class_weights(train_counts)
    print(
        "  weights    "
        + ",   ".join(f"{name} x{class_weights[i]:.2f}" for i, name in enumerate(CLASS_NAMES))
        + "   (inverse class frequency; report only)"
    )
    print(
        f"  pos_weight {pos_weight.item():.4f}   "
        f"(n_cat/n_dog → train loss only; dogs=positive class)"
    )
    print("  val_loss   unweighted   (Keras class_weight never touches validation)")
    print(LINE)

    # Train: weighted. Val/checkpoint: plain BCE — same split as mission-4 Keras.
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    val_criterion = nn.BCEWithLogitsLoss()  # no pos_weight
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, train_criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, val_criterion, optimizer=None, device=device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_path)
            marker = "   <- saved"

        print(
            f"  epoch {epoch:>2}/{EPOCHS}"
            f"   loss {train_loss:.4f}  acc {train_acc:.4f}"
            f"   |   val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
            f"{marker}"
        )
    elapsed = time.time() - start

    print_summary(history, baseline_class, baseline_acc, elapsed, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
