"""
Lightweight API tests — no HuggingFace weights loaded.

Run from repo root (venv active):
    pip install -r backend/requirements-dev.txt
    PYTHONPATH=backend pytest backend/tests -q
"""

from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def app_module(monkeypatch):
    """Import app; stub loaders so TestClient lifespan never touches real weights."""
    import app as app_module

    monkeypatch.setattr(app_module, "_load_catdog", lambda: None)
    monkeypatch.setattr(app_module, "_load_detr", lambda: None)
    monkeypatch.setattr(app_module, "_load_blip", lambda: None)

    app_module.state.clear()
    app_module.state.update(
        {
            "detr_processor": None,
            "detr_model": None,
            "blip_processor": None,
            "blip_model": None,
            "catdog_model": None,
            "errors": {},
        }
    )
    return app_module


@pytest.fixture()
def client(app_module):
    with TestClient(app_module.app) as c:
        # Lifespan may have cleared/re-touched state; reset after enter.
        app_module.state.update(
            {
                "detr_processor": None,
                "detr_model": None,
                "blip_processor": None,
                "blip_model": None,
                "catdog_model": None,
                "errors": {},
            }
        )
        yield c


class FakeCatDog(nn.Module):
    """Tiny stand-in that returns a fixed dog logit."""

    def forward(self, x):
        batch = x.shape[0]
        return torch.full((batch, 1), 2.0)


def _png_bytes(size=(32, 32), color=(40, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _gif_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buf, format="GIF")
    return buf.getvalue()


def test_health_ok_when_only_catdog_ready(app_module, client):
    app_module.state["catdog_model"] = FakeCatDog()
    app_module.state["errors"]["blip"] = "blip failed to load"

    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["core_ready"] is True
    assert body["status"] == "ok"
    assert body["models"]["catdog"] is True
    assert body["models"]["blip"] is False
    assert "blip" in body["errors"]


def test_health_503_when_catdog_missing(app_module, client):
    app_module.state["errors"]["catdog"] = "catdog failed to load"
    app_module.state["detr_model"] = object()

    res = client.get("/health")
    assert res.status_code == 503
    body = res.json()
    assert body["core_ready"] is False
    assert body["models"]["catdog"] is False


def test_empty_upload_400(app_module, client):
    app_module.state["catdog_model"] = FakeCatDog()
    res = client.post(
        "/classify",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "empty_file"


def test_invalid_bytes_400(app_module, client):
    app_module.state["catdog_model"] = FakeCatDog()
    res = client.post(
        "/classify",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "invalid_image"


def test_unsupported_format_400(app_module, client):
    app_module.state["catdog_model"] = FakeCatDog()
    res = client.post(
        "/classify",
        files={"file": ("anim.gif", _gif_bytes(), "image/gif")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "unsupported_file_type"


def test_oversized_upload_413(app_module, client, monkeypatch):
    app_module.state["catdog_model"] = FakeCatDog()
    monkeypatch.setattr(app_module, "MAX_UPLOAD_BYTES", 100)
    res = client.post(
        "/classify",
        files={"file": ("big.png", b"x" * 200, "image/png")},
    )
    assert res.status_code == 413
    assert res.json()["detail"]["error"] == "file_too_large"


def test_classify_contract(app_module, client):
    app_module.state["catdog_model"] = FakeCatDog().eval()
    res = client.post(
        "/classify",
        files={"file": ("cat.png", _png_bytes(), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {"prediction", "confidence", "p_dog"}
    assert body["prediction"] == "dog"
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["p_dog"] >= 0.5


def test_oversized_pixels_400(app_module, client, monkeypatch):
    """Excessive dimensions must become a clean 400, not a 500."""
    app_module.state["catdog_model"] = FakeCatDog()
    monkeypatch.setattr(app_module, "MAX_IMAGE_PIXELS", 1000)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
    # 64x64 = 4096 pixels > 1000
    res = client.post(
        "/classify",
        files={"file": ("big.png", _png_bytes(size=(64, 64)), "image/png")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "image_too_large"
    assert "traceback" not in res.text.lower()
    assert "/Users/" not in res.text


def test_health_errors_are_clean_strings(app_module, client):
    app_module.state["catdog_model"] = FakeCatDog()
    app_module.state["errors"]["blip"] = "blip failed to load"
    body = client.get("/health").json()
    assert body["errors"]["blip"] == "blip failed to load"
    assert "Traceback" not in body["errors"]["blip"]
    assert "/" not in body["errors"]["blip"]  # no filesystem paths


def test_detect_503_when_detr_missing(app_module, client):
    app_module.state["catdog_model"] = FakeCatDog()
    app_module.state["errors"]["detr"] = "detr failed to load"
    res = client.post(
        "/detect",
        files={"file": ("x.png", _png_bytes(), "image/png")},
    )
    assert res.status_code == 503
    assert res.json()["detail"]["error"] == "model_unavailable"


def test_fake_mime_cannot_bypass(app_module, client):
    """GIF bytes labeled as image/png must still be rejected by Pillow format check."""
    app_module.state["catdog_model"] = FakeCatDog()
    res = client.post(
        "/classify",
        files={"file": ("fake.png", _gif_bytes(), "image/png")},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["error"] == "unsupported_file_type"
