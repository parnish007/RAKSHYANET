"""Imagery classifier sidecar for RakshyaNet.

Implements contract C1 (docs/CONTRACT-imagery.md) on http://127.0.0.1:8011.

This process is deliberately separate from the RakshyaNet backend: torch and
transformers live only in this sidecar's virtualenv so that
backend/requirements.txt stays unchanged (contract C8.1).

The tiles it classifies are real Sentinel-2 RGB patches from the EuroSAT
benchmark (Helber et al.) bound to Nepali corridor ids for demonstration.
They are NOT imagery of the named corridor, and every surface says so.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("imagery_sidecar")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Candidate EuroSAT land-cover classifiers, tried in order until one loads.
MODEL_CANDIDATES = [
    os.getenv("IMAGERY_MODEL_ID") or "nielsr/vit-finetuned-eurosat-kornia",
    "mrm8488/convnext-tiny-finetuned-eurosat",
    "nielsr/swin-tiny-patch4-window7-224-finetuned-eurosat",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGERY_DIR = Path(os.getenv("IMAGERY_DIR") or (_REPO_ROOT / "backend" / "data" / "imagery"))
MANIFEST_PATH = IMAGERY_DIR / "manifest.json"

# C1: water_like is label in {"River", "SeaLake"} -- nothing else counts.
WATER_LABELS = {"River", "SeaLake"}


class ClassifyRequest(BaseModel):
    corridor_id: str = Field(..., min_length=1)
    incident_type: Optional[str] = None


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

STATE: Dict[str, Any] = {
    "pipe": None,
    "model_id": None,
    "device": "cpu",
    "warm": False,
    "tiles_by_corridor": {},
}


def _load_manifest() -> Dict[str, Dict[str, Any]]:
    """Map corridor_id -> tile entry. Missing manifest is not fatal for /health."""
    if not MANIFEST_PATH.is_file():
        logger.warning("manifest not found at %s", MANIFEST_PATH)
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    tiles = {t["corridor_id"]: t for t in manifest.get("tiles", [])}
    logger.info("manifest loaded: %d tiles from %s", len(tiles), MANIFEST_PATH)
    return tiles


def _build_pipeline():
    """Load the first candidate model that actually initialises. Returns
    (pipeline, model_id, device_string) or raises RuntimeError."""
    import torch
    from transformers import pipeline as hf_pipeline

    use_cuda = torch.cuda.is_available()
    device_index = 0 if use_cuda else -1
    device_name = "cuda" if use_cuda else "cpu"
    if use_cuda:
        logger.info("CUDA available: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("CUDA not available -- running on CPU")

    last_error: Optional[Exception] = None
    seen = set()
    for model_id in MODEL_CANDIDATES:
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        try:
            logger.info("loading %s ...", model_id)
            pipe = hf_pipeline(
                "image-classification",
                model=model_id,
                device=device_index,
            )
            return pipe, model_id, device_name
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            logger.warning("model %s failed to load: %s", model_id, exc)
            last_error = exc

    raise RuntimeError(f"no EuroSAT classifier could be loaded: {last_error}")


def _prewarm(pipe) -> None:
    """CUDA context init and the first cuDNN autotune cost 2-4 s. The backend
    allows 3 s per call with no retry (C4), so that cost must be paid at
    startup, not on the first real request."""
    dummy = Image.new("RGB", (224, 224), (72, 96, 84))
    started = time.perf_counter()
    pipe(dummy)
    pipe(dummy)
    logger.info("pre-warm complete in %.0f ms", (time.perf_counter() - started) * 1000.0)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE["tiles_by_corridor"] = _load_manifest()
    pipe, model_id, device_name = _build_pipeline()
    _prewarm(pipe)
    STATE.update({"pipe": pipe, "model_id": model_id, "device": device_name, "warm": True})
    logger.info("sidecar ready: model=%s device=%s", model_id, device_name)
    yield


app = FastAPI(title="RakshyaNet imagery sidecar", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if STATE["warm"] else "loading",
        "model_id": STATE["model_id"],
        "device": STATE["device"],
        "warm": bool(STATE["warm"]),
    }


@app.post("/classify")
def classify(request: ClassifyRequest) -> Dict[str, Any]:
    tile = STATE["tiles_by_corridor"].get(request.corridor_id)
    if tile is None:
        raise HTTPException(
            status_code=404,
            detail=f"no imagery tile bound to corridor '{request.corridor_id}'",
        )

    if not STATE["warm"] or STATE["pipe"] is None:
        raise HTTPException(status_code=503, detail="classifier not warm")

    tile_path = IMAGERY_DIR / tile["file"]
    if not tile_path.is_file():
        raise HTTPException(status_code=500, detail=f"tile file missing: {tile['file']}")

    started = time.perf_counter()
    with Image.open(tile_path) as image:
        predictions = STATE["pipe"](image.convert("RGB"))
    latency_ms = (time.perf_counter() - started) * 1000.0

    top = max(predictions, key=lambda p: p["score"])
    label = str(top["label"])

    return {
        "corridor_id": request.corridor_id,
        "tile_id": tile["tile_id"],
        "label": label,
        "confidence": round(float(top["score"]), 4),
        "water_like": label in WATER_LABELS,
        "reference_label": tile["reference_label"],
        "model_id": STATE["model_id"],
        "device": STATE["device"],
        "latency_ms": round(latency_ms, 1),
        "tile_relative_path": f"imagery/{tile['file']}",
        "acquired_at": tile["acquired_at"],
        "sensor": tile["sensor"],
    }
