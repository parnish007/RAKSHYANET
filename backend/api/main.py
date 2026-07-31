"""RakshyaNet FastAPI application.

The active runtime is deterministic: optimization requests execute the real
StateManager pipeline through OptimizationService. No mathematical value in
this module is randomly generated.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.api.hitl_routes import router as hitl_router
from backend.api.gemma_routes import router as gemma_router
from backend.api.imagery_routes import router as imagery_router
from backend.api.optimization_routes import router as optimization_router
from backend.api.scenario_routes import router as scenario_router
from backend.api.websocket_manager import WSMessage, ws_manager
from backend.api.websocket_routes import router as ws_router


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

app = FastAPI(
    title="RakshyaNet API",
    description="AI-assisted disaster logistics decision support for Nepal",
    version="1.1.0",
)

# The two dev origins are always allowed. A deployed frontend lives on a host
# this process cannot guess, so CORS_ALLOW_ORIGINS carries it in as a
# comma-separated list. Preview deployments get a fresh subdomain per push,
# which is what CORS_ALLOW_ORIGIN_REGEX is for.
_DEV_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*_DEV_ORIGINS, *_extra_origins],
    allow_origin_regex=os.getenv("CORS_ALLOW_ORIGIN_REGEX") or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

app.include_router(hitl_router)
app.include_router(gemma_router)
app.include_router(imagery_router)
app.include_router(optimization_router)
app.include_router(scenario_router)
app.include_router(ws_router)


@lru_cache(maxsize=8)
def _read_json_at(name: str, _stamp: int) -> dict:
    """Load bundled reference data once per (file, mtime) pair."""
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _read_json(name: str) -> dict:
    """Read bundled reference data, re-reading it when the file changes.

    Caching on mtime keeps the hot path cheap while letting a fixture edited
    during rehearsal take effect without restarting the API process.
    """
    return _read_json_at(name, (DATA_DIR / name).stat().st_mtime_ns)


@app.get("/")
async def root():
    return {
        "service": "RakshyaNet API",
        "status": "running",
        "optimization_runtime": "deterministic_state_manager",
    }


@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "rakshyanet",
        "math_engine": "available",
        "dispatch_requires_approval": True,
    }


@app.get("/api/villages")
async def get_villages():
    data = _read_json("nepal_villages.json")
    return {"depot": data["depot"], "villages": data["villages"]}


@app.get("/api/vehicles")
async def get_vehicles():
    fleet_data = _read_json("fleet_config.json")
    village_data = _read_json("nepal_villages.json")
    depot = village_data["depot"]
    location = {"lat": depot["lat"], "lng": depot["lng"]}

    vehicles = []
    for category, items in (
        ("helicopter", fleet_data["helicopters"]),
        ("truck", fleet_data["trucks"]),
    ):
        for item in items:
            vehicles.append({
                **item,
                "type": category,
                "status": "available",
                "current_location": item.get("current_location", location),
                "route": [location],
            })
    return {"vehicles": vehicles}


@app.post("/api/optimize/pause")
async def pause_optimization():
    return {
        "status": "idle",
        "detail": "Optimization runs are currently atomic; no background run is active.",
    }


@app.post("/api/optimize/stop")
async def stop_optimization():
    return {
        "status": "idle",
        "detail": "Optimization runs are currently atomic; no background run is active.",
    }


# Compatibility endpoints retained for the existing news-review modal.
_SIMULATION_REVIEWS: list[dict] = []


@app.get("/api/sim/hitl/pending")
async def get_simulation_reviews():
    return {"decisions": _SIMULATION_REVIEWS}


@app.post("/api/sim/hitl/approve")
async def approve_simulation_review(body: dict):
    review_id = body.get("id")
    _SIMULATION_REVIEWS[:] = [
        item for item in _SIMULATION_REVIEWS if item.get("id") != review_id
    ]
    await ws_manager.broadcast(WSMessage(
        type="hitl_approved",
        payload={
            "id": review_id,
            "pending_count": len(_SIMULATION_REVIEWS),
            "source": "compatibility_review",
        },
    ))
    return {"status": "approved"}


@app.post("/api/sim/hitl/reject")
async def reject_simulation_review(body: dict):
    review_id = body.get("id")
    _SIMULATION_REVIEWS[:] = [
        item for item in _SIMULATION_REVIEWS if item.get("id") != review_id
    ]
    await ws_manager.broadcast(WSMessage(
        type="hitl_rejected",
        payload={
            "id": review_id,
            "reason": body.get("reason", "Operator override"),
            "pending_count": len(_SIMULATION_REVIEWS),
            "source": "compatibility_review",
        },
    ))
    return {"status": "rejected"}
