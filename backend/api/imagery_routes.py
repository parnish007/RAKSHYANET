"""REST surface for overhead-imagery verification.

Three endpoints, and the reason each exists:

* `/api/imagery/status` — the operator can see, before the demo starts, which
  tier the next check will land in. A feature whose degradation is invisible is
  a feature that lies on stage.
* `/api/imagery/verify` — the escape hatch (ARCH §4 B2). One HTTP call, no
  model round trip, ~1 s instead of ~40 s. It exists so the capability is
  demonstrable on a day when the hosted model is refusing calls.
* `/api/imagery/tile/{tile_id}` — the actual JPEG, so a citation can be opened
  and the tile looked at rather than described.

Nothing here can close a road. The record this produces is subject to the same
imagery-only closure guard as one the model requested.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from backend.api.gemma_routes import broadcast_gemma_analysis
from backend.services.gemma_orchestrator import _load_corridors, append_evidence
from backend.services.gemma_service import gemma_service
from backend.services.imagery_verifier import (
    IMAGERY_DIR,
    MANIFEST_PATH,
    load_manifest,
    manifest_entry,
    manifest_entry_by_tile,
    satellite_tool_enabled,
    sidecar_health,
    sidecar_url,
    verify_corridor,
)

router = APIRouter(tags=["Imagery"])


class ImageryVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corridor_id: str = Field(min_length=1, max_length=120)
    incident_type: Literal["flood", "landslide"]
    evidence_id: Optional[str] = Field(default=None, max_length=120)


@router.get("/api/imagery/status")
async def get_imagery_status():
    """What the next check would actually do, without performing one."""
    health = await run_in_threadpool(sidecar_health)
    manifest = load_manifest()
    tiles = manifest.get("tiles", [])
    return {
        "enabled": satellite_tool_enabled(),
        "sidecar_url": sidecar_url(),
        "sidecar_reachable": health["reachable"],
        "sidecar_health": health.get("health"),
        "tiers": {
            "live": health["reachable"],
            "precomputed": any(
                isinstance(tile, dict) and tile.get("precomputed") for tile in tiles
            ),
            "unavailable": True,
        },
        "manifest_present": MANIFEST_PATH.exists(),
        "tile_count": len(tiles),
        "dataset": manifest.get("dataset"),
        "dataset_url": manifest.get("dataset_url"),
        "notice": manifest.get("notice"),
        "bound_corridors": [
            tile.get("corridor_id")
            for tile in tiles
            if isinstance(tile, dict) and tile.get("corridor_id")
        ],
        "authority": (
            "An imagery record corroborates. It can never, on its own, place a "
            "corridor in blocked_edge_ids; that is enforced in validation."
        ),
    }


@router.post("/api/imagery/verify")
async def verify_corridor_imagery(request: ImageryVerifyRequest):
    """Run one check without a model round trip and attach it to the analysis."""
    known_ids = {edge["id"] for edge in _load_corridors()}
    if request.corridor_id not in known_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown corridor '{request.corridor_id}'. "
                f"Known corridors: {', '.join(sorted(known_ids))}"
            ),
        )

    analysis = gemma_service.latest()
    if (
        request.evidence_id is not None
        and analysis is not None
        and request.evidence_id not in {item.evidence_id for item in analysis.evidence}
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"evidence_id '{request.evidence_id}' is not part of the latest "
                "analysis"
            ),
        )

    record, telemetry = await run_in_threadpool(
        verify_corridor,
        request.corridor_id,
        request.incident_type,
        "operator_request",
    )

    # No analysis yet is not an error: the operator can still see the tile and
    # the classification. The record simply has nothing to attach to, and the
    # response says so rather than inventing an analysis to hold it.
    if analysis is not None:
        append_evidence(analysis, record)
        await broadcast_gemma_analysis(analysis)

    return {
        "record": record.model_dump(mode="json"),
        "telemetry": telemetry,
        "analysis": analysis.model_dump(mode="json") if analysis is not None else None,
        "initiated_by": "operator",
        "attached": analysis is not None,
    }


@router.get("/api/imagery/tile/{tile_id}")
async def get_imagery_tile(tile_id: str):
    """Serve a bound tile. Only files named by the manifest are reachable."""
    tile = manifest_entry_by_tile(tile_id)
    if tile is None or not tile.get("file"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No imagery tile '{tile_id}' is bound in the manifest",
        )

    # The manifest is trusted input, but a traversal in it would still escape
    # the data directory, so the resolved path is checked rather than assumed.
    candidate = (IMAGERY_DIR / str(tile["file"])).resolve()
    try:
        candidate.relative_to(Path(IMAGERY_DIR).resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No imagery tile '{tile_id}' is available",
        )
    if not candidate.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Imagery tile '{tile_id}' is bound but its file is missing",
        )
    return FileResponse(candidate, media_type="image/jpeg")


@router.get("/api/imagery/corridor/{corridor_id}")
async def get_bound_tile_for_corridor(corridor_id: str):
    """Which tile, if any, is bound to a corridor — without classifying it."""
    tile = manifest_entry(corridor_id)
    if tile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No imagery tile is bound to corridor '{corridor_id}'",
        )
    return {
        "corridor_id": corridor_id,
        "tile_id": tile.get("tile_id"),
        "reference_label": tile.get("reference_label"),
        "acquired_at": tile.get("acquired_at"),
        "sensor": tile.get("sensor"),
        "has_precomputed": bool(tile.get("precomputed")),
        "notice": load_manifest().get("notice"),
    }
