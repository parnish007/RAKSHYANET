"""Three-tier overhead-imagery verification client.

This is the only module in the backend that knows an imagery sidecar exists.
Everything else asks it for an `EvidenceRecord` and gets one, always.

The design point worth stating plainly: **`verify_corridor` never raises.** A
verification tool that can throw is a tool that can take the orchestration loop
down with it, and an imagery check is the least important thing happening in
that loop. So every failure mode is expressed as a *tier* instead of an
exception:

* **tier 1 — live** — the local classifier answered. One POST, 3 s, no retry.
  A retry here would double the worst case for a corroborating signal that the
  system is explicitly forbidden from acting on alone.
* **tier 2 — precomputed** — the sidecar was unreachable, slow, or unhappy, so
  the cached result of a real earlier run is used. The record says so.
* **tier 3 — unavailable** — there is no cached result either. The record then
  states that *no corroboration was obtained*, which is a true and useful thing
  for the model to cite. Absence of information is not evidence of absence, and
  the record's own text says that too.

`reliability` is capped well below any human source (0.55 live/precomputed,
0.20 unavailable) so that an automated corroboration can never inflate system
confidence, and every caveat lives inside `text` so it travels with the citation
rather than depending on prompt discipline downstream.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

from backend.models.gemma import EvidenceRecord

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IMAGERY_DIR = DATA_DIR / "imagery"
MANIFEST_PATH = IMAGERY_DIR / "manifest.json"

# The exact string the closure guard keys on. Changing it silently disarms the
# one safety property in this feature that is enforced rather than promised.
IMAGERY_SOURCE_CATEGORY = "overhead_imagery_analysis"

INCIDENT_TYPES = ("flood", "landslide")
TRIGGER_REASONS = ("corroboration", "anticipatory", "operator_request")

# Live and precomputed results are the same model on the same tile; the only
# difference is when it ran, so they carry the same reliability. Both sit far
# below a witnessed field report, and nothing may push either above 0.6.
RELIABILITY_CLASSIFIED = 0.55
RELIABILITY_UNAVAILABLE = 0.20
MAX_RELIABILITY = 0.6

IMAGERY_TIMEOUT_SECONDS: float = 3.0


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def satellite_tool_enabled() -> bool:
    """Read the feature flag at call time.

    Deliberately not a frozen module constant: the orchestrator asks on every
    turn, and tests toggle it without reimporting half the backend.
    """
    return _env_flag("SATELLITE_TOOL_ENABLED", False)


def sidecar_url() -> str:
    return os.getenv("IMAGERY_SIDECAR_URL", "http://127.0.0.1:8011").rstrip("/")


# Contract-named module constants (C4). The callables above are what the running
# system consults, so an env change takes effect without a reimport.
SATELLITE_TOOL_ENABLED: bool = satellite_tool_enabled()
IMAGERY_SIDECAR_URL: str = sidecar_url()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _freshness_minutes(acquired_at: Any) -> int:
    """Minutes between acquisition and now, floored at 0.

    A tile stamped in the future is a fixture-authoring mistake, not a negative
    age, and `EvidenceRecord.freshness_minutes` is `ge=0` — so clamp rather than
    hand pydantic something it will reject.
    """
    acquired = _parse_iso(acquired_at)
    if acquired is None:
        return 0
    delta = (datetime.now(timezone.utc) - acquired).total_seconds() / 60.0
    return max(0, int(delta))


def load_manifest() -> Dict[str, Any]:
    """The tile manifest, or an empty one.

    Another workstream owns this file and it may legitimately not exist yet, so
    a missing or malformed manifest is an empty manifest — never an import-time
    or request-time failure.
    """
    try:
        raw = MANIFEST_PATH.read_text(encoding="utf-8")
    except OSError:
        return {"tiles": []}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"tiles": []}
    if not isinstance(data, dict):
        return {"tiles": []}
    if not isinstance(data.get("tiles"), list):
        data["tiles"] = []
    return data


def manifest_entry(corridor_id: str) -> Optional[Dict[str, Any]]:
    for tile in load_manifest().get("tiles", []):
        if isinstance(tile, dict) and tile.get("corridor_id") == corridor_id:
            return tile
    return None


def manifest_entry_by_tile(tile_id: str) -> Optional[Dict[str, Any]]:
    for tile in load_manifest().get("tiles", []):
        if isinstance(tile, dict) and tile.get("tile_id") == tile_id:
            return tile
    return None


# ---------------------------------------------------------------------------
# Record text. Every caveat is inside `text` on purpose: it is the only part of
# the record that reaches Gemma's summary, so honesty here is structural.
# ---------------------------------------------------------------------------

def _finding_sentence(water_like: bool, incident_type: str) -> str:
    if water_like:
        return (
            "The surface over this corridor now classifies as water, which is "
            "consistent with flood inundation."
        )
    return (
        "The surface classifies as expected for this corridor, which does not "
        f"corroborate the reported {incident_type}."
    )


def _trigger_sentence(trigger_reason: str, incident_type: str) -> str:
    if trigger_reason == "anticipatory":
        return (
            "No field report claims this corridor is affected; this check was "
            "prompted by a weather advisory and is precautionary."
        )
    if trigger_reason == "operator_request":
        return "A named operator requested this check."
    return (
        "This check was prompted by a field report claiming a "
        f"{incident_type} on this corridor."
    )


def _tier_sentence(tier: str, model_id: str, device: str) -> str:
    if tier == "live":
        return f"Classified live by {model_id} on {device}."
    return (
        f"This result was precomputed by {model_id}; the live classifier was "
        "not reachable."
    )


def _classified_text(
    *,
    corridor_id: str,
    incident_type: str,
    trigger_reason: str,
    sensor: str,
    acquired_at: str,
    freshness: int,
    label: str,
    confidence: float,
    reference_label: str,
    water_like: bool,
    tier: str,
    model_id: str,
    device: str,
) -> str:
    return (
        f"Automated land-cover classification over corridor {corridor_id} "
        f"({sensor}, acquired {acquired_at}, {freshness} minutes before this "
        f'analysis) returned "{label}" at {confidence:.0%} confidence against a '
        f'reference of "{reference_label}". '
        f"{_finding_sentence(water_like, incident_type)} "
        f"{_trigger_sentence(trigger_reason, incident_type)} "
        "This is a surface observation only: it does not measure water depth, "
        "does not see under cloud or tree canopy, and does not establish that "
        f"the corridor is impassable or that a {incident_type} has blocked it. "
        f"{_tier_sentence(tier, model_id, device)}"
    )


def _unavailable_text(corridor_id: str, incident_type: str) -> str:
    return (
        f"An overhead-imagery check of corridor {corridor_id} was requested but "
        "could not be completed: the classifier was unreachable and no "
        "precomputed result exists. No corroboration was obtained for the "
        f"reported {incident_type}. Treat this as absence of information, not "
        "as evidence of absence."
    )


_ID_SAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:/-"


def _slug(value: str) -> str:
    """Force a fragment into `EvidenceRecord.evidence_id`'s character class."""
    cleaned = "".join(char if char in _ID_SAFE else "-" for char in str(value))
    return cleaned.strip("-") or "unknown"


def _evidence_id(identifier: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"sat-{_slug(identifier)}-{stamp}"[:120]


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

def _call_sidecar(corridor_id: str, incident_type: str) -> Dict[str, Any]:
    """One POST, one attempt, hard 3 s ceiling. Raises on any failure."""
    with httpx.Client(timeout=IMAGERY_TIMEOUT_SECONDS) as client:
        response = client.post(
            f"{sidecar_url()}/classify",
            json={"corridor_id": corridor_id, "incident_type": incident_type},
        )
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError("Sidecar returned a non-object body")
    if not body.get("tile_id") or not body.get("label"):
        raise ValueError("Sidecar response is missing tile_id or label")
    return body


def _build_classified(
    *,
    tier: str,
    corridor_id: str,
    incident_type: str,
    trigger_reason: str,
    tile_id: str,
    label: str,
    confidence: float,
    water_like: bool,
    reference_label: str,
    model_id: str,
    device: str,
    latency_ms: Optional[float],
    sensor: str,
    acquired_at: str,
    tile_relative_path: Optional[str],
) -> Tuple[EvidenceRecord, Dict[str, Any]]:
    freshness = _freshness_minutes(acquired_at)
    record = EvidenceRecord(
        evidence_id=_evidence_id(tile_id),
        source_category=IMAGERY_SOURCE_CATEGORY,
        source_name="Overhead imagery land-cover classification",
        source_identifier=f"eurosat://{tile_id}",
        retrieved_at=utc_now_iso(),
        freshness_minutes=freshness,
        reliability=min(RELIABILITY_CLASSIFIED, MAX_RELIABILITY),
        text=_classified_text(
            corridor_id=corridor_id,
            incident_type=incident_type,
            trigger_reason=trigger_reason,
            sensor=sensor,
            acquired_at=acquired_at,
            freshness=freshness,
            label=label,
            confidence=confidence,
            reference_label=reference_label,
            water_like=water_like,
            tier=tier,
            model_id=model_id,
            device=device,
        ),
        provider=(
            "local_model_inference" if tier == "live" else "bundled_imagery_fixture"
        ),
        cache_status="live" if tier == "live" else "precomputed",
        simulated=True,
        operator_context=trigger_reason,
        gap_target=None,
        reported_latitude=None,
        reported_longitude=None,
    )
    telemetry = {
        "tier": tier,
        "corridor_id": corridor_id,
        "incident_type": incident_type,
        "trigger_reason": trigger_reason,
        "tile_id": tile_id,
        "label": label,
        "confidence": confidence,
        "water_like": water_like,
        "model_id": model_id,
        "device": device,
        "latency_ms": latency_ms,
        "tile_relative_path": tile_relative_path,
    }
    return record, telemetry


def _build_unavailable(
    corridor_id: str,
    incident_type: str,
    trigger_reason: str,
) -> Tuple[EvidenceRecord, Dict[str, Any]]:
    record = EvidenceRecord(
        evidence_id=_evidence_id(corridor_id),
        source_category=IMAGERY_SOURCE_CATEGORY,
        source_name="Overhead imagery land-cover classification",
        source_identifier=f"eurosat://unavailable/{corridor_id}",
        retrieved_at=utc_now_iso(),
        freshness_minutes=0,
        reliability=min(RELIABILITY_UNAVAILABLE, MAX_RELIABILITY),
        text=_unavailable_text(corridor_id, incident_type),
        provider="imagery_check_unavailable",
        cache_status="unavailable",
        error_status="imagery_classifier_unreachable_and_no_precomputed_result",
        simulated=True,
        operator_context=trigger_reason,
        gap_target=None,
        reported_latitude=None,
        reported_longitude=None,
    )
    telemetry = {
        "tier": "unavailable",
        "corridor_id": corridor_id,
        "incident_type": incident_type,
        "trigger_reason": trigger_reason,
        "tile_id": None,
        "label": None,
        "confidence": None,
        "water_like": None,
        "model_id": None,
        "device": None,
        "latency_ms": None,
        "tile_relative_path": None,
    }
    return record, telemetry


def _try_live(
    corridor_id: str,
    incident_type: str,
    trigger_reason: str,
) -> Optional[Tuple[EvidenceRecord, Dict[str, Any]]]:
    try:
        body = _call_sidecar(corridor_id, incident_type)
        return _build_classified(
            tier="live",
            corridor_id=corridor_id,
            incident_type=incident_type,
            trigger_reason=trigger_reason,
            tile_id=str(body["tile_id"]),
            label=str(body["label"]),
            confidence=float(body.get("confidence") or 0.0),
            water_like=bool(body.get("water_like")),
            reference_label=str(body.get("reference_label") or "unknown"),
            model_id=str(body.get("model_id") or "unknown model"),
            device=str(body.get("device") or "unknown device"),
            latency_ms=(
                float(body["latency_ms"])
                if isinstance(body.get("latency_ms"), (int, float))
                else None
            ),
            sensor=str(body.get("sensor") or "Sentinel-2 L2A (EuroSAT RGB patch)"),
            acquired_at=str(body.get("acquired_at") or utc_now_iso()),
            tile_relative_path=body.get("tile_relative_path"),
        )
    except Exception:  # noqa: BLE001 - every sidecar failure is simply tier 2.
        return None


def _try_precomputed(
    corridor_id: str,
    incident_type: str,
    trigger_reason: str,
) -> Optional[Tuple[EvidenceRecord, Dict[str, Any]]]:
    try:
        tile = manifest_entry(corridor_id)
        if not tile:
            return None
        precomputed = tile.get("precomputed")
        if not isinstance(precomputed, dict) or not precomputed.get("label"):
            return None
        tile_id = str(tile.get("tile_id") or corridor_id)
        file_name = tile.get("file")
        return _build_classified(
            tier="precomputed",
            corridor_id=corridor_id,
            incident_type=incident_type,
            trigger_reason=trigger_reason,
            tile_id=tile_id,
            label=str(precomputed["label"]),
            confidence=float(precomputed.get("confidence") or 0.0),
            water_like=bool(precomputed.get("water_like")),
            reference_label=str(tile.get("reference_label") or "unknown"),
            model_id=str(precomputed.get("model_id") or "unknown model"),
            device=str(precomputed.get("device") or "unknown device"),
            latency_ms=(
                float(precomputed["latency_ms"])
                if isinstance(precomputed.get("latency_ms"), (int, float))
                else None
            ),
            sensor=str(tile.get("sensor") or "Sentinel-2 L2A (EuroSAT RGB patch)"),
            acquired_at=str(tile.get("acquired_at") or utc_now_iso()),
            tile_relative_path=(
                f"imagery/{file_name}" if file_name else None
            ),
        )
    except Exception:  # noqa: BLE001 - a bad manifest entry is simply tier 3.
        return None


def verify_corridor(
    corridor_id: str,
    incident_type: str,
    trigger_reason: str,
) -> Tuple[EvidenceRecord, Dict[str, Any]]:
    """Classify the surface over one corridor. Never raises.

    Returns `(record, telemetry)` where telemetry carries the tier and the
    classifier detail the interface renders next to the tile.
    """
    corridor_id = str(corridor_id or "unknown_corridor")
    incident_type = (
        incident_type if incident_type in INCIDENT_TYPES else "flood"
    )
    trigger_reason = (
        trigger_reason if trigger_reason in TRIGGER_REASONS else "corroboration"
    )

    try:
        live = _try_live(corridor_id, incident_type, trigger_reason)
        if live is not None:
            return live

        precomputed = _try_precomputed(corridor_id, incident_type, trigger_reason)
        if precomputed is not None:
            return precomputed

        return _build_unavailable(corridor_id, incident_type, trigger_reason)
    except Exception:  # noqa: BLE001 - the contract is absolute: never raise.
        return _build_unavailable(corridor_id, incident_type, trigger_reason)


def sidecar_health() -> Dict[str, Any]:
    """Best-effort sidecar health probe for the status endpoint. Never raises."""
    try:
        with httpx.Client(timeout=IMAGERY_TIMEOUT_SECONDS) as client:
            response = client.get(f"{sidecar_url()}/health")
        response.raise_for_status()
        body = response.json()
        return {"reachable": True, "health": body if isinstance(body, dict) else None}
    except Exception as exc:  # noqa: BLE001
        return {"reachable": False, "health": None, "error": type(exc).__name__}
