# Frozen contracts — imagery verification

**Do not change anything in this file without telling the other agents.** Three
workstreams build against it in parallel. Everything here is authoritative.

---

## C1 · Sidecar HTTP API — `http://127.0.0.1:8011`

### `GET /health`
```json
{"status": "ok", "model_id": "<hf repo id>", "device": "cuda|cpu", "warm": true}
```

### `POST /classify`
Request:
```json
{"corridor_id": "east_west_bharatpur_nepalgunj", "incident_type": "flood"}
```
Response `200`:
```json
{
  "corridor_id": "east_west_bharatpur_nepalgunj",
  "tile_id": "nepal-corridor-07",
  "label": "River",
  "confidence": 0.87,
  "water_like": true,
  "reference_label": "Highway",
  "model_id": "nielsr/vit-finetuned-eurosat-kornia",
  "device": "cuda",
  "latency_ms": 41.2,
  "tile_relative_path": "imagery/nepal-corridor-07.jpg",
  "acquired_at": "2026-07-28T05:41:00+00:00",
  "sensor": "Sentinel-2 L2A (EuroSAT RGB patch)"
}
```
`404` if `corridor_id` has no bound tile. Any other failure → non-2xx; the
backend treats *everything* non-2xx as "sidecar unavailable".

`water_like` is `label in {"River", "SeaLake"}`.

---

## C2 · Tile manifest — `backend/data/imagery/manifest.json`

```json
{
  "notice": "Real Sentinel-2 RGB patches from the EuroSAT benchmark (Helber et al.), bound to Nepali corridors for demonstration. These tiles are NOT imagery of the named corridor.",
  "dataset": "EuroSAT RGB",
  "dataset_url": "https://github.com/phelber/EuroSAT",
  "tiles": [
    {
      "tile_id": "nepal-corridor-07",
      "corridor_id": "east_west_bharatpur_nepalgunj",
      "file": "nepal-corridor-07.jpg",
      "reference_label": "Highway",
      "acquired_at": "2026-07-28T05:41:00+00:00",
      "sensor": "Sentinel-2 L2A (EuroSAT RGB patch)",
      "precomputed": {
        "label": "River", "confidence": 0.87, "water_like": true,
        "model_id": "nielsr/vit-finetuned-eurosat-kornia", "device": "cpu",
        "latency_ms": 63.0
      }
    }
  ]
}
```

`precomputed` is **tier 2**: a real model result, cached. It must be produced by
an actual run, never hand-written.

**Corridor ids must exist in `backend/data/terrain_graph.json`.** Bind at least
these, and note the honest caveat above — the tiles are real Sentinel-2 imagery
from a published benchmark, not imagery of Nepal.

---

## C3 · `EvidenceRecord` produced by the tool

Must validate against `backend/models/gemma.py::EvidenceRecord` unchanged.

```python
{
  "evidence_id": "sat-<tile_id>-<yyyymmdd>",        # matches ^[A-Za-z0-9._:/-]+$
  "source_category": "overhead_imagery_analysis",    # EXACT — the closure guard keys on it
  "source_name": "Overhead imagery land-cover classification",
  "source_identifier": "eurosat://<tile_id>",
  "freshness_minutes": <int ≥ 0, from acquired_at>,
  "reliability": 0.55,                               # tier1/2; 0.20 for tier3. NEVER > 0.6
  "text": "<see below>",
  "provider": "local_model_inference"                # tier1
             | "bundled_imagery_fixture"             # tier2
             | "imagery_check_unavailable",          # tier3
  "cache_status": "live" | "precomputed" | "unavailable",
  "simulated": True,
  "operator_context": "<trigger_reason>",
  "gap_target": None,
  "reported_latitude": <float|None>, "reported_longitude": <float|None>
}
```

### `text` template — tiers 1 and 2

The literal word `flood` or `landslide` **must** appear, or
`_validate_incident_type_grounding`'s substring check breaks.

```
Automated land-cover classification over corridor {corridor_id} ({sensor},
acquired {acquired_at}, {freshness} minutes before this analysis) returned
"{label}" at {confidence:.0%} confidence against a reference of
"{reference_label}". {finding_sentence} {trigger_sentence} This is a surface
observation only: it does not measure water depth, does not see under cloud or
tree canopy, and does not establish that the corridor is impassable or that a
{incident_type} has blocked it. {tier_sentence}
```

- `finding_sentence` — water-like: `"The surface over this corridor now
  classifies as water, which is consistent with flood inundation."` Otherwise:
  `"The surface classifies as expected for this corridor, which does not
  corroborate the reported {incident_type}."`
- `trigger_sentence` — by `trigger_reason`:
  - `corroboration` → `"This check was prompted by a field report claiming a {incident_type} on this corridor."`
  - `anticipatory` → `"No field report claims this corridor is affected; this check was prompted by a weather advisory and is precautionary."`
  - `operator_request` → `"A named operator requested this check."`
- `tier_sentence` — tier 1: `"Classified live by {model_id} on {device}."`
  tier 2: `"This result was precomputed by {model_id}; the live classifier was
  not reachable."`

### `text` — tier 3
```
An overhead-imagery check of corridor {corridor_id} was requested but could not
be completed: the classifier was unreachable and no precomputed result exists.
No corroboration was obtained for the reported {incident_type}. Treat this as
absence of information, not as evidence of absence.
```

---

## C4 · Python entry point (backend agent owns; frontend does not call it)

`backend/services/imagery_verifier.py`
```python
SATELLITE_TOOL_ENABLED: bool          # env SATELLITE_TOOL_ENABLED, default False
IMAGERY_SIDECAR_URL: str              # env, default http://127.0.0.1:8011
IMAGERY_TIMEOUT_SECONDS: float = 3.0  # one attempt, no retry

def verify_corridor(corridor_id: str, incident_type: str,
                    trigger_reason: str) -> tuple[EvidenceRecord, dict]:
    """Never raises. Returns (record, telemetry).
    telemetry = {"tier": "live"|"precomputed"|"unavailable",
                 "label", "confidence", "water_like", "model_id", "device",
                 "latency_ms", "tile_relative_path", "tile_id"}"""
```

---

## C5 · REST endpoints (backend agent)

| Method | Path | Body | Purpose |
|---|---|---|---|
| `GET` | `/api/imagery/status` | — | sidecar health + flag + tier availability |
| `POST` | `/api/imagery/verify` | `{corridor_id, incident_type, evidence_id?}` | **B2 direct** — no Gemma. Appends record to latest analysis, returns `{record, telemetry, analysis}` |
| `POST` | `/api/optimization/orchestrate` | `+ operator_directive: {corridor_id, incident_type, evidence_id}` | **B1 forced** — directive into the turn |
| `GET` | `/api/imagery/tile/{tile_id}` | — | serves the JPEG, `image/jpeg` |

---

## C6 · `ToolCallRecord` additions — `backend/models/orchestration.py`

```python
initiated_by: str = "model"      # "model" | "operator"
model_complied: Optional[bool] = None   # only set on a forced turn
```
Both optional with defaults — existing records and tests must keep working.

---

## C7 · Frontend contract (frontend agent)

Everything needed is already on the run/analysis payloads:

- `run.orchestration.tool_calls[]` — a call with `name === 'verify_report_with_imagery'`;
  read `initiated_by`, `raw_arguments.trigger_reason`.
- `analysis.evidence[]` — a record with `source_category === 'overhead_imagery_analysis'`;
  read `provider` for the tier badge, `text` for the caveat.
- Tile image: `GET /api/imagery/tile/{tile_id}` where `tile_id` is parsed from
  `source_identifier` (`eurosat://<tile_id>`).
- `api.getImageryStatus()`, `api.verifyCorridorImagery(...)` to add to `services/api.js`.

**Tier badge:** `local_model_inference` → "live model" (green) ·
`bundled_imagery_fixture` → "precomputed" (amber) ·
`imagery_check_unavailable` → "unavailable" (red).

---

## C8 · Non-negotiables

1. **No new dependency in `backend/requirements.txt`.** Torch lives only in the sidecar venv.
2. **`SATELLITE_TOOL_ENABLED` defaults to `False`.** With it off the system must be byte-identical to today, and all 809 existing backend tests must pass.
3. **`verify_corridor` never raises.** Every failure is a tier.
4. **The closure guard** — a corridor whose only supporting evidence is
   `source_category == "overhead_imagery_analysis"` must be rejected from
   `blocked_edge_ids`. This is the headline test.
5. **`reliability` never exceeds 0.6.**
6. Never claim the tiles are imagery of Nepal. They are EuroSAT patches bound to
   corridors for demonstration, and every surface says so.
