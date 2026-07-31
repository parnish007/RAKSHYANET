# API

## Optimization

- `POST /api/optimization/run`
- `GET /api/optimization/runs`
- `GET /api/optimization/runs/{run_id}`
- `POST /api/optimization/runs/{run_id}/approve`
- `POST /api/optimization/runs/{run_id}/reject`

Compatibility aliases:

- `POST /api/optimize`
- `POST /api/optimize/start`
- `GET /api/optimize/history`
- `GET /api/vrp/solution`
- `GET /api/nash/equilibrium`
- `GET /api/allocation/proportional`
- `GET /api/allocation/social-welfare`
- `GET /api/allocation/compare`
- `GET /api/kkt/verify`

## Data and Health

- `GET /api/villages`
- `GET /api/vehicles`
- `GET /health`
- `GET /api/health`
- `WS /ws`

## Gemma Analysis

- `GET /api/gemma/status`
- `POST /api/gemma/analyze`
- `POST /api/gemma/analyze-submitted`
- `GET /api/gemma/analyses`
- `GET /api/gemma/analyses/latest`
- `GET /api/gemma/analyses/{analysis_id}`
- `POST /api/gemma/analyses/{analysis_id}/questions/{question_id}`

The status response reports requested provider, active provider, hosted-model
configuration, fallback state, the last provider error, and per-key pool health.
Analysis responses include provenance-tagged evidence, schema-validated extracted
values, separate model/system confidence, and inspectable decision-trace steps.

Fetch by id rather than `/latest` when displaying an analysis beside a run. The
backend retains every analysis and any scenario activation mints a new one, so
`/latest` and a given run's `analysis_id` diverge routinely — pairing a run with
the newest analysis instead of its own is how an interface ends up asking for a
field the displayed plan already has a value for. Route order matters: `/latest`
is declared before the `{analysis_id}` path so the literal wins.

## Overhead Imagery

Served only when `SATELLITE_TOOL_ENABLED=true`; otherwise these paths 404 and the
function is never declared to Gemma. Clients should treat a rejected request as
"the capability is absent" and render nothing rather than an error.

- `GET /api/imagery/status` — enablement, sidecar reachability, tier, tile count,
  dataset provenance and the authority boundary
- `GET /api/imagery/corridor/{corridor_id}`
- `GET /api/imagery/tile/{tile_id}` — the image itself, not JSON
- `POST /api/imagery/verify`

An imagery record corroborates. It can never, alone, place a corridor in
`blocked_edge_ids`; that is enforced in validation and covered by a test.

## Event Envelope

Every new event includes:

```json
{
  "event_id": "evt_...",
  "scenario_id": "nepal-national-demo",
  "timestamp": "ISO-8601",
  "type": "optimization_completed",
  "event_type": "optimization_completed",
  "schema_version": "1.0",
  "correlation_id": "corr_...",
  "payload": {}
}
```

Current real event types:

- `evidence_retrieved`
- `gemma_analysis_started`
- `gemma_analysis_completed`
- `optimization_started`
- `urgency_updated`
- `route_generated`
- `proportional_iteration`
- `allocation_generated`
- `validation_completed`
- `optimization_completed`
- `optimization_failed`
- `hitl_review_required`
- `hitl_approved`
- `hitl_rejected`
