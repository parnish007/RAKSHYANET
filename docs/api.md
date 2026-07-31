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
- `GET /api/gemma/analyses`
- `GET /api/gemma/analyses/latest`

The status response reports requested provider, active provider, hosted-model
configuration, fallback state, and the last provider error. Analysis responses
include provenance-tagged evidence, schema-validated extracted values, separate
model/system confidence, and inspectable decision-trace steps.

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
