# Architecture

## Operational Architecture

```text
React/Vite dashboard
  +-- MapLibre terrain + Three.js animated fleet
  +-- evidence/decision trace + HITL controls
  | REST + versioned WebSocket events
FastAPI transport
  +-- GemmaAnalysisRequest / GemmaAnalysisRecord
  |   +-- Gemini API hosted provider (default)
  |   +-- deterministic validated fallback
  | OptimizationRunRequest / OptimizationRunRecord
OptimizationService
  | evidence analysis + deterministic fixture loading + in-memory run registry
StateManager
  +-- UrgencyCalculator
  +-- VRPSolver (greedy heuristic)
  +-- NashSolver (legacy name; capped proportional allocation)
  +-- SocialWelfareOptimizer (continuous Nash bargaining comparison)
  +-- KKTVerifier (diagnostic)
```

Every run finishes in `awaiting_approval`. Approval and rejection are explicit
state transitions exposed by the optimization API.

## Repository Layout

- `backend/algorithms`: existing mathematical pipeline
- `backend/services`: application orchestration over domain algorithms
- `backend/models`: typed domain and optimization contracts
- `backend/api`: REST, HITL, and WebSocket transports
- `backend/data`: deterministic Nepal-wide incident, fleet, terrain, and evidence fixtures
- `backend/demo`: timeline and re-optimization simulation components
- `backend/hitl`: legacy report approval queue
- `backend/rag`: current rule-based report analyzer
- `backend/p2p`: gossip, topology, and serial bridge prototypes
- `frontend/src/components`: mission-control views
- `frontend/src/services`: REST client
- `frontend/src/hooks`: WebSocket state

## Next Boundaries

Scenario, incident, live evidence retrieval, inventory, graph routing, audit,
and persistence services remain future work. They should use the same typed
contracts and persist event envelopes so frontend state can be rebuilt after
restart.
