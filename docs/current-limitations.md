# Current Limitations

- Optimization state and event history are in memory.
- Scenario CRUD and durable audit persistence are not implemented.
- Routing is a greedy assignment and nearest-neighbour tour heuristic, not an
  exact OR-Tools/MILP solution. This is a deliberate choice, not an oversight:
  the underlying capacitated vehicle-routing problem is **NP-hard** (it contains
  TSP), so no fast algorithm is known that returns a provably optimal plan, and
  an exact solve is not the right engineering trade for a real-time dispatch
  tool. What the plan does guarantee is feasibility against the stated
  constraints, determinism, traceability of every number, and reproducibility —
  not optimality. Ground legs do follow the bundled road graph using
  capability-filtered Dijkstra with active closures removed; aircraft use direct
  geodesic corridors.
- The optimality gap of the greedy plan is **unmeasured**. The bundled instance
  is small enough to solve exactly as a MILP/CP-SAT model under the same
  constraints and objective; doing so and reporting the gap is roadmap, not
  something claimed today. No approximation ratio is claimed either.
- The at-most-two-stops-per-asset cap is a heuristic restriction of the search
  space, not a property of the problem or of the fleet.
- Dispatch allocation still uses the legacy proportional response. A continuous
  weighted Nash-social-welfare candidate is now computed for comparison, but it
  does not yet include discrete vehicle and route constraints.
- KKT checks are diagnostic rather than an independent proof. They apply to the
  continuous allocation only — a separate, convex problem — and three of their
  four conditions hold by construction for any primal-feasible allocation. They
  say nothing about the discrete routing decisions, and the API ships
  `independently_proves_optimality: false`.
- Enhanced configurable urgency scoring is not implemented.
- Terrain uses MapLibre raster DEM and degrades to a flat operational view when
  elevation tiles are unavailable. Cached/offline Nepal terrain tiles are not packaged.
- Three.js helicopter and truck motion follows solver legs and stop ETAs, but
  the simulation clock is not durable and no real fleet telemetry is connected.
- Bundled evidence and the five replay timelines are simulated hackathon data;
  authoritative live government/news/weather retrieval adapters are not yet
  connected.
- Hosted Gemma requires a user-supplied API key; deterministic fallback remains available.
- The backend compares capped proportional allocation against weighted Nash
  social welfare, but broader routing baselines and benchmark datasets are not
  implemented.
- Optimization-plan modification and locked assignments are not implemented.
- Legacy P2P visualization code is not part of the primary mission-control
  workflow and still contains cosmetic activity.
- Browser end-to-end coverage exercises all four premium workspaces and core
  review/closure flows; broader component-level frontend unit coverage is still
  needed.
- The production frontend dependency tree reports no known advisories. The
  legacy ESLint 8 development toolchain still reports audit findings and needs
  a separately tested major-version migration.
