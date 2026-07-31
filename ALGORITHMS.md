# RakshyaNet Algorithms and Execution Contracts

This is the authoritative algorithm-level continuation guide. Read
`docs/CONTEXT.md` first and `MATH.md` for equations and interpretation.

## 1. End-to-end lifecycle

The operational path is:

```text
submitted evidence
  -> strict evidence validation
  -> bounded Gemma analysis or deterministic safe fallback
  -> evidence-grounding and output-schema validation
  -> deterministic confidence and village signal handoff
  -> urgency ranking
  -> capability-constrained fleet assignment
  -> multimodal route construction
  -> proportional-allocation diagnostics
  -> continuous fairness comparison
  -> scoped KKT checks
  -> immutable optimization snapshot
  -> human approve/reject gate
```

New evidence or a road-network change does not mutate an approved proposal.
It creates a new analysis and a new optimization run. A closure run references
its predecessor through `parent_run_id`.

High-level pseudocode:

```text
function coordinate(evidence, elapsed_time, blocked_edges):
    analysis = analyze_with_gemma_boundary(evidence)
    villages = load_fresh_fixture_state()
    apply_supported_gemma_signal(analysis, villages)

    urgency = rank_villages(villages, elapsed_time)
    vrp = solve_routes(villages, fleet, urgency, blocked_edges)
    proportional = capped_proportional(villages, depot_stock, vrp)
    fair_candidate = weighted_social_welfare(
        villages, depot_stock, urgency, warm_start=proportional
    )
    diagnostics = verify_scoped_kkt(proportional)

    run = immutable_snapshot(
        analysis_id=analysis.id,
        urgency=urgency,
        vrp=vrp,
        proportional=proportional,
        fair_candidate=fair_candidate,
        diagnostics=diagnostics
    )
    run.status = awaiting_approval if run.completed else failed
    return run
```

## 2. Strict evidence ingestion

Implementation:

- `backend/models/gemma.py`
- `backend/services/gemma_service.py`
- `backend/api/gemma_routes.py`

Each evidence record carries:

- stable evidence ID;
- source category, name, and source identifier;
- retrieval time and freshness;
- reliability in `[0,1]`;
- report text;
- provider/cache/simulation provenance;
- optional operator context and the information gap it targets;
- optional paired latitude/longitude.

Unknown extra fields are rejected. Latitude and longitude must appear together.
Evidence IDs must be unique inside an analysis. Empty evidence is rejected.
Known prompt-injection patterns inside evidence are rejected before model use.

Pseudocode:

```text
function validate_evidence(records):
    require length(records) > 0
    require every evidence_id is unique
    for record in records:
        validate strict schema
        require both coordinates or neither coordinate
        reject prompt-injection pattern
```

Mocked data is never disguised as live data. Replay records set:

```text
provider = "timeline_scenario_fixture"
cache_status = "fixture"
simulated = true
retrieved_at = "mock://<scenario>/t+<seconds>"
```

## 3. Bounded Gemma analysis

Gemma is the evidence interpretation layer, not the decision authority.

The expected structured output contains:

- incident type;
- normalized severity range;
- affected-population range;
- normalized medical urgency;
- normalized accessibility risk;
- contradictions;
- missing information;
- follow-up questions;
- whether more evidence and human review are required;
- requested tools;
- concise evidence-grounded summary.

Every supported value must cite evidence IDs. An unknown value must have:

```text
value = null
confidence = 0
evidence_ids = []
```

A range must supply minimum, expected, and maximum together and satisfy:

```text
minimum <= expected <= maximum
```

Severity, medical urgency, and accessibility risk are constrained to `[0,1]`.
Affected-population values must be non-negative whole numbers.

Output post-validation enforces:

```text
if a field is unknown, contradictions exist, gaps exist, or tools are requested:
    needs_more_evidence must be true

if a field is unknown or contradictions exist:
    needs_human_review must be true

if missing_information exists and questions are empty:
    generate one concise verification question per missing item
```

The service also validates that claims are connected to cited evidence. The
model is not permitted to invent a population estimate merely because another
normalized number appears in cited text.

### Provider behavior

The hosted provider is used only when configured and healthy. Hosted output is
still parsed through the same strict contract and grounding checks. Invalid,
unsafe, unavailable, or unconfigured hosted execution falls back to the
deterministic safety implementation. The analysis record tells the UI which
provider/model was actually used; the UI must not label fallback output as a
hosted Gemma response.

Replays force the hosted key off so tests are deterministic, offline, and
credential-independent.

### Reasoning visibility

The product exposes bounded decision trace steps, cited inputs, output
summaries, warnings, and durations. It does not expose hidden chain-of-thought.
The human interface should show evidence-linked rationale and deterministic
solver substitutions instead.

## 4. Evidence-gap ownership

Gemma follow-up questions are actionable records, not decorative text.

An operator may mark each question:

- `assigned`: an identified person or team owns collection;
- `unavailable`: the evidence cannot currently be obtained.

Every disposition requires an owner and reason. It is recorded against the
immutable `analysis_id` and question index (`question-0`, `question-1`, ...).
This gives the missing-evidence UI a reviewable state without fabricating an
answer.

The evidence intake paired with a question uses deterministic UI routing, not a
second model call. Keywords in the evidence target select one of the location,
medical, road-access, population, resource, severity, contradiction, incident,
or general prompt sets. The exact Gemma question remains the required primary
answer. Supporting answers are optional and are serialized as explicit
question/answer facts; empty fields are omitted and therefore remain unknown.

The disposition action is disabled only while its request is in flight. Local
validation runs when the operator submits, explains the missing owner or
collection plan inline, and moves focus to the invalid field.

Algorithm:

```text
function disposition(analysis_id, question_id, status, owner, reason):
    require analysis exists
    require question exists on that analysis
    require status in {assigned, unavailable}
    append versioned disposition with timestamp
```

## 5. Gemma-to-optimizer handoff

Implementation: `OptimizationService._apply_gemma_signal`.

Algorithm:

```text
scores = [
    supported severity.expected,
    supported medical_urgency.value,
    supported accessibility_risk.value
]
signal = max(scores, default=0)
boost = round(signal * system_confidence, 4)

evidence_text = lowercase(concatenate(all evidence text))
for village in villages:
    if lowercase(village.id) or lowercase(village.name) occurs in evidence_text:
        village.external_urgency_boost = boost
```

The result records:

- all candidate inputs;
- unknown fields ignored;
- the selected maximum;
- system confidence;
- resulting boost;
- matched villages;
- source evidence IDs.

This handoff is intentionally small. Gemma cannot pass a route, allocation, or
approval directive into deterministic execution.

## 6. Urgency ranking

Implementation: `backend/algorithms/urgency_calculator.py`.

For each village:

```text
time_factor = 1 + 0.5 * (exp(0.3 * elapsed_hours) - 1)
base = 0

for each resource need:
    unmet = max(0, current_need - existing_allocated)
    ratio = clamp(unmet / current_need, 0, 1) if current_need > 0 else 0
    contribution = ratio * resource_urgency_multiplier * time_factor
    base += contribution

critical_penalty = 10 if any need is below min_need else 0
total = base + critical_penalty + external_gemma_boost

sort villages by total descending
assign one-based ranks
```

The output retains every component so the number can be audited in the Math
Engine. See `MATH.md` for equations and score limitations.

Complexity: `O(V * R)` time and `O(V * R)` explanation data.

## 7. Greedy multimodal resource assignment

Implementation: `backend/algorithms/vrp_solver.py`.

### 7.1 Complexity class, and why this is a heuristic by design

The dispatch problem solved here is a **capacitated vehicle-routing problem** over
a heterogeneous fleet with capability, payload, endurance and time constraints.
That problem is **NP-hard**: it contains the travelling salesman problem as a
special case, and the assignment layer above it — partitioning locations and
cargo across non-interchangeable assets — is harder still. No polynomial-time
algorithm is known that returns a provably optimal plan, and none is expected.

An exact solve is therefore **not the correct engineering choice** for a
real-time dispatch tool at this instance size and on this decision timescale.
The implemented method is a **greedy urgency-ordered assignment followed by a
nearest-neighbour tour**, over a graph already filtered for vehicle capability
and active closures. This is stated as a heuristic everywhere it appears — in
this document, in `MATH.md`, in the API payload, and in the interface.

**Choosing a heuristic here is a design decision, not a shortfall.** The claim
the system makes is not *"this is the best possible plan"*. It is:

- the plan is **feasible** against every stated constraint;
- it is **executable** by the fleet that exists, on the corridors that are open;
- it is **terrain- and closure-aware**;
- it is produced **fast enough to act on**;
- and **every number in it is traceable** to the evidence and the edges that
  produced it.

Two points of precision that a reviewer will check:

1. **Where the hardness actually sits.** With at most two village stops per
   asset (§9), ordering the stops inside a single route is trivial — there are
   two orderings. The combinatorial difficulty is in the layer above: which
   asset serves which locations with which cargo. The two-stop cap is itself a
   heuristic restriction of the search space, not a property of the problem.
2. **What is *not* proved.** The greedy result carries no optimality
   certificate, no approximation ratio, and no measured optimality gap. The
   continuous allocation (§12–§14) is a separate, convex problem; its KKT
   diagnostics say nothing about these discrete decisions.

**Roadmap, stated as roadmap.** The bundled instance is small enough to
formulate exactly. Solving it with a MILP/CP-SAT model under the same
constraints and the same coverage objective would yield an optimality gap that
could be reported honestly. That measurement does not exist yet, so the current
statement is "unmeasured", not "small".

### 7.2 Assignment pass

The assignment pass tracks:

- remaining depot stock per resource;
- remaining mass capacity per asset;
- village-resource quantities assigned;
- village stops already assigned to each asset.

Villages are processed in urgency order. For each unmet resource, candidate
assets must:

- have remaining mass capacity;
- support the village terrain/accessibility;
- have a viable road path, unless the asset is aircraft;
- have fewer than two already-assigned village stops unless returning to an
  existing stop;
- have a viable projected depot-to-stops-to-depot tour within fuel endurance.

`preferred_resources` is a soft specialty priority, not a hard cargo
restriction. Critical resources are processed before non-critical resources,
then by configured urgency multiplier. Every viable asset receives a
deterministic selection score:

```text
eta_score = 1 / (1 + direct_one_way_minutes / 120)
payload_fit = min(1, remaining_payload_kg / requested_payload_kg)
time_pressure = bounded(
    base
    + critical_shortage
    + incident_impact
    + village_urgency
    + time_sensitive_resource
    + bounded_gemma_signal
)
mode_bonus = air favors time pressure; road favors payload pressure

selection_score =
    time_pressure * eta_score
    + (1 - time_pressure) * payload_fit
    + specialty_bonus
    + same-stop consolidation_bonus
    + mode_bonus
```

Candidate sorting uses descending score, then fewer existing stops and a stable
asset ID tie-break. This makes a helicopter competitive for urgent medical
payloads because it arrives sooner, while a truck remains competitive for
large bulk payloads because of capacity. See `MATH.md` for exact coefficients.

The algorithm allocates the minimum of need, stock, and payload-convertible
quantity, then updates all ledgers. For every committed load it records:

- selected asset and air/road mode;
- quantity and converted payload mass;
- direct ETA and projected complete-tour time;
- score and all score components;
- a concise human-readable selection explanation.

Important distinction:

- resource quantities retain their own units;
- payload capacity is enforced after multiplying by `weight_per_unit_kg`.

Approximate complexity before route building is
`O(V * R * A * pathCheck)`. Path checks invoke graph search for ground assets.

## 8. Road and air path algorithms

### Aircraft

Aircraft receive direct Haversine legs between assigned locations. Their
geometry is a start/end geodesic corridor and is independent of road closures.

### Ground assets

Ground paths use Dijkstra over an undirected road adjacency list:

```text
function shortest_road_path(source, target, vehicle, blocked):
    adjacency = {}
    for edge in road_graph:
        if edge.id in blocked:
            continue
        if edge has no road or quality incompatible with vehicle:
            continue
        add edge in both directions

    costs[source] = 0
    priority_queue.push(0, source)

    while queue not empty:
        cost, node = pop_min()
        if node == target:
            break
        for neighbor, edge in adjacency[node]:
            risk = 1 + 0.06 * max(0, terrain_difficulty - 1)
            candidate = cost + edge.distance_km * risk
            relax neighbor if candidate is lower

    reconstruct edge IDs and oriented geometry
    report raw distance sum
```

Complexity per shortest-path call is `O((N + E) log N)` with the current
binary heap, plus `O(E)` adjacency construction.

An isolated unit-test fixture without a graph can use an explicitly marked
direct fallback. Production fixture runs use the bundled graph.

## 9. Multi-stop route construction

An asset may have at most two assigned village stops. Stop order is selected by
a nearest-neighbor heuristic over viable legs. Both the stop cap and the
nearest-neighbour ordering are heuristic restrictions on an NP-hard search
space (§7.1); neither is claimed to produce an optimal tour.

For each route:

```text
current = depot
elapsed = 0

for stop in nearest_neighbor_order:
    leg = viable leg(current, stop)
    require leg exists
    elapsed += travel_minutes(leg.distance, vehicle.speed)
    stop.eta_minutes = elapsed
    append leg and stop
    current = stop

return_leg = viable leg(current, depot)
elapsed += return_leg.travel_minutes

feasible = all legs exist and elapsed <= fuel_hours * 60
```

The returned route includes:

- transport mode;
- ordered village IDs;
- stop-level payload and ETA;
- every leg's geometry and edge IDs;
- road closures avoided;
- total distance and time;
- fuel-limit explanation;
- feasibility and reason.

The solver creates allocation explanations for shortages, including exhausted
stock, incompatible fleet capacity, terrain restrictions, and route
reachability.

## 10. Route interruption and reoptimization

A road closure is an input state change, not a visual-only map toggle.

```text
function reoptimize_for_closure(parent_run, closure_evidence, blocked_edge_ids):
    require every blocked edge exists in the road graph
    new_analysis = analyze(previous evidence + closure evidence)
    child_run = optimize(
        analysis_id=new_analysis.id,
        blocked_edge_ids=blocked_edge_ids,
        parent_run_id=parent_run.id,
        trigger="road_closure",
        disruption_reason=operator_reason
    )
    require every feasible ground route excludes blocked_edge_ids
    return child_run
```

Reopening a road similarly creates a fresh run with an updated blocked-edge
set. Approval does not transfer from parent to child. This prevents an operator
from approving a stale route after network conditions change.

## 11. Route objective and shortage explanation

The route result computes mean coverage over village-resource pairs using
existing allocation plus newly assigned payload. This objective is a
descriptive quality metric for the greedy result; the solver does not search
all route combinations to maximize it globally, and on an NP-hard instance
(§7.1) exhaustive search is not the intended behaviour. The reported objective
is therefore an achieved value, never a bound and never a maximum.

Each village allocation record distinguishes:

- current need;
- existing field allocation;
- newly assigned amount;
- unmet remainder;
- first-arrival ETA;
- assigned assets;
- reason codes and plain-language explanation.

These records should drive the resource-availability and allocation UI. The UI
must not infer resources from moving icons.

## 12. Capped proportional allocation

Implementation: `backend/algorithms/nash_solver.py`.

The historical `Nash*` names are compatibility names only.

```text
for resource in depot_stock:
    weight[village] = current_need[village, resource] * multiplier[resource]
    active = villages with positive weight
    remaining = depot_stock[resource]

    repeat at most village_count + 2:
        shares = proportional split of remaining across active weights
        cap each share at remaining village need
        collect surplus from newly capped villages
        remove newly capped villages from active set
        remaining = surplus
        stop if no new cap or no remaining stock
```

The solver warms this rule with the greedy route allocation, reapplies the
fixed rule, and records raw and normalized allocation changes until the
normalized residual is below `0.01`.

Complexity of cap redistribution is approximately `O(R * V^2)` in the current
small fixture implementation.

## 13. Continuous weighted social-welfare comparison

Implementation: `backend/algorithms/social_welfare_optimizer.py`.

The method builds one continuous variable for every positive
village-resource unmet need. It:

1. bounds each variable by unmet need;
2. adds one aggregate stock constraint per resource;
3. normalizes resource importance within each village;
4. normalizes village urgency weights to positive mean one;
5. maximizes weighted log coverage;
6. warm-starts from the capped proportional candidate;
7. calls SciPy SLSQP;
8. reports objective, coverage, stock use, iterations, status, runtime, and
   maximum constraint violation;
9. compares both candidates under the same continuous objective.

This candidate does not assign vehicles. Never substitute it into the dispatch
map without a separate feasible packing and routing stage.

Numerical complexity depends on decision-variable count, constraints, and SLSQP
iterations. It should be characterized empirically for larger data instead of
being advertised with a false closed-form runtime guarantee.

## 14. Scoped KKT verification

Implementation: `backend/algorithms/kkt_verifier.py`.

The verifier runs four diagnostics against the capped proportional continuous
allocation:

```text
stationarity arithmetic consistency
primal feasibility
dual feasibility of estimated aggregate multipliers
aggregate complementary slackness
```

Multipliers are inferred from the submitted allocation. Therefore the result
is partial consistency evidence, not an independent optimizer or proof.

Three of the four conditions hold **by construction** for any primal-feasible
allocation, because \(\lambda_r\) is *defined* as the stationarity ratio and is
*set* to zero exactly when the resource is slack; only primal feasibility
carries information. See `MATH.md` §7 for the algebra.

Scope, stated twice because it is the most misreadable panel in the product:
the diagnostics apply to the **continuous allocation**, which is a separate and
convex problem. They say nothing about the **discrete routing** decisions of
§7–§11, which are the NP-hard part. The payload ships
`independently_proves_optimality = false` and
`applies_to_discrete_route_decisions = false`.

Complexity is `O(V * R)` for each small number of diagnostic passes.

## 15. Optimization run state machine

Internal compute states:

```text
idle
  -> calculating_urgency
  -> solving_vrp
  -> solving_nash
  -> optimizing_social_welfare
  -> verifying_kkt
  -> complete
```

Any exception produces `error` with the partial result and resource/fleet
snapshots retained where available.

External review states:

```text
running -> awaiting_approval -> approved
                             \-> rejected
running -> failed
```

Only `awaiting_approval` can be reviewed.

Approval preconditions:

```text
request.expected_updated_at == run.updated_at
request.expected_analysis_id == run.analysis_id
run is the latest optimization snapshot
run.route_feasible is true
```

Both approve and reject are immutable one-time decisions. A second decision
against the same run is rejected.

## 16. Frontend data and motion algorithms

The premium frontend consumes the API snapshots; it should not duplicate
backend decision math.

### Progressive disclosure

The primary operational view presents status and exceptions first. Detail
surfaces open specific evidence, math, route, resource, or review information.
The full diagnostics action must open solver diagnostics, not silently switch
to an unrelated decision-trace panel.

### Map hover and selection

Village, route, fleet, and road-edge interactions should expose their own
records:

- village urgency and shortage;
- route mode, stops, ETA, payload, and feasibility;
- fleet capacity and assignment;
- edge state and closure effect.

### Solver-timed movement

Implementation: `frontend/src/components/TerrainMissionMap.jsx`.

```text
for each active route:
    derive ordered legs and stop ETAs
    at simulation minute now:
        find the current leg from ETA boundaries
        interpolate by (now - leg_start) / leg_duration
        calculate heading from nearby points on that leg geometry
```

Ground icons therefore turn along route geometry. Aircraft use their direct
leg. The final return uses total route time. PNG orientation is not used as a
substitute for calculated heading.

## 17. Five active-pipeline timeline fixtures

Location: `backend/demo/scenarios/`.

The five scenarios cover:

| Scenario | Sudden closure | Final HITL result |
|---|---|---|
| Taplejung landslide | `mechi_dharan_taplejung` | approved |
| Pokhara flood | `prithvi_bharatpur_pokhara` | rejected |
| Jumla bridge failure | `karnali_pokhara_jumla` | approved |
| Janakpur flood | `bp_kathmandu_janakpur` | rejected |
| Nepalgunj hospital disruption | `east_west_bharatpur_nepalgunj` | approved |

Every scenario is explicitly `simulated: true` and contains at least:

```text
t0 evidence_report
t1 optimization_requested
t2 road_block_report
t3 evidence_disposition
t4 review_decision
```

The Operations workspace exposes all five fixtures through a visible scenario
selector. An operator chooses either `Initial report` or `After road block`,
reviews the compact five-event timeline, and explicitly selects `Load this
scenario`. The disrupted stage creates a baseline run and then an immutable
closure-aware child run; selection alone never mutates the active plan.

Operator-facing endpoints:

```text
GET  /api/demo/scenarios
POST /api/demo/scenarios/{scenario_id}/activate
```

The strict fixture validator enforces:

- schema version `1.0`;
- chronological timestamps;
- unique step IDs and evidence IDs;
- all five lifecycle event types;
- evidence before optimization;
- baseline optimization before closure;
- review as the final event;
- event-specific required fields.

The replay engine then checks:

- analysis human-review expectation;
- baseline and child run states;
- immutable `analysis_id` handoff;
- parent/child linkage;
- minimum route count;
- route feasibility;
- active blocked edges;
- absence of blocked edges from feasible ground routes;
- evidence-question disposition;
- final approve/reject state.

Replay command:

```powershell
python scripts/replay_scenarios.py
```

Test command:

```powershell
python -m pytest backend/tests/test_scenario_replays.py -q
```

`backend/demo/timeline_simulator.py` is a separate legacy rule-based news
timeline. It must not be presented as the active Gemma-to-optimizer replay.

## 18. Deterministic and non-deterministic boundaries

Deterministic for fixed inputs:

- strict validation;
- system confidence;
- Gemma signal handoff;
- urgency calculation;
- graph filtering and Dijkstra;
- greedy fleet assignment and route order;
- capped proportional allocation;
- KKT arithmetic;
- offline scenario replay.

Potentially non-deterministic or environment-dependent:

- hosted Gemma wording/output before strict normalization;
- network availability and provider latency;
- SLSQP runtime and tiny platform-level floating-point variation;
- randomly generated record IDs and timestamps.

Tests should assert contracts, feasibility, relationships, and tolerances—not
hard-code volatile UUIDs, timestamps, or hosted wording.

## 19. Failure modes and required behavior

| Failure | Required behavior |
|---|---|
| no evidence | reject analysis request |
| duplicate evidence ID | reject analysis request |
| prompt injection in evidence | reject before provider use |
| hosted model unavailable/invalid | use labelled safe fallback |
| unsupported value | preserve `unknown`; ask for evidence |
| unknown blocked edge in replay | fail fixture replay |
| no route or infeasible route | add approval blocker |
| road closes | create child run and recompute ground paths |
| stale review snapshot | reject decision |
| wrong analysis ID | reject decision |
| older run reviewed after new run | reject decision |
| second decision on reviewed run | reject decision |
| optimizer exception | return failed run with error context |

## 20. Test strategy

The test suite should maintain four layers:

1. unit tests for formulas, schemas, graph paths, capacity, convergence, and
   diagnostics;
2. API tests for analysis, optimization, closure, evidence disposition, and
   review invariants;
3. scenario replays for complete chronological product stories;
4. browser tests for every important button, modal, view switch, closure flow,
   evidence form, math panel, and approval state.

Authoritative commands are listed in `docs/CONTEXT.md`.

## 21. Claims the project must not make

Do not claim:

- hidden Gemma chain-of-thought is shown;
- Gemma approves or dispatches resources;
- fixture evidence is live news;
- the greedy route is globally optimal;
- the route heuristic carries an approximation ratio or a measured optimality
  gap;
- `NashEquilibrium` proves a strategic Nash equilibrium;
- KKT diagnostics prove global optimality;
- continuous social-welfare allocation is automatically route-feasible;
- visual animation is telemetry from real vehicles.

Accurate phrasing is:

- evidence-grounded Gemma interpretation;
- deterministic, capability-constrained route intelligence;
- a **heuristic** solution to an NP-hard routing problem, chosen deliberately,
  producing a feasible and executable plan rather than a proven-optimal one;
- auditable allocation and fairness comparison;
- explicit missing-evidence ownership;
- immutable human-in-the-loop approval;
- simulated hackathon data with reproducible end-to-end replay.
