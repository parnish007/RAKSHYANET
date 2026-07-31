# RakshyaNet — how to run it and walk every feature

This is an operator's walkthrough, not an architecture document. It goes through
the product in the order the interface is designed to be used, and it names every
feature, where it lives, and what to say about it. If you only have five minutes,
read **§2 The five-minute demo path**.

For the mathematics and its stated limits see `MATH.md`. For algorithms as
implemented see `ALGORITHMS.md`. For interface invariants see `docs/CONTEXT.md`.

---

## 1. Starting the system

Two processes. Both must be running.

```bash
# Terminal 1 — backend (FastAPI, port 8000)
cd rakshyanet
python -m uvicorn backend.api.main:app --port 8000

# Terminal 2 — frontend (Vite, port 5173)
cd rakshyanet/frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>.

### Before you demo, check three things

```bash
# 1. Is a hosted Gemma key actually usable right now?
curl -s http://127.0.0.1:8000/api/gemma/status | python -m json.tool
```

Look at `key_pool.available_now`. It should be **at least 1**. The pool rotates
across every configured key and fails over on a rate limit, so one exhausted key
is survivable; zero means every call will fall back to the offline provider.

```bash
# 2. Are the declared function schemas being served?
curl -s http://127.0.0.1:8000/api/optimization/tools

# 3. Does the baseline comparison compute?
curl -s http://127.0.0.1:8000/api/optimization/baseline
```

If all three respond, the demo-critical paths are alive.

**Configuration** lives in `.env` (gitignored; copy `.env.example`). Multiple
Gemma keys are supported as `GEMMA_API_KEY`, `GEMMA_API_KEY_2`, `GEMMA_API_KEY_3`,
or as a comma-separated list in any one of them.

---

## 2. The five-minute demo path

The interface is four numbered stages, shown as tabs in the header. Walk them in
order — the narrative is the product.

| Stage | Tab | What you do |
|---|---|---|
| 1 | **Operations** | Start a plan. Show the map and the incident list. |
| 2 | **Gemma evidence** | Show what Gemma extracted, what it refused to guess, and the raw exchange. |
| 3 | **Math lab** | Show the math, then beat the naive baseline. |
| 4 | **Review & authorize** | Show the route manifest and record a human decision. |

### Stage 1 — Operations

You land on the **Operations** tab. At the top is the **mission launcher**, which is the entry point:

- **Run full pipeline** — you decide the inputs. Gemma extracts signals from the
  field reports, then the deterministic engine computes urgency, routes, and
  allocation. Takes ~15 seconds.
- **Let Gemma run the engine** — Gemma decides. This is the Route Intelligence
  track's headline path: the model calls `list_corridor_status`, reads the real
  road graph, then calls `run_optimization` through **native function calling**.
  Takes around a minute because it is several real round trips.

Say out loud: *neither button dispatches anything.* Both produce a plan that a
human must authorize in Stage 4.

Below the launcher:

- **Scenario deck** — five bundled incident timelines, each with a baseline stage
  and a road-closure stage. Switching one produces a new versioned plan.
- **Map** (`Geospatial twin · fixture road graph`) — the label says *fixture*
  because the road graph is bundled data, not a live feed. Click any incident to
  focus it. Click **Report map evidence** to add a report at a map point.
- **Mission clock** — always visible, but **locked** until a plan is authorized.
  Until then the readout reads `Mission clock locked · fleet held at depot` and the
  slider is disabled. Drag
  the slider to move mission time; vehicles move to where the solver's ETAs put
  them at that moment, and served stops are counted separately from pending ones.

### Stage 2 — Gemma evidence

This is the Gemma stage. Worth 30% of the rubric, so spend time here.

- **Grounded report analysis** — incident type, severity range, affected
  population range, medical urgency, accessibility risk, contradictions, missing
  information, follow-up questions, requested tools. Every non-null value cites
  the evidence IDs that support it.
- **The UNKNOWN track** — point at `medical_urgency`. On the bundled evidence it
  is `UNKNOWN`, not a number, because no source mentioned injuries. It renders
  with a dash **and a hatch texture**, not just a colour, so unknown is
  distinguishable from zero without relying on hue.
- **Contradictions** — both claims are shown with their citations. The system
  does not pick a winner; source credibility is a human judgement.
- **Follow-up questions** — each missing field generates a question you can
  assign to someone or mark **Unavailable**.
- **Model-orchestrated optimization** — the function-call panel. It shows each
  call Gemma emitted, the **raw arguments the model produced**, whether they
  passed validation, and the arguments the engine actually executed. If Gemma
  names a corridor that does not exist, you see the rejection here.
- **Raw prompt, reasoning, and response** — three tabs showing the literal wire
  content: the exact ~8,200-character prompt sent, whatever reasoning the
  provider exposed, and the exact JSON returned before any validation.
- **Model reasoning, verbatim** — inside the function-call panel. The
  function-calling turn returns real chain-of-thought, and it is shown unedited.

> **On chain-of-thought, precisely.** Reasoning availability depends on the call.
> In the **function-calling** turn the provider returns full readable reasoning —
> measured at ~1,900 characters, showing the model working through whether the
> corridor is genuinely impassable. That text is captured verbatim and rendered
> under **Model reasoning** in the function-call panel. In the **extraction** call
> the same model marks a thought segment but returns an **empty body**, because
> forcing `responseMimeType: application/json` suppresses it; there the panel says
> so rather than reconstructing reasoning that was never sent. The prompt and the
> raw response are real and complete in both cases.

### Stage 3 — Math lab

> **Say this before anyone asks it.** The dispatch problem is a capacitated
> vehicle-routing problem over a heterogeneous fleet — it contains TSP, so it is
> **NP-hard**. No fast algorithm is known that returns a provably optimal plan.
> So RakshyaNet uses a **heuristic on purpose**: greedy urgency-ordered
> assignment, then a nearest-neighbour tour, over a capability- and
> closure-filtered graph. The claim is not *"this is the best possible plan"* —
> it is *"this is a feasible, executable, closure-aware plan produced fast
> enough to act on, and every number in it is traceable"*. Saying that out loud
> is stronger than letting a judge discover it. What we guarantee is
> **feasibility, determinism, traceability and reproducibility**; what we do not
> guarantee is optimality, and the optimality gap against an exact solve is
> **unmeasured** — measuring it is roadmap.

- **Route plan** — assets, stops, distance, time, transport mode. Distances are
  the raw sum of edge lengths, never the terrain-weighted search cost.
- **Convergence** — the allocation's normalized residual on a log axis. If an
  iteration recorded no normalized residual it is **omitted from the curve**, not
  plotted as zero, and a footnote says how many were omitted.
- **Diagnostics** — KKT conditions. Say clearly that this is a *feasibility and
  consistency diagnostic*, not a proof of optimality: three of the four
  conditions hold for any primal-feasible allocation by construction. It applies
  to the **continuous allocation** — a separate, convex problem — and says
  nothing about the discrete NP-hard routing decisions. The API ships
  `independently_proves_optimality: false` and
  `applies_to_discrete_route_decisions: false` to make both machine-readable.
- **Shortest-path-only comparison** — press **Run comparison**. This runs the
  documented naive baseline and the production planner over identical inputs.

### Stage 4 — Review & authorize

- **Route manifest under authorization** — every asset, its state
  (feasible/excluded), mode, stops, distance, time. You are authorizing a plan,
  not an integer.
- **Approval scope** — states exactly what approval covers, and that it does not
  dispatch vehicles.
- **Server refuses approval** — if the backend has blocking reasons they appear
  verbatim. The server re-checks them on submit, so clearing them in the
  interface is not sufficient.
- **Override acknowledgement** — when there are unresolved warnings you must tick
  the acknowledgement and write a rationale of at least 12 characters.
- **Approve demo plan** / **Request changes**. (The primary button reads
  *Approve demo plan* because the bundled evidence is all simulated, and
  *Acknowledge below to authorize* until you tick the acknowledgement and write a
  rationale.)
- **Decision receipt** — after deciding, who decided, when, and their note.

---

## 3. The two Route Intelligence track requirements

Both are demonstrable in the running system, not just described in the writeup.

### Native function calling

```bash
curl -s http://127.0.0.1:8000/api/optimization/tools | python -m json.tool
curl -s -X POST http://127.0.0.1:8000/api/optimization/orchestrate \
     -H 'Content-Type: application/json' -d '{"scenario_id":"nepal-national-demo"}'
```

Two functions are declared to the model:

| Function | Purpose |
|---|---|
| `list_corridor_status()` | Returns every corridor in the road graph so any corridor Gemma names provably exists. |
| `run_optimization(analysis_id, blocked_edge_ids, time_elapsed_hours, rationale)` | Runs the deterministic engine and returns a plan awaiting human approval. |

Every argument is checked **before** the engine runs:

- `analysis_id` must be the analysis opened for this turn — the model cannot
  retarget a different evidence set;
- every blocked corridor must exist in the terrain graph — the model cannot
  invent a road closure;
- `time_elapsed_hours` must fall in 0–72;
- the free-text rationale is screened for allocation, dispatch, and approval
  language.

A rejected argument is returned to the model, not executed.

**The moment worth showing:** on the bundled evidence the police report says
heavy vehicles cannot pass but motorcycles can. Gemma calls
`list_corridor_status`, then calls `run_optimization` with **`blocked_edge_ids:
[]`** and a rationale explaining that the corridor is restricted but *not
established as impassable*. It refused to delete a usable corridor on
contradictory evidence. That is the behaviour the system prompt asks for, and it
is the right call — deleting a passable corridor can strand a district.

### The documented naive baseline

**Definition:** *shortest-path-only, no terrain weighting, closures ignored.* It
is the same engine with two behaviours switched off (`terrain_weighting=False`,
`honour_closures=False`), so it is not a strawman — it shares the road graph, the
Dijkstra search, the vehicle capability/capacity/fuel constraints, the urgency
model, and the allocation. Only the terrain reasoning is removed.

**Metric:** executable routes after a corridor on the plan closes. A route
through a closed corridor is not slower — it cannot be driven.

**Measured result** on the national scenario, closing
`east_west_bharatpur_nepalgunj` (which carries every ground route):

| | Naive | RakshyaNet |
|---|---|---|
| Routes through the closed corridor | **5** | **0** |
| Executable routes | **4 / 9** | **9 / 9** |
| Fleet distance | 9,782 km | 10,563 km (+8.0%) |
| Fleet time | 11,290 min | 12,462 min (+10.4%) |

All five trucks are stranded in the naive plan. RakshyaNet re-plans around the
closure and keeps every route executable, for 8% more distance.

**A measured negative result we report rather than hide:** on this 13-corridor
network, terrain-difficulty weighting **changes no path at all** — all nine
routes keep an identical edge sequence with it disabled. The corridors are too
sparse for the weighting to ever flip a choice. The entire measured advantage
comes from closure-aware re-planning. The track brief penalises inflated results,
so this is stated in the code, the API response, and the UI.

---

## 4. Feature index

Where everything lives, so nothing gets missed.

### Gemma boundary — `backend/services/gemma_service.py`

| Feature | Notes |
|---|---|
| Hosted extraction | `gemma-4-26b-a4b-it`, temperature 0, strict JSON schema |
| Prompt-injection screening | Runs **before** invocation; report text is data, never instruction |
| Evidence-ID citations | Every non-null field cites supporting evidence |
| UNKNOWN semantics | `null` + confidence 0 + empty citations; unknown ≠ zero ≠ unavailable |
| Population grounding | A figure is rejected unless it appears literally in cited text or is the midpoint of two stated bounds |
| Contradiction grounding | A claim must share 60% of its tokens with the record it cites |
| Incident-type grounding | The value must appear as a substring of cited text |
| Retrieval-tool allowlist | 19 names; anything else is rejected |
| Operational-authority screen | Any attempt to allocate, dispatch, or approve is rejected |
| No-digits-in-summary rule | Numeric claims must stay in cited structured fields |
| System confidence | `clamp[0,1](0.55·R̄ + 0.25·D + 0.20·F − P_contra − P_miss)` |
| Deterministic offline fallback | Derives its claims from whatever evidence it is given, obeys the same grounding rules, and reports UNKNOWN where the evidence does not carry the claim |
| Raw exchange capture | Exact prompt, exact response, thinking indicator |

### Function calling — `backend/services/gemma_orchestrator.py`
Declared schemas, the multi-turn loop, argument validation, and the
`OrchestrationRecord` persisted on the run.

### Key failover — `backend/services/api_key_pool.py`
Rotation across all keys; 429 parks a key for 65s; a revoked key parks for 900s;
a 5xx retries elsewhere without parking. Health on `/api/gemma/status`, never key
material.

### Math engine — `backend/algorithms/`

| Module | What it does |
|---|---|
| `urgency_calculator.py` | Unmet need × criticality × `T(t) = 1 + 0.5(e^{0.3t} − 1)`, plus a survival-threshold penalty that dominates proportional shortfalls |
| `vrp_solver.py` | Haversine air corridors; capability- and closure-filtered Dijkstra for ground; `cost_e = d_e(1 + 0.06·max(0, τ_e − 1))`; fuel feasibility; ≤2 stops per asset |
| `nash_solver.py` | Capped proportional allocation with a *normalized* convergence residual. Not an equilibrium, and named that way in `interpretation` |
| `social_welfare_optimizer.py` | SLSQP on `max Σ α_v log(1e-6 + c_v)`; a comparison only, never substituted into dispatch, because it ignores vehicle integrality |
| `kkt_verifier.py` | Scoped feasibility diagnostic with honesty flags |
| `baseline_service.py` | The documented naive baseline head-to-head |

### Run lifecycle — `backend/services/optimization_service.py`
Immutable versioned runs; `parent_run_id` child runs on closure; snapshot-token
approval (`expected_updated_at` + `expected_analysis_id`); a failed run never
supersedes the good run before it; `approval_blockers` surfaced to the interface.

---

## 5. Things to say out loud that a screenshot cannot show

- The **ETA audit is a shipped browser test**, not a CI job — this repo has no CI.
  `frontend/tests/mission-control.spec.js` pins
  `data-timeline-checkpoint-max-error` at `0.000000`: it walks every stop,
  compares the rendered position at that stop's `eta_minutes` against the stop's
  own coordinates, and requires zero error. The map is not an animation loosely
  inspired by the plan; it is the plan.
- **UNKNOWN uses a texture channel**, not only colour, so it survives
  colour-blindness and a bad projector.
- **The KKT panel is deliberately labelled as not a proof**, and the API says so
  in a machine-readable field. Most submissions claim the opposite.
- **The routing problem is NP-hard, so the solution is a heuristic — and that is
  the correct choice.** Volunteer it rather than defending it. What is
  guaranteed is feasibility, determinism, traceability and reproducibility; what
  is not is optimality, and the gap is unmeasured. Full argument, including why
  not OR-Tools, in `docs/QAs.md` §C1–C5.
- **Gemma's maximum influence on ranking is one bounded scalar in `[0,1]`**,
  against a survival-threshold penalty of `10.0`. The model can reorder
  priorities by at most a tenth of a single deterministic trigger.

---

## 6. Running the tests

```bash
# Backend: 861 tests
python -m pytest backend/tests -q

# Frontend
cd frontend
npm run lint
npm run build
npx playwright test          # end-to-end + axe accessibility + responsive
```

---

## 7. Known limitations, stated plainly

- The road graph, evidence, and scenarios are **bundled fixtures**, labelled as
  such in the interface. Nothing is a live government feed.
- Routing is a **heuristic**, deliberately — the underlying capacitated
  vehicle-routing problem is **NP-hard**, so an exact solve is not the right
  trade for real-time dispatch. Allocation is capped proportional, a rule rather
  than an optimum. The KKT output is a feasibility diagnostic on the continuous
  allocation, not a proof of global optimality.
- The **optimality gap is unmeasured**, and no approximation ratio is claimed.
  Solving the bundled instance exactly and reporting the gap is roadmap.
- Terrain weighting **changes no path on the bundled network** (see §3).
- The `Let Gemma run the engine` path takes ~30–60 seconds because it makes
  several real model round trips.
- Terrain tiles need network access; the map falls back to a schematic view
  without it.
- Nepali/Devanagari evidence is not yet supported; all bundled reports are in
  English.
- The imagery tool (§8) is **off by default**, uses EuroSAT tiles rather than
  imagery of the named corridors, and reports no accuracy figure.

---

## 8. The imagery verification tool (optional, off by default)

A third declared function lets Gemma ask for an independent overhead-imagery read
of a corridor. It runs a real Hugging Face image classifier **locally on this
machine**, in a **separate process**, and is **disabled unless you turn it on**.

### Turning it on

```bash
# 1. Start the classifier sidecar (its own venv on D:, never in the backend)
powershell -File tools/imagery_sidecar/start.ps1      # serves 127.0.0.1:8011

# 2. Enable the tool for the backend, then restart it
#    .env  ->  SATELLITE_TOOL_ENABLED=true
```

Check it: `curl http://127.0.0.1:8011/health` and
`curl http://127.0.0.1:8000/api/imagery/status`.

**With the flag off, the system is byte-identical to the version without this
feature.** The declaration is never sent to Gemma, the endpoints 404, and the
interface renders nothing about imagery. That is the default, and it is what a
judge cloning the repository sees.

### What it actually is

- A **land-cover classifier**, not a flood detector. It reports that the surface
  over a corridor now classifies as water where the reference is highway.
- Tiles are real Sentinel-2 patches from the published **EuroSAT** benchmark,
  **bound to Nepali corridors for demonstration**. They are *not* imagery of
  those corridors, and every surface says so.
- No accuracy figure is claimed. We have not validated on Nepali terrain.

### The three trigger paths

| Path | How it starts | What you see |
|---|---|---|
| **Model-initiated** | Gemma decides from the evidence — an uncorroborated claim, a contradiction, or **a weather advisory implying risk before anyone reports a blockage** | A `verify_report_with_imagery` call in the function-call panel |
| **Operator-directed** | *Ask Gemma to verify* — a directive enters the next turn; Gemma still emits the call | Same panel, tagged `operator-directed` |
| **Direct** | *Check imagery now* — skips Gemma entirely, ~1s | The record appears in the evidence ledger |

### Three tiers, always a valid answer

| Badge | Meaning |
|---|---|
| **live model** | Classifier answered within 3 s |
| **precomputed** | Sidecar unreachable; a cached result from a *real* model run |
| **unavailable** | No cached result either; the record says plainly that no corroboration was obtained |

`verify_corridor()` never raises. There is no path where the pipeline breaks
because the classifier is down, and no path where a result is invented.

### The safety property worth demonstrating

**A corridor whose only supporting evidence is an imagery record cannot enter
`blocked_edge_ids`.** Validation rejects the call and returns the reason to the
model. This is enforced in code with a test covering it — it is not prompt text.

Imagery raises confidence that something happened. It never establishes that a
truck cannot pass.
