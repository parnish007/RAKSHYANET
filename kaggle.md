# RakshyaNet — turning three disagreeing field reports into a route plan a human can sign

**Track:** Route Intelligence · **Model:** `gemma-4-26b-a4b-it` (hosted, Gemini API)

> **Gemma reads contradictory field reports and converts them — with citations and
> explicit `UNKNOWN`s — into the structured input a deterministic terrain-aware
> routing engine needs, then calls that engine itself through native function
> calling.** Every plan stops at a named human signature.

Gemma never allocates stock, never picks a vehicle, never dispatches.

---

## 🌍 Inspiration

In 2015 I was a child in Karnali. Our home was cut off, and what I remember is
not that there was nothing — it is that things existed *somewhere else* and did
not reach us. Trucks were on roads, supplies in warehouses. The road was gone,
and nobody with authority to re-route could see clearly enough to act.

That is the problem: **not scarcity, coordination.** Nepal has four national
helicopters, a road network one landslide can sever, and districts reporting by
radio, WhatsApp and paper.

Underneath sits a smaller problem. Two real sentences from one incident bundle:

> *"Heavy vehicles cannot pass the primary approach."*
> *"Motorcycles and trained foot teams may pass with caution."*

Both credible. Neither overrulable. Average them into "blocked" and you delete a
corridor motorcycles could use, stranding a district. Average into "open" and a
truck goes into a landslide.

**The gap is not routing. It is interpretation** — turning prose written by three
people who never spoke into numbers a solver can consume, *without silently
resolving the disagreement*.

That gap is real, not assumed. Humanitarian logistics packages (Sahana, HELIOS)
track inventory; a published review found they model no routing. The Logistics
Cluster runs on meetings. Disaster-VRP research, including a paper on this
earthquake (arXiv:1709.00162), *begins* with road status already clean.

**The optimiser is solved. Producing trustworthy input for it is not.**

---

## 🛠️ How we built it

**Technique: prompt engineering + native function calling. No fine-tuning, no
RAG.** The evidence for one decision is small, bounded and fully cited, so
retrieval adds a failure mode without adding recall. Fine-tuning was wrong too:
the task is not a style the model lacks but a *contract* it must obey — better
enforced by schema than by weights.

**Remove Gemma and there is no input.** The engine consumes structured risk
signals; nothing else produces them from a police transcript, a WhatsApp message
and a weather bulletin. Everything downstream is deterministic, bounding the one
probabilistic component.

### 1. Grounded extraction — Gemma as the interpreter

Each report carries an ID, source, reliability and age. Gemma returns **one
schema-validated object** — incident type, severity, affected population, medical
urgency, accessibility risk, contradictions, gaps — in which **every non-null
value cites the evidence IDs supporting it.**

**Citation validation, not trust.** A population figure is rejected unless the
number appears *literally* in cited text, or is the exact midpoint of two stated
bounds. Fail any check and the analysis is refused: the failure mode is *no
analysis*, never *a confident wrong number*.

**`UNKNOWN` instead of guessing.** On the bundled evidence `medical_urgency`
returns `null`, confidence 0, no citations, because no source mentions injuries —
rendered as a gap with a follow-up question, not a zero. Contradictions keep
**both** claims cited: source credibility is a human judgement.

### 2. Native function calling — Gemma as the operator

Three functions are declared. The model returns real `functionCall` parts and the
loop feeds `functionResponse` back; schemas are live at
`GET /api/optimization/tools`.

| function | what it does |
|---|---|
| `list_corridor_status()` | the real road graph — length, terrain difficulty, surface, landslide risk |
| `run_optimization(...)` | runs the engine, returns a versioned plan **awaiting human approval** |
| `verify_report_with_imagery(...)` | an **independent overhead read** — **off by default** |

**Every argument is validated before execution.** The `analysis_id` must be the
one opened for this turn, so the model cannot retarget another evidence set. Every
id in `blocked_edge_ids` must exist in the terrain graph, so **it cannot invent a
road closure.** A rejected argument goes back to the model, **never the engine**.

**What it did with the contradiction.** Gemma called `list_corridor_status` first
— it wanted the real graph before naming a corridor — then called
`run_optimization` with **`blocked_edge_ids: []`**, its rationale stating the
corridor is *restricted*, not *established as impassable*. It declined to delete
a usable corridor on contradictory evidence, and that turn is recorded verbatim
and replayable.

**The anticipatory behaviour.** One declared imagery trigger is a weather
advisory implying landslide risk **before anyone reports a blockage**. Each call
records a `trigger_reason`, because a precautionary look is a different claim
from corroborating a filed report.

**Imagery can corroborate a closure, never cause one.** A corridor whose *only*
support is an imagery record is **rejected from `blocked_edge_ids` during
validation** — in code, with a test. It is a **land-cover classifier, not a flood
detector**, on real Sentinel-2 **EuroSAT** patches *bound to* Nepali corridors,
not imagery *of* them. No accuracy is claimed.

### 3. The deterministic half, and the bound that makes it safe

Urgency scoring, capability-filtered Dijkstra over a terrain-weighted graph,
capped proportional allocation, KKT diagnostics. **Stack:** SciPy · NumPy ·
NetworkX · FastAPI · React + Vite · MapLibre.

Gemma's total influence on priority is **one scalar ≤ 1.0**, applied only to
locations the evidence names — against a deterministic survival penalty of
**10.0** from measured stock and population. A tenth of one deterministic
trigger: it can move an incident up the queue, never outweigh a measured
shortage. That ratio is on screen, checkable against the API. Report text is
**data, never instruction**; injection patterns are screened *before* invocation.

### 4. The mandatory naive baseline — and an honest result

**Definition:** shortest-path-only, terrain weighting off, closures ignored — not
a strawman, the **same engine with two flags off**. **Metric:** drivable ground
routes after a corridor closes (such a route is not slower — it cannot be driven).

| Closing `east_west_bharatpur_nepalgunj` | Naive | RakshyaNet |
|---|---|---|
| Drivable **ground** routes | **0 / 5** | **5 / 5** |
| Fleet distance | 9,782 km | 10,563 km (+8.0%) |

We count *ground* routes deliberately: the four aircraft are unaffected by road
closures, so including them would report 4/9 against 9/9 — and flatter us.

And the unflattering finding we report anyway: across all 36 node pairs,
**terrain weighting alone changes no path** — the network is too sparse to flip a
choice. The entire measured advantage is **closure-aware re-planning**.

---

## 🚀 The Prototype

**Live app:** https://rakshyanet.vercel.app · **Repo:**
https://github.com/parnish007/RAKSHYANET · **Demo video:** _[insert link]_

The hosted build names two absences on screen instead of hiding them: no live
telemetry (serverless cannot hold a socket open) and no overhead imagery (it
needs a GPU sidecar). Everything else runs.

Four workspaces follow one decision end to end: **Operations** (3D terrain twin,
incidents, closures), **Gemma evidence** (every field with confidence, citations
and `UNKNOWN`s; raw prompt and response one click away), **Math lab** (urgency,
routes, allocation, KKT, baseline), and **Review & authorize**, where a
still-`UNKNOWN` required field is **named aloud** and demands a source or a
written justification before the signature is accepted.

An **agent console** replays the recorded function-calling turn — verbatim
reasoning, emitted arguments, what validation accepted and rejected — labelled a
replay, not fake live streaming.

```bash
python -m uvicorn backend.api.main:app --port 8000
cd frontend && npm install && npm run dev      # http://localhost:5173
pytest backend/tests -q                        # 861 passed
```

Plus Playwright, axe and responsive suites. One pins the rendered vehicle
position to the solver's `eta_minutes` at **0.000000** error: the map is the
plan, not an animation of it.

### Stated limitations

Routing is a **heuristic on purpose** — the dispatch problem contains TSP and is
**NP-hard**, so an exact solve is the wrong trade for real-time dispatch; the
optimality gap is unmeasured. The KKT panel is a **diagnostic, not a proof**
(`independently_proves_optimality: false`). Road graph, fleet, evidence and
scenarios are **bundled fixtures**, labelled in-app. No Devanagari evidence yet.

---

## 🧩 Challenges we ran into

**1. Making uncertainty survive the whole pipeline.** The easy build averages
disagreeing reports into one number and loses the disagreement forever. Getting
`UNKNOWN` treated as first-class — propagated by the solver, the UI **and** the
approval gate rather than coerced to a default — touched every layer. The
authorization screen would happily approve a plan with required fields still
unknown until we made it name them aloud.

**2. Gemma's honesty cost us a feature, and we shipped the explanation.** Forcing
a JSON response MIME type on the extraction call **suppresses the model's thought
body** — it returns empty. We could have hidden that or invented reasoning.
Instead the panel says why, and shows the real 3,815 characters from the
function-calling turn, where the constraint does not apply.

**3. Enforcing the trust boundary in code, not in a prompt.** Every safeguard
exists because the first version let Gemma be *persuasive* instead of
*accountable*.

---

**Gemma is not a feature bolted onto a routing app.** It is the component that
turns three disagreeing humans into a decision a fourth human can sign.
