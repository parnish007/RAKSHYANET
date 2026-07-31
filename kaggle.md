# RakshyaNet — turning three disagreeing field reports into a route plan a human can sign

**Track:** Route Intelligence · **Model:** `gemma-4-26b-a4b-it` (hosted, Gemini API)

> **In one sentence:** RakshyaNet is a disaster-logistics decision system for
> Nepal in which **Gemma reads contradictory field reports and converts them —
> with citations, confidence scores and explicit `UNKNOWN`s — into the structured
> input a deterministic terrain-aware routing engine needs, then calls that
> engine itself through native function calling**, with every resulting plan
> stopping at a named human signature.

Gemma never allocates stock, never picks a vehicle, never dispatches.

---

## 🌍 Inspiration

In 2015 I was a child in Karnali. Our home was in one of the areas that got cut
off, and what I remember is not that there was nothing — it is that things
existed *somewhere else* and did not reach us. Trucks were on roads. Supplies
were in warehouses. The road was gone, and nobody with the authority to re-route
had a clear enough picture to act on quickly.

That is the local problem I am solving: **not scarcity, coordination.** Nepal has
four national helicopters, a road network a single monsoon landslide can sever,
and districts that report by radio, WhatsApp and paper.

Underneath it sits a smaller problem nothing addresses. Two real sentences from
one incident bundle:

> *"Heavy vehicles cannot pass the primary approach."*
> *"Motorcycles and trained foot teams may pass with caution."*

Both credible. Neither overrulable. Average them into "blocked" and you delete a
corridor motorcycles could still use — you strand a district. Average them into
"open" and you send a truck into a landslide.

**The gap is not the routing. The gap is interpretation** — turning prose written
by three people who never spoke to each other into numbers a solver can consume,
*without silently resolving the disagreement*.

And that gap is real, not assumed. Humanitarian logistics packages (Sahana Eden,
HELIOS, LSS, SUMA) track inventory and consignments — a published review found
they do not model routing or scheduling at all. The Logistics Cluster ran nine
hubs in Nepal in 2015 and runs on meetings; the Nepali Army's own lessons-learnt
review recorded relief being distributed randomly and uncoordinated. Academic
disaster-VRP work — including a Fisher–Jaikumar paper applied to this exact
earthquake (arXiv:1709.00162) — *begins* with demand and road status already
given as clean numbers.

**The optimiser is solved. Producing trustworthy input for it is not.** That is
the layer we built, and it is why the answer is a language model rather than a
better solver.

---

## 🛠️ How we built it

**Model:** `gemma-4-26b-a4b-it`, hosted via the Gemini API.
**Technique: prompt engineering + native function calling. No fine-tuning, no RAG
retrieval step** — the evidence set for one decision is small, bounded and fully
cited, so retrieval would add a failure mode without adding recall. Fine-tuning
was the wrong instrument too: the task is not a style the model lacks, it is a
*contract* it must obey, and a contract is better enforced by a schema and a
validator than by weights.

**Remove Gemma and there is no input.** The engine consumes structured risk
signals; nothing else produces them from a police transcript, a municipal
WhatsApp message and a weather bulletin. Everything downstream is deterministic
precisely so that the one probabilistic component can be bounded.

### 1. Grounded extraction — Gemma as the interpreter

Each report carries an ID, source category, reliability and age. Gemma
(`nepal-grounded-extraction-v3`) returns **one schema-validated object** —
incident type, severity range, affected-population range, medical urgency,
accessibility risk, contradictions, gaps, follow-ups, summary — in which **every
non-null value cites the evidence IDs supporting it.**

**Citation validation, not trust.** A population figure is rejected unless the
number appears *literally* in the cited text, or is the exact midpoint of two
stated bounds. A reported contradiction must share 60% of its content tokens with
the record it cites. The incident type must appear as a substring of cited text.
Fail any check and the analysis is refused — the failure mode is *no analysis*,
never *a confident wrong number*.

**`UNKNOWN` instead of guessing.** On the bundled evidence `medical_urgency`
returns `null`, confidence 0, no citations, because no source mentions injuries.
The UI renders a dash *and* a hatch texture, raises `needs_human_review`, lowers
confidence, and asks the follow-up question. Unknown, zero, unavailable and
not-applicable stay distinct states. Contradictions are preserved with **both**
claims cited — the system does not pick a winner, because source credibility is a
human judgement.

### 2. Native function calling — Gemma as the operator

Three functions are declared. The model returns real `functionCall` parts and the
loop feeds `functionResponse` back; the schemas are readable live at
`GET /api/optimization/tools`.

| function | what it does |
|---|---|
| `list_corridor_status()` | returns the real road graph — length, terrain difficulty, surface, landslide vulnerability |
| `run_optimization(analysis_id, blocked_edge_ids, time_elapsed_hours, rationale)` | runs the deterministic engine, returns a versioned plan **awaiting human approval** |
| `verify_report_with_imagery(corridor_id, incident_type, evidence_id, trigger_reason)` | requests an **independent overhead satellite read** of a corridor — **off by default** |

**Every argument is validated against the world before execution.** The
`analysis_id` must be the analysis opened for this turn, so the model cannot
retarget another evidence set. Every id in `blocked_edge_ids` must exist in the
terrain graph, so **it cannot invent a road closure.** `time_elapsed_hours` must
fall in 0–72. The free-text rationale is screened for allocation, dispatch and
approval language. A rejected argument goes back to the model, **never to the
engine**, and the rejection is displayed.

**What it actually did with the contradiction.** Gemma called
`list_corridor_status` first — it wanted the real graph before naming a corridor
— then called `run_optimization` with **`blocked_edge_ids: []`**, its rationale
stating the corridor is *restricted* but not *established as impassable*. It
declined to delete a usable corridor on contradictory evidence. That turn is
recorded verbatim — 3,815 characters of the model's own reasoning — and is
replayable in the app.

### 2b. The satellite tool — and the one genuinely anticipatory behaviour

The third declared function lets Gemma ask for **overhead imagery of a corridor**
as a second, independent source when the written reports are thin or disputed.
Two things about it matter more than the classifier itself.

**Gemma decides *when* to look, and one trigger is anticipatory.** The declared
triggers include a weather advisory implying elevated landslide risk **even
though nobody has reported a blockage yet** — so the model can check a corridor
*before* a report exists. Every call records a `trigger_reason`, and the UI
prints it in plain language, because a precautionary look off a forecast is a
different claim from corroborating a filed report. Nothing downstream is allowed
to confuse the two.

**Imagery can corroborate a closure. It can never cause one.** A corridor whose
*only* supporting evidence is an imagery record is **rejected from
`blocked_edge_ids` during validation** — enforced in code, with a test. Deleting
a road that motorcycles could still use is the exact failure this whole project
exists to prevent, and a land-cover classifier is not sufficient grounds to do
it.

**Stated honestly:** the tool is **off by default** — with the flag off the
declaration is never sent, the endpoints 404, and nothing renders, which is what
a judge cloning this repo sees. When it is on, it is a **land-cover classifier,
not a flood or landslide detector**, and the tiles are real Sentinel-2 patches
from the published **EuroSAT** benchmark *bound to Nepali corridors for
demonstration* — they are **not** live imagery of those corridors. We claim no
accuracy figure for it.

### 3. The deterministic half

**SciPy · NumPy · NetworkX · PuLP.** Urgency scoring, capability-filtered
Dijkstra over a terrain-weighted graph, capped proportional allocation, a Nash
bargaining comparison, and KKT verification of the allocation conditions.
**Stack:** FastAPI + Pydantic · React + Vite · MapLibre 3D terrain · WebSockets.

### 4. The bound that makes it safe

Gemma's total influence on priority is **one scalar ≤ 1.0**, applied only to
locations the evidence names — against a deterministic survival-threshold penalty
of **10.0** computed from measured stock and population. A tenth of one
deterministic trigger. It can move an incident up the queue; it cannot outweigh a
measured shortage. That ratio is on screen, in monospace, checkable against the
API. Report text is treated as **data, never instruction**: injection patterns are
screened *before* invocation. Approval is snapshot-token checked server-side
against the exact run reviewed.

### 5. The mandatory naive baseline — and an honest result

**Definition:** shortest-path-only, terrain weighting off, closures ignored. Not
a strawman — it is the **same engine with two flags off**, sharing the road
graph, the Dijkstra search, the vehicle constraints, the urgency model and the
allocation. **Metric:** drivable ground routes after a corridor on the plan
closes (such a route is not slower — it cannot be driven).

Closing `east_west_bharatpur_nepalgunj`, which carries every ground route:

| | Naive | RakshyaNet |
|---|---|---|
| Drivable **ground** routes | **0 / 5** | **5 / 5** |
| Fleet distance | 9,782 km | 10,563 km (+8.0%) |

We count *ground* routes deliberately: the four aircraft are unaffected by road
closures, so including them would report 4/9 against 9/9 and flatter us.

And the unflattering finding we report anyway: across all 36 node pairs,
**terrain weighting alone changes no path** — nine nodes and thirteen corridors
is too sparse to flip a choice. The entire measured advantage is **closure-aware
re-planning**, i.e. Gemma deciding which corridors the evidence actually removes.
Claiming otherwise would be the inflated baseline the brief penalises.

---

## 🚀 The Prototype

- **2-minute demo video:** _[insert link]_
- **GitHub repo:** https://github.com/parnish007/RAKSHYANET

Four workspaces, following one decision end to end:

1. **Operations** — 3D Nepal terrain twin, live incidents, road closures.
2. **Gemma evidence** — the extraction: every field with its confidence, its
   citations and its `UNKNOWN`s. The raw prompt and raw response are one click
   away.
3. **Math lab** — urgency arithmetic, routes, allocation, KKT diagnostics, and
   the naive-baseline comparison.
4. **Review & authorize** — a human signs. If a required field is still
   `UNKNOWN`, the interface **names it** — *"medical urgency is UNKNOWN"* — and
   the operator must either supply a source there and then, or record a written
   justification for proceeding without it.

An **agent console** replays the recorded function-calling turn — Gemma's
verbatim reasoning, the arguments it emitted, what validation accepted, what it
rejected — labelled as a replay of a stored record, not fake live streaming.

```bash
python -m uvicorn backend.api.main:app --port 8000
cd frontend && npm install && npm run dev      # http://localhost:5173
pytest backend/tests -q                        # 861 passed
```

Plus Playwright end-to-end, axe accessibility and responsive suites. One pins the
rendered vehicle position against the solver's `eta_minutes` at **0.000000**
error — the map is the plan, not an animation inspired by it. All five scenario
timelines replay offline via `python scripts/replay_scenarios.py`.

### Stated limitations

Routing is a **heuristic on purpose** — the dispatch problem contains TSP and is
NP-hard, so an exact solve is the wrong trade for real-time dispatch; the
optimality gap is unmeasured. The KKT panel is a **diagnostic, not a proof**, and
the API ships `independently_proves_optimality: false`. Road graph, fleet,
evidence and the five scenario timelines are **bundled fixtures**, labelled as
such in-app. The satellite tool is **off by default** and its tiles are EuroSAT
benchmark patches bound to corridors for demonstration, not live imagery. No Nepali/Devanagari evidence yet; no authentication or agency
integration.

---

## 🧩 Challenges we ran into

**1. Making the model's uncertainty survive the whole pipeline.**
The easy build averages disagreeing reports into one number and loses the
disagreement forever. Getting Gemma to emit a *bounded* answer with `UNKNOWN` as
a first-class value — and then getting the solver, the UI **and** the approval
gate to keep propagating that `UNKNOWN` instead of quietly coercing it to a
default — was the hardest single thing, and it touched every layer. The
authorization screen would happily approve a plan with required fields still
unknown until we made it name them out loud.

**2. Gemma's own honesty cost us a feature, and we shipped the explanation.**
Forcing a JSON response MIME type on the extraction call **suppresses the model's
thought body** — it returns empty. We could have hidden that, or invented
plausible reasoning. Instead the panel says exactly why it is empty, and shows
the real 3,815 characters from the function-calling turn, where the constraint
does not apply. Explaining a limitation was more work than papering over it.

**3. Trusting a model exactly as far as it should be trusted — no further.**
Every safeguard exists because the first working version let Gemma be *persuasive*
instead of *accountable*: the citation validator, the ≤1.0 cap, argument
validation against the road graph, the injection screen, the human gate. Deciding
where the line goes — and enforcing it in **code** rather than in a prompt — took
most of the day.

**4. One day, and a demo that must not lie.**
Under time pressure the tempting move is a scripted happy path. We kept the
disclosures instead: simulated evidence is labelled simulated, the replay is
labelled a replay, routing is called a heuristic, and the baseline reports the
result that does *not* flatter us.

---

**The one-line claim:** Gemma is not a feature bolted onto a routing app. It is
the component that turns three disagreeing humans into a decision a fourth human
can sign — and everything downstream is deterministic, bounded and checkable
precisely so that it can be.
