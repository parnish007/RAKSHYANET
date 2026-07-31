# Judge Q&A — RakshyaNet

Answers to what a panel is likely to ask, written to be said out loud. Where the
honest answer is a limitation, it is stated as a limitation — that is the position,
not a concession.

Grouped: **A** Gemma & hallucination · **B** reasoning & tool calling ·
**C** the math engine, optimality & proof · **D** design choices · **E** the baseline ·
**F** ethics & deployment · **G** engineering · **H** hard questions ·
**I** satellite imagery.

---

## A. Gemma, grounding, and hallucination

**A1. It's a small open model. How do you stop it hallucinating?**

We don't rely on the model not hallucinating. We make hallucination *detectable
and non-executable*, with four layers:

1. **Citation requirement.** Every non-null field must cite evidence IDs. A field
   with no citation is rejected at the schema boundary.
2. **Numeric traceability.** An affected-population figure is rejected unless the
   number appears *literally* in the cited text, or is the exact midpoint of two
   stated bounds. The model cannot introduce a population number that isn't in a
   report.
3. **Claim overlap.** A reported contradiction must share ≥60% of its content
   tokens with the record it cites. A paraphrase drifting away from the source
   fails.
4. **Substring check on categorical fields.** The incident type must appear as a
   substring of the cited text.

If any check fails, the analysis is refused and the deterministic fallback
answers. So the failure mode is "no analysis", never "a confident wrong number".

**A2. What if the model just says "unknown" to everything to pass your checks?**

Then it is behaving correctly and the product still works — you get an explicit
gap, a follow-up question, and a lower system confidence. Refusing to guess is the
designed behaviour, not a degradation. And it isn't free: UNKNOWN fields raise
`needs_human_review` and lower confidence, which is visible to the operator.

**A3. Show me a real case where it declined to guess.**

On the bundled evidence, `medical_urgency` returns `null` with confidence 0 and no
citations, because none of the three reports mentions injuries. The interface
renders UNKNOWN with a dash *and* a hatch texture and generates the follow-up
question "What is the latest verified count of residents requiring evacuation?"

**A4. What stops a malicious field report from hijacking the model?**

Report text is treated as data, never instruction. Prompt-injection patterns are
screened *before* invocation — "ignore previous instructions", "system prompt",
"reveal the prompt", role-change attempts — and the evidence is rejected at the
input boundary rather than being sent and hoped over. Separately, the model's
output is screened for any attempt to allocate, dispatch, or approve; that output
is refused even if it is well-formed.

**A5. Could the model quietly inflate severity to change priorities?**

It can move one bounded scalar in [0,1]. That scalar is
`B = round(max(severity, medical, accessibility) × system_confidence, 4)` and it
is added to urgency. The deterministic survival-threshold penalty is **10.0**. So
the maximum possible model influence on ranking is a tenth of a single
deterministic trigger, and it only reaches locations the evidence names.

**A6. Why not fine-tune?**

A one-day build, and more importantly the failure we care about is not "wrong
tone" but "invented facts". Fine-tuning does not give you traceability;
citation-level validation does. Fine-tuning would also make the safety properties
model-specific rather than architectural.

---

## B. Reasoning and native tool calling

**B1. How do you prevent hallucination in the *tool call* itself — wrong
arguments, invented corridors?**

Every argument is validated against the world before the engine executes:

| Argument | Guard |
|---|---|
| `analysis_id` | Must equal the analysis opened for this turn — it cannot retarget another evidence set |
| `blocked_edge_ids` | Every id must exist in the terrain graph — it cannot invent a road closure |
| `time_elapsed_hours` | Must be within 0–72 |
| `rationale` | Screened for allocation, dispatch, and approval language |

A rejected argument is returned to the model, never to the engine, and the
rejection is displayed. The model chooses *what to compute*; it never gets to
supply an unchecked value to a computation.

**B2. Do you actually use native function calling, or is it structured output
dressed up?**

Native. Two `functionDeclarations` are sent in the request by default — a third,
`verify_report_with_imagery`, is declared only when the imagery flag is on, and
it is off by default (see I8). The model returns real
`functionCall` parts, and the loop feeds `functionResponse` back. You can read the
declared schemas at `GET /api/optimization/tools` and watch the calls in the
interface. In the observed run it called `list_corridor_status` first — it wanted
the real graph before naming a corridor — then `run_optimization`.

**B3. Can I see its reasoning, or are you just asserting it reasoned?**

You can read it. The function-calling turn returns roughly 1,900 characters of the
model's own deliberation, shown verbatim under **Model reasoning**. It contains
the step where it decides the road is not completely blocked.

One honest caveat: on the *extraction* call the thought body comes back empty,
because forcing `responseMimeType: application/json` suppresses it. There the
interface says so rather than reconstructing anything. We also show the exact
8,200-character prompt and the raw pre-validation JSON, so you can audit the
exchange without trusting our summary of it.

**B4. Is the reasoning trustworthy?**

No, and we don't treat it as such. Reasoning text is not citation-validated, so it
is displayed for transparency and is explicitly excluded from every decision. Only
schema-checked, citation-backed fields and validated arguments have any effect.
Treating chain-of-thought as evidence would bypass every validator we built.

**B5. What if the model refuses to call the function, or loops?**

The loop is bounded at four turns. If no valid `run_optimization` call is produced,
the endpoint returns an error naming the rejections rather than executing a guess.
The operator-driven path remains available and produces the same plan.

**B6. It called `run_optimization` with an empty closure list. Isn't that the
model failing to act?**

It is the model acting correctly. The police report says heavy vehicles cannot
pass *and* that motorcycles can. That is a restriction, not an established
closure. Deleting a corridor that is actually passable can strand a district with
no ground access at all, so "don't delete on contradictory evidence" is the
instruction, and it followed it — and said why in its rationale.

---

## C. The math engine, optimality, and what is actually proved

**C1. What guarantees this is the optimal allocation, or the optimal route?**

Nothing does, and we say so on the record rather than being caught at it.

The dispatch problem here is a capacitated vehicle-routing problem with a
heterogeneous fleet, capability constraints, endurance limits and time pressure.
That problem is **NP-hard** — it contains the travelling salesman problem, and
the assignment layer on top of it is harder still. No fast algorithm is known
that returns a provably optimal plan, and unless P = NP none exists.

Be precise about *where* the difficulty sits, because a sharp reader will check.
With at most two stops per asset, ordering the stops inside one route is trivial
— there are two orderings. The hard part is the layer above: partitioning
locations and cargo across a heterogeneous fleet under capacity, endurance and
capability constraints. And the two-stop cap is itself a heuristic restriction
of the search space, not a property of the problem.

So we made a deliberate engineering choice: a **greedy urgency-ordered
assignment plus a nearest-neighbour tour**, over a graph already filtered for
vehicle capability and active closures. We call it a heuristic in the code, in
the API payload, in `MATH.md`, and here.

What we *do* guarantee, and what a dispatcher actually needs:

1. **Feasibility** against the stated constraints — every dispatched route
   respects payload mass, fuel endurance, terrain capability, depot stock and
   the active closure set. A run that fails route feasibility cannot be
   approved; the backend refuses it.
2. **Determinism** — same inputs, same plan, every time. No sampling, no
   tie-break by dictionary order, no hidden clock.
3. **Traceability** — every number on screen decomposes to its inputs: which
   report produced which signal, which edges produced which distance, which
   ledger produced which shortfall.
4. **Reproducibility** — the run is an immutable snapshot, replayable from
   bundled fixtures with `python scripts/replay_scenarios.py`.

"Optimal" is not on that list. "Feasible, fast, and auditable" is, and for a
plan a human has to sign in the next ten minutes that is the more useful
property.

**C2. So your math engine doesn't prove anything?**

It proves several things. Just not the one people assume. Four distinct claims,
which get conflated constantly:

1. **Arithmetic correctness** — *proved, in the ordinary software sense, and
   tested.* 861 backend tests cover the urgency composition, the edge-cost
   formula, Haversine distance, the allocation fixed point and the confidence
   formula against worked examples in `MATH.md`.
2. **Constraint feasibility** — *checked at runtime and reported.* Allocations
   are non-negative, within stated need, and within depot stock; the reported
   distance equals the raw sum of traversed edge lengths; no feasible ground
   route contains an active blocked edge. Violations surface as approval
   blockers rather than being smoothed over.
3. **Convergence of the allocation fixed point** — *measured, not asserted.* The
   normalized residual per iteration is returned and plotted; convergence means
   `ε_norm < 0.01`, and an iteration with no recorded residual is omitted from
   the curve rather than drawn as zero.
4. **Global optimality** — *not claimed, for either the routing or the
   allocation.* No certificate is produced and none is implied. The API ships
   `independently_proves_optimality: false`.

Points 1–3 are the ones that stop a wrong number reaching a dispatcher. Point 4
is the one that would be expensive and, on this problem, is not what makes the
plan trustworthy.

**C3. If it's only a heuristic, why should a dispatcher trust it?**

Four reasons, in the order they matter.

It is **feasible** — the plan can actually be executed by the fleet that exists,
which is more than the naive baseline manages after a closure, and more than the
continuous fairness candidate manages at all. It is **fast** — a plan in about
fifteen seconds, which is the timescale the decision happens on. It is **fully
traceable** — every figure decomposes to the evidence and the edges that
produced it, so a dispatcher who disagrees can see exactly where to disagree.
And **a named human authorizes it**; the system proposes, it never dispatches.

The last reason is the honest one. The alternative to this heuristic, in the
districts we are describing, is not an exact solver. It is a spreadsheet, a
radio and a phone call. Against that, "feasible, fast, and traceable" is not a
compromise — it is the whole improvement.

**C4. Why not use OR-Tools, or an exact MILP solver?**

Fair question, and it has a real answer rather than an excuse.

**What we would gain.** This instance is small — nine assets, eight locations,
thirteen corridors. A CP-SAT or MILP formulation would plausibly close it to
optimality or to a bounded gap, and it would give us something we currently
cannot report: a number for how far the greedy plan sits from the best possible
one.

**What it would cost.** A solver dependency, an encoding of every constraint the
greedy pass currently enforces procedurally, and an answer that is harder to
explain to the person signing it — "the solver says so" is a worse artifact than
a decomposable score. And the cost that decided it: a one-day build spends its
budget where the gap is.

**Why the gap is elsewhere.** Optimisation over a road network with known demand
is a solved field — there is published disaster-VRP work applied to this exact
earthquake. What that literature *starts* with is demand and road status already
given as clean numbers. Nothing produces those numbers from three contradictory
reports written by people who never spoke to each other. That interpretation
layer is this project's contribution; swapping the router for OR-Tools would
improve the part that was already solved.

Both are true at once, which is why an exact solve is on the roadmap rather than
dismissed.

**C5. How far from optimal are you?**

We have not measured it, and we are not going to guess.

Here is exactly how we would. The bundled instance is small enough to formulate
exactly — decision variables for asset-to-location-to-resource assignment,
constraints for payload mass, fuel endurance, capability and depot stock, and
the same coverage objective the API already reports. Solve it to optimality or
to a proven bound, then report `(greedy − optimal) / optimal` on the coverage
objective and on fleet distance, for the baseline scenario and for the
post-closure scenario.

That is a measurement, not a claim, and until it exists the honest statement is
"unmeasured". It is the third item on the roadmap in H3.

**C6. Then what is the KKT panel doing? Isn't that a proof?**

No, and this is the question we most want asked. Three of the four KKT conditions
hold for *any* primal-feasible allocation **by construction**:

- **Stationarity** — the multiplier λ_r is *defined* as
  `Σ(∂f/∂x)x / Σx`, so the residual `|Σ(∂f/∂x)x − λ_r Σx|` is algebraically zero.
- **Dual feasibility** — λ_r is a non-negatively-weighted mean of non-negative
  terms, so λ_r ≥ 0 always.
- **Complementary slackness** — λ_r is *set* to zero exactly when the resource is
  slack, so the product is always zero.

Only **primal feasibility** carries information. A zero allocation that delivers
nothing to anyone passes all four. So it is a feasibility-and-consistency
diagnostic on the continuous allocation, it says nothing about the discrete
routing decisions, and the API ships
`independently_proves_optimality: false` and
`applies_to_discrete_route_decisions: false` so nothing downstream can misread it.

Worth separating two problems that the panel's presence tends to blur. The
*continuous allocation* is a separate, convex problem, and that is the only
thing the KKT diagnostics look at. The *discrete routing* — which asset goes
where with what cargo — is the NP-hard one, and no condition on this panel
touches it. A green panel and a bad route plan are perfectly compatible states,
which is precisely why the flag is in the payload.

**C7. You call it a Nash solver. Is it a Nash equilibrium?**

No. The module is named `nash_solver` for backward compatibility and the object is
labelled *capped proportional allocation* in its own `interpretation` field. There
is no strategic game, no best-response dynamics over utilities, and no equilibrium
claim. Shares scale with need and criticality, cap at need, and surplus
redistributes to those still short. Calling that an equilibrium would be wrong,
and several documents in the repo that used to say so now carry a SUPERSEDED
banner.

**C8. Why exponential time decay rather than linear?**

`T(t) = 1 + 0.5(e^{0.3t} − 1)`. Hour eight is not four times worse than hour two —
untreated injury, dehydration and exposure compound. Measured values:
`T(2)=1.4111`, `T(4)=2.1601`, `T(8)=6.0116`. Separately, falling below a survival
threshold adds a fixed penalty of 10.0 that dominates proportional shortfalls,
because starvation is not a matter of degree.

**C9. Why is the convergence residual normalized?**

Because the raw per-iteration change is in each resource's own unit — litres of
water, medical kits, tarpaulin sheets — and those cannot be compared on one axis
or against one tolerance. The residual is divided by the larger of new demand, old
demand, depot stock, or 1, giving a dimensionless quantity. The interface plots
only that, and if an iteration recorded no normalized residual it is omitted from
the curve rather than plotted as zero — because missing is not converged.

**C10. Your social-welfare optimizer gives a better objective. Why don't you use
its answer?**

Because it ignores vehicle integrality. It maximises `Σ α_v log(1e-6 + c_v)` over
a continuous allocation, which can describe a distribution no fleet can actually
fly. It is computed and shown as a comparison and is never substituted into
dispatch. Also worth stating: on the bundled scenario it makes the worst-off
location *worse* off — 0.4808 versus 0.5426 minimum coverage — because weighted
log welfare is not max-min fair. That is a property of the objective, not a bug,
and it is exactly why the route-feasible allocation is the one that ships.

**C11. What was the hardest numerical bug you found?**

Per-village rounding to six decimals pushed the summed allocation about 1e-6 above
depot stock. The feasibility check uses a 1e-6 tolerance, so the demo was passing
that condition by roughly 6e-14 — any fixture change would have flipped it to FAIL
live. The residual is now returned to the largest recipient and the worst excess
is exactly 0.

**C12. How do I know the map isn't just an animation?**

There is a browser test in the suite asserting that the rendered vehicle position matches the
solver's `stop_details[].eta_minutes` to `0.000000`. When you scrub the mission
clock to T+45 you are looking at the plan's own arithmetic, not an interpolation.

---

## D. Design choices

**D1. Why a graph rather than a distance matrix?**

In Nepal, adjacency is not geometric. A settlement 40 km away may be four hours by
road, twenty minutes by air, or unreachable by ground once one corridor fails. We
measured it: road distance averages **1.43×** straight-line and reaches **1.80×**.
A planner using geometric proximity ranks the wrong locations as closest.

**D2. Why delete closed corridors instead of penalising them?**

A penalty can always be overcome by a large enough benefit, and a truck must never
be routed down a road it cannot use. Closed corridors and undriveable surfaces are
removed before the search begins. Terrain difficulty is a weighting; passability is
a filter — they are different kinds of constraint and are implemented differently.

**D3. Why is reported distance not the weighted cost?**

Because an inflated figure presented as a distance is a lie in kilometres. Terrain
difficulty inflates the *search* cost so safer corridors win; the reported number
is the raw sum of traversed edge lengths. We verified the separation numerically:
`depot→mahendranagar` reports **711.00 km** where the weighted cost is 723.29.

**D4. Why keep a human in the loop at all? Isn't that the bottleneck?**

The bottleneck in 2015 was not decision speed, it was interpretation and routing.
And the decisions here are irreversible commitments of a four-helicopter national
fleet against contradictory reports. The system's job is to make the human's
decision *fast and informed* — a route manifest, an explicit uncertainty account,
and the server's own refusal reasons — not to remove them.

**D5. Why an API rather than on-device?**

The hackathon explicitly permits API access, and a one-day build spends its budget
where it scores. That said the architecture is provider-shaped: extraction sits
behind a `GemmaProvider` protocol with a deterministic offline implementation
already in place, so a local runtime is a provider swap, not a redesign. The
motivation is real — districts lose bandwidth exactly when reporting volume peaks.

**D6. Why five bundled scenarios instead of live data?**

Because we would rather ship a reproducible demonstrator that says "fixture" on
its own map than an unreproducible one that implies a government feed it does not
have. Every fixture is labelled in the interface.

---

## E. The baseline

**E1. What is your baseline, exactly, and how do I know it isn't a strawman?**

*Shortest-path-only, no terrain weighting, closures ignored.* It is not a separate
implementation — it is the **same engine** with two flags off
(`terrain_weighting=False`, `honour_closures=False`). It shares the road graph, the
Dijkstra search, the vehicle capability/capacity/fuel constraints, the urgency
model and the allocation. Only the terrain reasoning is removed, so every measured
difference is attributable to that and nothing else.

**E2. What is the metric and the result?**

Executable routes after a corridor on the plan closes — because a route through a
closed corridor is not slower, it is undrivable. Closing
`east_west_bharatpur_nepalgunj`, which carries every ground route:

| | Naive | RakshyaNet |
|---|---|---|
| **Executable *ground* routes** | **0 / 5** | **5 / 5** |
| Routes through the closed corridor | 5 | 0 |
| Executable routes, all modes | 4 / 9 | 9 / 9 |
| Fleet distance | 9,782 km | 10,563 km (+8.0%) |
| Fleet time | 11,290 min | 12,462 min (+10.4%) |

The ground row is the one to quote, and it is the harsher of the two for the
naive planner rather than the kinder. Aircraft fly geodesic corridors, so a road
closure cannot strand them; every route the naive planner keeps is a helicopter.
Of the roads it planned, none is drivable — all five trucks are stranded. Both
denominators ship in the API response.

**E3. Isn't 8% more distance a *worse* result?**

Only if you score distance and ignore whether the plan can be executed. Not one
of the naive planner's five ground routes can be driven. Paying 8% to keep the
plan valid is the trade the operator wants; the interface shows both numbers so
they can see it.

**E4. Anything about the baseline that hurts your case?**

Yes, and we report it. On this 13-corridor network, terrain-difficulty weighting
**on its own changes no path** — all nine routes keep an identical edge sequence
with it disabled. The corridors are too sparse for it to flip a choice. So the
measured advantage is entirely closure-aware re-planning, not terrain weighting.
It is stated in the code, in the API response, and in the interface. On a denser
network we would expect the weighting to matter; on this one it does not, and we
are not going to claim otherwise.

---

## F. Ethics, safety, and deployment

**F1. Who is accountable if the plan is wrong?**

The human who authorized it, and the record says who. Approval writes
`reviewed_by`, `reviewed_at`, and a note; when there are unresolved exceptions the
operator must additionally tick an acknowledgement and write a rationale of at
least 12 characters. The decision leaves an artifact, and the run it authorized is
immutable.

**F2. Does the model ever decide who gets help?**

No. It contributes one bounded scalar to a priority score and it may select which
computation runs on which world state. Allocation, routing, approval and dispatch
are all outside its authority, enforced by an output filter that rejects any
attempt at them. The maximum ranking influence is a tenth of one deterministic
survival trigger.

**F3. Automating triage means ranking people. How is that defensible?**

The ranking already happens — today it happens in a dispatcher's head under time
pressure with contradictory radio traffic and no record. What we change is that
the ranking becomes inspectable: you can see which report produced which number,
where the evidence is missing, and what the alternative allocation would have
looked like. We would rather argue about a visible criterion than defer to an
invisible one.

**F4. What about the people in the reports — casualty counts, household
locations?**

That data is exactly why an eventually-local deployment matters, and why we kept
extraction behind a provider protocol. Today the demo runs on fixtures and no real
personal data leaves the machine. In a real deployment the honest requirement is
in-district inference, and we would not pretend otherwise to make the demo simpler.

**F5. What is the worst realistic failure of this system?**

Confidently mis-stating passability. If evidence wrongly establishes a corridor as
closed, the plan routes around a usable road and arrival is slower; if it wrongly
leaves a closed corridor open, vehicles are dispatched into it. That is why closure
is a filter with an explicit provenance trail, why contradictions are never
auto-resolved, and why the model is instructed not to treat a restriction as a
closure. It is mitigated, not eliminated.

**F6. Could this be misused?**

Any logistics prioritiser can be. The mitigations that matter are that every
number is traceable to a source, the run history is immutable, and a named human
authorizes each plan — so a misuse leaves evidence rather than disappearing into an
average.

---

## G. Engineering

**G1. What happens when the model is rate-limited mid-demo?**

There is a key pool. A 429 parks that key for 65 seconds and the request retries on
the next key before any error surfaces. A revoked key parks for 900 seconds. A 5xx
is treated as an upstream fault, retried without parking the key. Pool health,
never key material, is exposed at `/api/gemma/status`. We found the hard way that
one of our three keys is denied at the project level — so the effective pool is
two, and we check `available_now` before demoing.

**G2. What is the offline story?**

A deterministic provider derives its claims from whatever evidence it is given,
obeys the same grounding validators, and reports UNKNOWN where the evidence does
not support a field. It is labelled `Declared fallback` in the interface — we never
show a fallback result as a hosted one.

**G3. How is the demo tested?**

861 backend tests, plus end-to-end browser tests covering the four workspaces,
axe accessibility checks, and pinned responsive regressions at 390/768/900/1024/
1120/1280px. Several of those tests exist because verification found the bug
first: an approval deadlock behind a failed run, a mission clock whose Reset button
rendered outside a clipping parent, and a navigation bar that pushed the review tab
off-screen on a phone.

**G4. What is still a demonstrator rather than production?**

Routing is a greedy heuristic, not an exact VRP solve — a defensible choice on an
NP-hard problem, but the optimality gap is unmeasured (C5). Allocation is
proportional, not an optimum. The road graph, fleet, and evidence are fixtures. Nepali/Devanagari
evidence is not yet supported. The function-calling path takes 30–60 seconds. All
of that is in `docs/current-limitations.md`.

---

## H. The hardest questions

**H1. Strip away the framing — what have you actually built?**

An auditable boundary between a language model and irreversible logistics
arithmetic. Gemma interprets contradictory prose and selects the computation; a
deterministic engine computes; every number is traceable to a source; a human
authorizes. The novel part is not any single algorithm — it is that the model's
influence is bounded, measurable, and inspectable at every step.

**H2. Why should this win rather than a slicker demo?**

Because the interesting failure in a Gemma logistics tool is a confident wrong
number, and most demos have no mechanism that would catch one. We can show you the
model declining to guess, the arguments it produced being validated before
execution, a measured baseline comparison, and the places where our own approach
does *not* help. That is harder to build than a smooth animation and it is what
would matter if this were deployed.

**H3. What would you do with another week?**

In order: accept Nepali/Devanagari field reports; run extraction on a local Gemma
runtime so the data never leaves the district; replace the greedy router with an
exact solve on the small instance so we can report a genuine optimality gap; and
build the denser road graph on which terrain weighting would actually change paths,
since on the current network we measured that it does not.

**H4. What is the weakest part of the submission?**

That the closure comparison rests on one corridor in one bundled network, and that
terrain weighting contributes nothing measurable on that network. The safety
architecture is strong and tested; the empirical claim is narrow. We would rather
tell you where it is narrow than widen it with numbers we cannot defend.

---

## I. Satellite imagery and the vision tool

**I1. Is that real satellite imagery, and is it of Nepal?**

The tiles are real Sentinel-2 imagery from the **EuroSAT** benchmark (Helber et
al.) — a published dataset of 27,000 labelled patches. They are **bound to Nepali
corridors for demonstration; they are not imagery of those corridors.** Every
screen in the product says so, and the evidence record says so in its own text.
We would rather tell you that than have you find it.

**I2. What model, and where does it run?**

A EuroSAT-finetuned image classifier from Hugging Face, running **locally on this
laptop** in a separate process. It is a **land-cover classifier, not a flood
detector** — it reports that the surface over a corridor now classifies as water
where the reference is highway. That is a change-of-surface signal, and we use it
only as corroboration.

**I3. What's your accuracy?**

We don't quote one, deliberately. Published metrics for this class of model come
from benchmarks dominated by low-relief floodplains; domain shift to confined
Himalayan valleys is severe, and we have not validated on Nepali terrain.
Reporting someone else's benchmark F1 as if it were our performance here is
exactly the claim we refuse to make. Validating against ICIMOD or DHM ground
truth is the first item on the roadmap.

**I4. How do you get from "the model sees water" to "this road is impassable"?**

**You don't, and we don't.** Imagery tells you a valley is flooded; it does not
tell you a truck cannot pass. It measures neither depth nor duration. A landslide
scar on a slope does not establish that debris reached the carriageway. And
bridge scour — a leading mode of route severance in Nepal — is close to
invisible from a nadir view; the deck looks intact from directly above.

So imagery enters as **one more cited, bounded, low-reliability evidence record**
(`reliability` capped at 0.55), never as ground truth. And this is enforced:
**a corridor whose only supporting evidence is an imagery record is rejected from
the closure list before the engine sees it.** That is validation with a test
covering it, not a promise in a prompt.

**I5. Why not use SAR? Sentinel-1 sees through cloud.**

Because both halves are true and we should say both. C-band SAR does solve the
cloud problem, and Nepal's monsoon — which is precisely when these landslides
happen — puts the mid-hills under persistent cloud, so optical revisit stretches
from a nominal ~5 days to effectively weeks over a given valley.

But SAR is the *worst* instrument for our specific terrain. It is side-looking
and orders returns by range, so in Himalayan relief you get **layover** where the
top of a slope images before its base, and **radar shadow** where a ridge blocks
illumination entirely. Shadow preferentially hides **valley-floor roads** — the
corridors we care most about. That is orbital geometry, not a tuning problem.

Worth adding: the flagship NASA/IBM flood model, `Prithvi-EO-*-sen1floods11`,
takes Sentinel-2 **optical** input despite the name — the Sen1Floods11 dataset
contains both, and that finetune uses the optical half. So "we'll use Prithvi" is
not by itself a monsoon answer.

**I6. Why is it a tool call instead of just showing Gemma the image? Gemma 4 has
vision.**

It does — `gemma-4-26b-a4b-it` accepts image input on the endpoint we already
use. We chose not to, for a specific architectural reason.

Our grounding validator substring-matches an extracted value against the **text**
of the record it cites. An image has no text. So direct vision leaves two
options: cite a text record the image didn't contribute to, which is a fiction —
or manufacture a record whose text is Gemma's own description of the image, in
which case Gemma cites text Gemma wrote and **the validator passes while
certifying nothing.** The system would keep reporting green while the guarantee
silently became false.

The tool returns *text*, so the existing validators work unchanged and the audit
trail stays intact. Direct multimodal grounding — with a provenance record that
makes the citation auditable — is on the roadmap, not in the demo.

**I7. When does Gemma decide to call it?**

Four triggers, all in the declaration: an uncorroborated blockage claim; two
sources contradicting each other about passability; **a weather advisory
implying elevated risk even though nobody has reported a blockage yet**; or an
operator directive. The third is the interesting one — it means the model checks
a corridor *before* a report exists, and the record is tagged
`trigger_reason: anticipatory` so nothing downstream mistakes a precautionary
check for a confirmed one.

A negative result is recorded too. If the surface looks normal, that weakens an
uncorroborated claim, and the model is instructed to report it rather than omit
it.

**I8. What happens if the classifier is down during this demo?**

It degrades in tiers and says which one you're looking at. Live inference is
`live model`; if the classifier is unreachable within 3 seconds you get
`precomputed` — a cached result from a real model run, labelled as such; and if
there's no cached result either, `unavailable`, whose text states plainly that no
corroboration was obtained. There is no path where the system invents a result,
and the whole feature is behind a flag that is **off by default** — a judge
cloning the repo gets the system without it.
