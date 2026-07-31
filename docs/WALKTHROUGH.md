# RakshyaNet — demo walkthrough, start to finish

A single continuous script: from `npm run dev` to the closing line. Every feature
in the product appears somewhere below, in the order you should show it. Timings
assume you talk while things load.

*(The previous planning document that lived at this path is archived at
`docs/archive-walkthrough-planning.md`. It contained claims this system does not
make.)*

**Total: ~12 minutes.** Cut §7 and §8 first if you are short.

---

## 0. Setup — before anyone is watching (3 min)

```bash
# Terminal 1
cd rakshyanet && python -m uvicorn backend.api.main:app --port 8000

# Terminal 2
cd rakshyanet/frontend && npm run dev
```

Then the three-line pre-flight:

```bash
curl -s http://127.0.0.1:8000/api/gemma/status | python -m json.tool   # key_pool.available_now >= 1
curl -s http://127.0.0.1:8000/api/optimization/tools                   # 2 declared functions
curl -s "http://127.0.0.1:8000/api/optimization/baseline" | head -c 200 # baseline computes
```

Open `http://127.0.0.1:5173` and **leave it on the landing state**. Do not
pre-run the pipeline — the cold state is part of the story.

If `available_now` is 0, say so up front: "we're on the offline provider today,
here's what that changes." Never let a judge discover it.

---

## 1. The opening line (30 s)

> "A landslide closes a road in Sindhupalchok. The first three reports are a
> police radio call, a municipality WhatsApp message, and a weather bulletin —
> and they disagree. One says heavy vehicles can't pass, another says motorcycles
> can. Nobody knows how many households are cut off. A dispatcher has to commit a
> helicopter, and there are four for the entire country."

> "RakshyaNet is the layer between those reports and that decision. Gemma reads
> the reports and calls a deterministic route engine. A human authorizes the plan.
> Gemma never dispatches anything."

**On screen:** the landing state. Point at the **stage strip** under the header —
four numbered stages, and a **Next** button that always says what to do. Point at
the header chip: it reads **`Gemma not yet run`**, because nothing has run yet.

> "Three states, never collapsed: not run, hosted model, declared fallback. Most
> demos show you a green light before anything has happened."

---

## 2. Stage 1 · Operations — start the plan (1 min)

The **mission launcher** is the first thing in the workspace. Two buttons.

> "Two ways to start. Same deterministic engine. The difference is who decides
> what to compute."

Press **Let Gemma run the engine.**

> "That's the Route Intelligence track requirement. Gemma has been given two
> function schemas. It's deciding which computation to run, and on what world
> state. It takes about 40 seconds because these are real round trips."

While it runs, work the map:

- Point at the label: **`Geospatial twin · fixture road graph`**.
  > "Fixture. It's bundled data and we say so on the map. We don't call mocked
  > road data live."
- Click an incident → the camera focuses it, and the heading updates with how
  many feasible routes reach that selection.
- Point at the **mission clock**, currently **locked**.
  > "The fleet is held at the depot. The time slider is disabled until a human
  > authorizes a plan. Nothing moves on an unapproved plan — that's enforced, not
  > a convention."
- Below the map: the **scenario deck** — five bundled incident timelines, each
  with a baseline and a road-closure stage.

---

## 3. Stage 2 · Gemma evidence — the model's work (3 min)

This is 30% of the rubric. Spend the time.

Click **Gemma evidence** (or press the **Next** button).

**a. Grounded extraction.** Every non-null field cites the evidence IDs behind it.

**b. The UNKNOWN track.** Point at `medical_urgency`.
> "UNKNOWN. Not zero, not a guess. No source mentioned injuries, so the model
> reports the gap and asks a follow-up question instead of interpolating a
> plausible number. And it renders as a dash plus a hatch texture, not just a
> colour — so it survives colour-blindness and a bad projector."

**c. Contradiction preserved.**
> "Police say heavy vehicles can't pass. The same report says motorcycles can. We
> keep both claims with their citations. We don't pick a winner, because source
> credibility is a human judgement, not a model output."

**d. Follow-up questions.** Assign one, or mark it **Unavailable**.

**e. Model-orchestrated optimization** — the function-call panel. This is the
headline.
> "Two declared functions. Gemma called `list_corridor_status` first — it wanted
> the real road graph before naming any corridor. Then it called
> `run_optimization`."

Point at the **raw arguments the model produced**.
> "This is the model's own JSON, before validation. And here's what it chose:
> `blocked_edge_ids: []` — empty. Its rationale says the corridor is restricted
> but *not established as impassable*. It refused to delete a usable corridor on
> contradictory evidence. Deleting one can strand a whole district."

**f. Model reasoning, verbatim.** Scroll to it.
> "That's the model's actual chain-of-thought from the function-calling turn.
> About 1,900 characters, unedited. You can read it deciding the road isn't
> completely blocked. We didn't summarise it and we didn't write it."

**g. Raw prompt, reasoning, and response.**
> "And if you don't trust any of that: here's the literal 8,200-character prompt
> we sent, and the exact JSON that came back before validation. Nothing between
> you and the wire."

*(Honest aside if asked: on the extraction call the thought body comes back
empty, because forcing a JSON response mime type suppresses it. The panel says
so rather than inventing reasoning.)*

---

## 4. Stage 3 · Math lab — the engine (2.5 min)

Click **Math lab**.

**a. Route plan.** Assets, stops, distance, time, mode.
> "Distances are the raw sum of edge lengths. Never the terrain-weighted search
> cost. An inflated number presented as a distance is a lie in kilometres."

**b. Convergence.** The normalized residual on a log axis.
> "Normalized, because litres of water, medical kits and tarpaulin sheets cannot
> share one axis. And if an iteration recorded no normalized residual, it's
> omitted from the curve — not plotted as zero. Missing is not the same as
> converged."

**c. Diagnostics.** Say this before anyone asks:
> "This is a KKT feasibility diagnostic. It is *not* a proof of optimality —
> three of the four conditions hold for any feasible allocation by construction.
> The API ships `independently_proves_optimality: false` so nothing downstream
> can misread it. We'd rather tell you that than have you find it."

**d. The baseline.** Press **Run comparison**.
> "The track asks for a documented naive baseline. Ours is *shortest-path-only,
> no terrain weighting, closures ignored* — and it is not a strawman, because it's
> the same engine with two behaviours switched off. Same road graph, same
> Dijkstra, same capacity and fuel constraints, same urgency model, same
> allocation."

Read the headline off the screen:
> "Close the corridor that carries every ground route. The naive planner keeps
> **0 of 5** drivable ground routes — all five trucks are routed through a road
> nobody can drive. We keep **5 of 5**, for 8% more distance."

Then say why the denominator is five, before anyone asks:
> "We count ground routes on purpose. The four aircraft can't be affected by a
> road closure, so counting them would let us report 4 of 9 against 9 of 9 — and
> that would flatter us."

Then volunteer the negative result:
> "And here's the part we could have hidden. Terrain weighting on its own changes
> **no path** on this network — the corridors are too sparse for it to ever flip a
> choice. The entire measured advantage is closure-aware re-planning. The brief
> penalises inflated baselines, so we measured our own and told you."

---

## 5. Stage 4 · Review & authorize — the human gate (2 min)

Click **Review & authorize**.

**a. Route manifest.**
> "Every asset, and whether its route is feasible or excluded. An operator
> authorizes a plan — not the integer '9 routes'."

**b. Approval scope.**
> "This authorizes coordination for one versioned run. It does not dispatch
> vehicles."

**c. Guards.** If there are unresolved warnings, the override checkbox and a
12-character rationale are required.
> "And if the server has its own blocking reasons, they're printed verbatim. The
> server re-checks them on submit, so satisfying the UI isn't enough."

**d. Approve.** Tick the acknowledgement, write a rationale of at least 12
  characters, press **Approve demo plan**, then confirm in the dialog.

**e. Decision receipt** appears: who decided, when, their note.
> "The decision leaves an artifact. Who, when, and why."

---

## 6. The mission clock — time control (1.5 min)

Return to **Operations**. The clock is now **unlocked**.

- Drag to **T+45**: vehicles sit exactly where the solver's ETAs put them.
- Drag to **T+120**: the served/pending counter moves; the next ETA updates.
- Press **Reset**: everything returns to the depot.

> "This is not an animation loosely inspired by the plan. There's a browser test in the suite
> that asserts the rendered vehicle position matches the solver's `eta_minutes` to
> six decimal places. If you scrub to 45 minutes, you are looking at the plan's
> own arithmetic."

---

## 7. The closure — re-planning live (1 min) · *cut first if short*

Open the scenario deck, switch the active scenario to its **road-closure** stage.

> "A closure is a state change, not a map toggle. Blocking an edge produces a new
> versioned plan, linked to the one it replaces, recomputes every ground path
> without that edge, and resets approval. The previous plan survives as history.
> No approval survives a change to the facts it rested on."

Aircraft routes are unchanged.
> "Which is exactly the point — helicopters don't care about road closures, and
> the model shouldn't pretend they do."

---

## 8. Add a report yourself (1 min) · *cut second*

Press **Report map evidence** → a hint appears → click anywhere on the map → the
evidence drawer opens pre-filled with that location.

Fill a source name and a claim, press **Queue source**, then **Analyze and
recalculate plan**.

> "That report is now part of the analysis and is carried into every re-run. You
> never re-enter it."

---

## 9. Closing (30 s)

> "OCHA's review of the 2015 earthquake found rural communities cut off for days
> — communication and logistics failures, not a shortage of supplies. That's an
> interpretation and routing problem."

> "So: keep uncertainty visible instead of averaging it away. Make every number
> traceable to the report that produced it. Respect the terrain that decides who
> can actually be reached. And leave the decision with a human who can see all of
> it. Gemma reads, reasons, and calls the engine. It never dispatches."

---

## Recovery — if something breaks mid-demo

| Symptom | What to do and say |
|---|---|
| Model call fails | The offline provider answers, the chip reads `Declared fallback` with a reason. "That's the fallback doing its job — and it tells you it's the fallback." |
| Function calling times out | Press it once more; the pool retries transient upstream errors. If it fails twice, switch to **Run full pipeline** and show the function-call panel from an earlier run. |
| Approval refused as stale | A newer run exists. Re-run, approve the current one. A *failed* run no longer blocks the good one. |
| Terrain tiles don't load | The map drops to a schematic view. "Public tiles need network; the geometry is ours." |
| Clock stays locked | The plan isn't approved. Approve it in Stage 4. |

## Never claim

- Not a Nash equilibrium — it is capped proportional allocation.
- Not a proof of optimality — the KKT panel is a feasibility diagnostic.
- Not live data — the road graph, evidence, and scenarios are fixtures.
- Never quote `welfare_improvement_percent` — it compares a vehicle-constrained
  allocation against an unconstrained one and is labelled incomparable.
