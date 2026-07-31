# Demo Script

Verified against the shipped interface. Every control named here exists at the
click path given. Full feature reference: `HOW_TO_USE.md`.

## Before you start

```bash
curl -s http://127.0.0.1:8000/api/gemma/status | python -m json.tool
```

Check `key_pool.available_now` is at least 1. If it is 0 every model call falls
back to the offline provider, and you should say so rather than let a judge
discover it.

The interface is four numbered stages, selected by the tabs in the header.
**Approval lives in the Review & authorize tab, not in Operations.** An earlier version of this
script sent you to Operations to approve; the decision panel does not mount
there.

---

## Stage 1 · Operations — start the plan

1. Open `http://127.0.0.1:5173`. You land on the **Operations** tab.
2. Point at the header runtime chip. It reads the model name when hosted Gemma
   answered, `Declared fallback` when the offline provider did, and
   **`Gemma not yet run`** before anything has run — three distinct states, never
   collapsed into one.
3. At the top of the workspace is the launcher. Press **Let Gemma run the
   engine**. Say what is about to happen: *Gemma is given two declared function
   schemas and decides which computation to run.* It takes 30–60 seconds because
   these are real round trips.
4. While it runs, show the map. Note the label: **`Geospatial twin · fixture road
   graph`** — the road network is bundled data and the interface says so.
5. Click any incident to focus it.

## Stage 2 · Gemma evidence — the Gemma stage

6. Switch to **Gemma evidence**.
7. Show the grounded extraction. Every non-null value cites the evidence IDs
   behind it.
8. Point at **`medical_urgency: UNKNOWN`**. No source mentioned injuries, so the
   model reports the gap instead of interpolating a plausible number. It renders
   with a dash *and a hatch texture*, so unknown is distinguishable from zero
   without relying on colour.
9. Show the contradiction: heavy vehicles cannot pass, motorcycles can. **Both
   claims are kept, with citations.** The system does not pick a winner.
10. Scroll to **Model-orchestrated optimization**. This is the track requirement,
    made visible: the calls Gemma emitted, the **raw arguments the model
    produced**, whether they passed validation, and what the engine executed.
11. The line to land: Gemma called `run_optimization` with
    **`blocked_edge_ids: []`** and a rationale saying the corridor is restricted
    but *not established as impassable*. It refused to delete a usable corridor
    on contradictory evidence. Deleting one can strand a district.
12. Scroll to **Model reasoning, verbatim** in the function-call panel and read a
    few lines aloud. This is the model's actual chain-of-thought from the
    function-calling turn — roughly 1,900 characters, unedited — including the
    step where it decides the road is *not* completely blocked.
13. Scroll to **Raw prompt, reasoning, and response**. Open *Prompt sent*: the
    literal ~8,200-character prompt. Open *Raw response*: the exact JSON,
    pre-validation. Note that on the extraction call the thought body comes back
    empty because the forced JSON mime type suppresses it — the panel says so
    instead of reconstructing anything.

## Stage 3 · Math lab — the mathematics

13. Switch to **Math lab**.
14. Route plan: assets, stops, distance, time. Distances are the raw sum of edge
    lengths, never the terrain-weighted search cost.
15. Convergence: a *normalized* residual on a log axis. Iterations with no
    normalized residual are omitted from the curve, not plotted as zero.
16. Diagnostics: say plainly that KKT here is a **feasibility diagnostic, not a
    proof of optimality** — three of the four conditions hold for any feasible
    allocation by construction, and the API ships
    `independently_proves_optimality: false`.
17. Press **Run comparison** in the baseline panel. Read the headline: closing
    `east_west_bharatpur_nepalgunj`, which carries every ground route, the naive
    planner keeps **0 of 5** drivable ground routes — all five trucks routed
    through a road nobody can drive. RakshyaNet keeps **5 of 5**, for 8% more
    distance. Say why the denominator is five: the four aircraft cannot be
    affected by a road closure, so counting them would report 4 of 9 against
    9 of 9 and flatter us.
18. Then read the measured limitation out loud: terrain weighting alone changes
    **no path** on this network. The advantage is entirely closure-aware
    re-planning. Volunteering this is the point — the brief penalises inflated
    baselines.

## Stage 4 · Review & authorize — the human gate

19. Switch to **Review & authorize**.
20. Show the **route manifest**: every asset with its feasible/excluded state. An
    operator authorizes a plan, not an integer.
21. Show the approval scope, which states that approval does not dispatch
    vehicles.
22. If the server has blocking reasons they appear verbatim under **Server
    refuses approval**, and the server re-checks them on submit.
23. Press **Approve for coordination**. The **decision receipt** appears: who
    decided, when, and their note.
24. Return to **Operations**. The **mission clock** now exists. Drag it to T+45 and then
    T+120: vehicles sit where the solver's ETAs put them, and served stops are
    counted separately from pending ones. Press **Reset** to return to the depot.

---

## If something goes wrong

- **A model call fails.** The offline provider answers and the interface says
  `Declared fallback` with a reason. The rest of the demo still runs.
- **Approval refused as stale.** A newer run supersedes it. After a *rejection*
  there is no earlier plan to fall back on — press **Re-run pipeline**. A *failed*
  run does not block the good run before it.
- **Terrain tiles do not load.** The map falls back to a schematic view and
  reports `data-terrain-status="fallback"`.

## What not to claim

- Do not call the allocator a Nash equilibrium. It is capped proportional
  allocation.
- Do not call the KKT panel a proof of optimality.
- Do not describe the road graph, evidence, or scenarios as live feeds.
- Do not quote `welfare_improvement_percent`. It compares a vehicle-constrained
  allocation against an unconstrained one and is labelled incomparable in the
  API response.
