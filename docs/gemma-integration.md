# Gemma integration — a code map

This document exists so the integration can be **read in the source** rather than
taken on trust from prose. Every claim below names the file and line that
implements it, and the test that locks it.

Model: `gemma-4-26b-a4b-it`, served on `generativelanguage.googleapis.com`.
Track: Route Intelligence, Build With Gemma: Margadarshan.

> Line numbers were correct at the time of writing. If one has drifted, the
> symbol name beside it is the durable reference — grep for that.

---

## The one-paragraph version

Gemma does the **entire interpretation layer**: it turns contradictory, untrusted
field reports into a validated, evidence-cited analysis, and then it **drives the
deterministic engine itself** through native function calling — choosing which
computation to run and on what world state. It cannot allocate, route, approve or
dispatch. Its total influence on the plan is one bounded scalar of at most `1.0`,
against a deterministic survival penalty of `10.0`. A named human authorizes the
result.

Remove Gemma and there is no input at all: the engine consumes structured risk
signals, and nothing but a language model produces those from three reports
written by people who never spoke to each other.

---

## 1 · Native function calling — the track requirement

| What | Where |
|---|---|
| The declared function schemas sent to the model | `backend/services/gemma_orchestrator.py:137` — `_BASE_FUNCTION_DECLARATIONS`, resolved per call by `function_declarations()` at `:209` so the conditional third tool appears only when its flag is on |
| The multi-turn `functionCall` / `functionResponse` loop | `gemma_orchestrator.py:673` — `plan()`; response assembly at `:841` |
| Turn budget | `gemma_orchestrator.py:66` — `MAX_TOOL_TURNS` |
| The system prompt that steers tool choice | `gemma_orchestrator.py:585` — `_system_prompt()` |
| Inspect the declared contract live | `GET /api/optimization/tools` |

Three functions are declared:

- **`list_corridor_status()`** — called first when evidence mentions road access,
  so any corridor the model names provably exists in the graph.
- **`verify_report_with_imagery(...)`** — **conditional.** The model decides
  whether corroboration is needed. Behind `SATELLITE_TOOL_ENABLED`, off by default.
- **`run_optimization(...)`** — the deterministic engine.

**The safety property is not "the model cannot call anything."** It is that every
argument is checked against the world before the engine sees it:

`gemma_orchestrator.py:367` — `validate_run_optimization_arguments()`

- `analysis_id` must be the analysis this turn opened with — it cannot retarget a
  different evidence set.
- Every corridor in `blocked_edge_ids` must exist in `terrain_graph.json` — it
  cannot invent a road closure.
- Elapsed mission time must fall inside a bounded horizon.
- The free-text rationale is screened with the same operational-authority filter
  used on extraction output.

A rejected argument is returned **to the model**, never to the engine.

*Locked by* `backend/tests/test_baseline_and_orchestration.py:101` —
`test_a_model_invented_corridor_cannot_close_a_road`, and `:94` —
`test_invalid_arguments_never_reach_the_engine`.

---

## 2 · Grounded extraction — the model may not invent facts

`backend/services/gemma_service.py`

| Guard | Where |
|---|---|
| Every non-null field must cite evidence IDs that exist | `:299` — `_validate_grounding()` |
| Categorical values must appear as a **literal substring** of the cited record's text | `:248` — `_validate_incident_type_grounding()` |
| Output and tool rationales screened for allocation / dispatch / approval language | `:72` — `_OPERATIONAL_AUTHORITY_PATTERNS` |
| Hosted call timeout | `:436` — `GEMMA_TIMEOUT_SECONDS`, default `45` |

Population figures are rejected unless the number appears literally in a cited
report, or is the exact midpoint of two stated bounds. A reported contradiction
must share at least 60% of its content tokens with the record it cites.

**UNKNOWN is a first-class output.** A field the evidence does not support returns
`null` with confidence `0` and no citations — never zero, never a plausible guess.
The schema enforces that pairing rather than trusting the prompt.

*Locked by* `backend/tests/test_gemma_safety.py:96` —
`test_unknown_fields_must_use_null_zero_confidence_and_no_citation`, `:121` —
`test_model_cannot_downplay_unknown_or_contradictory_evidence`, `:129` —
`test_grounding_rejects_unknown_evidence_reference`.

---

## 3 · The authority boundary, in arithmetic

`backend/services/optimization_service.py:254` — `_apply_gemma_signal()`

```
B = max(supported severity, medical urgency, accessibility risk) × system_confidence
```

- Bounded to `[0, 1]`. UNKNOWN fields are **excluded from the max**, not read as
  zero.
- Applied only to villages whose name or id appears in the evidence text, so a
  valid score with no location match does not silently reweight every village.
- Compare with the deterministic critical-shortage penalty of `10.0`. Gemma's
  ceiling is one tenth of a single deterministic trigger.

System confidence is calibrated deterministically from source reliability,
source diversity, evidence freshness, contradiction count and missing-information
count. **It does not reuse the model's own confidence as operational confidence** —
the two are computed separately and displayed separately.

Gemma affects **ranking only**. It does not choose vehicles, compute routes,
allocate stock, or approve anything.

---

## 4 · Imagery verification — conditional, bounded, and unable to close a road

`backend/services/imagery_verifier.py:449` — `verify_corridor()`

Returns an `EvidenceRecord` that flows through the **same** validators as any
field report. It never raises: every failure is a labelled tier
(`local_model_inference` → `bundled_imagery_fixture` → `imagery_check_unavailable`),
and `reliability` is capped at `0.55` so an automated corroboration cannot inflate
system confidence. The record's own `text` carries its limitations, so the caveat
travels with the citation instead of depending on prompt discipline.

**The guard that matters** — `gemma_orchestrator.py:345` —
`_reject_imagery_only_closures()`:

A corridor whose only supporting evidence has
`source_category == "overhead_imagery_analysis"` **cannot enter
`blocked_edge_ids`.** The call is rejected and the reason returned to the model so
it can re-plan. Enforced in validation, not requested in a prompt.

*Locked by* `backend/tests/test_imagery_verifier.py:423` —
`test_a_corridor_supported_only_by_imagery_cannot_be_closed`, and `:436` —
`test_the_same_corridor_closes_once_a_field_report_corroborates_it`.

---

## 5 · Provider posture, stated honestly

Hosted Gemma is the default. When the hosted call fails, is malformed, or is not
grounded in known evidence IDs, the service records **why** and falls back to a
deterministic path — and the interface labels that state `Declared fallback`
rather than presenting it as a live model result.

Multi-key failover with per-key cooldowns lives in
`backend/services/api_key_pool.py`, with a wall-clock deadline so a slow upstream
cannot multiply one timeout across retries.

`GET /api/gemma/status` exposes the active provider, the last provider error and
per-key health, and never returns key material.

The model's chain-of-thought is captured and displayed verbatim where the provider
returns it. On the extraction call it comes back empty, because forcing a JSON
response mime type suppresses it — the interface says so rather than inventing
reasoning.

---

## What this integration does **not** claim

- **Not a Nash equilibrium.** The allocator is capped proportional allocation; the
  filename `nash_solver.py` is historical.
- **Not a proof of optimality.** Capacitated vehicle routing is **NP-hard**, and it
  is solved here with a documented greedy heuristic. The KKT panel is a
  feasibility diagnostic on the *continuous* allocation only — three of its four
  conditions hold by construction for any feasible allocation, and the API ships
  `independently_proves_optimality: false`.
- **Not live data.** The road graph, evidence, scenarios and imagery tiles are
  bundled fixtures, labelled as such on every screen.
- **Not validated reasoning.** Displayed chain-of-thought is not citation-checked
  and is never treated as a finding. Only validated, cited fields cross into
  application state.
