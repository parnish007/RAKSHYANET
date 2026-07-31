# Architecture — imagery verification as a Gemma tool

Companion to `docs/PLAN-satellite-tool.md` (build order, disk, isolation) and
`docs/research-satellite-vision.md` (why this shape and not direct vision).

This document is the *integration design*: who decides to call the tool, how the
model is steered into calling it at the right moment, what comes back, and what
the result is forbidden from causing.

---

## 1. The three ways a check can start

| # | Path | Who decides | Function call recorded? |
|---|---|---|---|
| **A** | **Model-initiated** | Gemma, from the evidence | Yes — Gemma emits it |
| **B** | **Operator-forced** | A human presses a button | Yes — Gemma still emits it, told to |
| **C** | ~~Background polling~~ | — | **Deliberately absent** |

**Why C is absent.** A scheduled sweep that checks every corridor would produce
imagery records nobody asked for, attached to no claim, with no reason. Every
record in this system exists because something prompted it, and that reason is
part of the audit trail. A tool that fires on a timer has no reason to cite.

---

## 2. Path A — Gemma decides

The steering is entirely in the declaration description and the system prompt.
There is no classifier-of-when-to-call; the model reads the evidence and judges.

### 2.1 When it SHOULD call — the four triggers

**T1 · Uncorroborated blockage claim.** One source says a corridor is flooded or
landslide-blocked and no independent source confirms it.

**T2 · Contradiction about passability.** Two sources disagree — the case slide 2
of the deck is built on. Imagery is a third, independent read.

**T3 · Forecast-implied risk with no observation yet.** *This is the one you
asked for.* A weather advisory says heavy rain and saturated slopes; nobody has
yet reported a blockage. Gemma checks the corridor **before** a report arrives.

> Your existing fixture already contains this trigger verbatim —
> `weather-dhm-003`: *"Continued rainfall is likely across eastern Nepal during
> the next six hours. Saturated slopes…"* No new demo data needed.

**T4 · Operator directive.** Path B, below.

### 2.2 When it must NOT call — the anti-triggers

Stated explicitly, because a model that calls a tool on every turn is a model
that isn't reasoning:

- Two or more independent sources already agree — corroboration adds nothing.
- The incident is not flood or landslide (a fuel shortage has no surface signature).
- The corridor id isn't in the graph.
- It has already checked that corridor this turn.

### 2.3 T3 is a different kind of claim, and the record must say so

A check triggered by a *forecast* is answering "is there anything visible yet?",
not "is this report true?". Those must not produce the same record. The tool
therefore carries a `trigger_reason`, and it changes the record's own text:

| `trigger_reason` | Meaning | Extra caveat written into `text` |
|---|---|---|
| `corroboration` | T1/T2 — a report exists | "This check was prompted by a field report claiming …" |
| `anticipatory` | T3 — forecast only | **"No field report claims this corridor is affected. This check was prompted by a rainfall advisory and is precautionary."** |
| `operator_request` | T4 | "A named operator requested this check." |

Without that distinction, an anticipatory check that happens to see water reads
downstream exactly like a confirmed report. The caveat is inside `text`, so it
travels with the citation into Gemma's summary — the honesty is structural, not
a matter of prompt discipline.

### 2.4 A negative result is a result

If the classifier returns `Highway`/`Residential` — normal surface — that is
**informative and must be recorded**, not discarded. It *weakens* an
uncorroborated claim. The prompt says so explicitly, otherwise the model learns
to ignore inconvenient answers.

---

## 3. The prompt work

### 3.1 Function declaration (appended to `FUNCTION_DECLARATIONS`)

```python
{
  "name": "verify_report_with_imagery",
  "description": (
    "Request an independent overhead-imagery read of one corridor. A local "
    "land-cover classifier examines a satellite tile covering that corridor "
    "and reports what the surface currently classifies as. Returns an evidence "
    "record you must cite.\n"
    "CALL THIS WHEN:\n"
    "(1) a report claims a corridor is flooded or landslide-blocked and no "
    "independent source confirms it;\n"
    "(2) two sources disagree about whether a corridor is passable;\n"
    "(3) a weather advisory or forecast indicates heavy rainfall, saturated "
    "slopes, or elevated landslide or flood risk affecting a corridor, even if "
    "nobody has yet reported a blockage there — a precautionary check is "
    "appropriate and you should set trigger_reason to 'anticipatory';\n"
    "(4) the operator directive in this turn instructs you to.\n"
    "DO NOT CALL IT when two or more independent sources already agree, when "
    "the incident has no visible surface signature, or for a corridor you have "
    "already checked this turn.\n"
    "CRITICAL LIMIT: imagery observes SURFACE CONDITIONS ONLY. It cannot "
    "measure water depth, cannot see under cloud or tree canopy, and CANNOT "
    "establish that a corridor is impassable. A positive result raises "
    "confidence that an event occurred. It is never sufficient on its own to "
    "place a corridor in blocked_edge_ids. A negative result is also "
    "informative: it weakens an uncorroborated claim, and you must report that "
    "rather than ignore it."
  ),
  "parameters": {
    "type": "object",
    "properties": {
      "corridor_id":    {"type": "string",
                         "description": "Exact id returned by list_corridor_status."},
      "incident_type":  {"type": "string", "enum": ["flood", "landslide"]},
      "evidence_id":    {"type": "string",
                         "description": "The record whose claim you are testing. "
                                        "For an anticipatory check, the advisory "
                                        "that prompted it."},
      "trigger_reason": {"type": "string",
                         "enum": ["corroboration", "anticipatory", "operator_request"]},
      "rationale":      {"type": "string",
                         "description": "One sentence: which evidence prompted "
                                        "this check. Describe evidence only."}
    },
    "required": ["corridor_id", "incident_type", "evidence_id", "trigger_reason"]
  }
}
```

### 3.2 System-prompt additions

Inserted into `_system_prompt()` under **HOW TO WORK**, keeping the existing voice:

```
- If a source claims a corridor is flooded or blocked and nothing independent
  confirms it, or if two sources disagree about passability, call
  verify_report_with_imagery before run_optimization.
- If a weather advisory reports heavy rain, saturated slopes, or elevated
  landslide risk over an area, you may check a corridor in that area even
  though no blockage has been reported. Set trigger_reason to 'anticipatory'
  and say in the rationale that no report claims a blockage.
- Imagery corroborates; it never establishes closure. A corridor whose only
  supporting evidence is an imagery record stays OPEN and out of
  blocked_edge_ids. Say so in your rationale.
- If the imagery shows nothing unusual, report that. A negative result weakens
  an uncorroborated claim and must not be omitted.
```

And under **AUTHORITY BOUNDARY**:

```
- You may not treat an imagery observation as a field observation. It has no
  witness, no depth measurement, and no ground contact.
```

---

## 4. Path B — the human forces a check

Two sub-cases, because the operator may or may not want to wait for a full model
turn.

### B1 · Directive (default) — Gemma still makes the call

The operator presses **"Ask Gemma to verify this corridor"**. The next
orchestration turn carries an operator directive appended to the user content:

```
OPERATOR DIRECTIVE: A named operator has requested imagery verification of
corridor <id> in connection with evidence <evidence_id>. Call
verify_report_with_imagery for that corridor with trigger_reason
'operator_request' before proposing an optimization.
```

Gemma emits the call, so the function-call panel shows a real model-emitted
tool call. The `ToolCallRecord` records `initiated_by: "operator"` so the audit
distinguishes it from a call the model chose. **This is the good demo beat:**
*"I can also force it — and notice the record says the operator asked, not the
model."*

**Guarantee if the model ignores the directive:** if a forced turn completes
without the call, the backend performs the check deterministically and appends
the record with `provider: operator_direct`, `initiated_by: "operator"`,
`model_complied: false`. The button always does something, and the record is
honest about how it happened.

### B2 · Direct (escape hatch)

**"Check imagery now"** skips the model entirely — one HTTP call to the sidecar,
record appended, no Gemma round trip. ~1 s instead of ~40 s. This exists so the
feature is demonstrable even if the model call is failing on the day.

---

## 5. Sequence

```
OPERATOR                BACKEND :8000              GEMMA                SIDECAR :8011
    │                        │                       │                       │
    │ Run pipeline ─────────▶│                       │                       │
    │                        │─ analysis + tools ───▶│                       │
    │                        │                       │ reads weather-dhm-003 │
    │                        │◀── functionCall ──────│ "saturated slopes"    │
    │                        │    list_corridor_status                       │
    │                        │─ corridors ──────────▶│                       │
    │                        │◀── functionCall ──────│ decides: precautionary│
    │                        │  verify_report_with_imagery(                  │
    │                        │    corridor, landslide, trigger=anticipatory) │
    │                        │                                               │
    │                   ┌────┴─ VALIDATE ARGS ─────────────────────┐         │
    │                   │ corridor ∈ terrain_graph?                │         │
    │                   │ evidence_id ∈ this analysis?             │         │
    │                   │ incident_type ∈ enum?                    │         │
    │                   └────┬─ reject → back to model, unexecuted─┘         │
    │                        │                                               │
    │                        │──────── POST /classify (3s, 1 try) ──────────▶│
    │                        │◀─────── label, confidence, model_id ──────────│
    │                        │                                               │
    │                        │  build EvidenceRecord (reliability ≤ 0.55,    │
    │                        │  caveats inside text, tier + trigger labelled)│
    │                        │─ functionResponse ───▶│                       │
    │                        │◀── functionCall ──────│ run_optimization(     │
    │                        │                       │   blocked_edge_ids=[])│
    │                   ┌────┴─ IMAGERY-ONLY CLOSURE GUARD ───────┐          │
    │                   │ any blocked id supported ONLY by an     │          │
    │                   │ imagery record? → reject, return reason │          │
    │                   └────┬───────────────────────────────────-┘          │
    │◀─ plan awaiting approval                                               │
```

---

## 6. The guard that makes this safe

Everything above is steering — prompt text the model may ignore. One rule is
enforced in code and cannot be talked around:

**A corridor whose only supporting evidence is an imagery record cannot enter
`blocked_edge_ids`.**

Implemented in `validate_run_optimization_arguments`: for each blocked corridor,
gather the evidence records that mention it; if every one of them has
`source_category == "overhead_imagery_analysis"`, reject the call and return the
reason to the model. It gets one chance to re-plan without that corridor.

This is the property to state to the geospatial judge: *"the system physically
cannot close a road on satellite imagery alone — not as policy, as validation."*
It is also directly testable, so it goes in the test suite as the headline case.

---

## 7. Where the record lands

The returned `EvidenceRecord` is appended to the analysis's evidence list, which
means it inherits everything already built:

- **Citable** — Gemma references it by `evidence_id`; `_validate_grounding` checks
  it like any other record.
- **Substring check works unchanged** — `text` contains the literal word
  `flood`/`landslide`, so `_validate_incident_type_grounding` passes without
  modification.
- **Confidence bounded** — `reliability` capped at **0.55**, so `_system_confidence`
  cannot be inflated by an automated corroboration.
- **Visible** — it appears in the evidence ledger and in the source-report dialog
  built this session, so clicking the citation shows the tile, the label, the
  confidence, the model id, and the tier badge.

Frontend surfaces, all reusing existing components:

| Surface | Addition |
|---|---|
| Function-call panel | Appears free — already renders any tool call |
| Source-report dialog | Tile thumbnail, label, confidence, model id, tier badge |
| Evidence ledger | Row with an `imagery` chip |
| Corridor / incident inspector | The two buttons from §4 |

---

## 8. Failure isolation, restated for this flow

Nothing in §5 can break the pipeline:

- Sidecar down/slow/OOM → 3 s timeout → **tier 2** precomputed record → the loop
  continues, and the record says it was precomputed.
- Fixture also missing → **tier 3** `imagery_check_unavailable`, whose `text`
  states no corroboration was obtained. Gemma cites it and correctly concludes
  nothing was confirmed.
- `SATELLITE_TOOL_ENABLED=false` → the declaration is never sent; the system is
  byte-identical to today.
- Model emits malformed args → rejected by validation, returned unexecuted —
  the mechanism that already exists for `run_optimization`.

**`MAX_TOOL_TURNS` rises 4 → 6.** The chain is now
`list_corridor_status → verify_report_with_imagery → run_optimization` = 3, and a
single rejected-argument retry would exhaust the old budget of 4.

---

## 9. Files

| File | Change |
|---|---|
| `tools/imagery_sidecar/service.py` | **new** — FastAPI on :8011, model loaded and pre-warmed at startup |
| `tools/imagery_sidecar/requirements.txt` | **new** — torch+cu126, transformers, pillow *(never enters the main requirements)* |
| `backend/services/imagery_verifier.py` | **new** — three-tier client, builds the `EvidenceRecord`. The only module that knows the sidecar exists |
| `backend/data/imagery/manifest.json` | **new** — tile ↔ corridor binding, acquisition metadata |
| `backend/data/imagery/*.jpg` | **new** — ~8 real Sentinel-2 tiles |
| `backend/services/gemma_orchestrator.py` | declaration (flag-gated), validation, closure guard, prompt text, `MAX_TOOL_TURNS` |
| `backend/api/optimization_routes.py` | operator-directive + direct-check endpoints |
| `backend/models/orchestration.py` | `initiated_by`, `model_complied` on `ToolCallRecord` |
| `frontend/src/PremiumApp.jsx` | two buttons, tier badge, tile in the source dialog |
| `backend/tests/test_imagery_verifier.py` | **new** — three tiers, arg rejection, **the imagery-only closure guard** |

---

## 10. The demo, as it will actually run

> "There's no report of a blockage here. But there *is* a DHM advisory —
> continued rainfall, saturated slopes. Watch what Gemma does with that."
>
> *(function-call panel)* "It called `verify_report_with_imagery` on its own,
> with `trigger_reason: anticipatory`. Nobody asked it to. It reasoned from a
> forecast that a corridor might be affected and went to check **before** anyone
> reported anything."
>
> "That ran a real EuroSAT-finetuned classifier, locally on this laptop's GPU,
> against a real Sentinel-2 tile. Here's the tile, here's the label."
>
> *(click the citation)* "And read the record's own text: *imagery observes
> surface conditions, not passability.* Gemma cites it, raises its confidence
> that something is happening — and still passes an **empty** closure list."
>
> "In fact it couldn't have closed it. If it tried to block a corridor whose
> only support was imagery, validation rejects the call before the engine sees
> it. That's not policy — it's a test in our suite."
>
> "And I can force it too." *(press the button)* "Same call, but the record now
> says the operator asked, not the model."

Four things a judge scores, in ninety seconds: autonomous tool use, anticipatory
reasoning from a forecast, a real local model on real imagery, and a safety
property that is enforced rather than promised.
