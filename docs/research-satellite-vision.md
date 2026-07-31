# Satellite imagery + vision for RakshyaNet — decision document

**Written:** 2026-07-30, ~12 hours before judging.
**Question asked:** should we integrate a Hugging Face image classifier for disaster detection from satellite data, or does Gemma have native vision? What integration shape maximises Innovation without breaking the grounding guarantees?

**Reading key used throughout:**

- **VERIFIED** — I fetched the source this session; URL given.
- **INFER** — domain knowledge, not fetched this session. Treat as a claim to check before saying it to a judge. I have flagged where the risk of being wrong is material.

---

## RECOMMENDATION

### The finding that reframes the question

**Gemma 4 has native vision, and `gemma-4-26b-a4b-it` — the exact model this project already calls — accepts image input on the exact endpoint we already use.** (VERIFIED, see §1.) The premise "we need an external classifier because Gemma cannot see" is false.

That does *not* mean option (b) "just show Gemma the satellite image" is the right call. It means the decision is now about **grounding**, not capability. See §3 — there is a concrete, code-level reason direct image input breaks the citation contract that makes this system defensible.

### Do tonight (in this order, stop when tired)

**1. Section 5 of this document — the limits table. ~45 min. Zero code, zero network, zero risk.**

Put the geospatial limits into the deck as a named slide (suggest: "What imagery can and cannot tell us"). A NASA geospatial reasoning scientist will probe these. **Naming the limits before the judge does converts your biggest vulnerability into your strongest credibility signal.** This is the single highest-value item in this document and it is the cheapest. Do it first, in case everything else gets abandoned.

**2. Declare `verify_report_with_imagery` as a third function declaration, returning a bundled, honestly-labelled evidence record. ~2–4 hours.**

This is the shippable innovation. Crucially, **the codebase already has the exact precedent**: `corridor_status_payload()` in `backend/services/gemma_orchestrator.py` returns `"source": "bundled_terrain_fixture"`, with the comment *"the model is allowed to reason about corridors but is never allowed to believe it has a live feed."* A satellite-verification tool that returns a bundled record is the same pattern, one layer out. It is not a mock pretending to be live — it is the architecture the project already committed to.

What it buys you: **Gemma autonomously decides that a text report needs independent corroboration and calls for it.** That is a real agentic behaviour, demonstrable live, and it is exactly what the Route Intelligence "native function-calling" clause rewards. Full signature in §3.

**3. Optional flourish if and only if 1 and 2 are done and you still have energy: a direct-vision capability probe. ~45–90 min.**

One screenshot of `gemma-4-26b-a4b-it` describing a real flood/landslide scene, shown as a *separate* slide labelled "native multimodal capability, not yet in the grounded pipeline." Do **not** wire it into the analysis path tonight (§3 explains why, §4 explains the demo risk).

### Present as roadmap, not as shipped

- Live Sentinel-1 SAR acquisition and inference (Prithvi-EO / Sen1Floods11-class model). Real, but a multi-day geospatial-stack job, not an overnight one (§4).
- Direct multimodal grounding — passing tiles to Gemma *with* a provenance record that makes the citation auditable (§3 sketches what that would require).

### Do NOT do tonight

**Do not attempt to download and run a geospatial foundation model.** §4 is blunt about why. The failure mode is not "the model underperforms", it is "you spend six hours on GDAL/rasterio/TerraTorch dependency resolution on Windows and ship nothing." An unshippable idea presented as shipped is worse than an honest roadmap slide.

---

## §1 — Does Gemma have native vision? (VERIFIED)

**Yes, and specifically for our model.**

| Gemma 4 variant | Supported input modalities |
|---|---|
| E2B (2.3B effective) | Text, Image, Audio |
| E4B (4.5B effective) | Text, Image, Audio |
| 12B Unified | Text, Image, Audio |
| 31B Dense | Text, Image |
| **26B A4B MoE (3.8B active)** | **Text, Image** (no audio) |

Source: [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) (VERIFIED). Confirmed independently on the [HF repo](https://huggingface.co/google/gemma-4-26B-A4B-it) (VERIFIED): 25.2B total / 3.8B active params, 8 of 128 experts, **~550M vision encoder**, hybrid local-sliding-window + global attention, 256K context, Apache 2.0. Audio is explicitly *not* available on 26B A4B — do not claim it.

**Served on our endpoint.** Google's [Run Gemma with the Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) page (VERIFIED) states exactly two Gemma 4 ids are served on `generativelanguage.googleapis.com`: `gemma-4-31b-it` and **`gemma-4-26b-a4b-it`** — the id hardcoded as the default in `backend/services/gemma_orchestrator.py` (`GEMMA_MODEL`, line 302). Same page confirms image understanding and, separately, function calling ("Define tools as function declarations. The model decides when to call them").

**API shape.** Two documented forms:

```jsonc
// Files API reference (documented on the Gemma-on-Gemini-API page)
{"file_data": {"mime_type": "image/jpeg", "file_uri": "<uri>"}}

// Inline bytes — SDK form types.Part.from_bytes(data=..., mime_type="image/png"),
// which on the wire is:
{"inlineData": {"mimeType": "image/png", "data": "<base64>"}}
```

The inline form is shown via the Python SDK in [philschmid's Gemma 4 + Gemini API writeup](https://www.philschmid.de/gemma-4-gemini-api) (VERIFIED). Our `_post()` builds raw REST dicts, so we would use `inlineData` camelCase, matching the existing `functionCall` / `functionResponse` camelCase convention in the orchestrator.

Also documented (VERIFIED, model card / overview): variable image resolution via a **configurable visual token budget** — supported budgets 70, 140, 280, 560, 1120 tokens per image. Higher budget preserves fine detail at more compute. Interleaved text+image in any order is supported; guidance is to **place image content before text** in the prompt.

### ⚠ The one thing I could NOT verify

**No source I fetched demonstrates image parts and `functionDeclarations` in the same request.** The docs cover vision and function calling in separate sections, with no combined example. This matters directly: our orchestrator sends `tools: [{functionDeclarations: [...]}]` on every call. Whether we can add an `inlineData` part to that same payload is **unknown**.

It may work fine. It may 400. It may silently degrade function-calling reliability. **Test it against a real key before anyone promises it on stage.** Treated as a demo risk in §4.

---

## §5 — The geospatial-rigour trap *(highest-value section; ordered first by importance)*

A geospatial reasoning scientist with a NASA background will not be impressed by "we run a model on satellite imagery." They will immediately probe whether you understand what the sensor can physically resolve, how often, and through what. Below: the questions, and an honest answer to each.

**The framing that wins the room:** *"We deliberately did not claim satellite verification as a live capability, because in the Nepal monsoon the sensor that can see through cloud is the one that struggles most in our terrain. Here is the analysis that led us to scope it as roadmap."* That answer is stronger than any working demo, because it demonstrates the reasoning the judge is actually assessing.

---

### Q1. "What's your revisit time, and what does that mean during a landslide event?"

**Honest answer: Sentinel-2 optical is ~5-day revisit at best, and effectively unusable during the Nepal monsoon.**

- Sentinel-2 is a 2-satellite constellation giving ~5-day revisit at the equator; higher latitudes get more frequent overlap from swath sidelap. (INFER — canonical reference: [ESA Sentinel-2 mission](https://sentiwiki.copernicus.eu/web/s2-mission), not fetched this session.)
- **The killer:** Nepal's monsoon (June–September) delivers the large majority of annual precipitation, and monsoon rainfall is precisely the trigger for the landslides RakshyaNet exists to route around. During monsoon, the mid-hills are under persistent cloud. A nominal 5-day revisit becomes an *effective* revisit of clear-sky observation that can stretch to weeks over a specific valley. (INFER, but this is well-established and low-risk to assert.)
- So: **the optical sensor is blind exactly when the disaster happens.** A disaster-response system built on Sentinel-2 optical would deliver its first usable image well after the response window closed.

Say this yourself. It is the single most likely question and the one where volunteering the limitation buys the most credit.

---

### Q2. "So use SAR. Sentinel-1 sees through cloud. Why isn't that your answer?"

**Honest answer: SAR is the right instrument for monsoon flood mapping, and it is the *worst* instrument for our specific terrain. Both are true and we should say both.**

What SAR gives you:
- C-band (~5.4 GHz) penetrates cloud and works day/night — it genuinely solves Q1's problem. (INFER, well-established.)
- Open water is a specular reflector: it scatters energy away from the sensor and appears dark. Flood mapping is fundamentally a low-backscatter threshold. This works well on floodplains.

What it costs us — **and this is severe for Nepal specifically:**

- **Layover and foreshortening.** SAR is side-looking and orders returns by range, not by ground position. Where terrain slope approaches or exceeds the incidence angle, the top of a slope is imaged *before* its base — the signal collapses into the same range bin and the geometry is unrecoverable. In the Nepal middle hills and High Himalaya, slopes routinely reach this regime. A meaningful fraction of any scene over our operating area is geometrically corrupted.
- **Radar shadow.** The far side of a ridge receives no illumination at all — zero signal, not weak signal. **Valley-floor roads are exactly what a ridge shadows.** The corridors RakshyaNet cares most about are disproportionately the pixels SAR cannot see. This is not a tuning problem; it is orbital geometry.
- **False positives on flood.** Any smooth surface is dark: dry sand and gravel river bars, tarmac, some bare soils. A naive threshold flags them as water.
- **False negatives on flood.** Wind roughens the water surface and raises backscatter — flooding disappears. Flooding *under vegetation canopy* is largely invisible to C-band (L-band penetrates better; Sentinel-1 is C-band). Flooding in built-up areas produces double-bounce *brightening*, not darkening, so simple thresholding systematically misses urban flood.
- **Revisit.** ⚠ **Verify before quoting a number.** Sentinel-1A+1B gave 6-day repeat until 1B failed in December 2021, leaving ~12-day. Sentinel-1C launched December 2024 to restore ~6-day. I am reasonably confident of this history but the *current* 2026 constellation state (including whether Sentinel-1D is operational) is past what I can confirm. State it as "6–12 day depending on constellation state" or check [Copernicus](https://sentiwiki.copernicus.eu/web/s1-mission) first. (INFER, flagged.)
- **Landslides are not a single-scene SAR problem anyway.** SAR landslide detection generally relies on interferometric coherence loss between an image *pair* — that is a processing pipeline with orbit files, coregistration and a DEM, not a classifier you point at one image.

**Both halves matter.** "SAR solves cloud" alone sounds naive. "SAR solves cloud but layover and shadow are severe in Himalayan relief, and shadow preferentially hides valley-floor roads" is the answer of someone who has actually thought about Nepal.

---

### Q3. "What's your spatial resolution versus the thing you're trying to detect?"

**Honest answer: off by roughly an order of magnitude. This is the quiet dealbreaker.**

- Sentinel-2: 10 m for VNIR bands (B2/3/4/8), 20 m for red-edge and SWIR, 60 m atmospheric. Sentinel-1 IW GRD: ~10 m pixel spacing, ~20 m true resolution. (INFER, standard figures.)
- A debris deposit blocking a rural Nepali road might be **5–30 m along the road and a few metres wide**. At 10 m pixels that is **one to three pixels**, partially mixed with road, vegetation and hillside within each pixel. That is below reliable detection, and far below reliable *classification*.
- A landslide **scar** on an open hillside is a different target — tens to hundreds of metres, spectrally distinct (vegetation stripped to bare earth), and genuinely detectable at 10 m. **But a visible scar above a road does not establish that debris reached the road.** See Q5.
- What would actually resolve a road blockage: sub-metre commercial optical (Maxar, Planet SkySat) or UAV imagery. Worth naming as the honest answer to "what would you need": Planet Dove offers ~3–5 m near-daily, and Maxar's Open Data Program releases high-resolution imagery after major disasters. (INFER — [Maxar Open Data](https://www.maxar.com/open-data) is real and citable; confirm current terms.)

---

### Q4. "Talk me through your georeferencing. What CRS, what DEM, what's your error budget in high relief?"

**Honest answer — and there is a specific Nepal gotcha worth volunteering:**

- **CRS mismatch is the classic silent bug.** Sentinel-2 products are delivered as ~110 km tiles in per-tile UTM/WGS84 (MGRS). **Nepal spans UTM zones 44N and 45N.** Any national-scale raster mosaic crosses a zone boundary. Meanwhile our road graph is lat/lon. Overlaying a UTM 45N raster on WGS84 geographic coordinates without explicit reprojection produces a *plausible-looking* result with a systematic offset — no exception, no error, just wrong answers. Nepal's official national grid is a Modified UTM, which is a third thing again. (INFER, but the zone-44/45 split is a hard geographic fact and safe to state.)
- **Geolocation accuracy.** Sentinel-2 nominal geolocation is order ~10 m (improved by the Global Reference Image refinement). At 10 m pixels, that is **one pixel of positional uncertainty before you start** — comparable to the entire width of the feature in Q3. (INFER.)
- **DEM quality drives SAR geolocation, and our DEMs are worst where we operate.** Terrain-correcting SAR requires a DEM, and DEM height error translates into horizontal position error scaled by the viewing geometry — the error grows precisely in steep terrain. SRTM 30 m has well-documented **voids in high mountains** (steep slopes and radar shadow — the same physics as Q2, one instrument generation earlier). Copernicus DEM GLO-30 (TanDEM-X derived) is materially better in the Himalaya and is what we would use. (INFER; verify at [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM).)
- **The compounding point:** in high relief, DEM error → terrain-correction error → a road corridor lands on the wrong pixels. Our features are 1–3 pixels wide (Q3). The error budget and the target size are the same magnitude. That is not a system you can build an impassability claim on.

---

### Q5. "Your model finds inundation. How do you get from that to 'this road is impassable'?"

**Honest answer: you don't. This is the crux, and we should say it in exactly these words.**

> **Imagery tells you a valley is flooded. It does not tell you that a truck cannot pass.**

The inferential gap, spelled out:

- **Inundation extent ≠ impassability.** 20 cm of water across a road is passable by a 4×4 and impassable by a loaded truck. The sensor measures neither depth nor duration. Water *depth* is the operative variable and is not observed.
- **A landslide scar ≠ a blocked road.** The scar is on the slope. Whether the debris runout reached the carriageway, and whether it is still there, are separate facts. Detecting the scar is the easy half.
- **Time of observation ≠ time of decision.** The overpass is a snapshot. Between acquisition and dispatch, a crew may have cleared the road, or a second slide may have come down. The system's claim decays with a half-life the imagery cannot report.
- **Roads fail in ways nadir imagery cannot see.** Bridge scour and pier undermining — the deck looks perfectly intact from directly above while the structure is unsafe. Sub-surface washout, collapsed culverts, a road that is intact but whose approach is severed. **Bridge failure is a leading cause of route severance in Nepal and is close to undetectable from a 10 m nadir view.** (INFER on the "leading cause" framing — soften to "a significant cause" if challenged.)

**This is why RakshyaNet's architecture is right.** The system never lets any single signal — model, sensor or report — assert a road closure. Corridors are removed from the graph only when validated against `terrain_graph.json`, and the resulting plan still requires human approval. A satellite record enters as *one more cited, bounded, low-reliability evidence record*, not as ground truth. **The correct role for imagery here is corroboration and prioritisation, not determination.** Say that sentence.

---

### Q6. "What's your accuracy, on what test set, and was Nepal in it?"

**Honest answer: we have not validated on Nepal, and we should not claim a number we have not measured.**

- Published flood-segmentation metrics come overwhelmingly from **Sen1Floods11** — a hand-labelled benchmark of a small number of global flood events, dominated by low-relief floodplains. **Domain shift to confined Himalayan valleys is severe**, and there is no reason to expect benchmark IoU to transfer.
- Reporting an upstream paper's F1 as if it were our performance on Nepal would be **exactly** the kind of claim this judge is trained to catch. Don't.
- The defensible position: *"We report the benchmark provenance, we have not validated on Nepali terrain, and that validation — with ICIMOD or DHM ground truth — is the first item on the roadmap."* (ICIMOD is Kathmandu-based and the natural partner; naming a real, correct institution signals local grounding.)

---

### Q7. "Cloud masking — how do you handle partial cloud, and what about cloud shadow?"

**Honest answer, briefly:** Sentinel-2 L2A ships a scene classification layer and cloud-probability bands, but **cloud *shadow* is the harder problem** — shadowed vegetation darkens and can mimic water in optical indices, producing false flood detections adjacent to real clouds. In monsoon conditions the usable-pixel fraction over a specific valley is often near zero regardless. This reinforces Q1 rather than rescuing it. (INFER.)

---

## §3 — Integration shape (rewritten around native vision)

### The guarantee we must not break

From `backend/services/gemma_service.py` and `backend/models/gemma.py`, the system's defensibility rests on three properties:

1. **Every non-null value cites evidence.** `GroundedValue` / `GroundedScore` / `GroundedRange` all enforce: `value is not None` ⇒ `evidence_ids` non-empty (`validate_unknown_contract`).
2. **Citations are checked against evidence *text*, not merely present.** `_validate_grounding` checks `set(field.evidence_ids) <= available_ids`, and `_validate_incident_type_grounding` goes further:

   ```python
   cited_text = _referenced_text(evidence_by_id, incident_type.evidence_ids).lower()
   if incident_type.value.lower() not in cited_text:
       raise GemmaProviderError("Incident type is not supported by its cited evidence")
   ```

   A literal substring check against the cited record's `text`. Contradictions face a token-coverage threshold of 0.6.
3. **Gemma has no authority.** `_OPERATIONAL_AUTHORITY_PATTERNS` screens output and tool rationales; the orchestrator validates every argument against the road graph before execution.

Property 2 is the one that decides this question.

---

### Option (b) — pass imagery directly to `gemma-4-26b-a4b-it`

**Capability: confirmed available (§1). Grounding: breaks it.**

`_validate_incident_type_grounding` performs a substring match against `EvidenceRecord.text`. **An image has no `text`.** So if Gemma looks at a tile and concludes "flood", there are only two outcomes:

- It cites an existing text record — in which case the image contributed nothing auditable, and the citation is a fiction.
- We manufacture an `EvidenceRecord` for the image whose `text` is *Gemma's own description of the image*. Then Gemma cites text Gemma wrote. **The validator still passes — and it now certifies nothing.** The check becomes circular: the model grades its own perception. Every downstream guarantee ("everything is cited and bounded") silently becomes false while continuing to *look* true.

That second failure mode is the dangerous one, because the system keeps reporting green. To a judge who understands the architecture, it is the most damaging thing they could find.

There is a legitimate version of (b) — a provenance record with the tile id, acquisition time, sensor, CRS and footprint, where the model's reading is stored *separately* from the machine-generated provenance text, and only the provenance is citable. That is a real design. **It is not a 12-hour design.**

**Verdict: do not put direct image input into the grounded analysis path tonight.** Use it as a clearly-separated capability probe (§4).

---

### Option (a) — expose a classifier as a function declaration ✅ **RECOMMENDED**

**Grounding: preserved exactly, because the tool returns *text*.**

The classifier's output is rendered into a deterministic, machine-written `EvidenceRecord`. That record then flows through the **same** validator as every field report. Gemma cites it by `evidence_id`; the substring check operates on our text; the audit trail is intact. Nothing about the grounding contract needs to change — which is precisely why this is the low-risk option.

**Precedent already in the codebase:** `corridor_status_payload()` returns `"source": "bundled_terrain_fixture"` with the comment *"the model is allowed to reason about corridors but is never allowed to believe it has a live feed."* This is that pattern, one layer out.

**Proposed declaration** (append to `FUNCTION_DECLARATIONS` in `backend/services/gemma_orchestrator.py`):

```python
{
    "name": "verify_report_with_imagery",
    "description": (
        "Request independent corroboration of a field report from overhead "
        "imagery analysis for one corridor. Call this when a report claims a "
        "corridor is flooded or landslide-blocked and no second source "
        "confirms it, or when two reports contradict each other about "
        "passability. Returns an evidence record describing what the imagery "
        "shows. IMPORTANT: imagery observes surface conditions, NOT "
        "passability. A positive result raises confidence that an event "
        "occurred; it does NOT establish that a corridor is impassable, and "
        "must never be used alone to block a corridor."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "corridor_id": {
                "type": "string",
                "description": (
                    "Corridor to examine. Must be an exact id returned by "
                    "list_corridor_status."
                ),
            },
            "incident_type": {
                "type": "string",
                "enum": ["flood", "landslide"],
                "description": (
                    "Which surface signature to look for. Only these two are "
                    "supported by the imagery analysis."
                ),
            },
            "evidence_id": {
                "type": "string",
                "description": (
                    "The evidence record whose claim you are corroborating, so "
                    "the check is traceable to a specific report."
                ),
            },
        },
        "required": ["corridor_id", "incident_type", "evidence_id"],
    },
}
```

**Validation before execution** — mirroring `validate_run_optimization_arguments`:

- `corridor_id` must exist in `terrain_graph.json` (reuse `_load_corridors()`), so the model cannot invent a location to inspect;
- `evidence_id` must be one of the current analysis's records, so it cannot retarget a different evidence set;
- `incident_type` must be in the enum — anything else is rejected and returned to the model, not executed.

**Return payload** — shaped so it validates as an `EvidenceRecord` (note `reported_latitude` / `reported_longitude` already exist on the model):

```jsonc
{
  "evidence_id": "sat-s1-20260728-NPR07",
  "source_category": "overhead_imagery_analysis",
  "source_name": "Sentinel-1 IW GRD water-extent classification",
  "source_identifier": "S1A_IW_GRDH_1SDV_20260728T....",
  "retrieved_at": "2026-07-28T05:41:00Z",
  "freshness_minutes": 2880,
  "reliability": 0.55,
  "simulated": true,
  "provider": "bundled_imagery_fixture",
  "cache_status": "fixture",
  "reported_latitude": 27.xx, "reported_longitude": 85.xx,
  "text": "Automated SAR water-extent classification over corridor NPR-07 (Sentinel-1 IW GRD, ~20 m resolution, acquired 2026-07-28T05:41Z, 48 h before this analysis). Approximately 380 m of the corridor centreline intersects the inundated class. This is a flood surface-extent observation only: it does not measure water depth, does not establish that the corridor is impassable, and does not reflect any change since acquisition. Terrain shadow affects part of this scene."
}
```

Four deliberate choices, each defensible to the judge:

1. **`text` states its own limits.** Because Gemma cites this record, **the caveat travels with the citation** into the summary. The honesty is structural, not a matter of prompt discipline.
2. **The word `flood` appears in `text`**, so `_validate_incident_type_grounding`'s substring check works as designed — no validator changes needed.
3. **`reliability` capped low (≤ 0.6)**, so `_system_confidence` cannot be inflated by an automated corroboration.
4. **`freshness_minutes` is honest about acquisition latency** (§5 Q1/Q2) — the record carries its own staleness.

**Why this scores on Innovation:** the demonstrable behaviour is *Gemma noticing that a single unconfirmed report is insufficient and autonomously calling for independent corroboration before allowing an optimization to run.* That is agentic reasoning over a real safety property, not a wrapper around a classifier. It is also the honest version — the payload says `"simulated": true` and `"bundled_imagery_fixture"` on its face.

---

### Option (c) — deterministic pre-processing, results injected as evidence

Architecturally this is the **same destination** as (a) — an `EvidenceRecord` in the pipeline — but the *model never decides* the check was needed; a pipeline step always runs it.

Strictly safer, and strictly less interesting: it forfeits the agentic behaviour that Innovation and the Route Intelligence function-calling clause reward. Option (a) *is* option (c) with Gemma choosing when to invoke it, and the safety properties are identical because validation happens on the return path either way.

**Take (a).** Fall back to (c) only if the extra function declaration destabilises the function-calling loop (see §4).

---

### Comparison

| | (a) function declaration | (b) direct image to Gemma | (c) pre-processing step |
|---|---|---|---|
| Preserves citation contract | ✅ returns text, existing validators unchanged | ❌ circular grounding, validator silently certifies nothing | ✅ |
| Gemma retains no authority | ✅ args validated, result is bounded evidence | ⚠️ model's perception enters uncontested | ✅ |
| Defensible to geospatial judge | ✅ provenance + limits in the record | ❌ no sensor metadata, no CRS, no acquisition time | ✅ |
| Innovation value | **High** — model decides corroboration is needed | Moderate — impressive but architecturally shallow | Low — invisible plumbing |
| Ships in 12 h | ✅ ~2–4 h | ⚠️ probe only | ✅ ~1–2 h |

---

## §4 — Honest risk assessment (12 hours, one exhausted person)

### Risk register

**R1 — `inlineData` + `functionDeclarations` in one request is UNVERIFIED. (§1)**
No fetched source shows both together. Our orchestrator always sends `tools`. Unknown whether adding an image part 400s, or silently degrades function-calling reliability.
*Mitigation:* the recommended option (a) **never sends an image to Gemma at all** — it returns text. This risk applies only to the optional probe (step 3), which must therefore be a *separate script and a separate slide*, never a dependency of the main demo path.

**R2 — a third function declaration destabilises the loop.**
Real but modest. `MAX_TOOL_TURNS = 4`; adding a tool that the model may call before `run_optimization` consumes a turn. With `list_corridor_status` → `verify_report_with_imagery` → `run_optimization` you are at 3 of 4, and a rejected-argument retry exhausts it.
*Mitigation:* raise `MAX_TOOL_TURNS` to 5–6, and rehearse the exact demo scenario at least twice. **Budget 30 min for this.** Don't discover it live.

**R3 — live model download and inference. NOT SHIPPABLE TONIGHT.**
The geospatial Python stack (GDAL, rasterio, TerraTorch/MMSegmentation, correct band stacking order) is a notorious Windows install. Prithvi-class models additionally expect specific band subsets and normalisation. Realistic: **6+ hours with a real chance of zero output.** With judging tomorrow this is a bet you cannot cover. **Do not start it.**

**R4 — over-claiming.**
The most damaging failure available to this team. If a slide says "satellite verification" and the judge asks "which scene, acquired when, what CRS, what's your false-positive rate on Himalayan terrain" and the answer is a fixture — the whole project's credibility, including the parts that genuinely work, is retroactively suspect.
*Mitigation:* the payload carries `"simulated": true` and `"bundled_imagery_fixture"`, the UI surfaces it, and the slide says "fixture" out loud. This project already made that choice for terrain data; stay consistent.

### Time estimates (assume tired, assume 1.5× optimism error)

| Task | Estimate | Ships? |
|---|---|---|
| §5 limits table → deck slide | 45 min | ✅ do first |
| (a) declaration + validation + fixture record + one test | 2–4 h | ✅ recommended |
| Rehearse demo path twice | 30 min | ✅ non-negotiable |
| (b) direct-vision probe, standalone script + screenshot | 45–90 min | ⚠️ only if time |
| (c) fallback if (a) destabilises | 1–2 h | ✅ safety net |
| Live HF geospatial inference | 6 h+ | ❌ **no** |

### Credible mock vs live integration — the blunt version

**For this deadline, the bundled fixture is not merely acceptable, it is the *more* defensible artifact** — provided it is labelled.

A live integration tonight would necessarily be: one hand-picked scene, no cloud handling, no terrain correction, no validation on Nepali data, and an accuracy figure borrowed from someone else's benchmark. **That is a mock with extra steps and worse honesty**, because it *looks* live and therefore invites exactly the §5 questions it cannot answer.

The bundled record is the same information with accurate provenance attached.

**Strongest available upgrade if you find spare time:** run a real model **offline, once**, on one real scene, and bundle *that* output as the fixture. Then the record describes a genuine model run, the pipeline is honest about it being pre-computed, and you can show the input scene on a slide. This gets you most of the credibility of a live integration at a fraction of the risk. **But it is strictly step 4, after 1–3.**

### What a safe demo looks like

> "A single unconfirmed report claims the corridor is blocked. Watch — Gemma doesn't route on it. It calls `verify_report_with_imagery` first. The record it gets back is bundled fixture data, labelled as such, and it says in its own text that imagery cannot establish passability. Gemma cites it, raises confidence that an event occurred, and *still* leaves the corridor open because no cited evidence establishes closure. The plan goes to a human."

Every sentence is true, the agentic behaviour is real and live, and the limitation is stated before it can be asked.

### What would be a bluff

> "We integrate NASA's Prithvi geospatial foundation model to verify reports against live Sentinel-1 imagery."

If the weights were never loaded, this is false. If a judge with a NASA background asks one follow-up — and they will — it collapses, and it takes the credible parts of the project with it.

---

## §2 — Hugging Face model inventory (partial, roadmap only)

Deprioritised behind §5/§3/§4 and superseded in practice by §4/R3 — nothing here can be integrated live tonight. Verified enough to make a roadmap slide **safe to say out loud**.

### ⚠ The finding that matters most here

**`Prithvi-EO-1.0-100M-sen1floods11` takes Sentinel-2 *optical* input, not Sentinel-1 SAR** — despite "Sen1" in the name. (VERIFIED: the model card lists six bands — Blue, Green, Red, Narrow NIR, SWIR 1, SWIR 2 — which are Sentinel-2/HLS multispectral.) The Sen1Floods11 *dataset* contains both S1 and S2 imagery; this fine-tune uses the optical half.

**Consequence: the flagship NASA/IBM flood model is subject to the entire §5 Q1 cloud problem.** Pointing it at Nepal during monsoon means pointing an optical model at a cloud deck. The instrument that solves monsoon (SAR, §5 Q2) is *not* the instrument this model consumes.

This is worth saying to the judge unprompted — it demonstrates you read past the model name. It also means **"we'll use Prithvi" is not by itself a coherent monsoon answer**, and the roadmap slide should not imply it is. The honest roadmap is: SAR-based flood mapping for monsoon, with Prithvi-class optical models useful only in the post-monsoon and dry-season window.

### Verified repositories (`ibm-nasa-geospatial`, VERIFIED this session)

Real published weights, Apache 2.0.

| Repo id | Type | Task |
|---|---|---|
| `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` | base foundation, 300M | general EO feature extraction (HLS) |
| `ibm-nasa-geospatial/Prithvi-EO-2.0-600M` | base foundation, 600M | general EO feature extraction |
| `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` | finetune | **flood segmentation** |
| `ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11` | finetune, 100M | flood segmentation (v1) |
| `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` | finetune | burn-scar mapping |
| `ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification` | finetune | crop classification |
| `ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M` | base, 2.3B | weather/climate (MERRA-2) |

Transfer-learning variants also exist: `Prithvi-EO-2.0-tiny-TL`, `-100M-TL`, `-300M-TL`, `-600M-TL`.

**Details for the v1 flood finetune (VERIFIED):** trained on Sen1Floods11 — *"446 labeled 512x512 chips that span all 14 biomes, 357 ecoregions, and 6 continents of the world across 11 flood events"*; input 512×512, six bands, **GeoTIFF required**; needs **TerraTorch** (PyTorch + mmsegmentation); fine-tuned in ~1 h on a V100.

Two things follow directly:

- **446 labelled chips across 11 global flood events.** That is a small, low-relief-dominated benchmark. §5 Q6 stands: **no Nepal validation, do not quote a transferred metric.**
- **GeoTIFF + 6 correctly-stacked bands + TerraTorch** confirms R3. This is a geospatial data-engineering job before it is a modelling job. Not an overnight task.

### Not verified this session — check before citing

- **Landslide4Sense** — Sentinel-2 multispectral + DEM slope, on-topic for landslides. Competition entries are frequently code/paper-only; **confirm a real checkpoint exists** before naming it.
- **xView2 / xBD** — building damage from pre/post RGB pairs. Mostly GitHub rather than HF, and **the task is wrong for us**: building damage, not road passability, and it needs a matched pre-event image.
- **MMFlood / Cloud to Street** — flood mapping; weight availability unconfirmed.

Stay strict on **real downloadable weights vs paper-only**. Naming a model on a roadmap slide that cannot actually be obtained is the same over-claim as R4.

---

## Sources

Verified this session:

- [Gemma 4 model card — Google AI for Developers](https://ai.google.dev/gemma/docs/core/model_card_4)
- [google/gemma-4-26B-A4B-it — Hugging Face](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [Run Gemma with the Gemini API — Google AI for Developers](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
- [Gemma 4 model overview — Google AI for Developers](https://ai.google.dev/gemma/docs/core)
- [How to use Gemma 4 with the Gemini API and Google AI Studio — philschmid.de](https://www.philschmid.de/gemma-4-gemini-api)
- [ibm-nasa-geospatial — Hugging Face organisation](https://huggingface.co/ibm-nasa-geospatial)
- [ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M-sen1floods11)

Referenced but **not** fetched this session (verify before quoting):

- [Copernicus Sentinel-2 mission](https://sentiwiki.copernicus.eu/web/s2-mission) · [Sentinel-1 mission](https://sentiwiki.copernicus.eu/web/s1-mission) · [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) · [Maxar Open Data Program](https://www.maxar.com/open-data)
