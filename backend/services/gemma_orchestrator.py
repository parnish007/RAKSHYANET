"""Gemma 4 native function-calling over the deterministic route engine.

The extraction stage in `gemma_service` turns untrusted field reports into a
validated, evidence-cited analysis. This module is the second stage: Gemma is
handed that validated analysis plus a set of *declared functions* and decides,
by emitting a real `functionCall` part, which corridor state to retrieve and
what optimization to run.

The safety property is not "the model cannot call anything". It is that every
argument the model produces is checked against the world before the engine is
allowed to see it:

* `analysis_id` must be the analysis this turn was opened with, so the model
  cannot retarget a different evidence set;
* every blocked corridor must exist in the terrain graph, so the model cannot
  invent a road closure;
* elapsed mission time must fall inside a bounded horizon;
* the free-text rationale is screened with the same operational-authority
  filter used on extraction output.

The model therefore *orchestrates* the engine, and still cannot dispatch
anything: the engine returns a versioned plan that remains subject to human
approval.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import httpx

from backend.models.gemma import EvidenceRecord, GemmaAnalysisRecord
from backend.services.api_key_pool import (
    ApiKeyPoolExhausted,
    gemma_key_pool,
    post_with_failover,
)
from backend.models.orchestration import (
    OperatorDirective,
    OrchestrationRecord,
    ToolCallRecord,
)
from backend.services.gemma_service import (
    _OPERATIONAL_AUTHORITY_PATTERNS,
    GemmaProviderError,
)
from backend.services.imagery_verifier import (
    IMAGERY_SOURCE_CATEGORY,
    INCIDENT_TYPES,
    TRIGGER_REASONS,
    satellite_tool_enabled,
    verify_corridor,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MAX_ELAPSED_HOURS = 72.0
# Raised from 4. The chain is now list_corridor_status →
# verify_report_with_imagery → run_optimization = three turns, so a single
# rejected-argument retry would have exhausted the old budget and the model
# would have lost its one chance to re-plan.
MAX_TOOL_TURNS = 6

_ORCHESTRATION_PROMPT_VERSION = "nepal-route-orchestration-v1"

IMAGERY_FUNCTION_NAME = "verify_report_with_imagery"


IMAGERY_FUNCTION_DECLARATION: Dict[str, Any] = {
    "name": IMAGERY_FUNCTION_NAME,
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
            "corridor_id": {
                "type": "string",
                "description": "Exact id returned by list_corridor_status.",
            },
            "incident_type": {"type": "string", "enum": ["flood", "landslide"]},
            "evidence_id": {
                "type": "string",
                "description": (
                    "The record whose claim you are testing. For an anticipatory "
                    "check, the advisory that prompted it."
                ),
            },
            "trigger_reason": {
                "type": "string",
                "enum": ["corroboration", "anticipatory", "operator_request"],
            },
            "rationale": {
                "type": "string",
                "description": (
                    "One sentence: which evidence prompted this check. "
                    "Describe evidence only."
                ),
            },
        },
        "required": [
            "corridor_id",
            "incident_type",
            "evidence_id",
            "trigger_reason",
        ],
    },
}


_BASE_FUNCTION_DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "list_corridor_status",
        "description": (
            "Return every ground corridor in the Nepal road graph with its "
            "identifier, endpoints, length in kilometres, terrain difficulty, "
            "surface quality, and whether it is landslide-vulnerable. Call this "
            "before proposing an optimization whenever the evidence refers to "
            "road access, blockage, or passability, so that any corridor you "
            "name is one that actually exists."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "run_optimization",
        "description": (
            "Run RakshyaNet's deterministic terrain-constrained routing and "
            "allocation engine and return a versioned plan awaiting human "
            "approval. This computes urgency, capability-filtered shortest "
            "paths over the road graph, capped proportional allocation, and "
            "diagnostics. It does NOT dispatch vehicles and does NOT approve "
            "anything; a human authorizes the resulting plan separately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "analysis_id": {
                    "type": "string",
                    "description": (
                        "The identifier of the validated Gemma analysis this "
                        "plan must be computed from. Use the analysis_id given "
                        "to you in this turn."
                    ),
                },
                "blocked_edge_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Corridor identifiers that the evidence establishes are "
                        "impassable and must be removed from the road graph "
                        "before the search runs. Use exact ids returned by "
                        "list_corridor_status. Pass an empty array if the "
                        "evidence does not establish a closure."
                    ),
                },
                "time_elapsed_hours": {
                    "type": "number",
                    "description": (
                        "Hours since the incident began, used to scale urgency "
                        "and to skip stops already served. Between 0 and 72."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "One sentence citing which evidence supports this "
                        "corridor set and elapsed time. Describe evidence only; "
                        "do not state an allocation, dispatch, or approval."
                    ),
                },
            },
            "required": [
                "analysis_id",
                "blocked_edge_ids",
                "time_elapsed_hours",
                "rationale",
            ],
        },
    },
]


def function_declarations() -> List[Dict[str, Any]]:
    """The schemas Gemma is given for this turn.

    Resolved per call rather than frozen at import, so the imagery tool can be
    switched on without restarting and so a test can exercise both shapes. With
    the flag off this returns exactly the two declarations that shipped before
    the imagery work existed.
    """
    if satellite_tool_enabled():
        return [*_BASE_FUNCTION_DECLARATIONS, IMAGERY_FUNCTION_DECLARATION]
    return list(_BASE_FUNCTION_DECLARATIONS)


# Import-time snapshot, kept because callers import the name directly.
FUNCTION_DECLARATIONS: List[Dict[str, Any]] = function_declarations()


class ToolArgumentError(ValueError):
    """Raised when a model-produced argument fails validation before execution."""


def _load_graph() -> Dict[str, Any]:
    return json.loads((DATA_DIR / "terrain_graph.json").read_text(encoding="utf-8"))


def _load_corridors() -> List[Dict[str, Any]]:
    return list(_load_graph().get("edges", []))


def corridor_status_payload() -> Dict[str, Any]:
    """The real return value of `list_corridor_status`.

    This is bundled fixture data, and it says so, because the model is allowed
    to reason about corridors but is never allowed to believe it has a live feed.
    """
    corridors = _load_corridors()
    return {
        "source": "bundled_terrain_fixture",
        "corridor_count": len(corridors),
        "corridors": [
            {
                "id": edge["id"],
                "name": edge.get("name"),
                "from": edge.get("from"),
                "to": edge.get("to"),
                "distance_km": edge.get("distance_km"),
                "terrain_difficulty": edge.get("terrain_difficulty"),
                "road_quality": edge.get("road_quality"),
                "vulnerable_to_landslide": edge.get("vulnerable_to_landslide"),
                "currently_open": True,
            }
            for edge in corridors
        ],
    }


# --------------------------------------------------------------------------
# The imagery-only closure guard.
#
# Everything the prompt says about imagery is steering the model may ignore.
# This is the one rule enforced in code: a corridor whose ONLY supporting
# evidence is an overhead-imagery record cannot enter `blocked_edge_ids`.
# Imagery sees a surface. It does not see passability, and deleting a usable
# corridor from the graph can strand a location that had one road left.
# --------------------------------------------------------------------------

# Terms too generic to identify a corridor. "depot" touches most of the graph,
# so matching on it would make almost any record count as support for almost
# any closure and quietly disarm the guard.
_GENERIC_REFERENCE_TERMS = {"depot", "nepal", "road", "highway", "corridor"}


def _corridor_reference_terms(edge: Dict[str, Any]) -> set:
    """Strings whose presence in a record means it speaks about this corridor.

    Field reports name places ("Landslide damage reported north of Taplejung"),
    never corridor ids, while the imagery record always carries the exact id.
    Matching on ids alone would therefore classify every real field report as
    "not about this corridor" and fire the guard on legitimate, well-evidenced
    closures. So endpoints — their ids and their display names — count too.

    Erring generous is the correct direction: a loose match can only make the
    guard *less* likely to reject, and the property being protected is "imagery
    was the ONLY support", not "this report is strong".
    """
    nodes = {
        str(node.get("id", "")).lower(): str(node.get("name", ""))
        for node in _load_graph().get("nodes", [])
        if isinstance(node, dict)
    }
    terms = {str(edge.get("id", "")).lower()}
    for endpoint in (edge.get("from"), edge.get("to")):
        key = str(endpoint or "").lower()
        if not key:
            continue
        terms.add(key)
        display = nodes.get(key, "")
        # Node names can be multi-word ("Kathmandu National Logistics Hub");
        # the leading token is the place, the rest is furniture.
        if display:
            terms.add(display.split()[0].lower())
    return {
        term
        for term in terms
        if len(term) >= 4 and term not in _GENERIC_REFERENCE_TERMS
    }


def _records_supporting_corridor(
    corridor_id: str,
    evidence: Sequence[EvidenceRecord],
) -> List[EvidenceRecord]:
    edge = next(
        (item for item in _load_corridors() if item.get("id") == corridor_id),
        None,
    )
    if edge is None:
        return []
    terms = _corridor_reference_terms(edge)
    supporting: List[EvidenceRecord] = []
    for record in evidence:
        haystack = " ".join(
            part
            for part in (
                getattr(record, "text", ""),
                getattr(record, "source_name", ""),
                getattr(record, "operator_context", "") or "",
                getattr(record, "gap_target", "") or "",
            )
            if part
        ).lower()
        if any(term in haystack for term in terms):
            supporting.append(record)
    return supporting


def _reject_imagery_only_closures(
    blocked: Sequence[str],
    evidence: Sequence[EvidenceRecord],
) -> None:
    for corridor_id in blocked:
        supporting = _records_supporting_corridor(corridor_id, evidence)
        if not supporting:
            continue
        if all(
            record.source_category == IMAGERY_SOURCE_CATEGORY
            for record in supporting
        ):
            cited = ", ".join(record.evidence_id for record in supporting)
            raise ToolArgumentError(
                f"Corridor '{corridor_id}' cannot be placed in blocked_edge_ids: "
                f"its only supporting evidence is overhead imagery ({cited}). "
                "Imagery observes surface conditions and cannot establish that a "
                "corridor is impassable. Re-plan without this corridor, or cite "
                "an independent ground or field observation for it."
            )


def validate_run_optimization_arguments(
    arguments: Dict[str, Any],
    *,
    expected_analysis_id: str,
    evidence: Optional[Sequence[EvidenceRecord]] = None,
) -> Dict[str, Any]:
    """Check every model-produced argument against the world before execution.

    Raises ToolArgumentError, which the caller records and reports back to the
    model, rather than executing anything on an unverified argument.

    `evidence` is the analysis's evidence list. When supplied, the imagery-only
    closure guard runs; when omitted the guard is inert, so callers that predate
    the imagery tool behave exactly as before.
    """
    if not isinstance(arguments, dict):
        raise ToolArgumentError("Function arguments must be an object")

    analysis_id = arguments.get("analysis_id")
    if analysis_id != expected_analysis_id:
        raise ToolArgumentError(
            "analysis_id must be the analysis supplied in this turn "
            f"('{expected_analysis_id}'), not '{analysis_id}'"
        )

    raw_blocked = arguments.get("blocked_edge_ids", [])
    if isinstance(raw_blocked, str):
        raw_blocked = [raw_blocked]
    if not isinstance(raw_blocked, list):
        raise ToolArgumentError("blocked_edge_ids must be an array of corridor ids")

    known_ids = {edge["id"] for edge in _load_corridors()}
    blocked: List[str] = []
    for item in raw_blocked:
        if not isinstance(item, str):
            raise ToolArgumentError("Every blocked corridor id must be a string")
        if item not in known_ids:
            raise ToolArgumentError(
                f"Corridor '{item}' does not exist in the road graph. "
                "Call list_corridor_status and use an exact id."
            )
        if item not in blocked:
            blocked.append(item)

    if evidence is not None:
        _reject_imagery_only_closures(blocked, evidence)

    try:
        elapsed = float(arguments.get("time_elapsed_hours", 0.0))
    except (TypeError, ValueError) as exc:
        raise ToolArgumentError("time_elapsed_hours must be a number") from exc
    if not 0.0 <= elapsed <= MAX_ELAPSED_HOURS:
        raise ToolArgumentError(
            f"time_elapsed_hours must be between 0 and {MAX_ELAPSED_HOURS:g}"
        )

    rationale = str(arguments.get("rationale", "")).strip()
    if not rationale:
        raise ToolArgumentError("rationale is required")
    if len(rationale) > 600:
        raise ToolArgumentError("rationale must be 600 characters or fewer")
    for pattern in _OPERATIONAL_AUTHORITY_PATTERNS:
        if pattern.search(rationale):
            raise ToolArgumentError(
                "rationale attempted an allocation, dispatch, or approval decision"
            )

    return {
        "analysis_id": analysis_id,
        "blocked_edge_ids": blocked,
        "time_elapsed_hours": elapsed,
        "rationale": rationale,
    }


def validate_imagery_arguments(
    arguments: Dict[str, Any],
    *,
    evidence: Sequence[EvidenceRecord],
) -> Dict[str, Any]:
    """Check an imagery-verification call against the world before executing it.

    Mirrors `validate_run_optimization_arguments`: the model may ask for a
    check, but it may not invent the corridor it checks, the record it claims
    prompted the check, or the reason it gives for it. A rejection goes back to
    the model unexecuted.
    """
    if not isinstance(arguments, dict):
        raise ToolArgumentError("Function arguments must be an object")

    corridor_id = arguments.get("corridor_id")
    known_ids = {edge["id"] for edge in _load_corridors()}
    if not isinstance(corridor_id, str) or corridor_id not in known_ids:
        raise ToolArgumentError(
            f"Corridor '{corridor_id}' does not exist in the road graph. "
            "Call list_corridor_status and use an exact id."
        )

    incident_type = arguments.get("incident_type")
    if incident_type not in INCIDENT_TYPES:
        raise ToolArgumentError(
            "incident_type must be one of " + ", ".join(INCIDENT_TYPES)
        )

    evidence_id = arguments.get("evidence_id")
    known_evidence = {record.evidence_id for record in evidence}
    if not isinstance(evidence_id, str) or evidence_id not in known_evidence:
        raise ToolArgumentError(
            f"evidence_id '{evidence_id}' is not part of this analysis. "
            "Cite a record you were given in this turn."
        )

    trigger_reason = arguments.get("trigger_reason")
    if trigger_reason not in TRIGGER_REASONS:
        raise ToolArgumentError(
            "trigger_reason must be one of " + ", ".join(TRIGGER_REASONS)
        )

    rationale = str(arguments.get("rationale", "") or "").strip()
    if len(rationale) > 600:
        raise ToolArgumentError("rationale must be 600 characters or fewer")
    for pattern in _OPERATIONAL_AUTHORITY_PATTERNS:
        if pattern.search(rationale):
            raise ToolArgumentError(
                "rationale attempted an allocation, dispatch, or approval decision"
            )

    return {
        "corridor_id": corridor_id,
        "incident_type": incident_type,
        "evidence_id": evidence_id,
        "trigger_reason": trigger_reason,
        "rationale": rationale,
    }


def append_evidence(
    analysis: GemmaAnalysisRecord,
    record: EvidenceRecord,
) -> EvidenceRecord:
    """Attach an imagery record to the analysis it was requested for.

    Idempotent on `evidence_id`: a repeated check of the same tile on the same
    day replaces its predecessor rather than stacking near-identical records
    that would each be independently citable.
    """
    for index, existing in enumerate(analysis.evidence):
        if existing.evidence_id == record.evidence_id:
            analysis.evidence[index] = record
            return record
    analysis.evidence.append(record)
    return record


def operator_directive_text(directive: OperatorDirective) -> str:
    return (
        "OPERATOR DIRECTIVE: A named operator has requested imagery "
        f"verification of corridor {directive.corridor_id} in connection with "
        f"evidence {directive.evidence_id}. Call {IMAGERY_FUNCTION_NAME} for "
        "that corridor with trigger_reason 'operator_request' before proposing "
        "an optimization."
    )


def _analysis_briefing(analysis: GemmaAnalysisRecord) -> str:
    output = analysis.output

    def _scalar(field) -> Any:
        return getattr(field, "value", None)

    def _range(field) -> Any:
        return getattr(field, "expected", None)

    briefing = {
        "analysis_id": analysis.analysis_id,
        "incident_type": _scalar(output.incident_type),
        "severity_expected": _range(output.severity),
        "medical_urgency": _scalar(output.medical_urgency),
        "accessibility_risk": _scalar(output.accessibility_risk),
        "system_confidence": analysis.system_confidence,
        "contradictions": [
            {"claim_a": item.claim_a, "claim_b": item.claim_b}
            for item in output.contradictions
        ],
        "missing_information": list(output.missing_information),
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "source_category": item.source_category,
                "text": item.text,
            }
            for item in analysis.evidence
        ],
    }
    return json.dumps(briefing, ensure_ascii=True, separators=(",", ":"))


_IMAGERY_HOW_TO_WORK = (
    "- If a source claims a corridor is flooded or blocked and nothing "
    "independent confirms it, or if two sources disagree about passability, "
    f"call {IMAGERY_FUNCTION_NAME} before run_optimization.\n"
    "- If a weather advisory reports heavy rain, saturated slopes, or elevated "
    "landslide risk over an area, you may check a corridor in that area even "
    "though no blockage has been reported. Set trigger_reason to 'anticipatory' "
    "and say in the rationale that no report claims a blockage.\n"
    "- Imagery corroborates; it never establishes closure. A corridor whose "
    "only supporting evidence is an imagery record stays OPEN and out of "
    "blocked_edge_ids. Say so in your rationale.\n"
    "- If the imagery shows nothing unusual, report that. A negative result "
    "weakens an uncorroborated claim and must not be omitted.\n"
)

_IMAGERY_AUTHORITY_BOUNDARY = (
    "- You may not treat an imagery observation as a field observation. It has "
    "no witness, no depth measurement, and no ground contact.\n"
)


def _system_prompt() -> str:
    imagery_how_to_work = _IMAGERY_HOW_TO_WORK if satellite_tool_enabled() else ""
    imagery_authority = (
        _IMAGERY_AUTHORITY_BOUNDARY if satellite_tool_enabled() else ""
    )
    return (
        "ROLE\n"
        "You orchestrate RakshyaNet's deterministic route-optimization engine "
        "for disaster logistics in Nepal. You decide WHICH computation to run "
        "and on WHAT world state. You never compute the plan yourself and never "
        "authorize it.\n\n"
        "HOW TO WORK\n"
        "- You are given one already-validated evidence analysis.\n"
        "- If the evidence refers to road access or blockage, first call "
        "list_corridor_status so that any corridor you name provably exists.\n"
        "- Then call run_optimization exactly once, passing the analysis_id you "
        "were given, only corridors the evidence establishes are impassable, and "
        "the elapsed mission time.\n"
        "- A corridor is blocked only if cited evidence says so. Contradictory "
        "evidence about passability is NOT an established closure; leave it open "
        "and say so in the rationale, because deleting a usable corridor can "
        "strand a location.\n"
        f"{imagery_how_to_work}"
        "\n"
        "AUTHORITY BOUNDARY\n"
        "- You may not allocate resources, choose vehicles, compute routes, "
        "approve, reject, or dispatch. The engine computes; a human authorizes.\n"
        "- Never invent a corridor id, an analysis id, or a numeric fact.\n"
        f"{imagery_authority}"
        "\n"
        "Every argument you produce is validated against the road graph before "
        "the engine runs. An invalid argument is rejected and returned to you, "
        "not executed."
    )


class GemmaFunctionCallingOrchestrator:
    """Drives the hosted Gemma function-calling loop over the route engine."""

    provider_name = "gemini_api_function_calling"

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMMA_API_KEY", "").strip()
        self.model_name = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
        # A function-calling turn is not comparable to an extraction call: the
        # model emits ~2k characters of reasoning before the call, and the loop
        # makes several round trips. Reusing extraction's 20s budget timed the
        # request out and surfaced as an opaque transport error, so this path
        # gets its own, larger budget.
        self.timeout_seconds = float(
            os.getenv("GEMMA_ORCHESTRATION_TIMEOUT_SECONDS", "90")
        )
        self.key_pool = gemma_key_pool

    @property
    def configured(self) -> bool:
        return self.key_pool.configured or bool(self.api_key)

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent"
        )
        # A function-calling turn makes several round trips, so it is several
        # times more likely than extraction to meet a quota wall. Failover
        # matters more here, not less.
        return post_with_failover(
            url,
            payload,
            self.timeout_seconds,
            pool=self.key_pool,
            fallback_key=self.api_key,
        )

    def _run_imagery_check(
        self,
        analysis: GemmaAnalysisRecord,
        validated: Dict[str, Any],
    ) -> tuple:
        """Execute a validated check and attach its record to the analysis."""
        record, telemetry = verify_corridor(
            validated["corridor_id"],
            validated["incident_type"],
            validated["trigger_reason"],
        )
        append_evidence(analysis, record)
        return record, telemetry

    def plan(
        self,
        analysis: GemmaAnalysisRecord,
        *,
        default_elapsed_hours: float = 0.0,
        operator_directive: Optional[OperatorDirective] = None,
    ) -> OrchestrationRecord:
        """Ask Gemma which optimization to run, and return the validated call.

        The engine is deliberately NOT executed here. This returns the arguments
        Gemma chose, already validated, so the caller owns execution and the
        interface can show the model's request and the checked arguments side by
        side.
        """
        if not self.configured:
            raise GemmaProviderError("GEMMA_API_KEY is not configured")

        started = time.perf_counter()
        declarations = function_declarations()
        imagery_enabled = satellite_tool_enabled()
        # A directive is only meaningful if the tool it names is declared. With
        # the flag off the button is inert rather than misleading.
        directive = operator_directive if imagery_enabled else None

        directive_block = (
            f"\n\n{operator_directive_text(directive)}" if directive else ""
        )
        contents: List[Dict[str, Any]] = [{
            "role": "user",
            "parts": [{
                "text": (
                    "VALIDATED_ANALYSIS:\n"
                    f"{_analysis_briefing(analysis)}\n\n"
                    f"MISSION_ELAPSED_HOURS: {default_elapsed_hours}\n\n"
                    "Decide what to retrieve, then run the optimization."
                    f"{directive_block}"
                ),
            }],
        }]
        payload_base = {
            "systemInstruction": {"parts": [{"text": _system_prompt()}]},
            "tools": [{"functionDeclarations": declarations}],
            "generationConfig": {"temperature": 0},
        }

        calls: List[ToolCallRecord] = []
        reasoning: List[str] = []
        accepted: Optional[Dict[str, Any]] = None
        model_called_imagery = False

        try:
            for _turn in range(MAX_TOOL_TURNS):
                body = self._post({**payload_base, "contents": contents})
                parts = body["candidates"][0]["content"].get("parts", [])
                # This turn does not force a JSON mime type, so the provider
                # returns its deliberation as readable text. Capture it verbatim.
                reasoning.extend(
                    part["text"].strip()
                    for part in parts
                    if part.get("thought") and part.get("text", "").strip()
                )
                function_calls = [
                    part["functionCall"] for part in parts if "functionCall" in part
                ]
                if not function_calls:
                    break

                contents.append({"role": "model", "parts": parts})
                response_parts: List[Dict[str, Any]] = []

                for call in function_calls:
                    name = call.get("name", "")
                    raw_args = call.get("args") or {}
                    record = ToolCallRecord(
                        name=name,
                        raw_arguments=raw_args,
                        accepted=False,
                    )

                    if name == "list_corridor_status":
                        result = corridor_status_payload()
                        record.accepted = True
                        record.result_summary = (
                            f"{result['corridor_count']} corridors returned from "
                            "the bundled terrain fixture"
                        )
                    elif name == IMAGERY_FUNCTION_NAME and imagery_enabled:
                        # A directive in this turn means the human asked for the
                        # check even though the model is the one emitting it.
                        # The audit must not read as model autonomy.
                        if directive is not None:
                            record.initiated_by = "operator"
                            record.model_complied = False
                        try:
                            validated = validate_imagery_arguments(
                                raw_args,
                                evidence=analysis.evidence,
                            )
                        except ToolArgumentError as exc:
                            record.rejection_reason = str(exc)[:600]
                            result = {"error": str(exc)}
                        else:
                            # Compliance means it checked the corridor it was
                            # told to. Checking a different one is a call the
                            # model chose to make, and the directive is still
                            # outstanding.
                            if directive is not None:
                                complied = (
                                    validated["corridor_id"] == directive.corridor_id
                                )
                                record.model_complied = complied
                                model_called_imagery = (
                                    model_called_imagery or complied
                                )
                            else:
                                model_called_imagery = True
                            evidence_record, telemetry = self._run_imagery_check(
                                analysis, validated
                            )
                            record.accepted = True
                            record.validated_arguments = validated
                            record.result_summary = (
                                f"imagery check returned tier '{telemetry['tier']}'"
                                f" for {validated['corridor_id']}; evidence "
                                f"{evidence_record.evidence_id} appended"
                            )[:600]
                            result = {
                                "evidence": evidence_record.model_dump(mode="json"),
                                "telemetry": telemetry,
                                "note": (
                                    "Cite this evidence_id. Imagery observes "
                                    "surface conditions only and is never "
                                    "sufficient on its own to place a corridor "
                                    "in blocked_edge_ids."
                                ),
                            }
                    elif name == "run_optimization":
                        try:
                            validated = validate_run_optimization_arguments(
                                raw_args,
                                expected_analysis_id=analysis.analysis_id,
                                evidence=analysis.evidence,
                            )
                        except ToolArgumentError as exc:
                            record.rejection_reason = str(exc)[:600]
                            result = {"error": str(exc)}
                        else:
                            record.accepted = True
                            record.validated_arguments = validated
                            accepted = validated
                            blocked = validated["blocked_edge_ids"]
                            record.result_summary = (
                                "arguments validated against the road graph; "
                                f"{len(blocked)} corridor(s) removed before search"
                            )
                            result = {
                                "status": "arguments_accepted",
                                "note": (
                                    "The engine will run with these arguments and "
                                    "produce a plan awaiting human approval."
                                ),
                            }
                    else:
                        record.rejection_reason = f"Unknown function '{name}'"
                        result = {"error": record.rejection_reason}

                    calls.append(record)
                    response_parts.append({
                        "functionResponse": {"name": name, "response": result},
                    })

                contents.append({"role": "user", "parts": response_parts})
                if accepted is not None:
                    break
        except ApiKeyPoolExhausted as exc:
            raise GemmaProviderError(
                f"Gemma function-calling request failed: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise GemmaProviderError(
                f"Gemma function-calling request failed: {detail}"
            ) from exc
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise GemmaProviderError(
                f"Gemma function-calling request failed: {exc}"
            ) from exc

        # The directive guarantee: the operator pressed a button, so a check
        # happens. If the model ignored the instruction the backend performs it
        # deterministically, and the record says the model did not comply rather
        # than dressing a backend call up as a model decision.
        if directive is not None and not model_called_imagery:
            fallback_args = {
                "corridor_id": directive.corridor_id,
                "incident_type": directive.incident_type,
                "evidence_id": directive.evidence_id,
                "trigger_reason": "operator_request",
                "rationale": "Operator directive; the model did not emit the call.",
            }
            fallback = ToolCallRecord(
                name=IMAGERY_FUNCTION_NAME,
                raw_arguments=fallback_args,
                initiated_by="operator",
                model_complied=False,
            )
            try:
                validated = validate_imagery_arguments(
                    fallback_args, evidence=analysis.evidence
                )
            except ToolArgumentError as exc:
                fallback.rejection_reason = str(exc)[:600]
            else:
                evidence_record, telemetry = self._run_imagery_check(
                    analysis, validated
                )
                fallback.accepted = True
                fallback.validated_arguments = validated
                fallback.result_summary = (
                    "operator directive executed by the backend after the model "
                    f"completed without calling the tool; tier '{telemetry['tier']}'"
                )[:600]
            calls.append(fallback)

        if accepted is None:
            raise GemmaProviderError(
                "Gemma did not produce a valid run_optimization call. "
                + "; ".join(
                    item.rejection_reason for item in calls if item.rejection_reason
                )
            )

        return OrchestrationRecord(
            analysis_id=analysis.analysis_id,
            provider=self.provider_name,
            model=self.model_name,
            prompt_version=_ORCHESTRATION_PROMPT_VERSION,
            declared_functions=[item["name"] for item in declarations],
            tool_calls=calls,
            reasoning=[item[:12000] for item in reasoning],
            chosen_arguments=accepted,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )


gemma_orchestrator = GemmaFunctionCallingOrchestrator()
