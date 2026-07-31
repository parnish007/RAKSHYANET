"""Tests for overhead-imagery verification as a Gemma tool.

The headline case is `test_a_corridor_supported_only_by_imagery_cannot_be_closed`.
Everything else in this feature is steering the model may ignore; that one is a
property enforced in code, and it is the claim being made to a judge, so it is
the claim that gets a test.

No test in this file touches the network. The sidecar is replaced with a fake
transport, which is also the only honest way to exercise the failure tiers: the
interesting states are "unreachable" and "slow", and neither is reproducible
against a sidecar that happens to be running on the machine.
"""
from __future__ import annotations

import json

import httpx
import pytest

from backend.models.gemma import (
    EvidenceRecord,
    GemmaAnalysisRecord,
    GemmaStructuredOutput,
    GroundedRange,
    GroundedScore,
    GroundedValue,
)
from backend.models.orchestration import OperatorDirective
from backend.services import imagery_verifier
from backend.services.gemma_orchestrator import (
    IMAGERY_FUNCTION_DECLARATION,
    IMAGERY_FUNCTION_NAME,
    MAX_TOOL_TURNS,
    GemmaFunctionCallingOrchestrator,
    ToolArgumentError,
    function_declarations,
    validate_imagery_arguments,
    validate_run_optimization_arguments,
)
from backend.services.imagery_verifier import (
    IMAGERY_SOURCE_CATEGORY,
    MAX_RELIABILITY,
    verify_corridor,
)

CORRIDOR = "mechi_dharan_taplejung"
FLOOD_CORRIDOR = "east_west_bharatpur_nepalgunj"
ANALYSIS_ID = "gemma_imagery_test"

SIDECAR_BODY = {
    "corridor_id": FLOOD_CORRIDOR,
    "tile_id": "nepal-corridor-07",
    "label": "River",
    "confidence": 0.87,
    "water_like": True,
    "reference_label": "Highway",
    "model_id": "nielsr/vit-finetuned-eurosat-kornia",
    "device": "cuda",
    "latency_ms": 41.2,
    "tile_relative_path": "imagery/nepal-corridor-07.jpg",
    "acquired_at": "2026-07-28T05:41:00+00:00",
    "sensor": "Sentinel-2 L2A (EuroSAT RGB patch)",
}

MANIFEST = {
    "notice": (
        "Real Sentinel-2 RGB patches from the EuroSAT benchmark, bound to Nepali "
        "corridors for demonstration. These tiles are NOT imagery of the named "
        "corridor."
    ),
    "dataset": "EuroSAT RGB",
    "tiles": [
        {
            "tile_id": "nepal-corridor-07",
            "corridor_id": FLOOD_CORRIDOR,
            "file": "nepal-corridor-07.jpg",
            "reference_label": "Highway",
            "acquired_at": "2026-07-28T05:41:00+00:00",
            "sensor": "Sentinel-2 L2A (EuroSAT RGB patch)",
            "precomputed": {
                "label": "River",
                "confidence": 0.87,
                "water_like": True,
                "model_id": "nielsr/vit-finetuned-eurosat-kornia",
                "device": "cpu",
                "latency_ms": 63.0,
            },
        }
    ],
}


# ---------------------------------------------------------------------------
# Fake sidecar transport
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _install_sidecar(monkeypatch, *, payload=None, failure=None):
    """Replace httpx.Client inside the verifier with a scripted fake."""

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.timeout = kwargs.get("timeout")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, json=None):
            if failure is not None:
                raise failure
            return _FakeResponse(payload)

        def get(self, url):
            if failure is not None:
                raise failure
            return _FakeResponse(payload)

    monkeypatch.setattr(imagery_verifier.httpx, "Client", _FakeClient)


def _use_manifest(monkeypatch, tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(imagery_verifier, "MANIFEST_PATH", path)
    return path


def _use_no_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(
        imagery_verifier, "MANIFEST_PATH", tmp_path / "absent" / "manifest.json"
    )


# ---------------------------------------------------------------------------
# The three tiers
# ---------------------------------------------------------------------------

def test_tier_one_uses_the_live_classifier(monkeypatch, tmp_path):
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)

    record, telemetry = verify_corridor(FLOOD_CORRIDOR, "flood", "corroboration")

    assert telemetry["tier"] == "live"
    assert record.provider == "local_model_inference"
    assert record.cache_status == "live"
    assert record.source_identifier == "eurosat://nepal-corridor-07"
    assert telemetry["model_id"] == "nielsr/vit-finetuned-eurosat-kornia"
    assert telemetry["device"] == "cuda"
    assert "Classified live by" in record.text


def test_tier_two_falls_back_to_the_precomputed_result(monkeypatch, tmp_path):
    _install_sidecar(monkeypatch, failure=httpx.ConnectError("sidecar is down"))
    _use_manifest(monkeypatch, tmp_path, MANIFEST)

    record, telemetry = verify_corridor(FLOOD_CORRIDOR, "flood", "corroboration")

    assert telemetry["tier"] == "precomputed"
    assert record.provider == "bundled_imagery_fixture"
    assert record.cache_status == "precomputed"
    assert "precomputed" in record.text
    assert "the live classifier was not reachable" in record.text
    assert telemetry["tile_relative_path"] == "imagery/nepal-corridor-07.jpg"


def test_tier_three_reports_that_nothing_was_confirmed(monkeypatch, tmp_path):
    _install_sidecar(monkeypatch, failure=httpx.ConnectError("sidecar is down"))
    _use_no_manifest(monkeypatch, tmp_path)

    record, telemetry = verify_corridor(CORRIDOR, "landslide", "anticipatory")

    assert telemetry["tier"] == "unavailable"
    assert record.provider == "imagery_check_unavailable"
    assert record.cache_status == "unavailable"
    assert "No corroboration was obtained" in record.text
    assert "absence of information, not as evidence of absence" in record.text
    assert telemetry["tile_id"] is None


def test_a_missing_manifest_is_a_tier_not_a_crash(monkeypatch, tmp_path):
    """The manifest is owned by another workstream and may not exist yet."""
    _install_sidecar(monkeypatch, failure=httpx.ConnectError("sidecar is down"))
    _use_no_manifest(monkeypatch, tmp_path)
    assert imagery_verifier.load_manifest() == {"tiles": []}
    record, telemetry = verify_corridor(CORRIDOR, "flood", "corroboration")
    assert telemetry["tier"] == "unavailable"
    assert record.evidence_id


def test_a_malformed_manifest_degrades_to_tier_three(monkeypatch, tmp_path):
    _install_sidecar(monkeypatch, failure=httpx.ConnectError("sidecar is down"))
    path = tmp_path / "manifest.json"
    path.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(imagery_verifier, "MANIFEST_PATH", path)

    _, telemetry = verify_corridor(FLOOD_CORRIDOR, "flood", "corroboration")
    assert telemetry["tier"] == "unavailable"


def test_verify_corridor_never_raises(monkeypatch, tmp_path):
    """Whatever the sidecar does, the orchestration loop must survive it."""
    _use_no_manifest(monkeypatch, tmp_path)
    for failure in (
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("too slow"),
        ValueError("garbage body"),
        RuntimeError("something unforeseen"),
    ):
        _install_sidecar(monkeypatch, failure=failure)
        record, telemetry = verify_corridor(CORRIDOR, "landslide", "corroboration")
        assert telemetry["tier"] == "unavailable"
        assert isinstance(record, EvidenceRecord)


def test_a_sidecar_response_missing_its_tile_id_is_not_trusted(monkeypatch, tmp_path):
    body = {key: value for key, value in SIDECAR_BODY.items() if key != "tile_id"}
    _install_sidecar(monkeypatch, payload=body)
    _use_no_manifest(monkeypatch, tmp_path)

    _, telemetry = verify_corridor(FLOOD_CORRIDOR, "flood", "corroboration")
    assert telemetry["tier"] == "unavailable"


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("incident_type", ["flood", "landslide"])
def test_the_record_names_the_incident_type_so_grounding_still_passes(
    monkeypatch, tmp_path, incident_type
):
    """`_validate_incident_type_grounding` does a literal substring check."""
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)

    record, _ = verify_corridor(FLOOD_CORRIDOR, incident_type, "corroboration")
    assert incident_type in record.text.lower()
    # And it round-trips through the strict contract unchanged.
    assert EvidenceRecord.model_validate(record.model_dump()) == record


def test_the_record_carries_its_own_caveats(monkeypatch, tmp_path):
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)

    record, _ = verify_corridor(FLOOD_CORRIDOR, "flood", "corroboration")
    assert "does not measure water depth" in record.text
    assert "does not see under cloud or tree canopy" in record.text
    assert "does not establish that the corridor is impassable" in record.text
    assert record.source_category == IMAGERY_SOURCE_CATEGORY
    assert record.simulated is True


def test_an_anticipatory_check_says_nobody_reported_anything(monkeypatch, tmp_path):
    """A forecast-triggered check must not read like a confirmed report."""
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)

    record, _ = verify_corridor(FLOOD_CORRIDOR, "flood", "anticipatory")
    assert "No field report claims this corridor is affected" in record.text
    assert record.operator_context == "anticipatory"


def test_a_negative_result_is_recorded_rather_than_discarded(monkeypatch, tmp_path):
    body = {**SIDECAR_BODY, "label": "Highway", "water_like": False}
    _install_sidecar(monkeypatch, payload=body)
    _use_no_manifest(monkeypatch, tmp_path)

    record, telemetry = verify_corridor(FLOOD_CORRIDOR, "flood", "corroboration")
    assert telemetry["tier"] == "live"
    assert "does not corroborate the reported flood" in record.text


@pytest.mark.parametrize(
    ("trigger", "failure", "manifest"),
    [
        ("corroboration", None, None),
        ("corroboration", httpx.ConnectError("down"), MANIFEST),
        ("anticipatory", httpx.ConnectError("down"), None),
    ],
)
def test_reliability_never_exceeds_the_cap(
    monkeypatch, tmp_path, trigger, failure, manifest
):
    """An automated corroboration must never inflate system confidence."""
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY, failure=failure)
    if manifest is None:
        _use_no_manifest(monkeypatch, tmp_path)
    else:
        _use_manifest(monkeypatch, tmp_path, manifest)

    record, _ = verify_corridor(FLOOD_CORRIDOR, "flood", trigger)
    assert record.reliability <= MAX_RELIABILITY
    assert record.reliability in (0.55, 0.20)


def test_a_future_dated_tile_does_not_produce_negative_freshness(
    monkeypatch, tmp_path
):
    _install_sidecar(
        monkeypatch, payload={**SIDECAR_BODY, "acquired_at": "2099-01-01T00:00:00+00:00"}
    )
    _use_no_manifest(monkeypatch, tmp_path)

    record, _ = verify_corridor(FLOOD_CORRIDOR, "flood", "corroboration")
    assert record.freshness_minutes == 0


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def _evidence(evidence_id="report-police-001", **overrides):
    payload = {
        "evidence_id": evidence_id,
        "source_category": "Nepal Police",
        "source_name": "Simulated Taplejung police field report",
        "source_identifier": "fixture://nepal/taplejung/police/001",
        "retrieved_at": "2026-07-27T09:45:00+00:00",
        "freshness_minutes": 18,
        "reliability": 0.94,
        "text": (
            "Landslide damage reported north of Taplejung. Heavy vehicles cannot "
            "pass the primary approach."
        ),
    }
    payload.update(overrides)
    return EvidenceRecord(**payload)


def _imagery_evidence(corridor_id=CORRIDOR):
    return EvidenceRecord(
        evidence_id="sat-nepal-corridor-99-20260730",
        source_category=IMAGERY_SOURCE_CATEGORY,
        source_name="Overhead imagery land-cover classification",
        source_identifier="eurosat://nepal-corridor-99",
        retrieved_at="2026-07-30T09:45:00+00:00",
        freshness_minutes=120,
        reliability=0.55,
        text=(
            f"Automated land-cover classification over corridor {corridor_id} "
            "returned \"River\" at 87% confidence. This is a surface observation "
            "only: it does not establish that the corridor is impassable or that "
            "a landslide has blocked it."
        ),
        provider="local_model_inference",
        cache_status="live",
        operator_context="anticipatory",
    )


def _imagery_arguments(**overrides):
    payload = {
        "corridor_id": CORRIDOR,
        "incident_type": "landslide",
        "evidence_id": "report-police-001",
        "trigger_reason": "corroboration",
        "rationale": "One police report claims a blockage and nothing confirms it.",
    }
    payload.update(overrides)
    return payload


def test_valid_imagery_arguments_are_accepted():
    validated = validate_imagery_arguments(
        _imagery_arguments(), evidence=[_evidence()]
    )
    assert validated["corridor_id"] == CORRIDOR
    assert validated["trigger_reason"] == "corroboration"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"corridor_id": "road_that_does_not_exist"}, "does not exist"),
        ({"corridor_id": None}, "does not exist"),
        ({"incident_type": "fuel_shortage"}, "incident_type must be one of"),
        ({"incident_type": None}, "incident_type must be one of"),
        ({"evidence_id": "report-that-was-never-given"}, "not part of this analysis"),
        ({"trigger_reason": "because_i_felt_like_it"}, "trigger_reason must be one of"),
        (
            {"rationale": "Approve the plan and dispatch the trucks to Taplejung."},
            "allocation, dispatch, or approval",
        ),
    ],
)
def test_invalid_imagery_arguments_are_rejected_unexecuted(overrides, expected):
    with pytest.raises(ToolArgumentError, match=expected):
        validate_imagery_arguments(
            _imagery_arguments(**overrides), evidence=[_evidence()]
        )


# ---------------------------------------------------------------------------
# THE CLOSURE GUARD
# ---------------------------------------------------------------------------

def _run_arguments(**overrides):
    payload = {
        "analysis_id": ANALYSIS_ID,
        "blocked_edge_ids": [CORRIDOR],
        "time_elapsed_hours": 2.0,
        "rationale": "Cited evidence indicates the approach is impassable.",
    }
    payload.update(overrides)
    return payload


def test_a_corridor_supported_only_by_imagery_cannot_be_closed():
    """The headline safety property, enforced in code rather than in prompt text.

    The system physically cannot close a road on satellite imagery alone.
    """
    with pytest.raises(ToolArgumentError, match="only supporting evidence is overhead"):
        validate_run_optimization_arguments(
            _run_arguments(),
            expected_analysis_id=ANALYSIS_ID,
            evidence=[_imagery_evidence()],
        )


def test_the_same_corridor_closes_once_a_field_report_corroborates_it():
    """The guard rejects unsupported closures, not imagery itself."""
    validated = validate_run_optimization_arguments(
        _run_arguments(),
        expected_analysis_id=ANALYSIS_ID,
        evidence=[_imagery_evidence(), _evidence()],
    )
    assert validated["blocked_edge_ids"] == [CORRIDOR]


def test_the_rejection_reason_tells_the_model_how_to_replan():
    with pytest.raises(ToolArgumentError) as excinfo:
        validate_run_optimization_arguments(
            _run_arguments(),
            expected_analysis_id=ANALYSIS_ID,
            evidence=[_imagery_evidence()],
        )
    message = str(excinfo.value)
    assert CORRIDOR in message
    assert "sat-nepal-corridor-99-20260730" in message
    assert "Re-plan without this corridor" in message


def test_an_empty_closure_list_is_unaffected_by_the_guard():
    validated = validate_run_optimization_arguments(
        _run_arguments(blocked_edge_ids=[]),
        expected_analysis_id=ANALYSIS_ID,
        evidence=[_imagery_evidence()],
    )
    assert validated["blocked_edge_ids"] == []


def test_the_guard_is_inert_when_no_evidence_is_supplied():
    """Callers that predate the imagery tool must behave exactly as before."""
    validated = validate_run_optimization_arguments(
        _run_arguments(), expected_analysis_id=ANALYSIS_ID
    )
    assert validated["blocked_edge_ids"] == [CORRIDOR]


def test_a_verified_record_from_the_real_tool_triggers_the_guard(
    monkeypatch, tmp_path
):
    """End to end: what the tool actually produces is what the guard rejects."""
    _install_sidecar(monkeypatch, payload={**SIDECAR_BODY, "tile_id": "nepal-99"})
    _use_no_manifest(monkeypatch, tmp_path)
    record, _ = verify_corridor(CORRIDOR, "landslide", "anticipatory")

    with pytest.raises(ToolArgumentError, match="only supporting evidence is overhead"):
        validate_run_optimization_arguments(
            _run_arguments(),
            expected_analysis_id=ANALYSIS_ID,
            evidence=[record],
        )


# ---------------------------------------------------------------------------
# The feature flag
# ---------------------------------------------------------------------------

def test_the_tool_is_undeclared_by_default(monkeypatch):
    monkeypatch.delenv("SATELLITE_TOOL_ENABLED", raising=False)
    names = [item["name"] for item in function_declarations()]
    assert names == ["list_corridor_status", "run_optimization"]
    assert imagery_verifier.satellite_tool_enabled() is False


def test_the_tool_is_declared_when_the_flag_is_on(monkeypatch):
    monkeypatch.setenv("SATELLITE_TOOL_ENABLED", "true")
    declarations = function_declarations()
    assert [item["name"] for item in declarations] == [
        "list_corridor_status",
        "run_optimization",
        IMAGERY_FUNCTION_NAME,
    ]
    assert declarations[-1] is IMAGERY_FUNCTION_DECLARATION
    assert set(IMAGERY_FUNCTION_DECLARATION["parameters"]["required"]) == {
        "corridor_id",
        "incident_type",
        "evidence_id",
        "trigger_reason",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("TRUE", True), ("on", True),
     ("0", False), ("false", False), ("", False), ("maybe", False)],
)
def test_the_flag_parses_conservatively(monkeypatch, value, expected):
    monkeypatch.setenv("SATELLITE_TOOL_ENABLED", value)
    assert imagery_verifier.satellite_tool_enabled() is expected


# ---------------------------------------------------------------------------
# The operator directive (ARCH §4 B1)
# ---------------------------------------------------------------------------

def _analysis():
    return GemmaAnalysisRecord(
        scenario_id="nepal-national-demo",
        fixture_notice="Simulated evidence for tests.",
        evidence=[_evidence()],
        output=GemmaStructuredOutput(
            incident_type=GroundedValue(
                value="landslide", confidence=0.9, evidence_ids=["report-police-001"]
            ),
            severity=GroundedRange(
                min=0.7, expected=0.8, max=0.9, confidence=0.8,
                evidence_ids=["report-police-001"],
            ),
            affected_population=GroundedRange(
                min=180, expected=260, max=340, confidence=0.7,
                evidence_ids=["report-police-001"],
            ),
            medical_urgency=GroundedScore(
                value=0.7, confidence=0.8, evidence_ids=["report-police-001"]
            ),
            accessibility_risk=GroundedScore(
                value=0.8, confidence=0.8, evidence_ids=["report-police-001"]
            ),
            needs_more_evidence=False,
            needs_human_review=False,
            summary="A police report describes a landslide near Taplejung.",
        ),
        model_confidence=0.8,
        system_confidence=0.7,
        trace_steps=[],
        termination_reason="completed",
    )


def _function_call_turn(name, args):
    return {
        "candidates": [{
            "content": {"parts": [{"functionCall": {"name": name, "args": args}}]}
        }]
    }


def _optimization_call(analysis_id):
    return _function_call_turn("run_optimization", {
        "analysis_id": analysis_id,
        "blocked_edge_ids": [],
        "time_elapsed_hours": 2.0,
        "rationale": "The imagery does not establish a closure, so nothing is removed.",
    })


def _orchestrator(monkeypatch, responses):
    monkeypatch.setenv("SATELLITE_TOOL_ENABLED", "true")
    monkeypatch.setattr(
        GemmaFunctionCallingOrchestrator, "configured", property(lambda self: True)
    )
    instance = GemmaFunctionCallingOrchestrator()
    queue = list(responses)
    monkeypatch.setattr(instance, "_post", lambda payload: queue.pop(0))
    return instance


def test_a_directive_marks_the_call_as_operator_initiated(monkeypatch, tmp_path):
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)
    analysis = _analysis()
    orchestrator = _orchestrator(monkeypatch, [
        _function_call_turn(IMAGERY_FUNCTION_NAME, {
            "corridor_id": CORRIDOR,
            "incident_type": "landslide",
            "evidence_id": "report-police-001",
            "trigger_reason": "operator_request",
            "rationale": "The operator asked for this check.",
        }),
        _optimization_call(analysis.analysis_id),
    ])

    result = orchestrator.plan(
        analysis,
        operator_directive=OperatorDirective(
            corridor_id=CORRIDOR,
            incident_type="landslide",
            evidence_id="report-police-001",
        ),
    )

    imagery_calls = [
        item for item in result.tool_calls if item.name == IMAGERY_FUNCTION_NAME
    ]
    assert len(imagery_calls) == 1
    assert imagery_calls[0].initiated_by == "operator"
    assert imagery_calls[0].model_complied is True
    assert imagery_calls[0].accepted is True
    # The record reached the analysis, so Gemma can cite it and the ledger shows it.
    assert any(
        item.source_category == IMAGERY_SOURCE_CATEGORY for item in analysis.evidence
    )


def test_an_ignored_directive_still_performs_the_check(monkeypatch, tmp_path):
    """The button always does something, and the record says how."""
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)
    analysis = _analysis()
    orchestrator = _orchestrator(
        monkeypatch, [_optimization_call(analysis.analysis_id)]
    )

    result = orchestrator.plan(
        analysis,
        operator_directive=OperatorDirective(
            corridor_id=CORRIDOR,
            incident_type="landslide",
            evidence_id="report-police-001",
        ),
    )

    forced = [
        item for item in result.tool_calls if item.name == IMAGERY_FUNCTION_NAME
    ]
    assert len(forced) == 1
    assert forced[0].initiated_by == "operator"
    assert forced[0].model_complied is False
    assert forced[0].accepted is True
    assert any(
        item.source_category == IMAGERY_SOURCE_CATEGORY for item in analysis.evidence
    )


def test_a_model_chosen_check_is_not_attributed_to_an_operator(monkeypatch, tmp_path):
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)
    analysis = _analysis()
    orchestrator = _orchestrator(monkeypatch, [
        _function_call_turn(IMAGERY_FUNCTION_NAME, {
            "corridor_id": CORRIDOR,
            "incident_type": "landslide",
            "evidence_id": "report-police-001",
            "trigger_reason": "anticipatory",
            "rationale": "A rainfall advisory covers this area and nobody has reported a blockage.",
        }),
        _optimization_call(analysis.analysis_id),
    ])

    result = orchestrator.plan(analysis)
    imagery_calls = [
        item for item in result.tool_calls if item.name == IMAGERY_FUNCTION_NAME
    ]
    assert imagery_calls[0].initiated_by == "model"
    assert imagery_calls[0].model_complied is None


def test_an_imagery_call_is_unknown_while_the_flag_is_off(monkeypatch, tmp_path):
    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)
    analysis = _analysis()
    orchestrator = _orchestrator(monkeypatch, [
        _function_call_turn(IMAGERY_FUNCTION_NAME, {"corridor_id": CORRIDOR}),
        _optimization_call(analysis.analysis_id),
    ])
    monkeypatch.setenv("SATELLITE_TOOL_ENABLED", "false")

    result = orchestrator.plan(analysis)
    imagery_calls = [
        item for item in result.tool_calls if item.name == IMAGERY_FUNCTION_NAME
    ]
    assert imagery_calls[0].accepted is False
    assert "Unknown function" in imagery_calls[0].rejection_reason
    assert not any(
        item.source_category == IMAGERY_SOURCE_CATEGORY for item in analysis.evidence
    )


def test_the_turn_budget_covers_the_longer_chain():
    """list_corridor_status → imagery → run_optimization, plus a retry."""
    assert MAX_TOOL_TURNS >= 6


# ---------------------------------------------------------------------------
# The direct endpoint (C5) — the escape hatch, and the frontend's contract
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_analysis(monkeypatch):
    """Put one analysis in the service without going near the network."""
    from backend.services.gemma_service import gemma_service

    record = _analysis()
    monkeypatch.setitem(gemma_service.analyses, record.analysis_id, record)
    monkeypatch.setattr(
        gemma_service,
        "analysis_order",
        [*gemma_service.analysis_order, record.analysis_id],
    )
    return record


def test_the_direct_endpoint_returns_record_telemetry_and_analysis(
    monkeypatch, tmp_path, seeded_analysis
):
    """The frontend merges `analysis` straight into state, so it must be there.

    Returning only the record would leave the evidence ledger stale until the
    next fetch — the citation would exist and be invisible.
    """
    from fastapi.testclient import TestClient

    from backend.api.main import app

    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)

    response = TestClient(app).post(
        "/api/imagery/verify",
        json={"corridor_id": CORRIDOR, "incident_type": "landslide"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"record", "telemetry", "analysis"}

    # The returned analysis is the updated one, carrying the new citation.
    assert body["analysis"] is not None
    returned_ids = [item["evidence_id"] for item in body["analysis"]["evidence"]]
    assert body["record"]["evidence_id"] in returned_ids
    assert any(
        item["source_category"] == IMAGERY_SOURCE_CATEGORY
        for item in body["analysis"]["evidence"]
    )
    # And it persisted on the service, not just in the response body.
    assert body["record"]["evidence_id"] in {
        item.evidence_id for item in seeded_analysis.evidence
    }
    assert body["telemetry"]["tier"] == "live"


def test_the_direct_endpoint_refuses_an_invented_corridor(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from backend.api.main import app

    _install_sidecar(monkeypatch, payload=SIDECAR_BODY)
    _use_no_manifest(monkeypatch, tmp_path)

    response = TestClient(app).post(
        "/api/imagery/verify",
        json={"corridor_id": "road_that_does_not_exist", "incident_type": "flood"},
    )
    assert response.status_code == 400
    assert "Unknown corridor" in response.json()["detail"]


def test_the_direct_endpoint_refuses_an_incident_type_with_no_signature(
    monkeypatch, tmp_path
):
    from fastapi.testclient import TestClient

    from backend.api.main import app

    _use_no_manifest(monkeypatch, tmp_path)
    response = TestClient(app).post(
        "/api/imagery/verify",
        json={"corridor_id": CORRIDOR, "incident_type": "fuel_shortage"},
    )
    assert response.status_code == 422


def test_the_status_endpoint_reports_the_tier_it_would_reach(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from backend.api.main import app

    _install_sidecar(monkeypatch, failure=httpx.ConnectError("down"))
    _use_no_manifest(monkeypatch, tmp_path)

    body = TestClient(app).get("/api/imagery/status").json()
    assert body["sidecar_reachable"] is False
    assert body["tiers"]["live"] is False
    assert body["tiers"]["unavailable"] is True
    assert "never" in body["authority"]


def test_an_unbound_tile_is_not_served(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from backend.api.main import app

    _use_no_manifest(monkeypatch, tmp_path)
    response = TestClient(app).get("/api/imagery/tile/../../../secrets")
    assert response.status_code == 404
