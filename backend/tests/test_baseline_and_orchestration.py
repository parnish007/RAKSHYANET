"""Tests for the two mandatory Route Intelligence behaviours.

1. Gemma's function-call arguments are validated against the world before the
   engine is allowed to execute them.
2. The naive baseline is a real configuration of the same engine, and the
   head-to-head result is reproducible.

The hosted model is not called here; the network round-trip belongs in a manual
check, not in a suite that must pass offline.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.services.baseline_service import (
    DEFAULT_CLOSURE_EDGE,
    BaselineComparisonService,
)
from backend.services.gemma_orchestrator import (
    FUNCTION_DECLARATIONS,
    ToolArgumentError,
    corridor_status_payload,
    function_declarations,
    validate_run_optimization_arguments,
)
from backend.services.imagery_verifier import satellite_tool_enabled

ANALYSIS_ID = "gemma_test_analysis"


def _arguments(**overrides):
    payload = {
        "analysis_id": ANALYSIS_ID,
        "blocked_edge_ids": [DEFAULT_CLOSURE_EDGE],
        "time_elapsed_hours": 0.75,
        "rationale": "The police report states heavy vehicles cannot pass.",
    }
    payload.update(overrides)
    return payload


def test_declared_functions_expose_the_engine_handoff() -> None:
    declarations = function_declarations()
    names = {item["name"] for item in declarations}
    # The imagery tool is declared only behind SATELLITE_TOOL_ENABLED, so the
    # expected set depends on the flag. With it off — the default — the contract
    # is still exactly these two and nothing else.
    expected = {"list_corridor_status", "run_optimization"}
    if satellite_tool_enabled():
        expected.add("verify_report_with_imagery")
    assert names == expected
    run = next(
        item for item in declarations if item["name"] == "run_optimization"
    )
    assert set(run["parameters"]["required"]) == {
        "analysis_id",
        "blocked_edge_ids",
        "time_elapsed_hours",
        "rationale",
    }


def test_valid_arguments_are_accepted() -> None:
    validated = validate_run_optimization_arguments(
        _arguments(), expected_analysis_id=ANALYSIS_ID
    )
    assert validated["blocked_edge_ids"] == [DEFAULT_CLOSURE_EDGE]
    assert validated["time_elapsed_hours"] == 0.75


def test_corridor_status_returns_only_real_corridors() -> None:
    payload = corridor_status_payload()
    assert payload["corridor_count"] > 0
    assert payload["source"] == "bundled_terrain_fixture"
    assert all(item["id"] for item in payload["corridors"])


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"analysis_id": "some_other_analysis"}, "analysis_id"),
        ({"blocked_edge_ids": ["road_that_does_not_exist"]}, "does not exist"),
        ({"time_elapsed_hours": 900}, "between 0 and"),
        ({"time_elapsed_hours": -1}, "between 0 and"),
        ({"rationale": ""}, "rationale is required"),
        (
            {"rationale": "Approve the plan and dispatch the trucks now."},
            "allocation, dispatch, or approval",
        ),
    ],
)
def test_invalid_arguments_never_reach_the_engine(overrides, expected) -> None:
    with pytest.raises(ToolArgumentError, match=expected):
        validate_run_optimization_arguments(
            _arguments(**overrides), expected_analysis_id=ANALYSIS_ID
        )


def test_a_model_invented_corridor_cannot_close_a_road() -> None:
    """The specific failure this guard exists for: a hallucinated closure."""
    with pytest.raises(ToolArgumentError):
        validate_run_optimization_arguments(
            _arguments(blocked_edge_ids=["mahakali_highway_west"]),
            expected_analysis_id=ANALYSIS_ID,
        )


def test_baseline_loses_routes_to_a_closure_that_rakshyanet_survives() -> None:
    report = BaselineComparisonService().compare()

    naive = report["after_closure"]["naive"]
    ours = report["after_closure"]["rakshyanet"]

    # The naive planner never removed the closed corridor, so its plan still
    # routes assets through a road that cannot be driven.
    assert naive["routes_traversing_closed_corridor"] > 0
    assert naive["executable_routes"] < naive["routes"]

    # The production planner deletes the corridor before searching.
    assert ours["routes_traversing_closed_corridor"] == 0
    assert ours["executable_routes"] == ours["routes"]
    assert ours["executable_routes"] > naive["executable_routes"]


def test_baseline_reports_its_own_measured_limitation() -> None:
    """The comparison must not overstate what terrain weighting achieved."""
    report = BaselineComparisonService().compare()
    definition = report["baseline_definition"]
    assert "measured_limitation" in definition
    assert report["undisrupted"]["naive"]["total_distance_km"] == pytest.approx(
        report["undisrupted"]["rakshyanet"]["total_distance_km"]
    )


def test_baseline_endpoint_is_reachable() -> None:
    client = TestClient(app)
    response = client.get("/api/optimization/baseline")
    assert response.status_code == 200
    body = response.json()
    assert body["headline"]["metric"]
    assert body["baseline_definition"]["name"].startswith("shortest-path-only")


def test_tools_endpoint_publishes_the_declared_schemas() -> None:
    client = TestClient(app)
    response = client.get("/api/optimization/tools")
    assert response.status_code == 200
    body = response.json()
    expected = {"list_corridor_status", "run_optimization"}
    if satellite_tool_enabled():
        expected.add("verify_report_with_imagery")
    assert {item["name"] for item in body["declared_functions"]} == expected
    assert body["validation"]
