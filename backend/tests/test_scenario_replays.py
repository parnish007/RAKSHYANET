"""Regression coverage for the active timeline-driven demo scenarios."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.demo.scenario_replay import (
    ScenarioFixture,
    ScenarioReplayEngine,
    load_scenarios,
)


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "demo" / "scenarios"
SCENARIOS = load_scenarios(SCENARIO_DIR)


def test_exactly_five_explicitly_simulated_scenarios_are_available():
    assert len(SCENARIOS) == 5
    assert all(scenario.simulated is True for scenario in SCENARIOS)
    assert len({scenario.scenario_id for scenario in SCENARIOS}) == 5


def test_scenarios_cover_distinct_road_closures_and_both_human_decisions():
    blocked_edges = {
        edge_id
        for scenario in SCENARIOS
        for step in scenario.timeline
        for edge_id in step.blocked_edge_ids
    }
    final_statuses = {
        scenario.expected_final_status for scenario in SCENARIOS
    }

    assert len(blocked_edges) == 5
    assert final_statuses == {"approved", "rejected"}


@pytest.mark.parametrize(
    "scenario",
    SCENARIOS,
    ids=[scenario.scenario_id for scenario in SCENARIOS],
)
def test_full_scenario_replay_exercises_active_product_pipeline(scenario):
    engine = ScenarioReplayEngine()
    result = engine.replay(scenario)

    assert result.baseline_run_id != result.child_run_id
    assert result.final_status == scenario.expected_final_status
    assert result.route_feasible is True
    assert result.blocked_edge_ids
    assert len(result.analysis_ids) == 2
    assert len(result.entries) == len(scenario.timeline)
    assert [entry.event_type for entry in result.entries] == [
        "evidence_report",
        "optimization_requested",
        "road_block_report",
        "evidence_disposition",
        "review_decision",
    ]

    baseline = result.entries[1]
    closure = result.entries[2]
    disposition = result.entries[3]
    decision = result.entries[4]

    assert baseline.run_id == result.baseline_run_id
    assert baseline.analysis_id == result.analysis_ids[0]
    assert baseline.parent_run_id is None
    assert baseline.run_status == "awaiting_approval"
    assert baseline.route_feasible is True
    assert baseline.route_count > 0

    assert closure.run_id == result.child_run_id
    assert closure.analysis_id == result.analysis_ids[1]
    assert closure.parent_run_id == result.baseline_run_id
    assert closure.run_status == "awaiting_approval"
    assert closure.route_feasible is True
    assert set(result.blocked_edge_ids) <= set(closure.blocked_edge_ids)

    assert disposition.question_status in {"assigned", "unavailable"}
    assert decision.decision_status == scenario.expected_final_status

    latest_analysis = engine.gemma.latest()
    assert latest_analysis is not None
    # Which provider answers depends on whether a hosted key is reachable from
    # the machine running the suite, so pinning one of them made this test
    # assert that hosted Gemma was DOWN. The property that actually matters is
    # below: whichever provider ran, the evidence stays marked simulated and
    # attributed to the fixture.
    assert latest_analysis.provider in {"mock_submitted_screening", "gemini_api"}
    assert all(item.simulated for item in latest_analysis.evidence)
    assert all(
        item.provider == "timeline_scenario_fixture"
        for item in latest_analysis.evidence
    )


def test_timeline_validation_rejects_non_chronological_steps():
    payload = SCENARIOS[0].model_dump(mode="json")
    payload["timeline"][1]["t_seconds"] = 999

    with pytest.raises(ValidationError, match="chronological"):
        ScenarioFixture.model_validate(payload)


def test_replay_rejects_an_unknown_road_edge():
    payload = SCENARIOS[0].model_dump(mode="json")
    payload["timeline"][2]["blocked_edge_ids"] = ["not-a-real-edge"]
    payload["timeline"][2]["expectation"]["blocked_edges_active"] = [
        "not-a-real-edge"
    ]
    scenario = ScenarioFixture.model_validate(payload)

    with pytest.raises(AssertionError, match="unknown blocked edges"):
        ScenarioReplayEngine().replay(scenario)

