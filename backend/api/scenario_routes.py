"""Operator-facing access to the five explicitly simulated timeline fixtures."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.api.gemma_routes import broadcast_gemma_analysis
from backend.api.optimization_routes import broadcast_run
from backend.demo.scenario_replay import ScenarioFixture, load_scenarios
from backend.models.optimization import OptimizationRunRequest
from backend.services.gemma_service import gemma_service
from backend.services.optimization_service import optimization_service


router = APIRouter(prefix="/api/demo/scenarios", tags=["Demo scenarios"])
SCENARIO_DIR = Path(__file__).resolve().parents[1] / "demo" / "scenarios"


class ScenarioActivationRequest(BaseModel):
    stage: Literal["baseline", "disrupted"] = "baseline"
    requested_by: str = "mission-control-scenario-switcher"


@lru_cache(maxsize=4)
def _fixtures_at(_stamp: int) -> tuple[ScenarioFixture, ...]:
    return tuple(load_scenarios(SCENARIO_DIR))


def _fixtures() -> tuple[ScenarioFixture, ...]:
    """Scenario fixtures, reloaded whenever any fixture file changes.

    Scenario definitions are edited while rehearsing; a process-lifetime cache
    silently served the old closure set until the server was restarted.
    """
    stamp = max(
        (path.stat().st_mtime_ns for path in SCENARIO_DIR.glob("*.json")),
        default=0,
    )
    return _fixtures_at(stamp)


def _fixture(scenario_id: str) -> ScenarioFixture:
    match = next(
        (
            fixture
            for fixture in _fixtures()
            if fixture.scenario_id == scenario_id
        ),
        None,
    )
    if match is None:
        raise KeyError(scenario_id)
    return match


def _summary(fixture: ScenarioFixture) -> dict:
    closure = next(
        step
        for step in fixture.timeline
        if step.event_type == "road_block_report"
    )
    first_evidence = fixture.timeline[0].evidence[0]
    return {
        "scenario_id": fixture.scenario_id,
        "title": fixture.title,
        "description": fixture.description,
        "simulated": True,
        "expected_final_status": fixture.expected_final_status,
        "location": (
            {
                "latitude": first_evidence.reported_latitude,
                "longitude": first_evidence.reported_longitude,
            }
            if first_evidence.reported_latitude is not None
            else None
        ),
        "closure": {
            "t_seconds": closure.t_seconds,
            "blocked_edge_ids": list(closure.blocked_edge_ids),
            "reason": closure.reason,
        },
        "timeline": [
            {
                "t_seconds": step.t_seconds,
                "step_id": step.step_id,
                "event_type": step.event_type,
                "label": step.step_id.replace("-", " ").title(),
            }
            for step in fixture.timeline
        ],
    }


# Extraction results for a scenario stage, keyed by scenario and stage.
#
# A scenario's evidence is fixture text that never changes, so re-extracting it
# is a pure re-computation of a known answer — and an expensive one: the
# disrupted stage makes two hosted calls and was measured at 63 seconds, with
# the interface blocked the whole time. Re-selecting a scenario during a demo
# now reuses the extraction and recomputes only the plan, which takes ~0.2s.
#
# Only the analysis is cached. Every activation still mints a fresh versioned
# run, so approval history and supersession behave exactly as before.
_ANALYSIS_CACHE: dict[tuple[str, str], object] = {}


def _extract_once(cache_key: tuple[str, str], scenario_id: str, evidence: list):
    cached = _ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    analysis = gemma_service.analyze_submitted(scenario_id, evidence)
    _ANALYSIS_CACHE[cache_key] = analysis
    return analysis


def _activate(
    fixture: ScenarioFixture,
    request: ScenarioActivationRequest,
):
    first_step = fixture.timeline[0]
    baseline_evidence = [
        item.to_record(fixture.scenario_id, first_step.t_seconds)
        for item in first_step.evidence
    ]
    baseline_analysis = _extract_once(
        (fixture.scenario_id, "baseline"),
        fixture.scenario_id,
        baseline_evidence,
    )
    baseline_analysis.fixture_notice = (
        "Selected mocked timeline scenario. Not a live government or field feed."
    )
    baseline_run = optimization_service.run(
        OptimizationRunRequest(
            scenario_id=fixture.scenario_id,
            analysis_id=baseline_analysis.analysis_id,
            requested_by=request.requested_by,
            trigger="scenario_baseline",
        ),
        baseline_analysis,
    )

    if request.stage == "baseline":
        return baseline_analysis, baseline_run, baseline_run

    closure = next(
        step
        for step in fixture.timeline
        if step.event_type == "road_block_report"
    )
    closure_evidence = [
        item.to_record(fixture.scenario_id, closure.t_seconds)
        for item in closure.evidence
    ]
    disrupted_analysis = _extract_once(
        (fixture.scenario_id, "disrupted"),
        fixture.scenario_id,
        baseline_evidence + closure_evidence,
    )
    disrupted_analysis.fixture_notice = (
        "Selected mocked timeline scenario with a simulated road disruption. "
        "Not a live government or field feed."
    )
    child_run = optimization_service.run(
        OptimizationRunRequest(
            scenario_id=fixture.scenario_id,
            analysis_id=disrupted_analysis.analysis_id,
            requested_by=request.requested_by,
            blocked_edge_ids=list(closure.blocked_edge_ids),
            parent_run_id=baseline_run.run_id,
            trigger="road_closure",
            disruption_reason=closure.reason,
        ),
        disrupted_analysis,
    )
    return disrupted_analysis, child_run, baseline_run


@router.get("")
async def list_demo_scenarios():
    """List the simulated stories available in the mission-control selector."""
    return {"scenarios": [_summary(fixture) for fixture in _fixtures()]}


@router.post("/{scenario_id}/activate")
async def activate_demo_scenario(
    scenario_id: str,
    request: ScenarioActivationRequest,
):
    """Load a baseline or post-disruption scenario into the active runtime."""
    try:
        fixture = _fixture(scenario_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Demo scenario '{scenario_id}' not found",
        )

    analysis, run, baseline_run = await run_in_threadpool(
        _activate,
        fixture,
        request,
    )
    await broadcast_gemma_analysis(analysis)
    await broadcast_run(run)
    return {
        "scenario": _summary(fixture),
        "stage": request.stage,
        "analysis": analysis.model_dump(mode="json"),
        "run": run.model_dump(mode="json"),
        "baseline_run_id": baseline_run.run_id,
    }
