"""Deterministic end-to-end replay for mocked route-intelligence timelines.

The legacy :mod:`timeline_simulator` exercises rule-based news routing. This
module exercises the active product path instead:

    submitted evidence -> Gemma safety boundary -> optimization -> road closure
    -> evidence ownership -> immutable human decision

Fixtures are intentionally marked as simulated. Hosted model calls are disabled
inside the replay engine so regression tests remain deterministic and offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, model_validator

from backend.models.gemma import (
    EvidenceQuestionDispositionRequest,
    EvidenceRecord,
    StrictModel,
)
from backend.models.optimization import (
    OptimizationDecisionRequest,
    OptimizationRunRequest,
)
from backend.services.gemma_service import GemmaAnalysisService
from backend.services.optimization_service import OptimizationService


TimelineEventType = Literal[
    "evidence_report",
    "optimization_requested",
    "road_block_report",
    "evidence_disposition",
    "review_decision",
]


class MockEvidence(StrictModel):
    """One explicitly simulated evidence record in a replay fixture."""

    evidence_id: str = Field(
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9._:/-]+$",
    )
    source_category: str = Field(min_length=1, max_length=100)
    source_name: str = Field(min_length=1, max_length=200)
    source_identifier: str = Field(min_length=1, max_length=500)
    freshness_minutes: int = Field(default=0, ge=0)
    reliability: float = Field(ge=0.0, le=1.0)
    text: str = Field(min_length=10, max_length=20_000)
    operator_context: Optional[str] = Field(default=None, max_length=2_000)
    gap_target: Optional[str] = Field(default=None, max_length=200)
    reported_latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)
    reported_longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)

    @model_validator(mode="after")
    def validate_coordinates(self) -> "MockEvidence":
        if (self.reported_latitude is None) != (
            self.reported_longitude is None
        ):
            raise ValueError(
                "Mock evidence latitude and longitude must be supplied together"
            )
        return self

    def to_record(self, scenario_id: str, t_seconds: int) -> EvidenceRecord:
        """Convert fixture evidence to the active strict evidence contract."""
        return EvidenceRecord(
            evidence_id=self.evidence_id,
            source_category=self.source_category,
            source_name=self.source_name,
            source_identifier=self.source_identifier,
            retrieved_at=f"mock://{scenario_id}/t+{t_seconds}",
            freshness_minutes=self.freshness_minutes,
            reliability=self.reliability,
            text=self.text,
            provider="timeline_scenario_fixture",
            cache_status="fixture",
            simulated=True,
            operator_context=self.operator_context,
            gap_target=self.gap_target,
            reported_latitude=self.reported_latitude,
            reported_longitude=self.reported_longitude,
        )


class StepExpectation(StrictModel):
    """Machine-checkable expected state after one timeline step."""

    analysis_needs_human_review: Optional[bool] = None
    run_status: Optional[str] = None
    route_feasible: Optional[bool] = None
    minimum_route_count: Optional[int] = Field(default=None, ge=0)
    child_run: Optional[bool] = None
    blocked_edges_active: List[str] = Field(default_factory=list)
    question_status: Optional[Literal["assigned", "unavailable"]] = None
    decision_status: Optional[Literal["approved", "rejected"]] = None


class ScenarioTimelineStep(StrictModel):
    """A single event in an active-pipeline scenario."""

    t_seconds: int = Field(ge=0)
    step_id: str = Field(min_length=1, max_length=120)
    event_type: TimelineEventType
    evidence: List[MockEvidence] = Field(default_factory=list, max_length=10)
    blocked_edge_ids: List[str] = Field(default_factory=list, max_length=20)
    reason: Optional[str] = Field(default=None, max_length=1_000)
    question_index: Optional[int] = Field(default=None, ge=0, le=9)
    disposition_status: Optional[
        Literal["assigned", "unavailable"]
    ] = None
    owner: Optional[str] = Field(default=None, max_length=120)
    decision: Optional[Literal["approve", "reject"]] = None
    reviewer: Optional[str] = Field(default=None, max_length=120)
    expectation: StepExpectation = Field(default_factory=StepExpectation)

    @model_validator(mode="after")
    def validate_event_payload(self) -> "ScenarioTimelineStep":
        if self.event_type in {"evidence_report", "road_block_report"}:
            if not self.evidence:
                raise ValueError(
                    f"{self.event_type} requires at least one evidence record"
                )
        elif self.evidence:
            raise ValueError(
                f"{self.event_type} cannot contain evidence records"
            )

        if self.event_type == "road_block_report":
            if not self.blocked_edge_ids:
                raise ValueError(
                    "road_block_report requires blocked_edge_ids"
                )
            if not self.reason:
                raise ValueError("road_block_report requires a reason")
        elif self.blocked_edge_ids:
            raise ValueError(
                f"{self.event_type} cannot contain blocked_edge_ids"
            )

        if self.event_type == "evidence_disposition":
            if self.question_index is None:
                raise ValueError(
                    "evidence_disposition requires question_index"
                )
            if self.disposition_status is None or not self.owner or not self.reason:
                raise ValueError(
                    "evidence_disposition requires status, owner, and reason"
                )

        if self.event_type == "review_decision":
            if self.decision is None or not self.reviewer or not self.reason:
                raise ValueError(
                    "review_decision requires decision, reviewer, and reason"
                )
        return self


class ScenarioFixture(StrictModel):
    """One complete, explicitly mocked product story."""

    schema_version: Literal["1.0"]
    scenario_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9-]+$",
    )
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1_000)
    simulated: Literal[True]
    expected_final_status: Literal["approved", "rejected"]
    timeline: List[ScenarioTimelineStep] = Field(min_length=5, max_length=20)

    @model_validator(mode="after")
    def validate_timeline(self) -> "ScenarioFixture":
        times = [step.t_seconds for step in self.timeline]
        if times != sorted(times):
            raise ValueError("Scenario timeline must be chronological")
        step_ids = [step.step_id for step in self.timeline]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Scenario step IDs must be unique")
        evidence_ids = [
            evidence.evidence_id
            for step in self.timeline
            for evidence in step.evidence
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Scenario evidence IDs must be unique")

        event_types = [step.event_type for step in self.timeline]
        required = {
            "evidence_report",
            "optimization_requested",
            "road_block_report",
            "evidence_disposition",
            "review_decision",
        }
        missing = required - set(event_types)
        if missing:
            raise ValueError(
                "Scenario is missing lifecycle steps: "
                + ", ".join(sorted(missing))
            )
        if event_types[0] != "evidence_report":
            raise ValueError("Scenario must begin with evidence_report")
        if event_types.index("optimization_requested") > event_types.index(
            "road_block_report"
        ):
            raise ValueError(
                "Baseline optimization must precede the road block"
            )
        if event_types[-1] != "review_decision":
            raise ValueError("Scenario must end with review_decision")
        return self


class ReplayEntry(StrictModel):
    """Compact audit record produced for every replayed timeline event."""

    t_seconds: int
    step_id: str
    event_type: TimelineEventType
    analysis_id: Optional[str] = None
    provider: Optional[str] = None
    run_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    run_status: Optional[str] = None
    route_feasible: Optional[bool] = None
    route_count: Optional[int] = None
    blocked_edge_ids: List[str] = Field(default_factory=list)
    question_status: Optional[str] = None
    decision_status: Optional[str] = None


class ScenarioReplayResult(StrictModel):
    """Final reproducible result of one fixture replay."""

    scenario_id: str
    baseline_run_id: str
    child_run_id: str
    analysis_ids: List[str]
    final_status: Literal["approved", "rejected"]
    route_feasible: bool
    blocked_edge_ids: List[str]
    entries: List[ReplayEntry]


def load_scenario(path: Path) -> ScenarioFixture:
    """Load and strictly validate one scenario JSON document."""
    return ScenarioFixture.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def load_scenarios(directory: Path) -> List[ScenarioFixture]:
    """Load all scenario fixtures in stable filename order."""
    return [
        load_scenario(path)
        for path in sorted(directory.glob("*.json"))
    ]


class ScenarioReplayEngine:
    """Replay mocked timelines through the active Gemma and planning services."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = (
            data_dir
            or Path(__file__).resolve().parents[1] / "data"
        )
        self.gemma = GemmaAnalysisService(data_dir=self.data_dir)
        # Tests and CLI replays must not depend on credentials or network.
        self.gemma.online_provider.api_key = ""
        self.optimizer = OptimizationService(data_dir=self.data_dir)
        terrain = json.loads(
            (self.data_dir / "terrain_graph.json").read_text(encoding="utf-8")
        )
        self.known_edge_ids = {
            edge["id"] for edge in terrain.get("edges", [])
        }

    def replay(self, scenario: ScenarioFixture) -> ScenarioReplayResult:
        """Execute one full timeline and enforce every fixture expectation."""
        self.gemma.analyses.clear()
        self.gemma.analysis_order.clear()
        self.optimizer.runs.clear()
        self.optimizer.run_order.clear()

        evidence: List[EvidenceRecord] = []
        analysis = None
        current_run = None
        baseline_run_id: Optional[str] = None
        child_run_id: Optional[str] = None
        analysis_ids: List[str] = []
        entries: List[ReplayEntry] = []
        all_blocked_edges: List[str] = []

        for step in scenario.timeline:
            entry = ReplayEntry(
                t_seconds=step.t_seconds,
                step_id=step.step_id,
                event_type=step.event_type,
            )

            if step.event_type == "evidence_report":
                evidence.extend(
                    item.to_record(scenario.scenario_id, step.t_seconds)
                    for item in step.evidence
                )
                analysis = self.gemma.analyze_submitted(
                    scenario.scenario_id,
                    evidence,
                )
                analysis_ids.append(analysis.analysis_id)
                entry.analysis_id = analysis.analysis_id
                entry.provider = analysis.provider
                self._assert_analysis(step, analysis)

            elif step.event_type == "optimization_requested":
                if analysis is None:
                    raise AssertionError(
                        f"{step.step_id}: optimization has no analysis"
                    )
                current_run = self.optimizer.run(
                    OptimizationRunRequest(
                        scenario_id=scenario.scenario_id,
                        analysis_id=analysis.analysis_id,
                        requested_by="timeline-replay",
                        trigger="initial_evidence",
                    ),
                    analysis,
                )
                baseline_run_id = current_run.run_id
                self._assert_run(step, current_run, parent_expected=False)
                entry = self._run_entry(entry, current_run)

            elif step.event_type == "road_block_report":
                if current_run is None:
                    raise AssertionError(
                        f"{step.step_id}: road block has no baseline run"
                    )
                unknown_edges = (
                    set(step.blocked_edge_ids) - self.known_edge_ids
                )
                if unknown_edges:
                    raise AssertionError(
                        f"{step.step_id}: unknown blocked edges "
                        + ", ".join(sorted(unknown_edges))
                    )
                evidence.extend(
                    item.to_record(scenario.scenario_id, step.t_seconds)
                    for item in step.evidence
                )
                analysis = self.gemma.analyze_submitted(
                    scenario.scenario_id,
                    evidence,
                )
                analysis_ids.append(analysis.analysis_id)
                parent_run_id = current_run.run_id
                current_run = self.optimizer.run(
                    OptimizationRunRequest(
                        scenario_id=scenario.scenario_id,
                        analysis_id=analysis.analysis_id,
                        requested_by="timeline-replay",
                        blocked_edge_ids=step.blocked_edge_ids,
                        parent_run_id=parent_run_id,
                        trigger="road_closure",
                        disruption_reason=step.reason,
                    ),
                    analysis,
                )
                child_run_id = current_run.run_id
                all_blocked_edges.extend(step.blocked_edge_ids)
                self._assert_run(step, current_run, parent_expected=True)
                self._assert_blocked_edges_absent(
                    step,
                    current_run,
                )
                entry = self._run_entry(entry, current_run)
                entry.analysis_id = analysis.analysis_id
                entry.provider = analysis.provider

            elif step.event_type == "evidence_disposition":
                if analysis is None:
                    raise AssertionError(
                        f"{step.step_id}: disposition has no analysis"
                    )
                questions = analysis.output.follow_up_questions
                index = step.question_index
                if index is None or index >= len(questions):
                    raise AssertionError(
                        f"{step.step_id}: question index {index} is unavailable"
                    )
                question_id = f"question-{index}"
                analysis = self.gemma.record_question_disposition(
                    analysis.analysis_id,
                    question_id,
                    EvidenceQuestionDispositionRequest(
                        status=step.disposition_status,
                        owner=step.owner,
                        reason=step.reason,
                    ),
                )
                disposition = next(
                    item
                    for item in analysis.question_dispositions
                    if item.question_id == question_id
                )
                entry.analysis_id = analysis.analysis_id
                entry.question_status = disposition.status
                expected = step.expectation.question_status
                if expected is not None and disposition.status != expected:
                    raise AssertionError(
                        f"{step.step_id}: expected question status "
                        f"{expected}, got {disposition.status}"
                    )

            elif step.event_type == "review_decision":
                if current_run is None:
                    raise AssertionError(
                        f"{step.step_id}: review has no run"
                    )
                decision = OptimizationDecisionRequest(
                    reviewer=step.reviewer,
                    notes=step.reason,
                    expected_updated_at=current_run.updated_at,
                    expected_analysis_id=current_run.analysis_id,
                )
                current_run = (
                    self.optimizer.approve(current_run.run_id, decision)
                    if step.decision == "approve"
                    else self.optimizer.reject(current_run.run_id, decision)
                )
                entry = self._run_entry(entry, current_run)
                entry.decision_status = current_run.status.value
                expected = step.expectation.decision_status
                if (
                    expected is not None
                    and current_run.status.value != expected
                ):
                    raise AssertionError(
                        f"{step.step_id}: expected decision {expected}, "
                        f"got {current_run.status.value}"
                    )

            entries.append(entry)

        if current_run is None or baseline_run_id is None or child_run_id is None:
            raise AssertionError(
                f"{scenario.scenario_id}: incomplete replay result"
            )
        if current_run.status.value != scenario.expected_final_status:
            raise AssertionError(
                f"{scenario.scenario_id}: expected final status "
                f"{scenario.expected_final_status}, got "
                f"{current_run.status.value}"
            )

        return ScenarioReplayResult(
            scenario_id=scenario.scenario_id,
            baseline_run_id=baseline_run_id,
            child_run_id=child_run_id,
            analysis_ids=analysis_ids,
            final_status=current_run.status.value,
            route_feasible=current_run.route_feasible is True,
            blocked_edge_ids=list(dict.fromkeys(all_blocked_edges)),
            entries=entries,
        )

    @staticmethod
    def _run_entry(
        entry: ReplayEntry,
        run,
    ) -> ReplayEntry:
        entry.analysis_id = run.analysis_id
        entry.run_id = run.run_id
        entry.parent_run_id = run.parent_run_id
        entry.run_status = run.status.value
        entry.route_feasible = run.route_feasible
        entry.route_count = len(run.result.vrp_solution.routes)
        entry.blocked_edge_ids = list(run.blocked_edge_ids)
        return entry

    @staticmethod
    def _assert_analysis(step, analysis) -> None:
        expected = step.expectation.analysis_needs_human_review
        if (
            expected is not None
            and analysis.output.needs_human_review != expected
        ):
            raise AssertionError(
                f"{step.step_id}: expected needs_human_review={expected}, "
                f"got {analysis.output.needs_human_review}"
            )

    @staticmethod
    def _assert_run(step, run, parent_expected: bool) -> None:
        expectation = step.expectation
        if (
            expectation.run_status is not None
            and run.status.value != expectation.run_status
        ):
            raise AssertionError(
                f"{step.step_id}: expected run status "
                f"{expectation.run_status}, got {run.status.value}"
            )
        if (
            expectation.route_feasible is not None
            and run.route_feasible != expectation.route_feasible
        ):
            raise AssertionError(
                f"{step.step_id}: expected route_feasible="
                f"{expectation.route_feasible}, got {run.route_feasible}"
            )
        route_count = len(run.result.vrp_solution.routes)
        if (
            expectation.minimum_route_count is not None
            and route_count < expectation.minimum_route_count
        ):
            raise AssertionError(
                f"{step.step_id}: expected at least "
                f"{expectation.minimum_route_count} routes, got {route_count}"
            )
        if expectation.child_run is not None:
            actual = run.parent_run_id is not None
            if actual != expectation.child_run:
                raise AssertionError(
                    f"{step.step_id}: expected child_run="
                    f"{expectation.child_run}, got {actual}"
                )
        if parent_expected and run.parent_run_id is None:
            raise AssertionError(
                f"{step.step_id}: road closure did not create a child run"
            )
        expected_blocks = set(expectation.blocked_edges_active)
        active_blocks = set(run.result.vrp_solution.active_road_blocks)
        if not expected_blocks <= active_blocks:
            raise AssertionError(
                f"{step.step_id}: missing active blocks "
                + ", ".join(sorted(expected_blocks - active_blocks))
            )

    @staticmethod
    def _assert_blocked_edges_absent(step, run) -> None:
        blocked = set(step.blocked_edge_ids)
        for route in run.result.vrp_solution.routes:
            if route.transport_mode != "road" or not route.feasible:
                continue
            used = blocked & set(route.road_edge_ids)
            if used:
                raise AssertionError(
                    f"{step.step_id}: feasible road route "
                    f"{route.vehicle_id} still uses blocked edges "
                    + ", ".join(sorted(used))
                )
