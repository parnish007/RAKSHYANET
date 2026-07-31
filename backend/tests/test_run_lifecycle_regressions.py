"""Regressions for three failures that could break the live demo.

Each test reproduces a defect that was observed against the running service:
the real solver error being replaced by an AttributeError, approval deadlocking
behind a failed run, and the offline fallback provider crashing on any evidence
count other than three.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.algorithms.state_manager import OptimizationResult, OptimizationState
from backend.models.gemma import EvidenceRecord
from backend.models.optimization import (
    OptimizationDecisionRequest,
    OptimizationRunRequest,
    OptimizationRunStatus,
)
from backend.services.gemma_service import DeterministicMockGemmaProvider
from backend.services.optimization_service import OptimizationService

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _request() -> OptimizationRunRequest:
    return OptimizationRunRequest(
        scenario_id="nepal-national-demo",
        analysis_id="ana_regression_fixture",
    )


def _decision(record) -> OptimizationDecisionRequest:
    return OptimizationDecisionRequest(
        reviewer="regression-operator",
        expected_updated_at=record.updated_at,
        expected_analysis_id=record.analysis_id or "no-analysis",
    )


def _fail_optimization(*_args, **_kwargs) -> OptimizationResult:
    return OptimizationResult(
        state=OptimizationState.ERROR,
        error_message="SIMULATED solver failure",
    )


def test_solver_failure_preserves_the_real_error(monkeypatch) -> None:
    """A failed stage must not have its cause overwritten by an AttributeError."""
    service = OptimizationService()
    monkeypatch.setattr(
        "backend.algorithms.state_manager.StateManager.run_full_optimization",
        _fail_optimization,
    )

    record = service.run(OptimizationRunRequest(scenario_id="nepal-national-demo"))

    assert record.status is OptimizationRunStatus.FAILED
    assert record.error == "SIMULATED solver failure"
    assert "NoneType" not in (record.error or "")
    assert record.approval_blockers, "operator must be told why approval is blocked"
    assert "SIMULATED solver failure" in record.approval_blockers[0]


def test_failed_run_does_not_deadlock_approval_of_the_good_run(monkeypatch) -> None:
    """A failed run never produced a plan, so it cannot supersede the run before it."""
    service = OptimizationService()
    good = service.run(_request())
    assert good.status is OptimizationRunStatus.AWAITING_APPROVAL

    monkeypatch.setattr(
        "backend.algorithms.state_manager.StateManager.run_full_optimization",
        _fail_optimization,
    )
    failed = service.run(_request())
    assert failed.status is OptimizationRunStatus.FAILED

    approved = service.approve(good.run_id, _decision(good))
    assert approved.status is OptimizationRunStatus.APPROVED


def test_a_successful_newer_run_still_supersedes() -> None:
    """The staleness rule itself must survive the deadlock fix."""
    service = OptimizationService()
    first = service.run(_request())
    service.run(_request())

    with pytest.raises(ValueError, match="stale"):
        service.approve(first.run_id, _decision(first))


def _fixture_evidence() -> list[EvidenceRecord]:
    payload = json.loads((DATA_DIR / "demo_evidence.json").read_text(encoding="utf-8"))
    return [EvidenceRecord.model_validate(item) for item in payload["evidence"]]


@pytest.mark.parametrize("count", [1, 2, 3, 4, 6])
def test_fallback_provider_accepts_any_evidence_count(count: int) -> None:
    """The offline rescue path must not crash when the fixture is edited."""
    base = _fixture_evidence()
    evidence: list[EvidenceRecord] = []
    for index in range(count):
        source = base[index % len(base)]
        evidence.append(
            source.model_copy(update={"evidence_id": f"{source.evidence_id}-{index}"})
        )

    record = DeterministicMockGemmaProvider().analyze(
        scenario_id="nepal-national-demo",
        evidence=evidence,
        fallback_reason="regression test",
    )

    cited = {
        evidence_id
        for evidence_id in record.output.incident_type.evidence_ids
    }
    known = {item.evidence_id for item in evidence}
    assert cited, "the fallback must still cite its evidence"
    assert cited <= known, "citations must reference evidence that was supplied"
