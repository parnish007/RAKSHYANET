"""
Tests for ImpactAnalyzer -- Prompt 4.1
Run: pytest backend/tests/test_impact_analyzer.py -v
"""
import pytest
from typing import Dict, List

from backend.hitl.approval_queue import ImpactPreview
from backend.hitl.impact_analyzer import ImpactAnalyzer
from backend.models.village import Village
from backend.rag.news_analyzer import NewsEvent
from backend.algorithms.vrp_solver import VRPSolution, VillageAllocation, Route


# ================================================================== #
#  Helpers and fixtures                                                #
# ================================================================== #

def _make_village(
    vid: str,
    urgency: float = 0.5,
    population: int = 1000,
) -> Village:
    return Village(
        id=vid,
        name=vid.capitalize(),
        lat=27.6,
        lng=85.5,
        population=population,
        terrain_difficulty=1.5,
        urgency_score=urgency,
        disaster_impact=0.5,
    )


def _make_event(
    event_id: str = "evt_t",
    severity: int = 7,
    confidence: float = 0.65,
    affected: list | None = None,
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        raw_text="Landslide near Dhulikhel — road blocked.",
        location=["Dhulikhel"],
        severity=severity,
        confidence=confidence,
        affected_villages=affected or ["dhulikhel", "panauti"],
        resource_implications={"medical_kit": 30.0, "rescue_equipment": 20.0},
        requires_hitl=True,
    )


def _make_vrp(villages: List[Village]) -> VRPSolution:
    allocations = [
        VillageAllocation(
            village_id=v.id,
            allocated_resources={"food": 50.0, "medical_kit": 30.0},
            vehicle_assignments=["v1"],
            eta_minutes=60.0,
            satisfied=True,
        )
        for v in villages
    ]
    route = Route(
        vehicle_id="v1",
        stops=[v.id for v in villages],
        total_distance_km=100.0,
        estimated_time_hours=2.0,
    )
    return VRPSolution(
        routes=[route],
        allocations=allocations,
        total_distance_km=100.0,
    )


@pytest.fixture
def villages() -> List[Village]:
    return [
        _make_village("dhulikhel", urgency=0.7),
        _make_village("panauti",   urgency=0.5),
        _make_village("banepa",    urgency=0.3),
    ]


@pytest.fixture
def vrp(villages) -> VRPSolution:
    return _make_vrp(villages)


@pytest.fixture
def analyzer(villages, vrp) -> ImpactAnalyzer:
    return ImpactAnalyzer(current_villages=villages, current_routes=vrp)


@pytest.fixture
def event() -> NewsEvent:
    return _make_event()


# ================================================================== #
#  Impact calculation tests                                            #
# ================================================================== #

class TestCalculateImpact:
    def test_returns_impact_preview(self, analyzer, event):
        preview = analyzer.calculate_impact(event)
        assert isinstance(preview, ImpactPreview)

    def test_urgency_changes_contains_affected_villages(self, analyzer, event):
        preview = analyzer.calculate_impact(event)
        for vid in event.affected_villages:
            assert vid in preview.urgency_changes

    def test_resource_reallocation_is_dict(self, analyzer, event):
        preview = analyzer.calculate_impact(event)
        assert isinstance(preview.resource_reallocation, dict)

    def test_eta_changes_are_integers(self, analyzer, event):
        preview = analyzer.calculate_impact(event)
        for vid, delta in preview.eta_changes.items():
            assert isinstance(delta, int)

    def test_welfare_improvement_estimate_is_float(self, analyzer, event):
        preview = analyzer.calculate_impact(event)
        assert isinstance(preview.welfare_improvement_estimate, float)

    def test_welfare_improvement_estimate_in_range(self, analyzer, event):
        preview = analyzer.calculate_impact(event)
        assert 0.0 <= preview.welfare_improvement_estimate <= 1.0

    def test_affected_villages_matches_event(self, analyzer, event):
        preview = analyzer.calculate_impact(event)
        assert set(preview.affected_villages) == set(event.affected_villages)


# ================================================================== #
#  Urgency estimation tests                                            #
# ================================================================== #

class TestUrgencyEstimation:
    def test_higher_severity_gives_larger_delta(self, analyzer, villages):
        low_sev  = _make_event("e1", severity=3, affected=["dhulikhel"])
        high_sev = _make_event("e2", severity=9, affected=["dhulikhel"])
        low_chg  = analyzer.estimate_urgency_changes(low_sev)
        high_chg = analyzer.estimate_urgency_changes(high_sev)
        assert high_chg["dhulikhel"] > low_chg["dhulikhel"]

    def test_multiple_affected_villages_all_show_delta(self, analyzer):
        event = _make_event(affected=["dhulikhel", "panauti", "banepa"])
        chg   = analyzer.estimate_urgency_changes(event)
        assert len(chg) == 3
        for vid in ["dhulikhel", "panauti", "banepa"]:
            assert vid in chg
            assert chg[vid] > 0.0

    def test_unaffected_village_not_in_urgency_changes(self, analyzer):
        event = _make_event(affected=["dhulikhel"])
        chg   = analyzer.estimate_urgency_changes(event)
        assert "banepa" not in chg

    def test_urgency_delta_is_positive_for_positive_severity(self, analyzer):
        event = _make_event(severity=5, affected=["panauti"])
        chg   = analyzer.estimate_urgency_changes(event)
        assert chg["panauti"] > 0.0

    def test_severity_zero_gives_zero_delta(self, analyzer):
        event = _make_event(severity=0, affected=["panauti"])
        chg   = analyzer.estimate_urgency_changes(event)
        assert chg["panauti"] == pytest.approx(0.0, abs=1e-6)

    def test_unknown_village_gets_default_weight(self, analyzer):
        event = _make_event(severity=5, affected=["unknown_village"])
        chg   = analyzer.estimate_urgency_changes(event)
        assert "unknown_village" in chg
        assert chg["unknown_village"] > 0.0


# ================================================================== #
#  Resource shift tests                                                #
# ================================================================== #

class TestResourceShifts:
    def test_shifts_non_negative(self, analyzer, event):
        chg    = analyzer.estimate_urgency_changes(event)
        shifts = analyzer.estimate_resource_shifts(chg)
        for village_shifts in shifts.values():
            for qty in village_shifts.values():
                assert qty >= 0.0

    def test_empty_urgency_changes_returns_empty(self, analyzer):
        shifts = analyzer.estimate_resource_shifts({})
        assert shifts == {}


# ================================================================== #
#  ETA impact tests                                                    #
# ================================================================== #

class TestETAImpacts:
    def test_positive_urgency_gives_negative_eta_delta(self, analyzer):
        chg  = {"dhulikhel": 0.5}
        etas = analyzer.estimate_eta_impacts(chg)
        assert etas["dhulikhel"] <= 0   # earlier arrival

    def test_zero_urgency_gives_zero_eta_delta(self, analyzer):
        chg  = {"panauti": 0.0}
        etas = analyzer.estimate_eta_impacts(chg)
        assert etas["panauti"] == 0


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def test_full_pipeline_with_real_event(self, analyzer):
        event   = _make_event("evt_real", severity=8, confidence=0.72,
                              affected=["dhulikhel", "panauti"])
        preview = analyzer.calculate_impact(event)
        assert len(preview.affected_villages) == 2
        assert preview.urgency_changes["dhulikhel"] > 0
        assert preview.urgency_changes["panauti"] > 0
        assert isinstance(preview.welfare_improvement_estimate, float)

    def test_preview_structure_complete(self, analyzer, event):
        preview = analyzer.calculate_impact(event)
        assert hasattr(preview, "urgency_changes")
        assert hasattr(preview, "affected_villages")
        assert hasattr(preview, "resource_reallocation")
        assert hasattr(preview, "eta_changes")
        assert hasattr(preview, "welfare_improvement_estimate")
