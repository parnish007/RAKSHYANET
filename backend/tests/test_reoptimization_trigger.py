"""
Tests for ReoptimizationTrigger -- Prompt 5.2
Run: pytest backend/tests/test_reoptimization_trigger.py -v
"""
import json
from datetime import timedelta
from pathlib import Path

import pytest

from backend.demo.reoptimization_trigger import (
    OptimizationChange,
    ReoptimizationConfig,
    ReoptimizationTrigger,
)
from backend.algorithms.state_manager import StateManager
from backend.algorithms.vrp_solver import Route, VRPSolution, VillageAllocation
from backend.models.resource import ResourceType
from backend.models.vehicle import Vehicle, VehicleCategory, VehicleType, TerrainCapability
from backend.models.village import Village
from backend.rag.news_analyzer import IntelligenceReport, NewsEvent

DATA = Path(__file__).parent.parent / "data"


# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

def _load_state_manager() -> StateManager:
    config        = json.loads((DATA / "config.json").read_text())
    terrain_graph = json.loads((DATA / "terrain_graph.json").read_text())
    resource_types = {
        k: ResourceType(**v)
        for k, v in config.get("resource_types", {}).items()
    }
    return StateManager(
        depot_location=(27.7172, 85.3240),
        depot_resources={rt: 500.0 for rt in resource_types},
        terrain_graph=terrain_graph,
        resource_types=resource_types,
        config=config,
    )


def _make_village(vid: str, urgency: float = 0.5) -> Village:
    return Village(
        id=vid, name=vid.capitalize(),
        lat=27.6, lng=85.5,
        population=2000,
        terrain_difficulty=1.5,
        urgency_score=urgency,
        disaster_impact=0.5,
    )


def _make_vehicle(vid: str) -> Vehicle:
    vtype = VehicleType(
        type_id="truck",
        name="Truck",
        category=VehicleCategory.GROUND_HEAVY,
        capacity_kg=2000.0,
        speed_kmh=60.0,
        fuel_hours=8.0,
        terrain_capability=TerrainCapability.ALL_ROADS,
    )
    return Vehicle(id=vid, name=f"Truck {vid}", vehicle_type=vtype)


def _make_report(
    event_id: str = "evt_test",
    urgency_change: dict | None = None,
    confidence: float = 0.9,
) -> IntelligenceReport:
    event = NewsEvent(
        event_id=event_id,
        raw_text="Test earthquake event.",
        severity=7,
        confidence=confidence,
        affected_villages=list((urgency_change or {}).keys()),
        resource_implications={"food": 20.0},
    )
    return IntelligenceReport(
        event=event,
        urgency_change=urgency_change or {},
        recommended_action="AUTO_OPTIMIZE",
    )


@pytest.fixture
def villages():
    return [
        _make_village("dhulikhel", urgency=0.6),
        _make_village("panauti",   urgency=0.5),
        _make_village("banepa",    urgency=0.4),
    ]


@pytest.fixture
def vehicles():
    return [_make_vehicle("v1"), _make_vehicle("v2")]


@pytest.fixture
def state_manager():
    return _load_state_manager()


@pytest.fixture
def trigger(state_manager, villages, vehicles):
    cfg = ReoptimizationConfig(
        urgency_change_threshold=0.10,
        enable_reoptimization=True,
        broadcast_via_p2p=False,
        log_optimization_changes=False,
    )
    return ReoptimizationTrigger(
        config=cfg,
        state_manager=state_manager,
        villages=villages,
        vehicles=vehicles,
        time_elapsed=timedelta(hours=2),
    )


# ================================================================== #
#  Initialization tests                                                #
# ================================================================== #

class TestInitialization:
    def test_trigger_creates_successfully(self, trigger):
        assert trigger is not None

    def test_history_empty_on_init(self, trigger):
        assert trigger.optimization_history == []

    def test_config_stored(self, trigger):
        assert trigger.config.urgency_change_threshold == pytest.approx(0.10)

    def test_last_vrp_solution_none_initially(self, trigger):
        assert trigger._last_vrp_solution is None


# ================================================================== #
#  Threshold tests                                                     #
# ================================================================== #

class TestThreshold:
    def test_delta_above_threshold_returns_true(self, trigger):
        assert trigger.should_trigger_reoptimization({"dhulikhel": 0.15}) is True

    def test_delta_equal_threshold_returns_true(self, trigger):
        assert trigger.should_trigger_reoptimization({"dhulikhel": 0.10}) is True

    def test_delta_below_threshold_returns_false(self, trigger):
        assert trigger.should_trigger_reoptimization({"dhulikhel": 0.05}) is False

    def test_empty_dict_returns_false(self, trigger):
        assert trigger.should_trigger_reoptimization({}) is False

    def test_disabled_reoptimization_always_false(self, state_manager, villages, vehicles):
        cfg = ReoptimizationConfig(
            enable_reoptimization=False,
            log_optimization_changes=False,
        )
        t = ReoptimizationTrigger(cfg, state_manager, villages, vehicles)
        assert t.should_trigger_reoptimization({"dhulikhel": 0.9}) is False

    def test_negative_delta_also_checked_by_abs(self, trigger):
        assert trigger.should_trigger_reoptimization({"dhulikhel": -0.15}) is True

    def test_multiple_villages_any_triggers(self, trigger):
        assert trigger.should_trigger_reoptimization(
            {"dhulikhel": 0.02, "panauti": 0.20}
        ) is True


# ================================================================== #
#  Urgency update tests                                                #
# ================================================================== #

class TestUrgencyUpdates:
    def test_apply_updates_modifies_urgency(self, trigger, villages):
        original = villages[0].urgency_score
        trigger.apply_urgency_updates({"dhulikhel": 0.20})
        assert villages[0].urgency_score == pytest.approx(original + 0.20, abs=1e-6)

    def test_urgency_clamped_at_1_0(self, trigger, villages):
        trigger.apply_urgency_updates({"dhulikhel": 999.0})
        assert villages[0].urgency_score == pytest.approx(1.0)

    def test_urgency_clamped_at_0_0(self, trigger, villages):
        trigger.apply_urgency_updates({"dhulikhel": -999.0})
        assert villages[0].urgency_score == pytest.approx(0.0)

    def test_unknown_village_does_not_raise(self, trigger):
        trigger.apply_urgency_updates({"nonexistent_village": 0.5})

    def test_multiple_villages_all_updated(self, trigger, villages):
        trigger.apply_urgency_updates({"dhulikhel": 0.10, "panauti": 0.05})
        assert villages[0].urgency_score == pytest.approx(0.70, abs=1e-6)
        assert villages[1].urgency_score == pytest.approx(0.55, abs=1e-6)


# ================================================================== #
#  Trigger tests                                                       #
# ================================================================== #

class TestTrigger:
    def test_trigger_returns_optimization_change(self, trigger):
        report = _make_report(urgency_change={"dhulikhel": 0.15})
        change = trigger.trigger_reoptimization(report)
        assert isinstance(change, OptimizationChange)

    def test_trigger_with_no_urgency_change_raises(self, trigger):
        report = _make_report(urgency_change={})
        with pytest.raises(ValueError):
            trigger.trigger_reoptimization(report)

    def test_trigger_stores_event_id(self, trigger):
        report = _make_report("evt_flood", urgency_change={"dhulikhel": 0.15})
        change = trigger.trigger_reoptimization(report)
        assert change.trigger_event_id == "evt_flood"

    def test_trigger_execution_time_measured(self, trigger):
        report = _make_report(urgency_change={"dhulikhel": 0.15})
        change = trigger.trigger_reoptimization(report)
        assert change.execution_time_ms >= 0.0

    def test_trigger_adds_to_history(self, trigger):
        report = _make_report(urgency_change={"dhulikhel": 0.15})
        trigger.trigger_reoptimization(report)
        assert len(trigger.optimization_history) == 1

    def test_trigger_triggered_at_is_set(self, trigger):
        report = _make_report(urgency_change={"dhulikhel": 0.15})
        change = trigger.trigger_reoptimization(report)
        assert change.triggered_at != ""

    def test_trigger_optimization_state_is_complete(self, trigger):
        report = _make_report(urgency_change={"dhulikhel": 0.15})
        change = trigger.trigger_reoptimization(report)
        assert change.optimization_state == "complete"


# ================================================================== #
#  History tests                                                       #
# ================================================================== #

class TestHistory:
    def test_history_empty_initially(self, trigger):
        assert trigger.get_optimization_history() == []

    def test_history_accumulates(self, trigger):
        for i in range(3):
            report = _make_report(f"evt_{i}", urgency_change={"dhulikhel": 0.12})
            trigger.trigger_reoptimization(report)
        assert len(trigger.get_optimization_history()) == 3

    def test_history_sorted_newest_first(self, trigger):
        for i in range(3):
            report = _make_report(f"evt_{i}", urgency_change={"panauti": 0.12})
            trigger.trigger_reoptimization(report)
        history = trigger.get_optimization_history()
        timestamps = [h.triggered_at for h in history]
        assert timestamps == sorted(timestamps, reverse=True)


# ================================================================== #
#  Route-change counting tests                                         #
# ================================================================== #

class TestRouteChangeCounting:
    def _make_vrp(self, stops_by_vehicle: dict) -> VRPSolution:
        routes = [
            Route(
                vehicle_id=vid,
                stops=stops,
                total_distance_km=50.0,
                estimated_time_hours=1.5,
            )
            for vid, stops in stops_by_vehicle.items()
        ]
        return VRPSolution(routes=routes)

    def test_no_old_solution_returns_new_route_count(self, trigger):
        new_sol = self._make_vrp({"v1": ["a", "b"], "v2": ["c"]})
        assert trigger._count_route_changes(None, new_sol) == 2

    def test_identical_solutions_returns_zero(self, trigger):
        old = self._make_vrp({"v1": ["a", "b"]})
        new = self._make_vrp({"v1": ["a", "b"]})
        assert trigger._count_route_changes(old, new) == 0

    def test_changed_order_detected(self, trigger):
        old = self._make_vrp({"v1": ["a", "b"]})
        new = self._make_vrp({"v1": ["b", "a"]})
        assert trigger._count_route_changes(old, new) == 1

    def test_new_vehicle_counted(self, trigger):
        old = self._make_vrp({"v1": ["a"]})
        new = self._make_vrp({"v1": ["a"], "v2": ["b"]})
        assert trigger._count_route_changes(old, new) == 1  # v2 is new


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def test_full_pipeline_runs(self, trigger):
        report = _make_report(urgency_change={"dhulikhel": 0.20, "panauti": 0.15})
        change = trigger.trigger_reoptimization(report)
        assert change.optimization_state == "complete"
        assert len(trigger.get_optimization_history()) == 1

    def test_multiple_triggers_accumulate(self, trigger):
        for i, delta in enumerate([0.15, 0.20, 0.12]):
            report = _make_report(f"evt_{i}", urgency_change={"dhulikhel": delta})
            trigger.trigger_reoptimization(report)
        assert len(trigger.optimization_history) == 3

    def test_urgency_actually_changes(self, trigger, villages):
        original = villages[0].urgency_score
        report = _make_report(urgency_change={"dhulikhel": 0.20})
        trigger.trigger_reoptimization(report)
        assert villages[0].urgency_score != pytest.approx(original)
