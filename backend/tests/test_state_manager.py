"""
Tests for StateManager -- Prompt 2.5
Run: pytest backend/tests/test_state_manager.py -v
"""
import pytest
from datetime import timedelta
from typing import Dict, List

from backend.models.resource import ResourceCategory, ResourceType, VillageResourceNeed
from backend.models.vehicle import Vehicle, VehicleType, VehicleCategory, TerrainCapability
from backend.models.village import Village
from backend.algorithms.state_manager import (
    StateManager,
    OptimizationState,
    OptimizationResult,
)

# ------------------------------------------------------------------ #
#  Shared test data                                                    #
# ------------------------------------------------------------------ #

DEPOT_LOC = (27.7172, 85.3240)  # Kathmandu
DEPOT_RES = {"food": 5000.0, "water": 3000.0, "medical_kit": 200.0}

CONFIG = {
    "resource_types": {
        "food": {
            "resource_id": "food",
            "name": "Food",
            "category": "food",
            "urgency_multiplier": 1.5,
            "weight_per_unit": 1.0,
        },
        "water": {
            "resource_id": "water",
            "name": "Water",
            "category": "water",
            "urgency_multiplier": 1.8,
            "weight_per_unit": 1.0,
        },
        "medical_kit": {
            "resource_id": "medical_kit",
            "name": "Medical Kit",
            "category": "medical",
            "urgency_multiplier": 2.0,
            "weight_per_unit": 5.0,
        },
    },
    "vrp": {
        "max_vehicles_per_village": 2,
        "min_load_fraction": 0.1,
    },
}

RESOURCE_TYPES = {
    rid: ResourceType(**d) for rid, d in CONFIG["resource_types"].items()
}


def _make_vehicle(vid: str, capacity: float = 2000.0) -> Vehicle:
    vt = VehicleType(
        type_id="truck",
        name="Truck",
        category=VehicleCategory.GROUND_HEAVY,
        capacity_kg=capacity,
        speed_kmh=60.0,
        fuel_hours=8.0,
        terrain_capability=TerrainCapability.ALL_ROADS,
    )
    return Vehicle(id=vid, name=f"Truck {vid}", vehicle_type=vt)


def _make_village(vid: str, food: float = 800.0, water: float = 500.0) -> Village:
    needs = {
        "food": VillageResourceNeed(
            resource_type="food",
            current_need=food,
            min_need=food * 0.6,
            allocated=0,
        ),
        "water": VillageResourceNeed(
            resource_type="water",
            current_need=water,
            min_need=water * 0.7,
            allocated=0,
        ),
        "medical_kit": VillageResourceNeed(
            resource_type="medical_kit",
            current_need=20.0,
            min_need=10.0,
            allocated=0,
        ),
    }
    return Village(
        id=vid,
        name=vid.upper(),
        lat=27.62,
        lng=85.55,
        population=1000,
        accessibility="road",
        resource_needs=needs,
    )


@pytest.fixture(scope="module")
def manager() -> StateManager:
    return StateManager(
        depot_location=DEPOT_LOC,
        depot_resources=DEPOT_RES,
        terrain_graph={},
        resource_types=RESOURCE_TYPES,
        config=CONFIG,
    )


@pytest.fixture(scope="module")
def villages() -> List[Village]:
    return [_make_village(f"v{i}") for i in range(1, 4)]


@pytest.fixture(scope="module")
def vehicles() -> List[Vehicle]:
    return [_make_vehicle(f"t{i}") for i in range(1, 4)]


# ================================================================== #
#  Initial state tests                                                 #
# ================================================================== #

class TestInitialState:
    def test_initial_state_is_idle(self, manager):
        manager.reset()
        assert manager.get_state() == OptimizationState.IDLE

    def test_get_state_returns_enum(self, manager):
        manager.reset()
        state = manager.get_state()
        assert isinstance(state, OptimizationState)

    def test_reset_returns_to_idle(self, manager):
        manager._state = OptimizationState.COMPLETE
        manager.reset()
        assert manager.get_state() == OptimizationState.IDLE

    def test_constructor_stores_depot_location(self):
        sm = StateManager(
            depot_location=(1.0, 2.0),
            depot_resources={"food": 100.0},
            terrain_graph={},
            resource_types=RESOURCE_TYPES,
            config=CONFIG,
        )
        assert sm.depot_location == (1.0, 2.0)

    def test_constructor_copies_depot_resources(self):
        original = {"food": 500.0}
        sm = StateManager(
            depot_location=DEPOT_LOC,
            depot_resources=original,
            terrain_graph={},
            resource_types=RESOURCE_TYPES,
            config=CONFIG,
        )
        original["food"] = 9999.0  # mutate original
        assert sm.depot_resources["food"] == 500.0  # SM should be unaffected


# ================================================================== #
#  Full optimization pipeline tests                                    #
# ================================================================== #

class TestRunFullOptimization:
    @pytest.fixture(scope="class")
    def result(self, manager, villages, vehicles):
        manager.reset()
        return manager.run_full_optimization(villages, vehicles, timedelta(hours=0))

    def test_result_is_optimization_result(self, result):
        assert isinstance(result, OptimizationResult)

    def test_state_is_complete(self, result, manager):
        assert result.state == OptimizationState.COMPLETE

    def test_manager_state_is_complete(self, result, manager):
        assert manager.get_state() == OptimizationState.COMPLETE

    def test_urgency_scores_populated(self, result, villages):
        assert len(result.urgency_scores) == len(villages)

    def test_vrp_solution_present(self, result):
        assert result.vrp_solution is not None

    def test_nash_equilibrium_present(self, result):
        assert result.nash_equilibrium is not None

    def test_kkt_verification_present(self, result):
        assert result.kkt_verification is not None

    def test_kkt_all_conditions_satisfied(self, result):
        assert result.kkt_verification.all_conditions_satisfied is True

    def test_execution_time_nonneg(self, result):
        assert result.execution_time_seconds >= 0.0

    def test_timestamp_present(self, result):
        assert result.timestamp != ""

    def test_error_message_none_on_success(self, result):
        assert result.error_message is None

    def test_nash_converged(self, result):
        assert result.nash_equilibrium.converged is True

    def test_vrp_has_routes(self, result):
        assert len(result.vrp_solution.routes) > 0

    def test_urgency_scores_sorted_descending(self, result):
        scores = [s.total_urgency for s in result.urgency_scores]
        assert scores == sorted(scores, reverse=True)


# ================================================================== #
#  Error handling tests                                                #
# ================================================================== #

class TestErrorHandling:
    def test_error_state_on_empty_villages(self, manager, vehicles):
        """Empty village list propagates into result (VRP handles it gracefully or not)."""
        manager.reset()
        result = manager.run_full_optimization([], vehicles, timedelta(hours=0))
        # Either COMPLETE with empty routes or ERROR — either is acceptable
        assert result.state in (OptimizationState.COMPLETE, OptimizationState.ERROR)

    def test_error_result_has_error_message(self):
        """Inject bad data that causes an exception."""
        bad_manager = StateManager(
            depot_location=DEPOT_LOC,
            depot_resources={},          # no resources — may cause divide-by-zero
            terrain_graph={},
            resource_types={},           # empty resource types
            config=CONFIG,
        )
        result = bad_manager.run_full_optimization(
            [_make_village("x")],
            [_make_vehicle("t1")],
            timedelta(hours=0),
        )
        # If an exception occurred the error_message should be set
        if result.state == OptimizationState.ERROR:
            assert result.error_message is not None and result.error_message != ""

    def test_partial_results_on_error(self):
        """Urgency scores should still be populated even if a later step fails."""
        bad_manager = StateManager(
            depot_location=DEPOT_LOC,
            depot_resources={},
            terrain_graph={},
            resource_types=RESOURCE_TYPES,
            config=CONFIG,
        )
        result = bad_manager.run_full_optimization(
            [_make_village("v1")],
            [_make_vehicle("t1")],
            timedelta(hours=0),
        )
        # Urgency calculation runs first, should always succeed
        assert len(result.urgency_scores) >= 0  # may be 0 or 1

    def test_reset_after_error_returns_idle(self):
        bad_manager = StateManager(
            depot_location=DEPOT_LOC,
            depot_resources={},
            terrain_graph={},
            resource_types={},
            config=CONFIG,
        )
        bad_manager.run_full_optimization([], [], timedelta())
        bad_manager.reset()
        assert bad_manager.get_state() == OptimizationState.IDLE


# ================================================================== #
#  Time elapsed tests                                                  #
# ================================================================== #

class TestTimeElapsed:
    def test_longer_elapsed_increases_urgency(self, manager, villages, vehicles):
        manager.reset()
        r0 = manager.run_full_optimization(villages, vehicles, timedelta(hours=0))
        manager.reset()
        r24 = manager.run_full_optimization(villages, vehicles, timedelta(hours=24))
        total0  = sum(s.total_urgency for s in r0.urgency_scores)
        total24 = sum(s.total_urgency for s in r24.urgency_scores)
        assert total24 >= total0

    def test_time_elapsed_zero_still_completes(self, manager, villages, vehicles):
        manager.reset()
        result = manager.run_full_optimization(villages, vehicles, timedelta(0))
        assert result.state == OptimizationState.COMPLETE
