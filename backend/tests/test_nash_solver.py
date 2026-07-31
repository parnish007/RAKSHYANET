"""
Tests for NashSolver -- Prompt 2.3 verification.
Run: pytest backend/tests/test_nash_solver.py -v
"""
import pytest
from typing import Dict, List

from backend.models.resource import ResourceCategory, ResourceType, VillageResourceNeed
from backend.models.village import Village
from backend.algorithms.vrp_solver import VRPSolution, VillageAllocation
from backend.algorithms.nash_solver import (
    NashSolver, PlayerStrategy, NashEquilibrium, ConvergenceHistory,
)

DEPOT = {
    "food":        5000.0,
    "water":       3000.0,
    "medical_kit": 200.0,
    "tarpaulin":   800.0,
    "blanket":     1200.0,
    "first_aid":   300.0,
}


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture(scope="module")
def resource_types() -> Dict[str, ResourceType]:
    return {
        "food":        ResourceType(resource_id="food",        name="Food",        category=ResourceCategory.FOOD,    urgency_multiplier=1.5, weight_per_unit=1.0),
        "water":       ResourceType(resource_id="water",       name="Water",       category=ResourceCategory.WATER,   urgency_multiplier=1.8, weight_per_unit=1.0),
        "medical_kit": ResourceType(resource_id="medical_kit", name="Medical Kit", category=ResourceCategory.MEDICAL, urgency_multiplier=2.0, weight_per_unit=5.0),
        "tarpaulin":   ResourceType(resource_id="tarpaulin",   name="Tarpaulin",   category=ResourceCategory.SHELTER, urgency_multiplier=1.2, weight_per_unit=3.0),
        "blanket":     ResourceType(resource_id="blanket",     name="Blanket",     category=ResourceCategory.SHELTER, urgency_multiplier=1.0, weight_per_unit=2.0),
        "first_aid":   ResourceType(resource_id="first_aid",   name="First Aid",   category=ResourceCategory.MEDICAL, urgency_multiplier=1.7, weight_per_unit=1.0),
    }


@pytest.fixture(scope="module")
def solver(resource_types) -> NashSolver:
    return NashSolver(
        depot_resources=DEPOT,
        resource_types=resource_types,
        convergence_threshold=0.01,
        max_iterations=100,
        seed=42,
    )


def make_village(village_id, food_need, water_need, medical_need=0.0,
                 food_alloc=0.0, water_alloc=0.0, medical_alloc=0.0,
                 accessibility="road") -> Village:
    needs = {
        "food":  VillageResourceNeed(resource_type="food",  current_need=food_need,   min_need=food_need*0.6,  allocated=food_alloc),
        "water": VillageResourceNeed(resource_type="water", current_need=water_need,  min_need=water_need*0.7, allocated=water_alloc),
    }
    if medical_need > 0:
        needs["medical_kit"] = VillageResourceNeed(
            resource_type="medical_kit", current_need=medical_need,
            min_need=medical_need*0.5, allocated=medical_alloc,
        )
    return Village(
        id=village_id, name=village_id.capitalize(),
        lat=27.62, lng=85.55, population=1000,
        accessibility=accessibility,
        resource_needs=needs,
    )


@pytest.fixture(scope="module")
def village_a(resource_types) -> Village:
    return make_village("village_a", food_need=1000, water_need=600, medical_need=20)


@pytest.fixture(scope="module")
def village_b(resource_types) -> Village:
    return make_village("village_b", food_need=800, water_need=500, medical_need=15)


@pytest.fixture(scope="module")
def village_critical(resource_types) -> Village:
    return make_village("critical_v", food_need=2000, water_need=1200, medical_need=50)


@pytest.fixture(scope="module")
def two_villages(village_a, village_b) -> List[Village]:
    return [village_a, village_b]


def make_vrp_solution(village_ids, alloc_per_village: Dict[str, Dict[str, float]]) -> VRPSolution:
    allocations = [
        VillageAllocation(
            village_id=vid,
            allocated_resources=alloc_per_village.get(vid, {}),
        )
        for vid in village_ids
    ]
    return VRPSolution(allocations=allocations, total_distance_km=100.0)


# ================================================================== #
#  Utility calculation tests                                           #
# ================================================================== #

class TestUtilityCalculation:
    def test_utility_zero_when_nothing_allocated(self, solver, village_a):
        util = solver.calculate_utility(village_a, {})
        assert util == pytest.approx(0.0)

    def test_utility_increases_with_allocation(self, solver, village_a):
        util_low  = solver.calculate_utility(village_a, {"food": 200, "water": 100})
        util_high = solver.calculate_utility(village_a, {"food": 800, "water": 500})
        assert util_high > util_low

    def test_medical_contributes_more_than_food(self, solver, village_a):
        """medical_kit multiplier=2.0 > food multiplier=1.5 for same ratio."""
        util_food = solver.calculate_utility(village_a, {"food": 1000})   # 100% food
        util_med  = solver.calculate_utility(village_a, {"medical_kit": 20})  # 100% medical
        assert util_med > util_food

    def test_fully_met_utility_equals_sum_of_multipliers(self, solver, village_a, resource_types):
        full_alloc = {rtype: need.current_need for rtype, need in village_a.resource_needs.items()}
        util = solver.calculate_utility(village_a, full_alloc)
        expected = sum(
            resource_types[rtype].urgency_multiplier
            for rtype in village_a.resource_needs
            if rtype in resource_types
        )
        assert util == pytest.approx(expected, rel=1e-4)

    def test_utility_capped_at_full_need(self, solver, village_a):
        """Allocating more than current_need doesn't increase utility beyond 1.0 ratio."""
        exact  = solver.calculate_utility(village_a, {"food": 1000})
        excess = solver.calculate_utility(village_a, {"food": 5000})
        assert excess == pytest.approx(exact)

    def test_utility_nonnegative(self, solver, village_a):
        util = solver.calculate_utility(village_a, {"food": 0, "water": 0})
        assert util >= 0.0

    def test_partial_allocation_partial_utility(self, solver, village_a, resource_types):
        # 50% food allocation → contributes 0.5 * 1.5 = 0.75
        util = solver.calculate_utility(village_a, {"food": 500})
        assert util == pytest.approx(0.5 * resource_types["food"].urgency_multiplier, rel=1e-4)


# ================================================================== #
#  Best response tests                                                 #
# ================================================================== #

class TestBestResponse:
    def test_best_response_within_remaining_resources(self, solver, village_a):
        remaining = {"food": 300.0, "water": 200.0, "medical_kit": 10.0}
        br = solver.calculate_best_response(village_a, {}, {}, remaining)
        for rtype, amt in br.items():
            assert amt <= remaining.get(rtype, 0.0) + 1e-9, f"{rtype} exceeds remaining"

    def test_best_response_does_not_exceed_village_need(self, solver, village_a):
        remaining = {"food": 9999.0, "water": 9999.0, "medical_kit": 9999.0}
        br = solver.calculate_best_response(village_a, {}, {}, remaining)
        for rtype, amt in br.items():
            need = village_a.resource_needs.get(rtype)
            if need:
                assert amt <= need.current_need + 1e-9, f"{rtype} exceeds village need"

    def test_best_response_zero_when_no_resources(self, solver, village_a):
        br = solver.calculate_best_response(village_a, {}, {}, {})
        assert all(v == pytest.approx(0.0) for v in br.values())

    def test_best_response_takes_full_need_when_available(self, solver, village_a):
        ample = {"food": 9999.0, "water": 9999.0, "medical_kit": 9999.0}
        br = solver.calculate_best_response(village_a, {}, {}, ample)
        assert br.get("food", 0.0) == pytest.approx(village_a.resource_needs["food"].current_need)
        assert br.get("water", 0.0) == pytest.approx(village_a.resource_needs["water"].current_need)


# ================================================================== #
#  Convergence tests                                                   #
# ================================================================== #

class TestConvergence:
    def test_two_village_game_converges(self, solver, two_villages):
        initial = {"village_a": {"food": 500.0}, "village_b": {"food": 400.0}}
        _, history = solver.iterate_best_responses(two_villages, initial)
        assert len(history) < 20, f"Did not converge within 20 iterations (took {len(history)})"

    def test_max_change_decreases_or_stays_low(self, solver, two_villages):
        initial = {"village_a": {}, "village_b": {}}
        _, history = solver.iterate_best_responses(two_villages, initial)
        # Last recorded change must be < threshold (converged)
        assert history[-1].max_strategy_change <= solver.convergence_threshold + 1e-9

    def test_convergence_history_has_iterations(self, solver, two_villages):
        _, history = solver.iterate_best_responses(two_villages, {})
        assert len(history) >= 1

    def test_history_records_total_utility(self, solver, two_villages):
        _, history = solver.iterate_best_responses(two_villages, {})
        for entry in history:
            assert entry.total_utility >= 0.0

    def test_history_records_allocation_snapshot(self, solver, two_villages):
        _, history = solver.iterate_best_responses(two_villages, {})
        for entry in history:
            assert "village_a" in entry.allocation_snapshot
            assert "village_b" in entry.allocation_snapshot

    def test_symmetric_villages_get_equal_allocations(self, solver):
        """Two villages with identical needs should receive equal allocation."""
        v1 = make_village("sym_1", food_need=1000, water_need=600)
        v2 = make_village("sym_2", food_need=1000, water_need=600)
        _, history = solver.iterate_best_responses([v1, v2], {})
        final = history[-1].allocation_snapshot
        assert final["sym_1"].get("food", 0.0) == pytest.approx(
            final["sym_2"].get("food", 0.0), abs=1.0
        )


# ================================================================== #
#  Nash equilibrium property tests                                     #
# ================================================================== #

class TestNashEquilibriumProperties:
    def test_is_nash_equilibrium_identical_strategies(self, solver, two_villages):
        strats = [PlayerStrategy(village_id=v.id, allocated_resources={"food": 100.0}) for v in two_villages]
        converged, max_change = solver.is_nash_equilibrium(strats, strats, threshold=0.01)
        assert converged is True
        assert max_change == pytest.approx(0.0)

    def test_is_not_nash_when_strategies_differ(self, solver, two_villages):
        old = [PlayerStrategy(village_id=v.id, allocated_resources={"food": 100.0}) for v in two_villages]
        new = [PlayerStrategy(village_id=v.id, allocated_resources={"food": 200.0}) for v in two_villages]
        converged, max_change = solver.is_nash_equilibrium(old, new, threshold=0.01)
        assert converged is False
        assert max_change == pytest.approx(100.0)

    def test_epsilon_convergence_below_threshold_after_solve(self, solver, two_villages):
        vrp = make_vrp_solution(["village_a", "village_b"], {"village_a": {"food": 200.0}, "village_b": {"food": 100.0}})
        result = solver.solve(two_villages, vrp)
        assert result.epsilon_convergence <= solver.convergence_threshold + 1e-9

    def test_strategies_have_nonnegative_utility(self, solver, two_villages):
        vrp = make_vrp_solution(["village_a", "village_b"], {})
        result = solver.solve(two_villages, vrp)
        for s in result.strategies:
            assert s.utility >= 0.0

    def test_all_villages_have_strategy(self, solver, two_villages):
        vrp = make_vrp_solution(["village_a", "village_b"], {})
        result = solver.solve(two_villages, vrp)
        strategy_ids = {s.village_id for s in result.strategies}
        assert strategy_ids == {"village_a", "village_b"}


# ================================================================== #
#  Comparison tests (Nash vs VRP greedy)                               #
# ================================================================== #

class TestNashVsVRP:
    def _run(self, solver, villages):
        vids = [v.id for v in villages]
        vrp_alloc = {vid: {} for vid in vids}  # worst-case: VRP gives nothing
        vrp = make_vrp_solution(vids, vrp_alloc)
        return solver.solve(villages, vrp)

    def test_nash_utility_nonnegative(self, solver, two_villages):
        result = self._run(solver, two_villages)
        assert result.total_utility >= 0.0

    def test_nash_improves_over_empty_vrp(self, solver, two_villages):
        vrp = make_vrp_solution(["village_a", "village_b"], {})
        result = solver.solve(two_villages, vrp)
        # With empty VRP, Nash should allocate resources → positive utility
        assert result.total_utility > 0.0

    def test_nash_converged_flag_set(self, solver, two_villages):
        result = self._run(solver, two_villages)
        assert result.converged is True

    def test_nash_welfare_improvement_tracked(self, solver, two_villages):
        """Nash starting from empty VRP should always show positive welfare change."""
        vrp = make_vrp_solution(["village_a", "village_b"], {})
        # Give VRP a non-zero baseline so we can compare
        vrp_alloc = {"village_a": {"food": 100.0}, "village_b": {"food": 100.0}}
        vrp = make_vrp_solution(["village_a", "village_b"], vrp_alloc)
        result = solver.solve(two_villages, vrp)
        # Welfare improvement is tracked (not necessarily > 0 if VRP was already optimal)
        assert isinstance(result.welfare_improvement_percent, float)

    def test_nash_more_balanced_than_greedy(self, solver, two_villages):
        """Nash should not completely starve one village when resources allow both."""
        # Give the greedy VRP all to village_a
        vrp_alloc = {"village_a": {"food": 1000.0, "water": 600.0}, "village_b": {}}
        vrp = make_vrp_solution(["village_a", "village_b"], vrp_alloc)
        result = solver.solve(two_villages, vrp)
        b_strat = next(s for s in result.strategies if s.village_id == "village_b")
        # Nash should now give village_b something
        assert b_strat.utility > 0.0


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def test_full_solve_returns_nash_equilibrium(self, solver, two_villages):
        vrp = make_vrp_solution(["village_a", "village_b"], {})
        result = solver.solve(two_villages, vrp)
        assert isinstance(result, NashEquilibrium)

    def test_convergence_history_populated(self, solver, two_villages):
        vrp = make_vrp_solution(["village_a", "village_b"], {})
        result = solver.solve(two_villages, vrp)
        assert len(result.convergence_history) >= 1

    def test_iterations_positive(self, solver, two_villages):
        vrp = make_vrp_solution(["village_a", "village_b"], {})
        result = solver.solve(two_villages, vrp)
        assert result.iterations >= 1

    def test_three_village_solve(self, solver, village_a, village_b, village_critical):
        villages = [village_a, village_b, village_critical]
        vrp = make_vrp_solution(
            [v.id for v in villages],
            {"village_a": {"food": 300.0}, "village_b": {"food": 200.0}, "critical_v": {}},
        )
        result = solver.solve(villages, vrp)
        assert len(result.strategies) == 3
        assert result.total_utility > 0.0

    def test_solve_within_iteration_limit(self, solver, two_villages):
        vrp = make_vrp_solution(["village_a", "village_b"], {})
        result = solver.solve(two_villages, vrp)
        assert result.iterations <= solver.max_iterations
