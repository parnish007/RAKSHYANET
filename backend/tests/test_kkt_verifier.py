"""
Tests for KKTVerifier -- Prompt 2.4 verification.
Run: pytest backend/tests/test_kkt_verifier.py -v
"""
import pytest
from typing import Dict, List

from backend.models.resource import ResourceCategory, ResourceType, VillageResourceNeed
from backend.models.village import Village
from backend.algorithms.nash_solver import NashEquilibrium, PlayerStrategy
from backend.algorithms.kkt_verifier import KKTVerifier, KKTCondition, KKTVerificationResult

DEPOT = {
    "food":        5000.0,
    "water":       3000.0,
    "medical_kit": 200.0,
}

TOL = 1e-6


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture(scope="module")
def resource_types() -> Dict[str, ResourceType]:
    return {
        "food":        ResourceType(resource_id="food",        name="Food",    category=ResourceCategory.FOOD,    urgency_multiplier=1.5, weight_per_unit=1.0),
        "water":       ResourceType(resource_id="water",       name="Water",   category=ResourceCategory.WATER,   urgency_multiplier=1.8, weight_per_unit=1.0),
        "medical_kit": ResourceType(resource_id="medical_kit", name="Medical", category=ResourceCategory.MEDICAL, urgency_multiplier=2.0, weight_per_unit=5.0),
    }


@pytest.fixture(scope="module")
def verifier(resource_types) -> KKTVerifier:
    return KKTVerifier(resource_types=resource_types, tolerance=TOL)


def make_village(vid, food=1000, water=600, medical=20, accessibility="road") -> Village:
    needs = {
        "food":  VillageResourceNeed(resource_type="food",  current_need=food,    min_need=food*0.6,    allocated=0),
        "water": VillageResourceNeed(resource_type="water", current_need=water,   min_need=water*0.7,   allocated=0),
        "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=medical, min_need=medical*0.5, allocated=0),
    }
    return Village(id=vid, name=vid.capitalize(), lat=27.62, lng=85.55, population=1000,
                   accessibility=accessibility, resource_needs=needs)


def make_nash(villages, alloc_map: Dict[str, Dict[str, float]]) -> NashEquilibrium:
    """Build a NashEquilibrium from a dict of allocations."""
    strategies = [
        PlayerStrategy(
            village_id=v.id,
            demanded_resources={r: n.current_need for r, n in v.resource_needs.items()},
            allocated_resources=alloc_map.get(v.id, {}),
            utility=0.0,
            best_response=True,
        )
        for v in villages
    ]
    return NashEquilibrium(strategies=strategies, converged=True, iterations=2, total_utility=10.0)


@pytest.fixture(scope="module")
def three_villages() -> List[Village]:
    return [
        make_village("v1", food=1000, water=600, medical=20),
        make_village("v2", food=800,  water=500, medical=15),
        make_village("v3", food=1200, water=700, medical=25),
    ]


@pytest.fixture(scope="module")
def proportional_nash(three_villages) -> NashEquilibrium:
    """
    Proportional Nash allocation for 3 villages:
      food:  depot=5000, total_need=3000 → each village gets 100% (surplus)
      water: depot=3000, total_need=1800 → each village gets 100% (surplus)
      medical_kit: depot=200, total_need=60 → each village gets 100% (surplus)
    """
    alloc_map = {
        "v1": {"food": 1000.0, "water": 600.0, "medical_kit": 20.0},
        "v2": {"food": 800.0,  "water": 500.0, "medical_kit": 15.0},
        "v3": {"food": 1200.0, "water": 700.0, "medical_kit": 25.0},
    }
    return make_nash(three_villages, alloc_map)


@pytest.fixture(scope="module")
def tight_nash() -> NashEquilibrium:
    """
    Nash where all depot resources are FULLY used (tight constraints).
    food: v1=1666, v2=1666, v3=1668 (total=5000 = depot)
    """
    v1 = make_village("t1", food=2500, water=1500, medical=50)
    v2 = make_village("t2", food=1500, water=900,  medical=30)
    v3 = make_village("t3", food=1000, water=600,  medical=20)
    villages = [v1, v2, v3]

    total_food = 2500 + 1500 + 1000
    total_water = 1500 + 900 + 600
    total_medical = 50 + 30 + 20

    alloc_map = {
        "t1": {
            "food":  round(5000 * 2500 / total_food,  4),
            "water": round(3000 * 1500 / total_water, 4),
            "medical_kit": round(200 * 50 / total_medical, 4),
        },
        "t2": {
            "food":  round(5000 * 1500 / total_food,  4),
            "water": round(3000 * 900  / total_water, 4),
            "medical_kit": round(200 * 30 / total_medical, 4),
        },
        "t3": {
            "food":  round(5000 * 1000 / total_food,  4),
            "water": round(3000 * 600  / total_water, 4),
            "medical_kit": round(200 * 20 / total_medical, 4),
        },
    }
    return make_nash(villages, alloc_map), villages


# ================================================================== #
#  Gradient computation tests                                          #
# ================================================================== #

class TestGradientComputation:
    def test_gradient_equals_multiplier_over_need(self, verifier, three_villages):
        v = three_villages[0]
        grads = verifier.compute_gradient(v, {"food": 500.0})
        assert grads["food"] == pytest.approx(1.5 / 1000.0, rel=1e-9)

    def test_gradient_water_correct(self, verifier, three_villages):
        v = three_villages[0]
        grads = verifier.compute_gradient(v, {"water": 300.0})
        assert grads["water"] == pytest.approx(1.8 / 600.0, rel=1e-9)

    def test_gradient_medical_correct(self, verifier, three_villages):
        v = three_villages[0]
        grads = verifier.compute_gradient(v, {"medical_kit": 10.0})
        assert grads["medical_kit"] == pytest.approx(2.0 / 20.0, rel=1e-9)

    def test_gradient_positive_for_needed_resource(self, verifier, three_villages):
        v = three_villages[0]
        grads = verifier.compute_gradient(v, {})
        assert all(g > 0.0 for g in grads.values())

    def test_gradient_medical_greater_than_food(self, verifier, three_villages):
        v = three_villages[0]
        grads = verifier.compute_gradient(v, {})
        assert grads["medical_kit"] > grads["food"]

    def test_gradient_independent_of_current_allocation(self, verifier, three_villages):
        v = three_villages[0]
        g0 = verifier.compute_gradient(v, {})
        g1 = verifier.compute_gradient(v, {"food": 999.0})
        assert g0["food"] == pytest.approx(g1["food"])


# ================================================================== #
#  Constraint violation tests                                          #
# ================================================================== #

class TestConstraintViolations:
    def test_no_violations_within_bounds(self, verifier, three_villages):
        v = three_villages[0]
        viols = verifier.compute_constraint_violations(v, {"food": 500.0, "water": 300.0}, DEPOT)
        assert viols == {}

    def test_violation_detected_over_need(self, verifier, three_villages):
        v = three_villages[0]
        viols = verifier.compute_constraint_violations(v, {"food": 9999.0}, DEPOT)
        assert "food" in viols
        assert viols["food"] > 0

    def test_violation_detected_negative_allocation(self, verifier, three_villages):
        v = three_villages[0]
        viols = verifier.compute_constraint_violations(v, {"food": -10.0}, DEPOT)
        assert "food" in viols

    def test_zero_allocation_no_violation(self, verifier, three_villages):
        v = three_villages[0]
        viols = verifier.compute_constraint_violations(v, {"food": 0.0}, DEPOT)
        assert viols == {}


# ================================================================== #
#  Primal feasibility tests                                            #
# ================================================================== #

class TestPrimalFeasibility:
    def test_proportional_nash_is_primal_feasible(self, verifier, three_villages, proportional_nash):
        cond = verifier.check_primal_feasibility(proportional_nash, three_villages, DEPOT)
        assert cond.satisfied is True
        assert cond.constraint_value <= TOL

    def test_infeasible_over_depot(self, verifier, three_villages):
        # Give every village more than depot/3 → total exceeds depot
        bad_alloc = {
            "v1": {"food": 3000.0},
            "v2": {"food": 3000.0},
            "v3": {"food": 3000.0},
        }
        bad_nash = make_nash(three_villages, bad_alloc)
        cond = verifier.check_primal_feasibility(bad_nash, three_villages, DEPOT)
        assert cond.satisfied is False
        assert cond.constraint_value > 0

    def test_exact_need_allocation_feasible(self, verifier, three_villages):
        exact_alloc = {
            "v1": {"food": 1000.0, "water": 600.0, "medical_kit": 20.0},
            "v2": {"food": 800.0,  "water": 500.0, "medical_kit": 15.0},
            "v3": {"food": 1200.0, "water": 700.0, "medical_kit": 25.0},
        }
        nash = make_nash(three_villages, exact_alloc)
        cond = verifier.check_primal_feasibility(nash, three_villages, DEPOT)
        assert cond.satisfied is True


# ================================================================== #
#  Dual feasibility tests                                              #
# ================================================================== #

class TestDualFeasibility:
    def test_nash_dual_feasible(self, verifier, three_villages, proportional_nash):
        cond = verifier.check_dual_feasibility(proportional_nash, three_villages, DEPOT)
        assert cond.satisfied is True
        assert cond.constraint_value <= TOL

    def test_lagrange_multipliers_nonnegative(self, verifier, three_villages, proportional_nash):
        lambdas = verifier._estimate_lagrange_multipliers(proportional_nash, three_villages, DEPOT)
        for rtype, lam in lambdas.items():
            assert lam >= -TOL, f"λ_{rtype} = {lam} < 0"

    def test_slack_resource_has_zero_lambda(self, verifier, three_villages, proportional_nash):
        """All 3 villages fully met → resources have slack → λ should be 0."""
        lambdas = verifier._estimate_lagrange_multipliers(proportional_nash, three_villages, DEPOT)
        for rtype, lam in lambdas.items():
            # proportional_nash has slack in all resources
            assert lam == pytest.approx(0.0, abs=TOL)


# ================================================================== #
#  Stationarity tests                                                  #
# ================================================================== #

class TestStationarity:
    def test_nash_satisfies_stationarity(self, verifier, three_villages, proportional_nash):
        cond = verifier.check_stationarity(proportional_nash, three_villages, DEPOT)
        assert cond.satisfied is True
        assert cond.constraint_value < TOL

    def test_stationarity_residual_near_zero(self, verifier, three_villages, proportional_nash):
        cond = verifier.check_stationarity(proportional_nash, three_villages, DEPOT)
        assert cond.constraint_value == pytest.approx(0.0, abs=TOL)


# ================================================================== #
#  Complementary slackness tests                                       #
# ================================================================== #

class TestComplementarySlackness:
    def test_proportional_nash_complementary_slackness(self, verifier, three_villages, proportional_nash):
        cond = verifier.check_complementary_slackness(proportional_nash, three_villages, DEPOT)
        assert cond.satisfied is True

    def test_slack_times_lambda_near_zero(self, verifier, three_villages, proportional_nash):
        """Slack resource → λ=0 → product = 0."""
        lambdas = verifier._estimate_lagrange_multipliers(proportional_nash, three_villages, DEPOT)
        for rtype, lam in lambdas.items():
            total_alloc = sum(
                proportional_nash.strategies[i].allocated_resources.get(rtype, 0.0)
                for i in range(len(proportional_nash.strategies))
            )
            slack = max(0.0, DEPOT.get(rtype, 0.0) - total_alloc)
            product = abs(lam * slack)
            assert product < TOL, f"Product λ·slack = {product} for {rtype}"


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def test_full_verify_returns_result(self, verifier, three_villages, proportional_nash):
        result = verifier.verify(proportional_nash, three_villages, DEPOT)
        assert isinstance(result, KKTVerificationResult)

    def test_verify_has_four_conditions(self, verifier, three_villages, proportional_nash):
        result = verifier.verify(proportional_nash, three_villages, DEPOT)
        assert len(result.conditions) == 4

    def test_all_conditions_satisfied_for_nash(self, verifier, three_villages, proportional_nash):
        result = verifier.verify(proportional_nash, three_villages, DEPOT)
        assert result.all_conditions_satisfied is True

    def test_lagrange_multipliers_present(self, verifier, three_villages, proportional_nash):
        result = verifier.verify(proportional_nash, three_villages, DEPOT)
        assert isinstance(result.lagrange_multipliers, dict)
        assert len(result.lagrange_multipliers) > 0

    def test_objective_value_recorded(self, verifier, three_villages, proportional_nash):
        result = verifier.verify(proportional_nash, three_villages, DEPOT)
        assert result.objective_value == pytest.approx(proportional_nash.total_utility)

    def test_timestamp_present(self, verifier, three_villages, proportional_nash):
        result = verifier.verify(proportional_nash, three_villages, DEPOT)
        assert result.verification_timestamp != ""

    def test_condition_names_correct(self, verifier, three_villages, proportional_nash):
        result = verifier.verify(proportional_nash, three_villages, DEPOT)
        names = {c.condition_name for c in result.conditions}
        assert "Stationarity" in names
        assert "Primal Feasibility" in names
        assert "Dual Feasibility" in names
        assert "Complementary Slackness" in names
