"""Tests for the continuous Nash-social-welfare allocation comparison."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.algorithms.nash_solver import NashEquilibrium, PlayerStrategy
from backend.algorithms.social_welfare_optimizer import SocialWelfareOptimizer
from backend.algorithms.urgency_calculator import UrgencyScore
from backend.api.main import app
from backend.models.resource import ResourceCategory, ResourceType, VillageResourceNeed
from backend.models.village import Village


def resource_types():
    return {
        "water": ResourceType(
            resource_id="water",
            name="Water",
            category=ResourceCategory.WATER,
            urgency_multiplier=1.5,
        )
    }


def village(village_id: str, need: float = 100.0, allocated: float = 0.0) -> Village:
    return Village(
        id=village_id,
        name=village_id,
        lat=27.5,
        lng=85.3,
        population=100,
        resource_needs={
            "water": VillageResourceNeed(
                resource_type="water",
                current_need=need,
                min_need=need * 0.2,
                allocated=allocated,
            )
        },
    )


def proportional_half() -> NashEquilibrium:
    return NashEquilibrium(
        strategies=[
            PlayerStrategy(village_id="a", allocated_resources={"water": 50.0}),
            PlayerStrategy(village_id="b", allocated_resources={"water": 50.0}),
        ]
    )


def test_weighted_social_welfare_responds_to_operational_urgency():
    villages = [village("a"), village("b")]
    urgency = [
        UrgencyScore(village_id="a", total_urgency=20.0),
        UrgencyScore(village_id="b", total_urgency=10.0),
    ]
    optimizer = SocialWelfareOptimizer({"water": 100.0}, resource_types())

    result, comparison = optimizer.solve(villages, urgency, proportional_half())

    assert result.solver_success is True
    assert result.allocations["a"]["water"] == pytest.approx(66.6667, abs=0.02)
    assert result.allocations["b"]["water"] == pytest.approx(33.3333, abs=0.02)
    assert comparison.objective_improvement > 0.0


def test_social_welfare_respects_stock_and_unmet_need_caps():
    villages = [village("a", need=100.0, allocated=80.0), village("b", need=40.0)]
    urgency = [
        UrgencyScore(village_id="a", total_urgency=10.0),
        UrgencyScore(village_id="b", total_urgency=10.0),
    ]
    optimizer = SocialWelfareOptimizer({"water": 50.0}, resource_types())

    result, _ = optimizer.solve(villages, urgency)

    assert result.max_constraint_violation <= 1e-6
    assert result.allocations["a"]["water"] <= 20.0 + 1e-6
    assert result.allocations["b"]["water"] <= 40.0 + 1e-6
    assert sum(row.get("water", 0.0) for row in result.allocations.values()) <= 50.0 + 1e-6


def test_social_welfare_is_not_reported_as_strategic_equilibrium():
    result, _ = SocialWelfareOptimizer(
        {"water": 100.0},
        resource_types(),
    ).solve([village("a"), village("b")])

    assert "not a strategic Nash equilibrium" in result.interpretation
    assert result.continuous_problem is True
    assert result.route_feasibility_included is False
    assert result.kkt_applicable_to_continuous_allocation is True


def test_legacy_result_self_identifies_as_proportional():
    legacy = proportional_half()
    assert legacy.allocation_method == "capped_proportional_allocation"
    assert "not a strategic Nash equilibrium" in legacy.interpretation


def test_api_exposes_social_welfare_candidate_and_comparison():
    client = TestClient(app)
    run = client.post("/api/optimization/run", json={}).json()

    candidate = run["result"]["social_welfare_allocation"]
    comparison = run["result"]["allocation_comparison"]
    assert candidate["method"] == "weighted_nash_social_welfare"
    assert candidate["solver_status"] == "converged"
    assert candidate["route_feasibility_included"] is False
    assert comparison["optimized_method"] == "weighted_nash_social_welfare"

    assert client.get("/api/allocation/social-welfare").status_code == 200
    assert client.get("/api/allocation/compare").status_code == 200


def test_kkt_api_disclaims_independent_and_discrete_proof():
    client = TestClient(app)
    client.post("/api/optimization/run", json={})
    diagnostics = client.get("/api/kkt/verify").json()

    assert diagnostics["independently_proves_optimality"] is False
    assert diagnostics["applies_to_discrete_route_decisions"] is False
