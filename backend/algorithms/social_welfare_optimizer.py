"""Continuous, fairness-aware resource allocation.

This module solves a weighted Nash social-welfare (Nash bargaining) problem.
It does not model a strategic game and does not claim a Nash equilibrium.
Vehicle assignment and route feasibility remain separate discrete decisions.
"""
from __future__ import annotations

from math import log
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, LinearConstraint, minimize

from backend.algorithms.nash_solver import NashEquilibrium
from backend.algorithms.urgency_calculator import UrgencyScore
from backend.models.resource import ResourceType
from backend.models.village import Village


_LOG_EPSILON = 1e-6
_FEASIBILITY_TOLERANCE = 1e-6


class SocialWelfareAllocationResult(BaseModel):
    method: str = "weighted_nash_social_welfare"
    interpretation: str = (
        "Continuous Nash bargaining/social-welfare allocation; "
        "not a strategic Nash equilibrium."
    )
    solver: str = "scipy_slsqp"
    solver_status: str
    solver_success: bool
    solver_message: str
    objective_value: float
    allocations: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    village_coverage: Dict[str, float] = Field(default_factory=dict)
    village_weights: Dict[str, float] = Field(default_factory=dict)
    resource_stock_used: Dict[str, float] = Field(default_factory=dict)
    max_constraint_violation: float = 0.0
    iterations: int = 0
    runtime_seconds: float = 0.0
    continuous_problem: bool = True
    kkt_applicable_to_continuous_allocation: bool = True
    route_feasibility_included: bool = False
    diagnostic_scope: str = (
        "Solver status applies only to the continuous depot-stock and "
        "unmet-need allocation model. Vehicle and route decisions are excluded."
    )


class AllocationMethodComparison(BaseModel):
    proportional_method: str = "capped_proportional_allocation"
    optimized_method: str = "weighted_nash_social_welfare"
    proportional_social_welfare_objective: float
    optimized_social_welfare_objective: float
    objective_improvement: float
    proportional_minimum_coverage: float
    optimized_minimum_coverage: float
    proportional_mean_coverage: float
    optimized_mean_coverage: float
    comparison_scope: str = (
        "Both candidates are evaluated against the same continuous coverage "
        "utility; routing and integer vehicle decisions are not compared here."
    )


class SocialWelfareOptimizer:
    """Solve the weighted concave social-welfare allocation subproblem."""

    def __init__(
        self,
        depot_resources: Dict[str, float],
        resource_types: Dict[str, ResourceType],
        tolerance: float = _FEASIBILITY_TOLERANCE,
        max_iterations: int = 500,
    ) -> None:
        self.depot_resources = {
            resource_id: max(0.0, float(quantity))
            for resource_id, quantity in depot_resources.items()
        }
        self.resource_types = resource_types
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def _resource_weight(self, resource_id: str) -> float:
        resource = self.resource_types.get(resource_id)
        return max(0.0, resource.urgency_multiplier if resource else 1.0)

    @staticmethod
    def _urgency_weights(
        villages: List[Village],
        urgency_scores: Optional[List[UrgencyScore]],
    ) -> Dict[str, float]:
        score_map = {
            score.village_id: max(0.0, score.total_urgency)
            for score in (urgency_scores or [])
        }
        raw = {village.id: score_map.get(village.id, 1.0) for village in villages}
        positive = [value for value in raw.values() if value > 0.0]
        if not positive:
            return {village.id: 1.0 for village in villages}

        # Normalization changes only objective scale, not the maximizer.
        mean = sum(positive) / len(positive)
        return {
            village.id: (raw[village.id] / mean if raw[village.id] > 0.0 else 0.0)
            for village in villages
        }

    def _problem(
        self,
        villages: List[Village],
    ) -> Tuple[
        List[Tuple[str, str]],
        np.ndarray,
        Dict[str, List[int]],
        Dict[str, List[Tuple[int, float]]],
    ]:
        variables: List[Tuple[str, str]] = []
        upper_bounds: List[float] = []
        by_resource: Dict[str, List[int]] = {}
        utility_terms: Dict[str, List[Tuple[int, float]]] = {
            village.id: [] for village in villages
        }

        for village in villages:
            positive_needs = [
                (resource_id, need.unmet_need)
                for resource_id, need in village.resource_needs.items()
                if need.unmet_need > 0.0
            ]
            weight_sum = sum(
                self._resource_weight(resource_id)
                for resource_id, _ in positive_needs
            )
            if weight_sum <= 0.0:
                continue

            for resource_id, unmet_need in positive_needs:
                index = len(variables)
                variables.append((village.id, resource_id))
                upper_bounds.append(unmet_need)
                by_resource.setdefault(resource_id, []).append(index)
                normalized_weight = self._resource_weight(resource_id) / weight_sum
                utility_terms[village.id].append(
                    (index, normalized_weight / unmet_need)
                )

        return (
            variables,
            np.asarray(upper_bounds, dtype=float),
            by_resource,
            utility_terms,
        )

    @staticmethod
    def _coverage(
        vector: np.ndarray,
        utility_terms: Dict[str, List[Tuple[int, float]]],
    ) -> Dict[str, float]:
        return {
            village_id: min(
                1.0,
                max(0.0, sum(vector[index] * coefficient for index, coefficient in terms)),
            )
            for village_id, terms in utility_terms.items()
        }

    @staticmethod
    def _social_welfare_objective(
        coverage: Dict[str, float],
        village_weights: Dict[str, float],
    ) -> float:
        return sum(
            village_weights.get(village_id, 0.0) * log(_LOG_EPSILON + value)
            for village_id, value in coverage.items()
            if village_weights.get(village_id, 0.0) > 0.0
        )

    def _initial_vector(
        self,
        variables: List[Tuple[str, str]],
        upper_bounds: np.ndarray,
        proportional: Optional[NashEquilibrium],
    ) -> np.ndarray:
        allocation_map = {
            strategy.village_id: strategy.allocated_resources
            for strategy in (proportional.strategies if proportional else [])
        }
        initial = np.asarray(
            [
                min(
                    upper_bounds[index],
                    max(0.0, allocation_map.get(village_id, {}).get(resource_id, 0.0)),
                )
                for index, (village_id, resource_id) in enumerate(variables)
            ],
            dtype=float,
        )

        # Ensure the warm start respects stock even if the legacy candidate does not.
        for resource_id in {resource_id for _, resource_id in variables}:
            indices = [
                index
                for index, (_, candidate_resource) in enumerate(variables)
                if candidate_resource == resource_id
            ]
            used = float(initial[indices].sum())
            available = self.depot_resources.get(resource_id, 0.0)
            if used > available and used > 0.0:
                initial[indices] *= available / used
        return initial

    def solve(
        self,
        villages: List[Village],
        urgency_scores: Optional[List[UrgencyScore]] = None,
        proportional: Optional[NashEquilibrium] = None,
    ) -> Tuple[SocialWelfareAllocationResult, AllocationMethodComparison]:
        started = perf_counter()
        variables, upper_bounds, by_resource, utility_terms = self._problem(villages)
        village_weights = self._urgency_weights(villages, urgency_scores)

        if not variables:
            empty_result = SocialWelfareAllocationResult(
                solver_status="no_decision_variables",
                solver_success=True,
                solver_message="No positive unmet demand variables.",
                objective_value=0.0,
                allocations={village.id: {} for village in villages},
                village_coverage={village.id: 1.0 for village in villages},
                village_weights=village_weights,
                runtime_seconds=perf_counter() - started,
            )
            comparison = AllocationMethodComparison(
                proportional_social_welfare_objective=0.0,
                optimized_social_welfare_objective=0.0,
                objective_improvement=0.0,
                proportional_minimum_coverage=1.0,
                optimized_minimum_coverage=1.0,
                proportional_mean_coverage=1.0,
                optimized_mean_coverage=1.0,
            )
            return empty_result, comparison

        rows: List[np.ndarray] = []
        limits: List[float] = []
        for resource_id, indices in by_resource.items():
            row = np.zeros(len(variables), dtype=float)
            row[indices] = 1.0
            rows.append(row)
            limits.append(self.depot_resources.get(resource_id, 0.0))

        constraint = LinearConstraint(
            np.vstack(rows),
            lb=np.zeros(len(rows)),
            ub=np.asarray(limits, dtype=float),
        )
        initial = self._initial_vector(variables, upper_bounds, proportional)

        def objective(vector: np.ndarray) -> float:
            coverage = self._coverage(vector, utility_terms)
            return -self._social_welfare_objective(coverage, village_weights)

        def objective_gradient(vector: np.ndarray) -> np.ndarray:
            gradient = np.zeros(len(vector), dtype=float)
            coverage = self._coverage(vector, utility_terms)
            for village_id, terms in utility_terms.items():
                weight = village_weights.get(village_id, 0.0)
                if weight <= 0.0:
                    continue
                denominator = _LOG_EPSILON + coverage[village_id]
                for index, coefficient in terms:
                    gradient[index] -= weight * coefficient / denominator
            return gradient

        scipy_result = minimize(
            objective,
            initial,
            jac=objective_gradient,
            method="SLSQP",
            bounds=Bounds(np.zeros(len(variables)), upper_bounds),
            constraints=[constraint],
            options={"ftol": 1e-9, "maxiter": self.max_iterations, "disp": False},
        )
        vector = np.clip(np.asarray(scipy_result.x, dtype=float), 0.0, upper_bounds)

        allocations = {village.id: {} for village in villages}
        for index, (village_id, resource_id) in enumerate(variables):
            allocations[village_id][resource_id] = round(float(vector[index]), 6)

        stock_used = {
            resource_id: float(vector[indices].sum())
            for resource_id, indices in by_resource.items()
        }
        stock_violation = max(
            (
                stock_used[resource_id] - self.depot_resources.get(resource_id, 0.0)
                for resource_id in by_resource
            ),
            default=0.0,
        )
        bound_violation = max(
            float(np.max(-vector)),
            float(np.max(vector - upper_bounds)),
            0.0,
        )
        max_violation = max(0.0, stock_violation, bound_violation)
        solver_success = bool(scipy_result.success) and max_violation <= self.tolerance
        solver_status = "converged" if solver_success else "solver_failed"
        optimized_coverage = self._coverage(vector, utility_terms)
        optimized_objective = self._social_welfare_objective(
            optimized_coverage,
            village_weights,
        )

        proportional_vector = self._initial_vector(
            variables,
            upper_bounds,
            proportional,
        )
        proportional_coverage = self._coverage(proportional_vector, utility_terms)
        proportional_objective = self._social_welfare_objective(
            proportional_coverage,
            village_weights,
        )
        comparable_villages = [
            village_id for village_id, terms in utility_terms.items() if terms
        ]

        result = SocialWelfareAllocationResult(
            solver_status=solver_status,
            solver_success=solver_success,
            solver_message=str(scipy_result.message),
            objective_value=optimized_objective,
            allocations=allocations,
            village_coverage=optimized_coverage,
            village_weights=village_weights,
            resource_stock_used=stock_used,
            max_constraint_violation=max_violation,
            iterations=int(getattr(scipy_result, "nit", 0)),
            runtime_seconds=perf_counter() - started,
        )
        comparison = AllocationMethodComparison(
            proportional_social_welfare_objective=proportional_objective,
            optimized_social_welfare_objective=optimized_objective,
            objective_improvement=optimized_objective - proportional_objective,
            proportional_minimum_coverage=min(
                (proportional_coverage[vid] for vid in comparable_villages),
                default=1.0,
            ),
            optimized_minimum_coverage=min(
                (optimized_coverage[vid] for vid in comparable_villages),
                default=1.0,
            ),
            proportional_mean_coverage=(
                sum(proportional_coverage[vid] for vid in comparable_villages)
                / len(comparable_villages)
                if comparable_villages
                else 1.0
            ),
            optimized_mean_coverage=(
                sum(optimized_coverage[vid] for vid in comparable_villages)
                / len(comparable_villages)
                if comparable_villages
                else 1.0
            ),
        )
        return result, comparison
