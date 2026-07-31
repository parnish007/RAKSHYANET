"""Allocation feasibility and partial KKT consistency diagnostics.

This verifier evaluates a submitted continuous allocation against depot-stock
and need-cap constraints. It estimates aggregate resource multipliers from the
submitted allocation; it does not independently solve every lower/upper-bound
dual variable and therefore is not an independent proof of optimality.

Problem formulation
-------------------
Maximize  f(x) = Σ_v Σ_r  (alloc_vr / need_vr) * mult_r
subject to
  (C1)  Σ_v alloc_vr  ≤  depot_r          ∀r   [resource capacity]
  (C2)  alloc_vr      ≤  need_vr           ∀v,r [village need cap]
  (C3)  alloc_vr      ≥  0                 ∀v,r [non-negativity]

Lagrangian
----------
L = f(x) + Σ_r λ_r (depot_r − Σ_v alloc_vr)
          + Σ_{v,r} ν_vr (need_vr − alloc_vr)
          + Σ_{v,r} μ_vr  alloc_vr

KKT conditions evaluated for the continuous allocation model
-------------------------------------------------------------
1. Stationarity      ∂L/∂alloc_vr = 0  →  grad_vr = λ_r + ν_vr − μ_vr
2. Primal feasibility  C1, C2, C3 all satisfied
3. Dual feasibility    λ_r ≥ 0,  ν_vr ≥ 0,  μ_vr ≥ 0
4. Complementary slackness
        λ_r · (depot_r − Σ_v alloc_vr) = 0
        ν_vr · (need_vr − alloc_vr)    = 0
        μ_vr · alloc_vr                = 0

Shadow-price estimation
-----------------------
We don't solve the dual problem.  Instead we estimate λ_r as the
allocation-weighted mean marginal utility across all active villages:

    λ_r = (Σ_v grad_vr · alloc_vr)  /  (Σ_v alloc_vr)

This is the Lagrange multiplier implied by the primal allocation.
By construction the aggregate stationarity residual equals zero.

For a resource not fully used by the submitted allocation, the
complementary slackness condition forces λ_r = 0.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from backend.models.resource import ResourceType
from backend.models.village import Village
from backend.algorithms.nash_solver import NashEquilibrium, PlayerStrategy


# ------------------------------------------------------------------ #
#  Output models                                                       #
# ------------------------------------------------------------------ #

class KKTCondition(BaseModel):
    condition_name: str
    satisfied: bool
    constraint_value: float          # Max residual / violation magnitude
    tolerance: float
    description: str


class KKTVerificationResult(BaseModel):
    all_conditions_satisfied: bool
    conditions: List[KKTCondition] = Field(default_factory=list)
    objective_value: float
    diagnostic_scope: str = (
        "Continuous allocation feasibility and partial KKT consistency; "
        "not an independent optimality proof."
    )
    independently_proves_optimality: bool = False
    applies_to_discrete_route_decisions: bool = False
    lagrange_multipliers: Dict[str, float] = Field(default_factory=dict)
    complementary_slackness_violations: int = 0
    verification_timestamp: str = ""


# ------------------------------------------------------------------ #
#  KKTVerifier                                                         #
# ------------------------------------------------------------------ #

class KKTVerifier:
    """Evaluate scoped KKT diagnostics for a continuous allocation candidate.

    Args:
        resource_types: Dict[resource_id -> ResourceType] with urgency multipliers.
        tolerance:      Numerical tolerance for floating-point comparisons.
    """

    def __init__(
        self,
        resource_types: Dict[str, ResourceType],
        tolerance: float = 1e-6,
    ) -> None:
        self.resource_types = resource_types
        self.tolerance = tolerance

    # ---------------------------------------------------------------- #
    #  Gradient  ∂utility/∂alloc_vr  =  mult_r / need_vr               #
    # ---------------------------------------------------------------- #

    def compute_gradient(
        self,
        village: Village,
        allocation: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute ∂f/∂alloc_vr = urgency_multiplier_r / current_need_vr
        for each resource the village needs.

        This is the marginal utility of allocating one additional unit
        of resource r to village v.
        """
        gradients: Dict[str, float] = {}
        for rtype_id, need in village.resource_needs.items():
            if need.current_need <= 0:
                continue
            mult = self._multiplier(rtype_id)
            gradients[rtype_id] = mult / need.current_need
        return gradients

    def _multiplier(self, rtype_id: str) -> float:
        rt = self.resource_types.get(rtype_id)
        return rt.urgency_multiplier if rt else 1.0

    # ---------------------------------------------------------------- #
    #  Constraint violations                                             #
    # ---------------------------------------------------------------- #

    def compute_constraint_violations(
        self,
        village: Village,
        allocation: Dict[str, float],
        depot_resources: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Return per-resource violation magnitudes (positive = violated).

        Violations checked:
          - alloc_vr > need_vr  → upper bound C2 violated
          - alloc_vr < 0        → non-negativity C3 violated
        Aggregate resource capacity C1 is checked separately.
        """
        violations: Dict[str, float] = {}
        for rtype_id, need in village.resource_needs.items():
            alloc = allocation.get(rtype_id, 0.0)
            if alloc < -self.tolerance:
                violations[rtype_id] = -alloc          # negativity violation
            elif alloc > need.current_need + self.tolerance:
                violations[rtype_id] = alloc - need.current_need  # over-need
        return violations

    # ---------------------------------------------------------------- #
    #  Lagrange multiplier estimation                                    #
    # ---------------------------------------------------------------- #

    def _estimate_lagrange_multipliers(
        self,
        nash_solution: NashEquilibrium,
        villages: List[Village],
        depot_resources: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Estimate λ_r = allocation-weighted mean marginal utility.

            λ_r = Σ_v (grad_vr · alloc_vr) / Σ_v alloc_vr

        For resources with slack (not fully used): force λ_r = 0.
        """
        village_map = {v.id: v for v in villages}
        lambdas: Dict[str, float] = {}

        for rtype_id, depot_qty in depot_resources.items():
            total_alloc = sum(
                s.allocated_resources.get(rtype_id, 0.0)
                for s in nash_solution.strategies
            )

            if total_alloc <= self.tolerance:
                lambdas[rtype_id] = 0.0
                continue

            # Complementary slackness: slack resource → λ = 0
            slack = depot_qty - total_alloc
            if slack > self.tolerance:
                lambdas[rtype_id] = 0.0
                continue

            # Tight resource: estimate λ as weighted mean gradient
            weighted_sum = 0.0
            for strat in nash_solution.strategies:
                alloc = strat.allocated_resources.get(rtype_id, 0.0)
                if alloc <= 0:
                    continue
                village = village_map.get(strat.village_id)
                if village is None:
                    continue
                need = village.resource_needs.get(rtype_id)
                if need is None or need.current_need <= 0:
                    continue
                grad = self._multiplier(rtype_id) / need.current_need
                weighted_sum += grad * alloc

            lambdas[rtype_id] = weighted_sum / total_alloc

        return lambdas

    # ---------------------------------------------------------------- #
    #  KKT Condition 1: Stationarity                                    #
    # ---------------------------------------------------------------- #

    def check_stationarity(
        self,
        nash_solution: NashEquilibrium,
        villages: List[Village],
        depot_resources: Dict[str, float],
    ) -> KKTCondition:
        """
        KKT stationarity: the aggregate weighted-gradient residual = 0.

        Residual_r = |Σ_v grad_vr · alloc_vr  -  λ_r · Σ_v alloc_vr|

        By construction of λ_r (weighted mean), this is identically zero
        for the submitted proportional allocation. The check degenerates to
        verifying that the arithmetic is consistent (residual < tolerance);
        it is not an independent stationarity proof.
        """
        village_map = {v.id: v for v in villages}
        lambdas = self._estimate_lagrange_multipliers(nash_solution, villages, depot_resources)

        max_residual = 0.0
        for rtype_id, lambda_r in lambdas.items():
            depot_qty = depot_resources.get(rtype_id, 0.0)
            total_alloc = sum(
                s.allocated_resources.get(rtype_id, 0.0)
                for s in nash_solution.strategies
            )
            if total_alloc <= self.tolerance:
                continue

            # For slack resources λ=0 by complementary slackness.
            # Stationarity holds trivially (no binding capacity constraint).
            slack = depot_qty - total_alloc
            if slack > self.tolerance:
                # No capacity constraint active → skip (always satisfied)
                continue

            weighted_grad_sum = 0.0
            for strat in nash_solution.strategies:
                alloc = strat.allocated_resources.get(rtype_id, 0.0)
                if alloc <= 0:
                    continue
                village = village_map.get(strat.village_id)
                if village is None:
                    continue
                need = village.resource_needs.get(rtype_id)
                if need is None or need.current_need <= 0:
                    continue
                grad = self._multiplier(rtype_id) / need.current_need
                weighted_grad_sum += grad * alloc

            # Residual = 0 by construction of λ_r (weighted mean)
            residual = abs(weighted_grad_sum - lambda_r * total_alloc) / max(total_alloc, 1.0)
            max_residual = max(max_residual, residual)

        satisfied = max_residual < self.tolerance
        return KKTCondition(
            condition_name="Stationarity",
            satisfied=satisfied,
            constraint_value=max_residual,
            tolerance=self.tolerance,
            description=(
                "Weighted gradient residual |sum(grad*alloc) - lam*sum(alloc)| / sum(alloc). "
                "Must be < tolerance to confirm marginal utilities match shadow prices."
            ),
        )

    # ---------------------------------------------------------------- #
    #  KKT Condition 2: Primal Feasibility                              #
    # ---------------------------------------------------------------- #

    def check_primal_feasibility(
        self,
        nash_solution: NashEquilibrium,
        villages: List[Village],
        depot_resources: Dict[str, float],
    ) -> KKTCondition:
        """
        KKT primal feasibility:
          (C1) Σ_v alloc_vr  ≤  depot_r     for each resource r
          (C2) alloc_vr      ≤  need_vr     for each village v, resource r
          (C3) alloc_vr      ≥  0
        """
        village_map = {v.id: v for v in villages}
        max_violation = 0.0

        # C1: aggregate resource constraint
        for rtype_id, depot_qty in depot_resources.items():
            total = sum(
                s.allocated_resources.get(rtype_id, 0.0)
                for s in nash_solution.strategies
            )
            viol = total - depot_qty
            if viol > max_violation:
                max_violation = viol

        # C2 + C3: per-village bounds
        for strat in nash_solution.strategies:
            village = village_map.get(strat.village_id)
            if village is None:
                continue
            for rtype_id, alloc in strat.allocated_resources.items():
                # Non-negativity
                if alloc < -self.tolerance:
                    max_violation = max(max_violation, -alloc)
                # Need cap
                need = village.resource_needs.get(rtype_id)
                if need and alloc > need.current_need + self.tolerance:
                    max_violation = max(max_violation, alloc - need.current_need)

        satisfied = max_violation <= self.tolerance
        return KKTCondition(
            condition_name="Primal Feasibility",
            satisfied=satisfied,
            constraint_value=max_violation,
            tolerance=self.tolerance,
            description=(
                "Max constraint violation across: sum(alloc_v_r) <= depot_r (C1), "
                "alloc_v_r <= need_v_r (C2), alloc_v_r >= 0 (C3). Must be 0 +/- tol."
            ),
        )

    # ---------------------------------------------------------------- #
    #  KKT Condition 3: Dual Feasibility                                #
    # ---------------------------------------------------------------- #

    def check_dual_feasibility(
        self,
        nash_solution: NashEquilibrium,
        villages: List[Village],
        depot_resources: Dict[str, float],
    ) -> KKTCondition:
        """
        KKT dual feasibility: all Lagrange multipliers λ_r ≥ 0.
        Shadow prices cannot be negative (resources have non-negative value).
        """
        lambdas = self._estimate_lagrange_multipliers(nash_solution, villages, depot_resources)
        min_lambda = min(lambdas.values()) if lambdas else 0.0
        violation = max(0.0, -min_lambda)

        satisfied = violation <= self.tolerance
        return KKTCondition(
            condition_name="Dual Feasibility",
            satisfied=satisfied,
            constraint_value=violation,
            tolerance=self.tolerance,
            description=(
                "All Lagrange multipliers lam_r >= 0. "
                "Negative shadow price would imply a resource has negative value."
            ),
        )

    # ---------------------------------------------------------------- #
    #  KKT Condition 4: Complementary Slackness                        #
    # ---------------------------------------------------------------- #

    def check_complementary_slackness(
        self,
        nash_solution: NashEquilibrium,
        villages: List[Village],
        depot_resources: Dict[str, float],
    ) -> KKTCondition:
        """
        Complementary slackness:  λ_r · slack_r = 0

        Where slack_r = depot_r − Σ_v alloc_vr.

        If slack > 0 (resource not fully used)  → λ must be 0.
        If λ > 0 (resource has positive value) → slack must be 0.
        Product of both must be ≈ 0.
        """
        village_map = {v.id: v for v in villages}
        lambdas = self._estimate_lagrange_multipliers(nash_solution, villages, depot_resources)

        max_product = 0.0
        n_violations = 0

        for rtype_id, lambda_r in lambdas.items():
            depot_qty = depot_resources.get(rtype_id, 0.0)
            total_alloc = sum(
                s.allocated_resources.get(rtype_id, 0.0)
                for s in nash_solution.strategies
            )
            slack = max(0.0, depot_qty - total_alloc)
            product = abs(lambda_r * slack)
            if product > max_product:
                max_product = product
            if product > self.tolerance:
                n_violations += 1

        satisfied = max_product <= self.tolerance
        return KKTCondition(
            condition_name="Complementary Slackness",
            satisfied=satisfied,
            constraint_value=max_product,
            tolerance=self.tolerance,
            description=(
                "Max |lam_r * (depot_r - sum(alloc_v_r))|. "
                "Slack resources must have zero shadow price and vice versa."
            ),
        )

    # ---------------------------------------------------------------- #
    #  Public API                                                        #
    # ---------------------------------------------------------------- #

    def verify(
        self,
        nash_solution: NashEquilibrium,
        villages: List[Village],
        depot_resources: Dict[str, float],
    ) -> KKTVerificationResult:
        """
        Run four scoped checks on the submitted continuous allocation.

        Returns a KKTVerificationResult with per-condition details,
        estimated Lagrange multipliers, and the overall pass/fail flag.
        """
        lambdas = self._estimate_lagrange_multipliers(nash_solution, villages, depot_resources)

        cond_stationarity    = self.check_stationarity(nash_solution, villages, depot_resources)
        cond_primal          = self.check_primal_feasibility(nash_solution, villages, depot_resources)
        cond_dual            = self.check_dual_feasibility(nash_solution, villages, depot_resources)
        cond_complementary   = self.check_complementary_slackness(nash_solution, villages, depot_resources)

        conditions = [cond_stationarity, cond_primal, cond_dual, cond_complementary]
        all_satisfied = all(c.satisfied for c in conditions)

        # Count complementary slackness violations from the detailed check
        cs_violations = sum(
            1
            for rtype_id, lambda_r in lambdas.items()
            if abs(lambda_r * max(0.0, depot_resources.get(rtype_id, 0.0) - sum(
                s.allocated_resources.get(rtype_id, 0.0) for s in nash_solution.strategies
            ))) > self.tolerance
        )

        return KKTVerificationResult(
            all_conditions_satisfied=all_satisfied,
            conditions=conditions,
            objective_value=nash_solution.total_utility,
            lagrange_multipliers=lambdas,
            complementary_slackness_violations=cs_violations,
            verification_timestamp=datetime.now(timezone.utc).isoformat(),
        )
