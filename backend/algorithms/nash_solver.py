"""Capped proportional allocation (legacy compatibility module).

The historical ``NashSolver`` and ``NashEquilibrium`` names remain importable
for compatibility. The implementation allocates each resource in proportion
to fixed need weights, applies need caps, and redistributes surplus.

This is a deterministic allocation rule, not a strategic game and not an
established Nash equilibrium. Its repeated pass only checks fixed-point
stability of the allocation rule.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from backend.models.resource import ResourceType
from backend.models.village import Village
from backend.algorithms.vrp_solver import VRPSolution, VillageAllocation


# ------------------------------------------------------------------ #
#  Output models                                                       #
# ------------------------------------------------------------------ #

class PlayerStrategy(BaseModel):
    village_id: str
    demanded_resources: Dict[str, float] = Field(default_factory=dict)
    allocated_resources: Dict[str, float] = Field(default_factory=dict)
    utility: float = 0.0
    best_response: bool = False


class ConvergenceHistory(BaseModel):
    iteration: int
    max_strategy_change: float
    max_normalized_change: float = 0.0
    total_utility: float
    allocation_snapshot: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class NashEquilibrium(BaseModel):
    allocation_method: str = "capped_proportional_allocation"
    interpretation: str = (
        "Legacy response name; deterministic capped proportional allocation, "
        "not a strategic Nash equilibrium."
    )
    strategies: List[PlayerStrategy] = Field(default_factory=list)
    converged: bool = False
    iterations: int = 0
    epsilon_convergence: float = 0.0
    normalized_epsilon_convergence: float = 0.0
    convergence_threshold: float = 0.01
    convergence_metric: str = "max_normalized_allocation_change"
    convergence_normalization: str = (
        "abs(delta) / max(village demand, depot stock, 1) for each village-resource pair"
    )
    total_utility: float = 0.0
    # Legacy field name, retained for client compatibility. See comparison_basis:
    # this is not an improvement and must not be reported as one.
    welfare_improvement_percent: float = 0.0
    comparison_basis: str = ""
    convergence_history: List[ConvergenceHistory] = Field(default_factory=list)


# ------------------------------------------------------------------ #
#  Legacy allocator                                                    #
# ------------------------------------------------------------------ #

class NashSolver:
    """Capped proportional allocator under a legacy class name.

    Args:
        depot_resources:       Full depot stock {resource_type: amount_kg}.
        resource_types:        Dict[resource_id -> ResourceType].
        convergence_threshold: Stop when max allocation change < this value.
        max_iterations:        Hard cap on outer iterations.
        seed:                  Optional RNG seed (unused here, kept for API compat).
    """

    def __init__(
        self,
        depot_resources: Dict[str, float],
        resource_types: Dict[str, ResourceType],
        convergence_threshold: float = 0.01,
        max_iterations: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.depot_resources = dict(depot_resources)
        self.resource_types = resource_types
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self._rng = random.Random(seed)

    # ---------------------------------------------------------------- #
    #  Utility                                                           #
    # ---------------------------------------------------------------- #

    def calculate_utility(
        self,
        village: Village,
        allocated_resources: Dict[str, float],
    ) -> float:
        """
        utility = sum( min(alloc[r], need[r]) / need[r] * multiplier[r] )
        """
        total = 0.0
        for rtype_id, need in village.resource_needs.items():
            if need.current_need <= 0:
                continue
            got   = min(allocated_resources.get(rtype_id, 0.0), need.current_need)
            ratio = got / need.current_need
            total += ratio * self._urgency_multiplier(rtype_id)
        return total

    def _urgency_multiplier(self, resource_type: str) -> float:
        rt = self.resource_types.get(resource_type)
        return rt.urgency_multiplier if rt else 1.0

    # ---------------------------------------------------------------- #
    #  Single-village best response (for tests / API completeness)       #
    # ---------------------------------------------------------------- #

    def calculate_best_response(
        self,
        village: Village,
        current_allocation: Dict[str, float],
        other_allocations: Dict[str, Dict[str, float]],
        remaining_resources: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Single-village best response: take as much as possible from
        remaining stock up to village need.

        Used in unit tests; the public solver uses the global proportional
        mechanism.
        """
        best: Dict[str, float] = {}
        for rtype_id, need in village.resource_needs.items():
            if need.current_need <= 0:
                continue
            avail = max(0.0, remaining_resources.get(rtype_id, 0.0))
            best[rtype_id] = min(need.current_need, avail)
        return best

    # ---------------------------------------------------------------- #
    #  Core proportional mechanism                                       #
    # ---------------------------------------------------------------- #

    def _nash_allocate(self, villages: List[Village]) -> Dict[str, Dict[str, float]]:
        """Compute capped proportional allocation for all villages.

        For each resource r:
          weight_v = current_need_v_r * urgency_multiplier_r   (fixed per village)
          share_v  = (weight_v / sum(weights)) * depot_r
          alloc_v  = min(share_v, current_need_v_r)

        Surplus from capped villages is redistributed iteratively.
        """
        result: Dict[str, Dict[str, float]] = {v.id: {} for v in villages}

        for rtype_id, depot_qty in self.depot_resources.items():
            if depot_qty <= 0:
                continue

            # Fixed weights (independent of current allocation → stable fixed point)
            base_weights: Dict[str, float] = {}
            caps: Dict[str, float] = {}
            for v in villages:
                need = v.resource_needs.get(rtype_id)
                if need is None or need.current_need <= 0:
                    base_weights[v.id] = 0.0
                    caps[v.id] = 0.0
                else:
                    base_weights[v.id] = need.current_need * self._urgency_multiplier(rtype_id)
                    caps[v.id] = need.current_need

            # Iterative proportional fill with cap redistribution
            allocated: Dict[str, float] = {v.id: 0.0 for v in villages}
            active_weights = dict(base_weights)
            remaining = float(depot_qty)

            for _ in range(len(villages) + 2):
                total_w = sum(w for w in active_weights.values() if w > 0)
                if total_w <= 0 or remaining <= 1e-9:
                    break

                leftover = 0.0
                newly_capped: List[str] = []
                for vid, w in active_weights.items():
                    if w <= 0:
                        continue
                    share = (w / total_w) * remaining
                    head_room = caps[vid] - allocated[vid]
                    if share >= head_room - 1e-9:
                        leftover   += share - head_room
                        allocated[vid] = caps[vid]
                        newly_capped.append(vid)
                    else:
                        allocated[vid] += share

                remaining = leftover
                for vid in newly_capped:
                    active_weights[vid] = 0.0

                if not newly_capped or remaining <= 1e-9:
                    break

            # Rounding each village independently can push their sum above the
            # depot stock: the per-village errors are all positive often enough
            # that the total broke the supply constraint by ~1e-6. The KKT
            # feasibility check uses a 1e-6 tolerance, so the demo was passing
            # that condition by roughly 1e-13. Give the residual back to the
            # largest recipient, where it is proportionally negligible, so the
            # constraint holds exactly rather than by luck.
            stock = float(depot_qty)
            rounded = {v.id: round(allocated[v.id], 6) for v in villages}
            overshoot = round(sum(rounded.values()) - stock, 9)
            if overshoot > 0 and rounded:
                largest = max(rounded, key=lambda vid: rounded[vid])
                rounded[largest] = round(max(0.0, rounded[largest] - overshoot), 6)

            for v in villages:
                result[v.id][rtype_id] = rounded[v.id]

        return result

    # ---------------------------------------------------------------- #
    #  Convergence check                                                 #
    # ---------------------------------------------------------------- #

    def is_nash_equilibrium(
        self,
        old_strategies: List[PlayerStrategy],
        new_strategies: List[PlayerStrategy],
        threshold: float,
    ) -> Tuple[bool, float]:
        """Compatibility name for a fixed-point allocation-delta check.

        This does not test unilateral deviations or establish a strategic
        Nash equilibrium.
        """
        old_map = {s.village_id: s.allocated_resources for s in old_strategies}
        new_map = {s.village_id: s.allocated_resources for s in new_strategies}

        max_change = 0.0
        for vid, new_alloc in new_map.items():
            old_alloc = old_map.get(vid, {})
            for rtype in set(new_alloc) | set(old_alloc):
                change = abs(new_alloc.get(rtype, 0.0) - old_alloc.get(rtype, 0.0))
                if change > max_change:
                    max_change = change

        return max_change < threshold, max_change

    def _max_normalized_change(
        self,
        old_strategies: List[PlayerStrategy],
        new_strategies: List[PlayerStrategy],
    ) -> float:
        """Return a dimensionless residual across heterogeneous resource units."""
        old_map = {strategy.village_id: strategy for strategy in old_strategies}
        new_map = {strategy.village_id: strategy for strategy in new_strategies}
        maximum = 0.0
        for village_id, new_strategy in new_map.items():
            old_strategy = old_map.get(village_id, PlayerStrategy(village_id=village_id))
            resource_ids = (
                set(new_strategy.allocated_resources)
                | set(old_strategy.allocated_resources)
                | set(new_strategy.demanded_resources)
                | set(old_strategy.demanded_resources)
            )
            for resource_id in resource_ids:
                change = abs(
                    new_strategy.allocated_resources.get(resource_id, 0.0)
                    - old_strategy.allocated_resources.get(resource_id, 0.0)
                )
                scale = max(
                    abs(new_strategy.demanded_resources.get(resource_id, 0.0)),
                    abs(old_strategy.demanded_resources.get(resource_id, 0.0)),
                    abs(self.depot_resources.get(resource_id, 0.0)),
                    1.0,
                )
                maximum = max(maximum, change / scale)
        return maximum

    # ---------------------------------------------------------------- #
    #  Outer iteration loop                                              #
    # ---------------------------------------------------------------- #

    def iterate_best_responses(
        self,
        villages: List[Village],
        initial_allocation: Dict[str, Dict[str, float]],
    ) -> Tuple[List[PlayerStrategy], List[ConvergenceHistory]]:
        """
        Iterate the deterministic allocation rule until it stabilizes.

        Typical behaviour:
          Iter 1: VRP -> proportional rule (large change)
          Iter 2: same rule -> same result (fixed point)

        Args:
            villages:           All village players.
            initial_allocation: Warm-start {village_id: {resource: amount}}.

        Returns:
            (final strategies, convergence history)
        """
        # Warm-start state
        current: Dict[str, Dict[str, float]] = {
            v.id: dict(initial_allocation.get(v.id, {}))
            for v in villages
        }
        history: List[ConvergenceHistory] = []

        for iteration in range(1, self.max_iterations + 1):
            old_current = {v.id: dict(current[v.id]) for v in villages}

            # Apply capped proportional allocation (fixed-point computation).
            current = self._nash_allocate(villages)

            # Measure convergence vs previous state
            old_strats = [
                PlayerStrategy(
                    village_id=v.id,
                    demanded_resources={
                        resource_id: need.current_need
                        for resource_id, need in v.resource_needs.items()
                    },
                    allocated_resources=old_current[v.id],
                )
                for v in villages
            ]
            new_strats = [
                PlayerStrategy(
                    village_id=v.id,
                    demanded_resources={
                        resource_id: need.current_need
                        for resource_id, need in v.resource_needs.items()
                    },
                    allocated_resources=current[v.id],
                )
                for v in villages
            ]
            _, max_change = self.is_nash_equilibrium(
                old_strats, new_strats, self.convergence_threshold
            )
            max_normalized_change = self._max_normalized_change(old_strats, new_strats)
            converged = max_normalized_change < self.convergence_threshold

            total_util = sum(self.calculate_utility(v, current[v.id]) for v in villages)
            history.append(ConvergenceHistory(
                iteration=iteration,
                max_strategy_change=max_change,
                max_normalized_change=max_normalized_change,
                total_utility=total_util,
                allocation_snapshot={v.id: dict(current[v.id]) for v in villages},
            ))

            if converged:
                break

        # Build final PlayerStrategy objects
        final_strategies: List[PlayerStrategy] = []
        for village in villages:
            alloc = current[village.id]
            util  = self.calculate_utility(village, alloc)
            final_strategies.append(PlayerStrategy(
                village_id=village.id,
                demanded_resources={r: n.current_need for r, n in village.resource_needs.items()},
                allocated_resources=alloc,
                utility=util,
                best_response=False,
            ))

        return final_strategies, history

    # ---------------------------------------------------------------- #
    #  Public API                                                        #
    # ---------------------------------------------------------------- #

    def solve(
        self,
        villages: List[Village],
        vrp_solution: VRPSolution,
    ) -> NashEquilibrium:
        """
        Compute proportional allocation, warm-starting from greedy routing.

        ``NashEquilibrium`` is retained only as a backward-compatible response
        type name.
        """
        # VRP warm-start
        vrp_alloc_map: Dict[str, Dict[str, float]] = {
            a.village_id: dict(a.allocated_resources)
            for a in vrp_solution.allocations
        }

        # Greedy-routing baseline utility (legacy comparison metric).
        village_map = {v.id: v for v in villages}
        vrp_utility = sum(
            self.calculate_utility(village_map[vid], alloc)
            for vid, alloc in vrp_alloc_map.items()
            if vid in village_map
        )

        strategies, history = self.iterate_best_responses(villages, vrp_alloc_map)

        final_utility = sum(s.utility for s in strategies)
        epsilon = history[-1].max_strategy_change if history else 0.0
        normalized_epsilon = history[-1].max_normalized_change if history else 0.0
        converged = normalized_epsilon < self.convergence_threshold
        n_iter    = len(history)

        # NOT an improvement figure, despite the legacy field name. The two
        # utilities are not comparable: the VRP baseline is constrained by
        # fleet capacity and route feasibility and allocates against *unmet*
        # need, while this allocation is unconstrained by vehicles and caps at
        # *current* need. A large positive number here mostly measures "what if
        # trucks had unlimited capacity", not a better decision. It is retained
        # only because existing clients read the field, and it is explicitly
        # labelled as incomparable in `comparison_basis` below so it can never
        # be quoted as a result.
        unconstrained_utility_ratio = (
            ((final_utility - vrp_utility) / vrp_utility * 100.0)
            if vrp_utility > 0
            else 0.0
        )

        return NashEquilibrium(
            strategies=strategies,
            converged=converged,
            iterations=n_iter,
            epsilon_convergence=epsilon,
            normalized_epsilon_convergence=normalized_epsilon,
            convergence_threshold=self.convergence_threshold,
            total_utility=final_utility,
            welfare_improvement_percent=unconstrained_utility_ratio,
            comparison_basis=(
                "NOT a performance improvement. The vehicle-constrained routing "
                "allocation and the unconstrained proportional allocation solve "
                "different problems against different demand bases (unmet need "
                "versus current need), so their utilities are not comparable. "
                "Do not quote this figure as a result."
            ),
            convergence_history=history,
        )
