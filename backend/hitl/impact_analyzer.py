"""
HITL Impact Analyzer -- Prompt 4.1

Estimates the effect of approving a medium-confidence news event on:
  - Village urgency scores
  - Resource allocation shifts
  - ETA changes
  - Welfare improvement

All outputs are ESTIMATES for the coordinator preview.
Actual optimization runs only AFTER approval.

Typical usage
-------------
    analyzer = ImpactAnalyzer(current_villages, current_vrp_solution)
    preview  = analyzer.calculate_impact(news_event)
"""
from __future__ import annotations

import math
from typing import Dict, List

from backend.models.village import Village
from backend.algorithms.vrp_solver import VRPSolution
from backend.rag.news_analyzer import NewsEvent
from backend.hitl.approval_queue import ImpactPreview


# Severity → urgency delta scale: severity 10 → +1.0, severity 0 → 0.0
_SEVERITY_SCALE = 0.10


class ImpactAnalyzer:
    """
    Estimates the operational impact of approving a queued news event.

    Args:
        current_villages: Live village list with current urgency scores.
        current_routes:   Current VRP solution (routes + allocations).
    """

    def __init__(
        self,
        current_villages: List[Village],
        current_routes:   VRPSolution,
    ) -> None:
        self.villages_by_id: Dict[str, Village] = {v.id: v for v in current_villages}
        self.current_routes = current_routes

    # -------------------------------------------------------------- #
    #  Public API                                                     #
    # -------------------------------------------------------------- #

    def calculate_impact(self, event: NewsEvent) -> ImpactPreview:
        """
        Compute an estimated ImpactPreview for the given news event.

        Steps:
          1. Estimate urgency deltas per affected village.
          2. Estimate which resources shift (gain / lose).
          3. Estimate ETA changes from potential route re-prioritisation.
          4. Estimate overall welfare improvement.
        """
        urgency_changes    = self.estimate_urgency_changes(event)
        resource_shifts    = self.estimate_resource_shifts(urgency_changes)
        eta_changes        = self.estimate_eta_impacts(urgency_changes)
        welfare_estimate   = self._estimate_welfare(urgency_changes, resource_shifts)

        return ImpactPreview(
            urgency_changes=urgency_changes,
            affected_villages=list(urgency_changes.keys()),
            resource_reallocation=resource_shifts,
            eta_changes=eta_changes,
            welfare_improvement_estimate=welfare_estimate,
        )

    def estimate_urgency_changes(self, event: NewsEvent) -> Dict[str, float]:
        """
        Estimate urgency deltas for each affected village.

        Formula:
            delta = severity * SEVERITY_SCALE * village_weight

        where village_weight = 1.0 + (current_urgency - 0.5) * 0.5.
        Villages already at high urgency get a slightly larger delta.

        Villages NOT in our known set get a default delta.
        """
        urgency_changes: Dict[str, float] = {}
        base_delta = event.severity * _SEVERITY_SCALE

        for village_id in event.affected_villages:
            village = self.villages_by_id.get(village_id)
            if village is not None:
                weight = 1.0 + (village.urgency_score - 0.5) * 0.5
            else:
                weight = 1.0  # unknown village — use neutral weight

            delta = round(base_delta * weight, 4)
            urgency_changes[village_id] = delta

        return urgency_changes

    def estimate_resource_shifts(
        self,
        urgency_changes: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Estimate resource reallocation triggered by urgency changes.

        Villages with positive urgency delta are likely to receive MORE
        resources.  We estimate based on the event's resource_implications
        proportionally distributed across affected villages.

        Returns:
            Dict[village_id, Dict[resource_type, estimated_shift]]
            Positive values = more resources flowing to this village.
        """
        if not urgency_changes:
            return {}

        # Build a simple allocation lookup from current routes
        current_alloc: Dict[str, Dict[str, float]] = {
            a.village_id: dict(a.allocated_resources)
            for a in self.current_routes.allocations
        }

        result: Dict[str, Dict[str, float]] = {}
        total_delta = sum(max(0.0, d) for d in urgency_changes.values()) or 1.0

        for village_id, delta in urgency_changes.items():
            if delta <= 0.0:
                result[village_id] = {}
                continue

            fraction      = delta / total_delta
            current       = current_alloc.get(village_id, {})
            shifts: Dict[str, float] = {}

            for resource, current_qty in current.items():
                # Estimate 10% more resources per unit urgency share
                additional = round(current_qty * fraction * 0.10, 2)
                if additional > 0:
                    shifts[resource] = additional

            result[village_id] = shifts

        return result

    def estimate_eta_impacts(
        self,
        urgency_changes: Dict[str, float],
    ) -> Dict[str, int]:
        """
        Estimate ETA changes in minutes.

        Logic: if a village's urgency rises significantly, it moves up
        in route priority → earlier arrival (negative delta = better ETA).
        Low-urgency villages may be pushed back (positive delta = worse ETA).

        Returns:
            Dict[village_id, delta_minutes]
            Negative = arrives sooner, Positive = arrives later.
        """
        eta_changes: Dict[str, int] = {}
        alloc_by_village = {a.village_id: a for a in self.current_routes.allocations}

        for village_id, delta in urgency_changes.items():
            alloc = alloc_by_village.get(village_id)
            if alloc is None:
                # Village not in current plan — estimate it gets added
                eta_changes[village_id] = int(30 * (1.0 - min(delta, 1.0)))
                continue

            current_eta = alloc.eta_minutes
            # Larger urgency increase → proportionally earlier arrival
            # Scale: +1.0 urgency delta → up to -20 min ETA improvement
            eta_shift = int(-delta * 20)
            # Clamp: can't arrive before t=0, and shift is at most ±current_eta
            eta_shift = max(eta_shift, -int(current_eta * 0.5))
            eta_changes[village_id] = eta_shift

        return eta_changes

    # -------------------------------------------------------------- #
    #  Internal helpers                                               #
    # -------------------------------------------------------------- #

    def _estimate_welfare(
        self,
        urgency_changes: Dict[str, float],
        resource_shifts: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Estimate the welfare improvement percentage if the event is approved.

        Heuristic:
          - Each village with a positive urgency delta contributes a welfare gain
            proportional to how much more resources it would receive.
          - Clamped to [0.0, 1.0] (returned as a 0–1 fraction, not %).
        """
        if not urgency_changes:
            return 0.0

        total_gain = 0.0
        n          = len(urgency_changes)

        for village_id, delta in urgency_changes.items():
            shifts = resource_shifts.get(village_id, {})
            resource_gain = sum(shifts.values())
            # Normalize: gain capped at 1.0 per village
            village_gain = min(1.0, delta * 0.3 + resource_gain * 0.01)
            total_gain  += village_gain

        welfare = round(total_gain / n, 4) if n > 0 else 0.0
        return min(1.0, max(0.0, welfare))
