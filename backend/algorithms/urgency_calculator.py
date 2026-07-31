"""
Urgency Calculator — Prompt 2.1

Ranks villages by multi-resource need urgency with exponential time decay.

Formula:
  time_factor        = 1.0 + (exp(0.3 * hours_elapsed) - 1.0) * 0.5
  resource_urgency   = (unmet_need / current_need) * urgency_multiplier * time_factor
  village_urgency    = Σ resource_urgency + critical_penalty
  critical_penalty   = +10.0 if any resource below min_need
"""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from backend.models.resource import ResourceType, VillageResourceNeed
from backend.models.village import Village

CRITICAL_PENALTY = 10.0


# ------------------------------------------------------------------ #
#  Output model                                                        #
# ------------------------------------------------------------------ #

class UrgencyComponent(BaseModel):
    """Auditable contribution of one resource to a village urgency score."""

    resource_type: str
    current_need: float = Field(ge=0.0)
    existing_allocated: float = Field(ge=0.0)
    unmet_need: float = Field(ge=0.0)
    unmet_ratio: float = Field(ge=0.0, le=1.0)
    urgency_multiplier: float = Field(ge=0.0)
    time_factor: float = Field(ge=1.0)
    contribution: float = Field(ge=0.0)
    below_survival_threshold: bool = False


class UrgencyScore(BaseModel):
    village_id: str
    total_urgency: float = Field(..., ge=0.0)
    resource_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-resource urgency scores keyed by resource_type ID"
    )
    has_critical_shortage: bool = False
    time_elapsed_hours: float = Field(default=0.0, ge=0.0)
    time_factor: float = Field(default=1.0, ge=1.0)
    base_resource_urgency: float = Field(default=0.0, ge=0.0)
    critical_penalty: float = Field(default=0.0, ge=0.0)
    ranking: int = Field(default=0, description="1 = most urgent; set by rank_villages()")
    external_signal: float = Field(default=0.0, ge=0.0)
    components: Dict[str, UrgencyComponent] = Field(default_factory=dict)
    formula: str = (
        "sum((unmet/current_need) x multiplier x time_factor) "
        "+ critical_penalty + Gemma external signal"
    )

    @property
    def urgency_without_penalty(self) -> float:
        """Base urgency before the critical-shortage penalty is added."""
        return self.total_urgency - (CRITICAL_PENALTY if self.has_critical_shortage else 0.0)

    def top_resource(self) -> Optional[str]:
        """Resource type with the highest urgency score (or None if empty)."""
        if not self.resource_scores:
            return None
        return max(self.resource_scores, key=self.resource_scores.__getitem__)

    def __repr__(self) -> str:
        return (
            f"UrgencyScore(rank={self.ranking}, village={self.village_id!r}, "
            f"urgency={self.total_urgency:.3f}, critical={self.has_critical_shortage})"
        )


# ------------------------------------------------------------------ #
#  Calculator                                                          #
# ------------------------------------------------------------------ #

class UrgencyCalculator:
    """
    Calculates and ranks village urgency scores.

    Args:
        resource_types: Dict mapping resource_id → ResourceType (loaded from config.json).
                        If a village has a resource not in this dict, urgency_multiplier=1.0 is used.
    """

    def __init__(self, resource_types: Dict[str, ResourceType]) -> None:
        self.resource_types = resource_types

    # ---------------------------------------------------------------- #
    #  Core formulas                                                    #
    # ---------------------------------------------------------------- #

    def calculate_time_factor(self, hours_elapsed: float) -> float:
        """
        Exponential urgency growth over time.

          time_factor = 1.0 + (exp(0.3 * hours) - 1.0) * 0.5

        Benchmarks (approximate):
          t=0 hr  → 1.00
          t=2 hr  → 1.41
          t=4 hr  → 2.16
          t=8 hr  → 6.01

        Note: the formula produces ~1.41 at t=2, not 1.5 exactly — the spec's
        "≈1.5" targets are illustrative of the growth direction, not exact.
        """
        if hours_elapsed <= 0.0:
            return 1.0
        return 1.0 + (math.exp(0.3 * hours_elapsed) - 1.0) * 0.5

    def calculate_resource_urgency(
        self,
        unmet_need: float,
        current_need: float,
        urgency_multiplier: float,
        time_factor: float,
    ) -> float:
        """
        Urgency score for a single resource type at a single village.

          resource_urgency = (unmet_need / current_need) * urgency_multiplier * time_factor

        Returns 0.0 when:
          - current_need is zero (nothing expected)
          - unmet_need is zero (fully satisfied)
        """
        if current_need <= 0.0:
            return 0.0
        unmet_ratio = min(1.0, max(0.0, unmet_need / current_need))
        return unmet_ratio * urgency_multiplier * time_factor

    # ---------------------------------------------------------------- #
    #  Village-level calculation                                        #
    # ---------------------------------------------------------------- #

    def calculate_village_urgency(
        self,
        village: Village,
        time_elapsed: timedelta,
    ) -> UrgencyScore:
        """
        Compute total urgency for one village at a given elapsed time.

        Sums resource urgencies across all entries in village.resource_needs,
        then adds CRITICAL_PENALTY (+10) if any resource is below min_need.
        """
        hours = max(0.0, time_elapsed.total_seconds() / 3600.0)
        time_factor = self.calculate_time_factor(hours)

        resource_scores: Dict[str, float] = {}
        components: Dict[str, UrgencyComponent] = {}
        base_total = 0.0

        for rtype_id, need in village.resource_needs.items():
            rtype = self.resource_types.get(rtype_id)
            multiplier = rtype.urgency_multiplier if rtype is not None else 1.0

            score = self.calculate_resource_urgency(
                need.unmet_need,
                need.current_need,
                multiplier,
                time_factor,
            )
            resource_scores[rtype_id] = score
            unmet_ratio = (
                min(1.0, max(0.0, need.unmet_need / need.current_need))
                if need.current_need > 0
                else 0.0
            )
            components[rtype_id] = UrgencyComponent(
                resource_type=rtype_id,
                current_need=need.current_need,
                existing_allocated=need.allocated,
                unmet_need=need.unmet_need,
                unmet_ratio=unmet_ratio,
                urgency_multiplier=multiplier,
                time_factor=time_factor,
                contribution=score,
                below_survival_threshold=need.critical,
            )
            base_total += score

        critical = village.has_critical_shortage
        critical_penalty = CRITICAL_PENALTY if critical else 0.0
        total_urgency = base_total + village.external_urgency_boost + critical_penalty

        return UrgencyScore(
            village_id=village.id,
            total_urgency=total_urgency,
            resource_scores=resource_scores,
            has_critical_shortage=critical,
            time_elapsed_hours=hours,
            time_factor=time_factor,
            base_resource_urgency=base_total,
            critical_penalty=critical_penalty,
            ranking=0,  # assigned by rank_villages()
            external_signal=village.external_urgency_boost,
            components=components,
        )

    # ---------------------------------------------------------------- #
    #  Fleet-level ranking                                              #
    # ---------------------------------------------------------------- #

    def rank_villages(
        self,
        villages: List[Village],
        time_elapsed: timedelta,
    ) -> List[UrgencyScore]:
        """
        Score all villages and return them sorted descending by total_urgency.
        Rankings are 1-based (1 = most urgent).
        """
        scores = [self.calculate_village_urgency(v, time_elapsed) for v in villages]
        scores.sort(key=lambda s: s.total_urgency, reverse=True)
        for rank, score in enumerate(scores, start=1):
            score.ranking = rank
        return scores

    def detect_reoptimization_trigger(
        self,
        old_scores: List[UrgencyScore],
        new_scores: List[UrgencyScore],
        threshold: float = 0.10,
    ) -> List[Dict]:
        """
        Compare two ranking snapshots.
        Returns list of {village_id, old_urgency, new_urgency, delta} for villages
        where urgency changed by more than threshold.
        Used by the RAG pipeline to decide whether to re-optimize.
        """
        old_map = {s.village_id: s.total_urgency for s in old_scores}
        triggers = []
        for score in new_scores:
            old_val = old_map.get(score.village_id, score.total_urgency)
            delta = abs(score.total_urgency - old_val)
            if delta > threshold:
                triggers.append({
                    "village_id": score.village_id,
                    "old_urgency": old_val,
                    "new_urgency": score.total_urgency,
                    "delta": delta,
                })
        return triggers
