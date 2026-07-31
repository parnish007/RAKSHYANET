"""
Village data model with dynamic urgency tracking and multi-resource needs.
"""
import math
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator, computed_field, model_validator

from .resource import VillageResourceNeed


class Village(BaseModel):
    # Identity
    id: str
    name: str
    lat: float = Field(..., ge=-90.0, le=90.0)
    lng: float = Field(..., ge=-180.0, le=180.0)
    population: int = Field(..., gt=0)
    terrain_difficulty: float = Field(default=1.0, ge=1.0, le=5.0)

    # ── Legacy single-need fields (kept for backward compat) ──────────
    # These are still used when resource_needs is empty (e.g. tests).
    current_need: Optional[float] = Field(default=None, ge=0.0)
    min_need: Optional[float] = Field(default=None, ge=0.0)
    allocated: float = Field(default=0.0, ge=0.0)

    # ── Multi-resource needs (preferred; replaces single-need fields) ──
    resource_needs: Dict[str, VillageResourceNeed] = Field(
        default_factory=dict,
        description="Per-resource needs keyed by resource_type ID"
    )

    # Dynamic urgency (updated by RAG pipeline)
    urgency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    previous_urgency: float = Field(default=0.5, ge=0.0, le=1.0)
    urgency_updated_at: datetime = Field(default_factory=datetime.utcnow)
    external_urgency_boost: float = Field(default=0.0, ge=0.0, le=2.0)

    # Disaster context
    disaster_impact: float = Field(default=0.5, ge=0.0, le=1.0)
    has_medical_facility: bool = False
    accessibility: str = Field(default="road")

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("min_need")
    @classmethod
    def min_need_lte_current(cls, v, info):
        if v is not None and "current_need" in info.data:
            cn = info.data.get("current_need")
            if cn is not None and v > cn:
                raise ValueError("min_need cannot exceed current_need")
        return v

    # ------------------------------------------------------------------ #
    #  Methods                                                             #
    # ------------------------------------------------------------------ #

    def calculate_distance_from(self, lat: float, lng: float) -> float:
        """Haversine distance in km from a given coordinate to this village."""
        R = 6371.0
        phi1 = math.radians(lat)
        phi2 = math.radians(self.lat)
        dphi = math.radians(self.lat - lat)
        dlambda = math.radians(self.lng - lng)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def update_urgency(self, new_score: float) -> None:
        """Update urgency, preserving previous value for delta detection."""
        self.previous_urgency = self.urgency_score
        self.urgency_score = max(0.0, min(1.0, new_score))
        self.urgency_updated_at = datetime.utcnow()

    def urgency_delta(self) -> float:
        """Return absolute change in urgency since last update."""
        return abs(self.urgency_score - self.previous_urgency)

    def get_resource_need(self, resource_type: str) -> Optional[VillageResourceNeed]:
        return self.resource_needs.get(resource_type)

    # ------------------------------------------------------------------ #
    #  Computed fields                                                     #
    # ------------------------------------------------------------------ #

    @computed_field
    @property
    def unmet_need(self) -> float:
        """Sum of unmet need across resources, in MIXED NATIVE UNITS.

        This is NOT kilograms. It adds litres of water to medical kits to
        tarpaulin sheets, which MATH.md section 1 forbids comparing. It is kept
        only as a coarse "is anything outstanding here" indicator for sorting
        and display; never use it as a quantity, and never present it with a
        unit. Per-resource figures in `resource_needs` are the real numbers.

        Uses resource_needs when populated; falls back to legacy current_need.
        """
        if self.resource_needs:
            return sum(n.unmet_need for n in self.resource_needs.values())
        if self.current_need is not None:
            return max(0.0, self.current_need - self.allocated)
        return 0.0

    @computed_field
    @property
    def total_unmet_need_mixed_units(self) -> float:
        """Alias for `unmet_need`, named so the mixed units are unmissable.

        The former name was `total_unmet_need_kg`, which asserted a unit this
        quantity does not have.
        """
        return self.unmet_need

    @computed_field
    @property
    def has_critical_shortage(self) -> bool:
        """True if any resource is below its survival threshold."""
        if self.resource_needs:
            return any(n.critical for n in self.resource_needs.values())
        # Legacy fallback
        if self.current_need is not None and self.min_need is not None:
            return self.allocated < self.min_need
        return False

    @computed_field
    @property
    def fairness_weight(self) -> float:
        """Raw population — normalised by the solver across all villages."""
        return float(self.population)

    def __repr__(self) -> str:
        return (
            f"Village({self.id!r}, urgency={self.urgency_score:.2f}, "
            f"unmet={self.unmet_need:.0f} (mixed units), "
            f"critical={self.has_critical_shortage})"
        )
