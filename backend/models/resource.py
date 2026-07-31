"""
Resource type definitions and village-level resource needs.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ResourceCategory(str, Enum):
    FOOD = "food"
    WATER = "water"
    MEDICAL = "medical"
    SHELTER = "shelter"
    SAFETY = "safety"
    COMMUNICATION = "communication"


class ResourceType(BaseModel):
    """Defines a category of relief resource (template, not an instance)."""

    resource_id: str = Field(..., description="Unique identifier, e.g. 'medical_kit'")
    name: str
    category: ResourceCategory
    unit: str = Field(default="kg", description="Unit of measurement")
    urgency_multiplier: float = Field(
        default=1.0, ge=0.0, le=2.0,
        description="How critical this resource is (0–2). Used in Nash objective weighting."
    )
    weight_per_unit: float = Field(
        default=1.0, gt=0,
        description="Physical weight in kg per unit (for capacity planning)"
    )
    shelf_life_hours: Optional[float] = Field(
        default=None,
        description="Hours until spoilage. None = non-perishable."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "resource_id": "medical_kit",
                "name": "Emergency Medical Kit",
                "category": "medical",
                "unit": "kit",
                "urgency_multiplier": 1.8,
                "weight_per_unit": 5.0,
                "shelf_life_hours": None,
            }
        }
    }

    @property
    def is_perishable(self) -> bool:
        return self.shelf_life_hours is not None

    def __repr__(self) -> str:
        return f"ResourceType({self.resource_id!r}, category={self.category.value})"


class VillageResourceNeed(BaseModel):
    """Specific need for one resource type at one village."""

    resource_type: str = Field(..., description="References ResourceType.resource_id")
    current_need: float = Field(..., ge=0.0, description="Total need in resource units")
    min_need: float = Field(..., ge=0.0, description="Survival threshold")
    allocated: float = Field(default=0.0, ge=0.0, description="Amount allocated so far")

    @property
    def unmet_need(self) -> float:
        """Need above what has already been allocated."""
        return max(0.0, self.current_need - self.allocated)

    @property
    def critical(self) -> bool:
        """True when allocated supply is below the survival threshold."""
        return self.allocated < self.min_need

    @property
    def satisfaction_ratio(self) -> float:
        """0.0 = nothing allocated, 1.0 = fully met."""
        if self.current_need == 0:
            return 1.0
        return min(1.0, self.allocated / self.current_need)

    def __repr__(self) -> str:
        return (
            f"VillageResourceNeed({self.resource_type!r}, "
            f"{self.allocated:.0f}/{self.current_need:.0f}, "
            f"critical={self.critical})"
        )
