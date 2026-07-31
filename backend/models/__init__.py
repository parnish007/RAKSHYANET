from .resource import ResourceCategory, ResourceType, VillageResourceNeed
from .village import Village
from .vehicle import (
    VehicleState, VehicleCategory, TerrainCapability,
    VehicleType, Vehicle, Helicopter, Truck, CargoItem,
    HELICOPTER_TYPE, TRUCK_TYPE,
)
from .allocation import AllocationResult, VehicleRoute, RouteWaypoint, KKTConditions, ConvergencePoint
from .news import NewsEvent, TRUSTED_SOURCES, SEVERITY_KEYWORDS
from .hitl import HITLDecision, HITLRequest, HITLDecisionType, HITLStatus

__all__ = [
    # Resource
    "ResourceCategory", "ResourceType", "VillageResourceNeed",
    # Village
    "Village",
    # Vehicle
    "VehicleState", "VehicleCategory", "TerrainCapability",
    "VehicleType", "Vehicle", "Helicopter", "Truck", "CargoItem",
    "HELICOPTER_TYPE", "TRUCK_TYPE",
    # Allocation
    "AllocationResult", "VehicleRoute", "RouteWaypoint", "KKTConditions", "ConvergencePoint",
    # News
    "NewsEvent", "TRUSTED_SOURCES", "SEVERITY_KEYWORDS",
    # HITL
    "HITLDecision", "HITLRequest", "HITLDecisionType", "HITLStatus",
]
"""RakshyaNet domain models.

Modules are intentionally imported directly to avoid circular imports between
optimization result contracts and the algorithm pipeline.
"""
