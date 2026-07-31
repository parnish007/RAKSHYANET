"""
Vehicle models: extensible type-system with configurable fleet composition.

Design:
  VehicleType   — config template (Pydantic model), defines a class of vehicle
  Vehicle       — runtime instance with state, cargo, and location
  Helicopter()  — factory shorthand returning a Vehicle with helicopter VehicleType
  Truck()       — factory shorthand returning a Vehicle with truck VehicleType
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ------------------------------------------------------------------ #
#  Enums                                                               #
# ------------------------------------------------------------------ #

class VehicleState(str, Enum):
    AVAILABLE  = "AVAILABLE"
    IN_TRANSIT = "IN_TRANSIT"
    DEPLOYED   = "DEPLOYED"
    REFUELING  = "REFUELING"


class VehicleCategory(str, Enum):
    """Broad capability group — used for routing and UI filtering."""
    AIRCRAFT      = "aircraft"        # Helicopters, drones
    GROUND_HEAVY  = "ground_heavy"    # Trucks, cargo lorries
    GROUND_LIGHT  = "ground_light"    # Vans, motorcycles, bicycles
    AMPHIBIOUS    = "amphibious"      # Boats for flood scenarios


class TerrainCapability(str, Enum):
    """What terrain a vehicle can traverse."""
    ANY           = "any"             # Helicopters / drones
    PAVED_ROADS   = "paved_roads"     # Heavy trucks (paved only)
    ALL_ROADS     = "all_roads"       # 4×4 vans (paved + dirt roads)
    DIRT_PATHS    = "dirt_paths"      # Motorcycles / bicycles
    WATER         = "water"           # Boats
    MULTI_TERRAIN = "multi_terrain"   # Amphibious vehicles


# Village accessibility → set of TerrainCapabilities that can reach it
_ACCESSIBILITY_MAP: Dict[str, set] = {
    "road":       {TerrainCapability.ANY, TerrainCapability.PAVED_ROADS,
                   TerrainCapability.ALL_ROADS, TerrainCapability.DIRT_PATHS,
                   TerrainCapability.MULTI_TERRAIN},
    "dirt_road":  {TerrainCapability.ANY, TerrainCapability.ALL_ROADS,
                   TerrainCapability.DIRT_PATHS, TerrainCapability.MULTI_TERRAIN},
    "water":      {TerrainCapability.ANY, TerrainCapability.WATER,
                   TerrainCapability.MULTI_TERRAIN},
    "any":        {TerrainCapability.ANY, TerrainCapability.MULTI_TERRAIN},
}


# ------------------------------------------------------------------ #
#  VehicleType  (config template)                                      #
# ------------------------------------------------------------------ #

class VehicleType(BaseModel):
    """
    Template that defines a class of vehicle.
    Stored in config.json; instances are created from this spec.
    """
    type_id: str = Field(..., description="Unique type identifier, e.g. 'heavy_helicopter'")
    name: str
    category: VehicleCategory
    capacity_kg: float = Field(..., gt=0)
    speed_kmh: float = Field(..., gt=0)
    fuel_hours: float = Field(..., gt=0)
    terrain_capability: TerrainCapability
    cost_per_km: float = Field(default=1.0, gt=0, description="Relative operating cost")
    preferred_resources: List[str] = Field(
        default_factory=list,
        description="Resource type IDs this vehicle is optimised for. Empty = no preference."
    )

    @property
    def fuel_range_km(self) -> float:
        return self.speed_kmh * self.fuel_hours

    def __repr__(self) -> str:
        return f"VehicleType({self.type_id!r}, {self.capacity_kg}kg, {self.terrain_capability.value})"


# ------------------------------------------------------------------ #
#  Pre-built standard types (used by Helicopter / Truck factories)     #
# ------------------------------------------------------------------ #

HELICOPTER_TYPE = VehicleType(
    type_id="helicopter",
    name="Helicopter",
    category=VehicleCategory.AIRCRAFT,
    capacity_kg=500.0,
    speed_kmh=200.0,
    fuel_hours=2.0,
    terrain_capability=TerrainCapability.ANY,
    cost_per_km=3.5,
)

TRUCK_TYPE = VehicleType(
    type_id="truck",
    name="Truck",
    category=VehicleCategory.GROUND_HEAVY,
    capacity_kg=2000.0,
    speed_kmh=40.0,
    fuel_hours=8.0,
    terrain_capability=TerrainCapability.PAVED_ROADS,
    cost_per_km=1.5,
)


# ------------------------------------------------------------------ #
#  Vehicle  (runtime instance)                                         #
# ------------------------------------------------------------------ #

class Vehicle(BaseModel):
    """A single vehicle with live state, location, and cargo."""

    id: str
    name: str
    vehicle_type: VehicleType

    # Runtime state
    state: VehicleState = VehicleState.AVAILABLE
    current_location: Tuple[float, float] = Field(
        default=(27.7172, 85.3240),
        description="(lat, lng) — defaults to Kathmandu depot"
    )
    destination: Optional[str] = None          # village_id currently heading to
    cargo_manifest: Dict[str, float] = Field(
        default_factory=dict,
        description="{resource_type: amount_kg}"
    )
    deployed_at: Optional[datetime] = None

    model_config = {"arbitrary_types_allowed": True}

    # ---------------------------------------------------------------- #
    #  Backward-compat properties (delegate to vehicle_type)           #
    # ---------------------------------------------------------------- #

    @property
    def capacity_kg(self) -> float:
        return self.vehicle_type.capacity_kg

    @property
    def speed_kmh(self) -> float:
        return self.vehicle_type.speed_kmh

    @property
    def fuel_hours(self) -> float:
        return self.vehicle_type.fuel_hours

    @property
    def terrain(self) -> str:
        """Legacy string alias for terrain_capability (for backward compat)."""
        cap = self.vehicle_type.terrain_capability
        if cap == TerrainCapability.ANY:
            return "any"
        if cap in (TerrainCapability.PAVED_ROADS, TerrainCapability.ALL_ROADS):
            return "roads_only"
        return cap.value

    # ---------------------------------------------------------------- #
    #  Computed helpers                                                  #
    # ---------------------------------------------------------------- #

    @property
    def remaining_capacity(self) -> float:
        loaded = sum(self.cargo_manifest.values())
        return max(0.0, self.vehicle_type.capacity_kg - loaded)

    @property
    def fuel_range_km(self) -> float:
        return self.vehicle_type.fuel_range_km

    @property
    def is_available_for_assignment(self) -> bool:
        return self.state == VehicleState.AVAILABLE

    # ---------------------------------------------------------------- #
    #  Methods                                                          #
    # ---------------------------------------------------------------- #

    def can_deliver_to(self, village_accessibility: str = "road") -> bool:
        """
        True if this vehicle's terrain capability allows reaching a village
        with the given accessibility type ('road', 'dirt_road', 'water', 'any').
        Helicopters (TerrainCapability.ANY) always return True.
        """
        cap = self.vehicle_type.terrain_capability
        if cap == TerrainCapability.ANY:
            return True
        allowed = _ACCESSIBILITY_MAP.get(village_accessibility, set())
        return cap in allowed

    def can_carry_resource(self, resource_type: str) -> bool:
        """True if vehicle has no preference (carries anything) or explicitly lists this type."""
        prefs = self.vehicle_type.preferred_resources
        return not prefs or resource_type in prefs

    def load_cargo(self, resource_type: str, amount_kg: float) -> float:
        """
        Load a resource type. Returns actual amount loaded (capped by remaining capacity).
        Raises ValueError if vehicle is DEPLOYED.
        """
        if self.state == VehicleState.DEPLOYED:
            raise ValueError(f"Vehicle {self.id} is DEPLOYED — cannot load cargo")
        loadable = min(amount_kg, self.remaining_capacity)
        if loadable > 0:
            self.cargo_manifest[resource_type] = (
                self.cargo_manifest.get(resource_type, 0.0) + loadable
            )
        return loadable

    def deploy(self, destination_village_id: str) -> None:
        self.state = VehicleState.IN_TRANSIT
        self.destination = destination_village_id
        self.deployed_at = datetime.utcnow()

    def mark_deployed(self, location: Tuple[float, float]) -> None:
        self.state = VehicleState.DEPLOYED
        self.current_location = location

    def return_to_depot(self, depot_location: Tuple[float, float]) -> None:
        self.state = VehicleState.AVAILABLE
        self.current_location = depot_location
        self.cargo_manifest = {}
        self.destination = None
        self.deployed_at = None

    def __repr__(self) -> str:
        return (
            f"Vehicle({self.id!r}, type={self.vehicle_type.type_id!r}, "
            f"state={self.state.value}, "
            f"capacity={self.remaining_capacity:.0f}/{self.vehicle_type.capacity_kg:.0f}kg)"
        )


# ------------------------------------------------------------------ #
#  Backward-compat factories                                           #
# ------------------------------------------------------------------ #

def Helicopter(
    id: str,
    name: Optional[str] = None,
    capacity_kg: float = 500.0,
    speed_kmh: float = 200.0,
    fuel_hours: float = 2.0,
    **kwargs,
) -> Vehicle:
    """Factory: create a Vehicle with helicopter specs."""
    vtype = VehicleType(
        type_id="helicopter",
        name="Helicopter",
        category=VehicleCategory.AIRCRAFT,
        capacity_kg=capacity_kg,
        speed_kmh=speed_kmh,
        fuel_hours=fuel_hours,
        terrain_capability=TerrainCapability.ANY,
        cost_per_km=3.5,
    )
    return Vehicle(id=id, name=name or id, vehicle_type=vtype, **kwargs)


def Truck(
    id: str,
    name: Optional[str] = None,
    capacity_kg: float = 2000.0,
    speed_kmh: float = 40.0,
    fuel_hours: float = 8.0,
    **kwargs,
) -> Vehicle:
    """Factory: create a Vehicle with truck specs."""
    vtype = VehicleType(
        type_id="truck",
        name="Truck",
        category=VehicleCategory.GROUND_HEAVY,
        capacity_kg=capacity_kg,
        speed_kmh=speed_kmh,
        fuel_hours=fuel_hours,
        terrain_capability=TerrainCapability.PAVED_ROADS,
        cost_per_km=1.5,
    )
    return Vehicle(id=id, name=name or id, vehicle_type=vtype, **kwargs)


# Keep CargoItem for any code that imports it (now unused internally)
class CargoItem(BaseModel):
    """Deprecated — cargo_manifest is now Dict[str, float]. Kept for import compat."""
    village_id: str
    amount_kg: float = Field(..., gt=0)
    item_type: str = Field(default="supplies")
