"""
VRP Solver — Prompt 2.2

Greedy urgency-first resource assignment + nearest-neighbour route construction.
Continuous allocation is computed separately from these heuristic routes.

Algorithm:
  1. Sort villages by urgency score (descending).
  2. Greedy assignment: process critical resources first, then score every
     viable asset by response time, payload fit, incident impact, specialty,
     consolidation, transport-mode fit, and complete-tour fuel feasibility.
     Allocate as much as possible from depot stock in descending score order.
  3. Build routes: for each vehicle that received cargo, construct a
     depot → [stops] → depot tour using nearest-neighbour ordering.
  4. Assemble VRPSolution with satisfaction metrics.
"""
from __future__ import annotations

import math
from heapq import heappop, heappush
from typing import Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from backend.models.resource import ResourceType
from backend.models.vehicle import (
    TerrainCapability,
    Vehicle,
    VehicleCategory,
    VehicleType,
)
from backend.models.village import Village
from backend.algorithms.urgency_calculator import UrgencyScore


# ------------------------------------------------------------------ #
#  Terrain access table                                                #
# ------------------------------------------------------------------ #
# Maps TerrainCapability → set of village accessibility values it can reach.
TERRAIN_ACCESS: Dict[TerrainCapability, set] = {
    TerrainCapability.ANY:           {"road", "dirt_road", "helicopter_only", "water", "any"},
    TerrainCapability.ALL_ROADS:     {"road", "dirt_road"},
    TerrainCapability.PAVED_ROADS:   {"road"},
    TerrainCapability.DIRT_PATHS:    {"road", "dirt_road", "helicopter_only"},
    TerrainCapability.WATER:         {"water"},
    TerrainCapability.MULTI_TERRAIN: {"road", "dirt_road", "helicopter_only", "water", "any"},
}


# ------------------------------------------------------------------ #
#  Output models                                                       #
# ------------------------------------------------------------------ #

class RouteStop(BaseModel):
    village_id: str
    lat: float
    lng: float
    distance_from_prev_km: float
    cumulative_distance_km: float
    eta_minutes: float
    cargo_delivered: Dict[str, float] = Field(default_factory=dict)


class RouteLeg(BaseModel):
    from_node_id: str
    to_node_id: str
    mode: str
    distance_km: float
    edge_ids: List[str] = Field(default_factory=list)
    geometry: List[Tuple[float, float]] = Field(default_factory=list)
    blocked_edges_avoided: List[str] = Field(default_factory=list)
    explanation: str


class AssetSelectionReason(BaseModel):
    """Auditable reason one asset received part of a resource assignment."""

    vehicle_id: str
    transport_mode: str
    quantity: float
    payload_kg: float
    estimated_one_way_minutes: float
    projected_route_minutes: float
    selection_score: float
    time_pressure: float
    eta_score: float
    payload_fit_score: float
    incident_impact: float
    specialty_bonus: float
    consolidation_bonus: float
    explanation: str


class ResourceAllocationDecision(BaseModel):
    resource_type: str
    unit: str
    current_need: float
    existing_allocated: float
    survival_threshold: float
    unmet_before_plan: float
    depot_available_at_start: float
    proposed_now: float
    post_plan_total: float
    unmet_after_plan: float
    survival_gap_after_plan: float
    assigned_vehicle_ids: List[str] = Field(default_factory=list)
    asset_selection: List[AssetSelectionReason] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    explanation: str


class Route(BaseModel):
    vehicle_id: str
    stops: List[str] = Field(default_factory=list, description="Village IDs in visit order")
    stop_details: List[RouteStop] = Field(default_factory=list)
    total_distance_km: float = 0.0
    total_time_minutes: float = 0.0
    cargo_manifest: Dict[str, float] = Field(default_factory=dict)
    total_cargo_kg: float = 0.0
    transport_mode: str = "air"
    path_coordinates: List[Tuple[float, float]] = Field(default_factory=list)
    legs: List[RouteLeg] = Field(default_factory=list)
    road_edge_ids: List[str] = Field(default_factory=list)
    rerouted_due_to: List[str] = Field(default_factory=list)
    routing_explanation: str = ""
    feasible: bool = True
    infeasibility_reason: Optional[str] = None


class VillageAllocation(BaseModel):
    """Per-village allocation result (VRP level — distinct from models.allocation)."""
    model_config = {"populate_by_name": True}

    village_id: str
    allocated_resources: Dict[str, float] = Field(default_factory=dict)
    vehicle_assignments: List[str] = Field(default_factory=list)
    eta_minutes: float = 0.0          # ETA of the first arriving vehicle
    satisfied: bool = False           # True if all resources ≥ min_need


    resource_decisions: List[ResourceAllocationDecision] = Field(default_factory=list)
    allocation_explanation: str = ""


# backward-compat alias expected by the prompt spec
AllocationResult = VillageAllocation


class VRPSolution(BaseModel):
    routes: List[Route] = Field(default_factory=list)
    allocations: List[VillageAllocation] = Field(default_factory=list)
    total_distance_km: float = 0.0
    unmet_villages: List[str] = Field(default_factory=list)
    convergence_iterations: int = 1   # Greedy routing is a single pass.
    objective_value: float = 0.0      # Σ(allocated/need) across all villages × resources
    active_road_blocks: List[str] = Field(default_factory=list)
    road_network: List[Dict[str, object]] = Field(default_factory=list)
    routing_source_kind: str = "bundled_scenario_fixture"
    routing_source_label: str = "Mocked deterministic road graph"


# ------------------------------------------------------------------ #
#  VRPSolver                                                           #
# ------------------------------------------------------------------ #

class VRPSolver:
    """
    Greedy VRP solver.

    Args:
        depot_location: (lat, lng) of the central depot.
        terrain_graph:  Raw terrain_graph.json dict (used for node coord lookup).
        resource_types: Dict[resource_id → ResourceType] from config.json.
        config:         Full config.json dict (currently informational).
    """

    DEPOT_ID = "depot"
    MAX_STOPS_PER_ASSET = 2

    def __init__(
        self,
        depot_location: Tuple[float, float],
        terrain_graph: Dict,
        resource_types: Dict[str, ResourceType],
        config: Dict,
        blocked_edge_ids: Optional[List[str]] = None,
        terrain_weighting: bool = True,
        honour_closures: bool = True,
    ) -> None:
        self.depot_location = depot_location
        self.terrain_graph = terrain_graph
        self.resource_types = resource_types
        self.config = config
        self._village_map: Dict[str, Village] = {}
        self.blocked_edge_ids: Set[str] = set(blocked_edge_ids or [])
        # These two switches exist so the documented naive baseline is the SAME
        # engine with its terrain reasoning turned off, rather than a separately
        # written strawman. A baseline nobody would actually build proves
        # nothing; "shortest path, no terrain weighting, closures ignored" is
        # what a reasonable first implementation does.
        self.terrain_weighting = terrain_weighting
        self.honour_closures = honour_closures
        self._graph_nodes = {
            node["id"]: node for node in self.terrain_graph.get("nodes", [])
        }
        self._graph_edges = [
            {
                **edge,
                "id": edge.get("id") or f"{edge['from']}__{edge['to']}",
            }
            for edge in self.terrain_graph.get("edges", [])
        ]
        self._urgency_by_id: Dict[str, UrgencyScore] = {}

    # ---------------------------------------------------------------- #
    #  Primitive helpers                                                 #
    # ---------------------------------------------------------------- #

    @staticmethod
    def calculate_distance(
        coord1: Tuple[float, float],
        coord2: Tuple[float, float],
    ) -> float:
        """Haversine great-circle distance in km."""
        R = 6371.0
        lat1, lon1 = map(math.radians, coord1)
        lat2, lon2 = map(math.radians, coord2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def calculate_travel_time(distance_km: float, vehicle_type: VehicleType) -> float:
        """One-way travel time in minutes."""
        if vehicle_type.speed_kmh <= 0:
            return float("inf")
        return (distance_km / vehicle_type.speed_kmh) * 60.0

    def check_terrain_accessibility(
        self,
        village: Village,
        vehicle_type: VehicleType,
    ) -> bool:
        """Return True if the vehicle's terrain capability allows reaching the village."""
        accessible = TERRAIN_ACCESS.get(vehicle_type.terrain_capability, set())
        return village.accessibility in accessible

    def can_vehicle_carry(self, vehicle: Vehicle, resource_type: str) -> bool:
        """True if vehicle has no resource preference, or explicitly lists this type."""
        prefs = vehicle.vehicle_type.preferred_resources
        return not prefs or resource_type in prefs

    @staticmethod
    def _is_aircraft(vehicle_type: VehicleType) -> bool:
        return vehicle_type.category == VehicleCategory.AIRCRAFT

    @staticmethod
    def _edge_allowed_for_vehicle(edge: Dict, vehicle_type: VehicleType) -> bool:
        if not edge.get("has_road", True):
            return False
        quality = edge.get("road_quality", "paved")
        capability = vehicle_type.terrain_capability
        if capability == TerrainCapability.PAVED_ROADS:
            return quality == "paved"
        if capability in {
            TerrainCapability.ALL_ROADS,
            TerrainCapability.DIRT_PATHS,
            TerrainCapability.MULTI_TERRAIN,
        }:
            return quality in {"paved", "mixed", "dirt"}
        return False

    def _origin_node_for(self, vehicle: Vehicle) -> str:
        """Graph node a vehicle departs from when re-planning mid-mission.

        Routes used to always begin at the depot, so a plan recomputed while the
        fleet was already moving routed every asset from where it started rather
        than from where it is.

        Aircraft can depart from an arbitrary point, but ground vehicles travel
        on edges and a position part-way along a corridor is not a node. Both are
        therefore snapped to the nearest node in the graph, which is exact when
        the asset is at or near a stop — the realistic mid-mission case — and an
        approximation otherwise. The depot is returned when the vehicle has not
        moved or nothing is closer.
        """
        location = getattr(vehicle, "current_location", None)
        if not location:
            return self.DEPOT_ID
        lat, lng = float(location[0]), float(location[1])
        depot_lat, depot_lng = self.depot_location
        best_id = self.DEPOT_ID
        best_distance = self.calculate_distance((lat, lng), (depot_lat, depot_lng))
        # Only nodes that the graph can actually route from are candidates.
        for node_id in self._graph_nodes:
            coordinate = self._node_coordinate(node_id)
            if coordinate is None:
                continue
            # _node_coordinate returns (lng, lat); calculate_distance takes (lat, lng).
            distance = self.calculate_distance((lat, lng), (coordinate[1], coordinate[0]))
            if distance < best_distance:
                best_distance = distance
                best_id = node_id
        return best_id

    def _node_coordinate(self, node_id: str) -> Optional[Tuple[float, float]]:
        node = self._graph_nodes.get(node_id)
        if node is not None:
            return (float(node["lng"]), float(node["lat"]))
        village = self._village_map.get(node_id)
        if village is not None:
            return (village.lng, village.lat)
        if node_id == self.DEPOT_ID:
            return (self.depot_location[1], self.depot_location[0])
        return None

    def _edge_geometry(
        self,
        edge: Dict,
        from_node_id: str,
    ) -> List[Tuple[float, float]]:
        raw = edge.get("geometry") or []
        if raw:
            geometry = [(float(point[0]), float(point[1])) for point in raw]
        else:
            start = self._node_coordinate(edge["from"])
            end = self._node_coordinate(edge["to"])
            geometry = [point for point in (start, end) if point is not None]
        if from_node_id != edge["from"]:
            geometry.reverse()
        return geometry

    def _shortest_road_path(
        self,
        from_node_id: str,
        to_node_id: str,
        vehicle_type: VehicleType,
        blocked_edge_ids: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, object]]:
        """Dijkstra over the deterministic road graph with closure filtering."""
        if from_node_id == to_node_id:
            coordinate = self._node_coordinate(from_node_id)
            return {
                "distance_km": 0.0,
                "edge_ids": [],
                "coordinates": [coordinate] if coordinate else [],
            }

        if not self._graph_nodes or not self._graph_edges:
            start = self._node_coordinate(from_node_id)
            end = self._node_coordinate(to_node_id)
            if start is None or end is None:
                return None
            return {
                "distance_km": self.calculate_distance(
                    (start[1], start[0]),
                    (end[1], end[0]),
                ),
                "edge_ids": [],
                "coordinates": [start, end],
                "fixture_fallback": True,
            }

        blocked = self.blocked_edge_ids if blocked_edge_ids is None else blocked_edge_ids
        if not self.honour_closures:
            blocked = set()
        adjacency: Dict[str, List[Tuple[str, Dict]]] = {}
        for edge in self._graph_edges:
            if edge["id"] in blocked or not self._edge_allowed_for_vehicle(edge, vehicle_type):
                continue
            adjacency.setdefault(edge["from"], []).append((edge["to"], edge))
            adjacency.setdefault(edge["to"], []).append((edge["from"], edge))

        queue: List[Tuple[float, str]] = [(0.0, from_node_id)]
        costs: Dict[str, float] = {from_node_id: 0.0}
        previous: Dict[str, Tuple[str, Dict]] = {}
        while queue:
            current_cost, node_id = heappop(queue)
            if current_cost > costs.get(node_id, float("inf")):
                continue
            if node_id == to_node_id:
                break
            for neighbour, edge in adjacency.get(node_id, []):
                if self.terrain_weighting:
                    difficulty = float(edge.get("terrain_difficulty", 1.0))
                    risk_factor = 1.0 + max(0.0, difficulty - 1.0) * 0.06
                else:
                    risk_factor = 1.0
                next_cost = current_cost + float(edge["distance_km"]) * risk_factor
                if next_cost >= costs.get(neighbour, float("inf")):
                    continue
                costs[neighbour] = next_cost
                previous[neighbour] = (node_id, edge)
                heappush(queue, (next_cost, neighbour))

        if to_node_id not in previous:
            return None

        steps: List[Tuple[str, str, Dict]] = []
        cursor = to_node_id
        while cursor != from_node_id:
            prior, edge = previous[cursor]
            steps.append((prior, cursor, edge))
            cursor = prior
        steps.reverse()

        coordinates: List[Tuple[float, float]] = []
        distance = 0.0
        edge_ids: List[str] = []
        for step_from, _step_to, edge in steps:
            geometry = self._edge_geometry(edge, step_from)
            if coordinates and geometry and coordinates[-1] == geometry[0]:
                coordinates.extend(geometry[1:])
            else:
                coordinates.extend(geometry)
            distance += float(edge["distance_km"])
            edge_ids.append(str(edge["id"]))
        return {
            "distance_km": distance,
            "edge_ids": edge_ids,
            "coordinates": coordinates,
        }

    def _leg_between(
        self,
        from_node_id: str,
        to_node_id: str,
        vehicle_type: VehicleType,
    ) -> Optional[RouteLeg]:
        start = self._node_coordinate(from_node_id)
        end = self._node_coordinate(to_node_id)
        if start is None or end is None:
            return None
        if self._is_aircraft(vehicle_type):
            distance = self.calculate_distance(
                (start[1], start[0]),
                (end[1], end[0]),
            )
            return RouteLeg(
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                mode="air",
                distance_km=distance,
                geometry=[start, end],
                explanation="Aircraft uses a direct geodesic corridor; road closures do not constrain this leg.",
            )

        path = self._shortest_road_path(from_node_id, to_node_id, vehicle_type)
        if path is None:
            return None
        baseline = self._shortest_road_path(
            from_node_id,
            to_node_id,
            vehicle_type,
            blocked_edge_ids=set(),
        )
        avoided = sorted(
            set(baseline.get("edge_ids", [])) & self.blocked_edge_ids
            if baseline
            else set()
        )
        fallback = bool(path.get("fixture_fallback"))
        explanation = (
            "Road-graph fixture unavailable in this isolated solver test; direct fallback used."
            if fallback
            else (
                f"Shortest capability-compatible road path avoids {len(avoided)} active closure(s)."
                if avoided
                else "Shortest capability-compatible path on the deterministic road graph."
            )
        )
        return RouteLeg(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            mode="road",
            distance_km=float(path["distance_km"]),
            edge_ids=list(path.get("edge_ids", [])),
            geometry=list(path.get("coordinates", [])),
            blocked_edges_avoided=avoided,
            explanation=explanation,
        )

    def _has_viable_route(
        self,
        village: Village,
        vehicle_type: VehicleType,
    ) -> bool:
        if self._is_aircraft(vehicle_type):
            return True
        if village.id not in self._graph_nodes and not self._graph_nodes:
            return True
        return self._shortest_road_path(
            self.DEPOT_ID,
            village.id,
            vehicle_type,
        ) is not None

    def _assignment_candidate(
        self,
        vehicle: Vehicle,
        village: Village,
        score: UrgencyScore,
        resource_type: str,
        requested_quantity: float,
        remaining_capacity_kg: float,
        existing_stops: List[str],
    ) -> Optional[Dict[str, object]]:
        """Score an asset using response time, impact, specialty, and payload fit.

        The ETA is a depot-to-village estimate used only for asset selection.
        Route construction later recomputes exact multi-stop ETAs.
        """
        leg = self._leg_between(
            self.DEPOT_ID,
            village.id,
            vehicle.vehicle_type,
        )
        if leg is None:
            return None

        projected_stops = list(existing_stops)
        if village.id not in projected_stops:
            projected_stops.append(village.id)
        ordered_stops = self._nearest_neighbour_order(
            projected_stops,
            vehicle.vehicle_type,
        )
        projected_route_minutes = 0.0
        previous_node_id = self.DEPOT_ID
        for stop_id in ordered_stops:
            projected_leg = self._leg_between(
                previous_node_id,
                stop_id,
                vehicle.vehicle_type,
            )
            if projected_leg is None:
                return None
            projected_route_minutes += self.calculate_travel_time(
                projected_leg.distance_km,
                vehicle.vehicle_type,
            )
            previous_node_id = stop_id
        return_leg = self._leg_between(
            previous_node_id,
            self.DEPOT_ID,
            vehicle.vehicle_type,
        )
        if return_leg is None:
            return None
        projected_route_minutes += self.calculate_travel_time(
            return_leg.distance_km,
            vehicle.vehicle_type,
        )
        if projected_route_minutes > vehicle.vehicle_type.fuel_hours * 60.0:
            return None

        resource = self.resource_types.get(resource_type)
        weight_per_unit = resource.weight_per_unit if resource else 1.0
        requested_mass = max(0.0, requested_quantity * weight_per_unit)
        one_way_minutes = self.calculate_travel_time(
            leg.distance_km,
            vehicle.vehicle_type,
        )
        eta_score = 1.0 / (1.0 + one_way_minutes / 120.0)
        payload_fit_score = min(
            1.0,
            remaining_capacity_kg / max(requested_mass, 1.0),
        )
        incident_impact = min(1.0, max(0.0, village.disaster_impact))
        urgency_pressure = min(1.0, score.total_urgency / 15.0)
        external_pressure = min(1.0, score.external_signal)
        resource_is_time_sensitive = bool(
            resource
            and resource.category.value in {
                "medical",
                "safety",
                "communication",
            }
        )
        need = village.resource_needs.get(resource_type)
        survival_shortage = bool(need and need.critical)
        time_pressure = min(
            0.90,
            0.12
            + 0.20 * float(survival_shortage)
            + 0.18 * incident_impact
            + 0.15 * urgency_pressure
            + 0.20 * float(resource_is_time_sensitive)
            + 0.15 * external_pressure,
        )
        specialty_bonus = (
            0.08
            if resource_type in vehicle.vehicle_type.preferred_resources
            else 0.0
        )
        consolidation_bonus = (
            0.08 if village.id in existing_stops else 0.0
        )
        is_aircraft = self._is_aircraft(vehicle.vehicle_type)
        mode_fit_bonus = (
            0.07 * time_pressure
            if is_aircraft
            else 0.07 * (1.0 - time_pressure)
        )
        selection_score = min(
            1.0,
            time_pressure * eta_score
            + (1.0 - time_pressure) * payload_fit_score
            + specialty_bonus
            + consolidation_bonus
            + mode_fit_bonus,
        )
        return {
            "vehicle": vehicle,
            "selection_score": selection_score,
            "time_pressure": time_pressure,
            "eta_score": eta_score,
            "payload_fit_score": payload_fit_score,
            "incident_impact": incident_impact,
            "specialty_bonus": specialty_bonus,
            "consolidation_bonus": consolidation_bonus,
            "estimated_one_way_minutes": one_way_minutes,
            "projected_route_minutes": projected_route_minutes,
            "transport_mode": "air" if is_aircraft else "road",
        }

    # ---------------------------------------------------------------- #
    #  Step 2: Greedy resource assignment                               #
    # ---------------------------------------------------------------- #

    def assign_resources_to_vehicles(
        self,
        urgency_scores: List[UrgencyScore],
        vehicles: List[Vehicle],
        available_resources: Dict[str, float],
    ) -> Dict:
        """
        Greedy urgency-first assignment.

        Returns a dict with:
          vehicle_cargo[vehicle_id]              → {resource_type: amount_kg}
          vehicle_village_assignments[vehicle_id] → [village_id, ...]
          village_allocated[village_id]           → {resource_type: amount_kg}
        """
        # Runtime state (mutable copies — do not mutate the original Vehicle objects)
        vehicle_remaining_kg: Dict[str, float] = {
            v.id: v.vehicle_type.capacity_kg for v in vehicles
        }
        vehicle_cargo: Dict[str, Dict[str, float]] = {v.id: {} for v in vehicles}
        vehicle_cargo_kg: Dict[str, float] = {v.id: 0.0 for v in vehicles}
        vehicle_village_assignments: Dict[str, List[str]] = {v.id: [] for v in vehicles}
        vehicle_deliveries: Dict[str, Dict[str, Dict[str, float]]] = {
            v.id: {} for v in vehicles
        }
        asset_selection: Dict[str, Dict[str, List[Dict[str, object]]]] = {
            s.village_id: {} for s in urgency_scores
        }
        village_allocated: Dict[str, Dict[str, float]] = {
            s.village_id: {} for s in urgency_scores
        }
        depot_stock_at_start: Dict[str, float] = dict(available_resources)
        depot_stock: Dict[str, float] = dict(available_resources)

        # Process villages in urgency order (highest first)
        for score in urgency_scores:
            village = self._village_map.get(score.village_id)
            if village is None:
                continue

            resource_needs = sorted(
                village.resource_needs.items(),
                key=lambda item: (
                    -float(item[1].critical),
                    -(
                        self.resource_types.get(item[0]).urgency_multiplier
                        if self.resource_types.get(item[0])
                        else 1.0
                    ),
                    item[0],
                ),
            )
            for rtype_id, need in resource_needs:
                unmet = need.unmet_need
                if unmet <= 0 or depot_stock.get(rtype_id, 0) <= 0:
                    continue

                # Score every viable asset using ETA, payload fit, incident
                # impact, resource criticality, specialty, and consolidation.
                candidates: List[Dict[str, object]] = []
                for v in vehicles:
                    if vehicle_remaining_kg[v.id] <= 0:
                        continue
                    if not self.check_terrain_accessibility(village, v.vehicle_type):
                        continue
                    existing_stops = vehicle_village_assignments[v.id]
                    if (
                        score.village_id not in existing_stops
                        and len(existing_stops) >= self.MAX_STOPS_PER_ASSET
                    ):
                        continue
                    candidate = self._assignment_candidate(
                        vehicle=v,
                        village=village,
                        score=score,
                        resource_type=rtype_id,
                        requested_quantity=unmet,
                        remaining_capacity_kg=vehicle_remaining_kg[v.id],
                        existing_stops=existing_stops,
                    )
                    if candidate is not None:
                        candidate["stop_count"] = len(existing_stops)
                        candidates.append(candidate)

                candidates.sort(
                    key=lambda item: (
                        -float(item["selection_score"]),
                        int(item["stop_count"]),
                        str(item["vehicle"].id),
                    ),
                )

                remaining_to_fill = unmet
                resource = self.resource_types.get(rtype_id)
                weight_per_unit = resource.weight_per_unit if resource is not None else 1.0
                for candidate in candidates:
                    if remaining_to_fill <= 0:
                        break
                    v = candidate["vehicle"]
                    allocatable = min(
                        vehicle_remaining_kg[v.id] / weight_per_unit,
                        remaining_to_fill,
                        depot_stock.get(rtype_id, 0.0),
                    )
                    if allocatable <= 0:
                        continue

                    # Commit allocation
                    vehicle_cargo[v.id][rtype_id] = (
                        vehicle_cargo[v.id].get(rtype_id, 0.0) + allocatable
                    )
                    allocated_kg = allocatable * weight_per_unit
                    vehicle_cargo_kg[v.id] += allocated_kg
                    vehicle_remaining_kg[v.id] -= allocated_kg
                    depot_stock[rtype_id] = depot_stock.get(rtype_id, 0.0) - allocatable
                    village_allocated[score.village_id][rtype_id] = (
                        village_allocated[score.village_id].get(rtype_id, 0.0) + allocatable
                    )
                    deliveries = vehicle_deliveries[v.id].setdefault(
                        score.village_id,
                        {},
                    )
                    deliveries[rtype_id] = deliveries.get(rtype_id, 0.0) + allocatable
                    selection_entries = asset_selection[
                        score.village_id
                    ].setdefault(rtype_id, [])
                    selection_entries.append({
                        "vehicle_id": v.id,
                        "transport_mode": candidate["transport_mode"],
                        "quantity": allocatable,
                        "payload_kg": allocated_kg,
                        "estimated_one_way_minutes": candidate[
                            "estimated_one_way_minutes"
                        ],
                        "projected_route_minutes": candidate[
                            "projected_route_minutes"
                        ],
                        "selection_score": candidate["selection_score"],
                        "time_pressure": candidate["time_pressure"],
                        "eta_score": candidate["eta_score"],
                        "payload_fit_score": candidate["payload_fit_score"],
                        "incident_impact": candidate["incident_impact"],
                        "specialty_bonus": candidate["specialty_bonus"],
                        "consolidation_bonus": candidate[
                            "consolidation_bonus"
                        ],
                        "explanation": (
                            f"{v.name} selected for {candidate['transport_mode']} "
                            f"delivery: estimated direct ETA "
                            f"{candidate['estimated_one_way_minutes']:.0f} min, "
                            f"projected full tour "
                            f"{candidate['projected_route_minutes']:.0f} min, "
                            f"payload fit {candidate['payload_fit_score']:.2f}, "
                            f"time pressure {candidate['time_pressure']:.2f}, "
                            f"incident impact {candidate['incident_impact']:.2f}."
                        ),
                    })
                    remaining_to_fill -= allocatable

                    if score.village_id not in vehicle_village_assignments[v.id]:
                        vehicle_village_assignments[v.id].append(score.village_id)

        return {
            "vehicle_cargo": vehicle_cargo,
            "vehicle_cargo_kg": vehicle_cargo_kg,
            "vehicle_village_assignments": vehicle_village_assignments,
            "vehicle_deliveries": vehicle_deliveries,
            "asset_selection": asset_selection,
            "village_allocated": village_allocated,
            "depot_stock_at_start": depot_stock_at_start,
            "depot_stock_remaining": depot_stock,
        }

    # ---------------------------------------------------------------- #
    #  Step 3: Route construction                                       #
    # ---------------------------------------------------------------- #

    def _nearest_neighbour_order(
        self,
        village_ids: List[str],
        vehicle_type: VehicleType,
    ) -> List[str]:
        """Order stops by the next feasible multimodal leg, not a straight chord."""
        if not village_ids:
            return []
        current_node_id = self.DEPOT_ID
        remaining = list(village_ids)
        ordered: List[str] = []
        while remaining:
            def leg_distance(village_id: str) -> float:
                leg = self._leg_between(
                    current_node_id,
                    village_id,
                    vehicle_type,
                )
                return leg.distance_km if leg is not None else float("inf")

            nearest = min(
                remaining,
                key=leg_distance,
            )
            ordered.append(nearest)
            remaining.remove(nearest)
            current_node_id = nearest
        return ordered

    def build_routes(self, vehicle_assignments: Dict) -> List[Route]:
        """Construct Route objects for every vehicle that has cargo."""
        vehicle_cargo: Dict[str, Dict] = vehicle_assignments["vehicle_cargo"]
        vehicle_village_assignments: Dict[str, List[str]] = vehicle_assignments[
            "vehicle_village_assignments"
        ]
        vehicle_deliveries: Dict[str, Dict[str, Dict[str, float]]] = (
            vehicle_assignments["vehicle_deliveries"]
        )
        vehicle_cargo_kg: Dict[str, float] = vehicle_assignments["vehicle_cargo_kg"]

        routes: List[Route] = []

        for v_id, cargo in vehicle_cargo.items():
            if not cargo:
                continue  # Vehicle not used

            # Find the Vehicle object
            vehicle = next(
                (v for v in self._vehicles if v.id == v_id), None
            )
            if vehicle is None:
                continue

            assigned_villages = vehicle_village_assignments.get(v_id, [])
            if not assigned_villages:
                continue

            ordered_stops = self._nearest_neighbour_order(
                assigned_villages,
                vehicle.vehicle_type,
            )

            stop_details: List[RouteStop] = []
            legs: List[RouteLeg] = []
            path_coordinates: List[Tuple[float, float]] = []
            road_edge_ids: List[str] = []
            rerouted_due_to: Set[str] = set()
            cumulative_dist = 0.0
            cumulative_time = 0.0
            previous_node_id = self._origin_node_for(vehicle)
            infeasibility_reason: Optional[str] = None

            def append_leg(leg: RouteLeg) -> None:
                nonlocal cumulative_dist, cumulative_time
                legs.append(leg)
                cumulative_dist += leg.distance_km
                cumulative_time += self.calculate_travel_time(
                    leg.distance_km,
                    vehicle.vehicle_type,
                )
                road_edge_ids.extend(leg.edge_ids)
                rerouted_due_to.update(leg.blocked_edges_avoided)
                if path_coordinates and leg.geometry and path_coordinates[-1] == leg.geometry[0]:
                    path_coordinates.extend(leg.geometry[1:])
                else:
                    path_coordinates.extend(leg.geometry)

            for vid in ordered_stops:
                village = self._village_map[vid]
                leg = self._leg_between(
                    previous_node_id,
                    vid,
                    vehicle.vehicle_type,
                )
                if leg is None:
                    infeasibility_reason = (
                        f"No capability-compatible path from {previous_node_id} to {vid} "
                        f"after applying closures: {', '.join(sorted(self.blocked_edge_ids)) or 'none'}"
                    )
                    break
                append_leg(leg)

                stop_details.append(
                    RouteStop(
                        village_id=vid,
                        lat=village.lat,
                        lng=village.lng,
                        distance_from_prev_km=leg.distance_km,
                        cumulative_distance_km=cumulative_dist,
                        eta_minutes=cumulative_time,
                        cargo_delivered=vehicle_deliveries.get(v_id, {}).get(vid, {}),
                    )
                )
                previous_node_id = vid

            if infeasibility_reason is None:
                return_leg = self._leg_between(
                    previous_node_id,
                    self.DEPOT_ID,
                    vehicle.vehicle_type,
                )
                if return_leg is None:
                    infeasibility_reason = (
                        f"No capability-compatible return path from {previous_node_id} "
                        "to the depot."
                    )
                else:
                    append_leg(return_leg)

            max_time_minutes = vehicle.vehicle_type.fuel_hours * 60.0
            if infeasibility_reason is None and cumulative_time > max_time_minutes:
                infeasibility_reason = (
                f"Route time {cumulative_time:.0f} min > fuel limit {max_time_minutes:.0f} min"
                )
            feasible = infeasibility_reason is None
            transport_mode = (
                "air" if self._is_aircraft(vehicle.vehicle_type) else "road"
            )
            routing_explanation = (
                "Direct aircraft corridor selected because the asset is not road-constrained."
                if transport_mode == "air"
                else (
                    f"Road graph rerouted around {len(rerouted_due_to)} active closure(s); "
                    "every truck waypoint follows capability-compatible graph edges."
                    if rerouted_due_to
                    else "Every truck waypoint follows the shortest capability-compatible road-graph path."
                )
            )

            routes.append(
                Route(
                    vehicle_id=v_id,
                    stops=ordered_stops,
                    stop_details=stop_details,
                    total_distance_km=cumulative_dist,
                    total_time_minutes=cumulative_time,
                    cargo_manifest=cargo,
                    total_cargo_kg=vehicle_cargo_kg.get(v_id, 0.0),
                    transport_mode=transport_mode,
                    path_coordinates=path_coordinates,
                    legs=legs,
                    road_edge_ids=road_edge_ids,
                    rerouted_due_to=sorted(rerouted_due_to),
                    routing_explanation=routing_explanation,
                    feasible=feasible,
                    infeasibility_reason=infeasibility_reason,
                )
            )

        return routes

    # ---------------------------------------------------------------- #
    #  Step 4: Solution assembly                                        #
    # ---------------------------------------------------------------- #

    def _build_allocations(
        self,
        vehicle_assignments: Dict,
        villages: List[Village],
        routes: List[Route],
    ) -> List[VillageAllocation]:
        """Create per-village allocation summaries."""
        village_allocated: Dict[str, Dict] = vehicle_assignments["village_allocated"]
        vehicle_village_assignments: Dict[str, List] = vehicle_assignments[
            "vehicle_village_assignments"
        ]
        vehicle_deliveries: Dict[str, Dict[str, Dict[str, float]]] = (
            vehicle_assignments["vehicle_deliveries"]
        )
        depot_stock_at_start: Dict[str, float] = vehicle_assignments[
            "depot_stock_at_start"
        ]
        depot_stock_remaining: Dict[str, float] = vehicle_assignments[
            "depot_stock_remaining"
        ]
        asset_selection: Dict[str, Dict[str, List[Dict[str, object]]]] = (
            vehicle_assignments.get("asset_selection", {})
        )

        # Map: village_id → list of vehicle_ids serving it
        village_vehicles: Dict[str, List[str]] = {v.id: [] for v in villages}
        for v_id, vids in vehicle_village_assignments.items():
            for vid in vids:
                if vid in village_vehicles:
                    village_vehicles[vid].append(v_id)

        # Map: village_id → earliest ETA across all serving routes
        village_eta: Dict[str, float] = {}
        for route in routes:
            for stop in route.stop_details:
                if stop.village_id not in village_eta or stop.eta_minutes < village_eta[stop.village_id]:
                    village_eta[stop.village_id] = stop.eta_minutes

        allocations: List[VillageAllocation] = []
        for village in villages:
            allocated = village_allocated.get(village.id, {})
            score = self._urgency_by_id.get(village.id)
            resource_decisions: List[ResourceAllocationDecision] = []
            for rtype_id, need in village.resource_needs.items():
                proposed = allocated.get(rtype_id, 0.0)
                post_plan = need.allocated + proposed
                unmet_after = max(0.0, need.current_need - post_plan)
                survival_gap = max(0.0, need.min_need - post_plan)
                assigned_vehicle_ids = [
                    vehicle_id
                    for vehicle_id, deliveries_by_village in vehicle_deliveries.items()
                    if deliveries_by_village.get(village.id, {}).get(rtype_id, 0.0) > 0
                ]
                selection_reasons = [
                    AssetSelectionReason(**item)
                    for item in asset_selection
                    .get(village.id, {})
                    .get(rtype_id, [])
                ]
                reason_codes = [f"urgency_rank_{score.ranking if score else 'unranked'}"]
                if need.critical:
                    reason_codes.append("below_survival_threshold")
                if proposed >= need.unmet_need and need.unmet_need > 0:
                    reason_codes.append("unmet_need_covered")
                elif depot_stock_remaining.get(rtype_id, 0.0) <= 1e-9:
                    reason_codes.append("depot_stock_exhausted")
                elif proposed < need.unmet_need:
                    reason_codes.append("compatible_fleet_capacity_exhausted")
                if not assigned_vehicle_ids:
                    reason_codes.append("no_vehicle_assignment")
                if any(
                    item.transport_mode == "air"
                    for item in selection_reasons
                ):
                    reason_codes.append("rapid_air_response")
                if any(
                    item.transport_mode == "road"
                    for item in selection_reasons
                ):
                    reason_codes.append("bulk_ground_lift")

                if proposed <= 0:
                    explanation = (
                        f"No new {rtype_id.replace('_', ' ')} was assigned. "
                        f"The village is urgency rank {score.ranking if score else 'unranked'}; "
                        "the limiting condition is recorded in the reason codes."
                    )
                else:
                    selected_assets = "; ".join(
                        (
                            f"{item.vehicle_id} via {item.transport_mode} "
                            f"(direct ETA {item.estimated_one_way_minutes:.0f} min, "
                            f"tour {item.projected_route_minutes:.0f} min, "
                            f"selection {item.selection_score:.2f})"
                        )
                        for item in selection_reasons
                    )
                    explanation = (
                        f"Proposed {proposed:.1f} {self.resource_types.get(rtype_id).unit if self.resource_types.get(rtype_id) else 'units'} "
                        f"because {village.name} is urgency rank {score.ranking if score else 'unranked'}"
                        f"{' and was below its survival threshold' if need.critical else ''}. "
                        f"Loaded onto {', '.join(assigned_vehicle_ids)} after scoring "
                        "response time, payload fit, incident impact, stock, terrain, "
                        f"and route reachability. {selected_assets}"
                    )
                resource = self.resource_types.get(rtype_id)
                resource_decisions.append(
                    ResourceAllocationDecision(
                        resource_type=rtype_id,
                        unit=resource.unit if resource is not None else "unit",
                        current_need=need.current_need,
                        existing_allocated=need.allocated,
                        survival_threshold=need.min_need,
                        unmet_before_plan=need.unmet_need,
                        depot_available_at_start=depot_stock_at_start.get(rtype_id, 0.0),
                        proposed_now=proposed,
                        post_plan_total=post_plan,
                        unmet_after_plan=unmet_after,
                        survival_gap_after_plan=survival_gap,
                        assigned_vehicle_ids=assigned_vehicle_ids,
                        asset_selection=selection_reasons,
                        reason_codes=reason_codes,
                        explanation=explanation,
                    )
                )
            satisfied = all(
                decision.survival_gap_after_plan <= 1e-9
                for decision in resource_decisions
            )

            allocations.append(
                VillageAllocation(
                    village_id=village.id,
                    allocated_resources=allocated,
                    vehicle_assignments=village_vehicles.get(village.id, []),
                    eta_minutes=village_eta.get(village.id, 0.0),
                    satisfied=satisfied,
                    resource_decisions=resource_decisions,
                    allocation_explanation=(
                        f"Urgency-first deterministic allocation at rank "
                        f"{score.ranking if score else 'unranked'}; "
                        f"{len(village_vehicles.get(village.id, []))} compatible asset(s) assigned."
                    ),
                )
            )

        return allocations

    @staticmethod
    def _calculate_objective(
        allocations: List[VillageAllocation],
        villages: List[Village],
    ) -> float:
        """
        Simple satisfaction ratio objective:
          Σ (allocated / current_need) for each village × resource type.
        Score of 1.0 per resource = perfectly met.
        """
        total = 0.0
        count = 0
        village_map = {v.id: v for v in villages}
        for alloc in allocations:
            village = village_map.get(alloc.village_id)
            if village is None:
                continue
            for rtype_id, need in village.resource_needs.items():
                if need.current_need > 0:
                    ratio = min(
                        1.0,
                        (
                            need.allocated
                            + alloc.allocated_resources.get(rtype_id, 0.0)
                        ) / need.current_need,
                    )
                    total += ratio
                    count += 1
        return total / count if count > 0 else 0.0

    # ---------------------------------------------------------------- #
    #  Public API                                                       #
    # ---------------------------------------------------------------- #

    def solve(
        self,
        villages: List[Village],
        vehicles: List[Vehicle],
        urgency_scores: List[UrgencyScore],
        available_resources: Dict[str, float],
    ) -> VRPSolution:
        """
        Full VRP solve.

        Args:
            villages:            All villages to serve.
            vehicles:            Available fleet.
            urgency_scores:      Ranked list from UrgencyCalculator (highest urgency first).
            available_resources: Depot stock: {resource_type: amount_kg}.

        Returns:
            VRPSolution with routes, allocations, and metrics.
        """
        self._village_map = {v.id: v for v in villages}
        self._vehicles = vehicles

        # Handle urgency scores that may not cover all villages
        scored_ids = {s.village_id for s in urgency_scores}
        for village in villages:
            if village.id not in scored_ids:
                urgency_scores = list(urgency_scores) + [
                    UrgencyScore(
                        village_id=village.id,
                        total_urgency=0.0,
                        ranking=len(urgency_scores) + 1,
                    )
                ]
        self._urgency_by_id = {
            score.village_id: score for score in urgency_scores
        }

        # Step 2: Assign resources
        assignment = self.assign_resources_to_vehicles(
            urgency_scores, vehicles, available_resources
        )

        # Step 3: Build routes
        routes = self.build_routes(assignment)

        # Step 4: Assemble solution
        allocations = self._build_allocations(assignment, villages, routes)
        objective = self._calculate_objective(allocations, villages)
        unmet = [a.village_id for a in allocations if not a.satisfied]
        total_dist = sum(r.total_distance_km for r in routes)

        return VRPSolution(
            routes=routes,
            allocations=allocations,
            total_distance_km=total_dist,
            unmet_villages=unmet,
            convergence_iterations=1,
            objective_value=objective,
            active_road_blocks=sorted(self.blocked_edge_ids),
            road_network=[
                {
                    "edge_id": edge["id"],
                    "from_node_id": edge["from"],
                    "to_node_id": edge["to"],
                    "name": edge.get(
                        "name",
                        f"{edge['from'].replace('_', ' ').title()} - "
                        f"{edge['to'].replace('_', ' ').title()}",
                    ),
                    "distance_km": edge["distance_km"],
                    "road_quality": edge.get("road_quality", "unknown"),
                    "terrain_difficulty": edge.get("terrain_difficulty", 1.0),
                    "vulnerable_to_landslide": edge.get(
                        "vulnerable_to_landslide",
                        False,
                    ),
                    "status": (
                        "blocked"
                        if edge["id"] in self.blocked_edge_ids
                        else "open"
                    ),
                    "geometry": self._edge_geometry(edge, edge["from"]),
                    "source_kind": "bundled_scenario_fixture",
                }
                for edge in self._graph_edges
            ],
        )
