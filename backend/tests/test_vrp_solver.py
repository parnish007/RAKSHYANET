"""
Tests for VRPSolver — Prompt 2.2 verification.
Run: pytest backend/tests/test_vrp_solver.py -v
"""
import pytest
from typing import Dict, List

from backend.models.resource import ResourceCategory, ResourceType, VillageResourceNeed
from backend.models.vehicle import (
    Vehicle, VehicleType, VehicleCategory, TerrainCapability, VehicleState,
)
from backend.models.village import Village
from backend.algorithms.urgency_calculator import UrgencyScore
from backend.algorithms.vrp_solver import (
    VRPSolver, Route, VillageAllocation, VRPSolution, TERRAIN_ACCESS,
)

DEPOT = (27.7172, 85.3240)


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture(scope="module")
def resource_types() -> Dict[str, ResourceType]:
    return {
        "food":        ResourceType(resource_id="food",        name="Food",        category=ResourceCategory.FOOD,    urgency_multiplier=1.5, weight_per_unit=1.0),
        "water":       ResourceType(resource_id="water",       name="Water",       category=ResourceCategory.WATER,   urgency_multiplier=1.8, weight_per_unit=1.0),
        "medical_kit": ResourceType(resource_id="medical_kit", name="Medical Kit", category=ResourceCategory.MEDICAL, urgency_multiplier=2.0, weight_per_unit=5.0),
        "tarpaulin":   ResourceType(resource_id="tarpaulin",   name="Tarpaulin",   category=ResourceCategory.SHELTER, urgency_multiplier=1.2, weight_per_unit=3.0),
        "blanket":     ResourceType(resource_id="blanket",     name="Blanket",     category=ResourceCategory.SHELTER, urgency_multiplier=1.0, weight_per_unit=2.0),
        "first_aid":   ResourceType(resource_id="first_aid",   name="First Aid",   category=ResourceCategory.MEDICAL, urgency_multiplier=1.7, weight_per_unit=1.0),
    }


def make_vehicle_type(type_id, category, capacity, speed, fuel_hours, terrain, prefs=None):
    return VehicleType(
        type_id=type_id, name=type_id,
        category=category, capacity_kg=capacity,
        speed_kmh=speed, fuel_hours=fuel_hours,
        terrain_capability=terrain,
        preferred_resources=prefs or [],
    )


@pytest.fixture(scope="module")
def vehicles() -> List[Vehicle]:
    heli_type  = make_vehicle_type("helicopter",  VehicleCategory.AIRCRAFT,     500,  200, 2.0, TerrainCapability.ANY,         ["medical_kit", "first_aid"])
    truck_type = make_vehicle_type("heavy_truck",  VehicleCategory.GROUND_HEAVY, 2000, 40,  8.0, TerrainCapability.PAVED_ROADS, ["food", "water", "tarpaulin", "blanket"])
    van_type   = make_vehicle_type("van_4x4",      VehicleCategory.GROUND_LIGHT, 800,  60,  8.0, TerrainCapability.ALL_ROADS,   [])
    moto_type  = make_vehicle_type("motorcycle",   VehicleCategory.GROUND_LIGHT, 50,   80,  4.0, TerrainCapability.DIRT_PATHS,  ["medical_kit", "first_aid"])

    return [
        Vehicle(id="heli_1", name="Heli 1",  vehicle_type=heli_type,  current_location=DEPOT),
        Vehicle(id="heli_2", name="Heli 2",  vehicle_type=heli_type,  current_location=DEPOT),
        Vehicle(id="truck_1",name="Truck 1", vehicle_type=truck_type, current_location=DEPOT),
        Vehicle(id="van_1",  name="Van 1",   vehicle_type=van_type,   current_location=DEPOT),
        Vehicle(id="moto_1", name="Moto 1",  vehicle_type=moto_type,  current_location=DEPOT),
    ]


@pytest.fixture(scope="module")
def village_road() -> Village:
    """Paved-road-accessible village."""
    return Village(
        id="banepa", name="Banepa",
        lat=27.6317, lng=85.5206, population=6500,
        accessibility="road",
        resource_needs={
            "food":        VillageResourceNeed(resource_type="food",        current_need=3250, min_need=1950, allocated=0),
            "water":       VillageResourceNeed(resource_type="water",       current_need=1950, min_need=1365, allocated=0),
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=65,   min_need=39,   allocated=0),
            "tarpaulin":   VillageResourceNeed(resource_type="tarpaulin",   current_need=260,  min_need=130,  allocated=0),
            "blanket":     VillageResourceNeed(resource_type="blanket",     current_need=390,  min_need=195,  allocated=0),
            "first_aid":   VillageResourceNeed(resource_type="first_aid",   current_need=104,  min_need=52,   allocated=0),
        },
    )


@pytest.fixture(scope="module")
def village_dirt_road() -> Village:
    """Dirt-road-only village."""
    return Village(
        id="namobuddha", name="Namobuddha",
        lat=27.5833, lng=85.5833, population=2800,
        accessibility="dirt_road",
        resource_needs={
            "food":  VillageResourceNeed(resource_type="food",  current_need=1400, min_need=840, allocated=0),
            "water": VillageResourceNeed(resource_type="water", current_need=840,  min_need=588, allocated=0),
        },
    )


@pytest.fixture(scope="module")
def village_helicopter_only() -> Village:
    """Helicopter-only village."""
    return Village(
        id="remote_peak", name="Remote Peak",
        lat=27.55, lng=85.65, population=800,
        accessibility="helicopter_only",
        resource_needs={
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=20, min_need=10, allocated=0),
        },
    )


@pytest.fixture(scope="module")
def available_resources() -> Dict[str, float]:
    return {
        "food": 5000.0,
        "water": 3000.0,
        "medical_kit": 200.0,
        "tarpaulin": 800.0,
        "blanket": 1200.0,
        "first_aid": 300.0,
    }


@pytest.fixture(scope="module")
def solver(resource_types) -> VRPSolver:
    return VRPSolver(
        depot_location=DEPOT,
        terrain_graph={},
        resource_types=resource_types,
        config={},
    )


def make_scores(*village_ids) -> List[UrgencyScore]:
    return [
        UrgencyScore(village_id=vid, total_urgency=10.0 - i, ranking=i + 1)
        for i, vid in enumerate(village_ids)
    ]


# ================================================================== #
#  Distance calculation tests                                          #
# ================================================================== #

class TestDistanceCalculation:
    def test_kathmandu_to_dhulikhel_approx_25km(self, solver):
        dist = solver.calculate_distance(DEPOT, (27.62, 85.55))
        assert 20.0 < dist < 35.0, f"Expected ~25 km, got {dist:.2f}"

    def test_distance_is_symmetric(self, solver):
        a = (27.72, 85.32)
        b = (27.62, 85.55)
        assert solver.calculate_distance(a, b) == pytest.approx(
            solver.calculate_distance(b, a), rel=1e-9
        )

    def test_same_point_distance_is_zero(self, solver):
        assert solver.calculate_distance(DEPOT, DEPOT) == pytest.approx(0.0, abs=1e-6)

    def test_distance_positive_for_distinct_points(self, solver):
        assert solver.calculate_distance((27.72, 85.32), (27.58, 85.52)) > 0


# ================================================================== #
#  Travel time tests                                                   #
# ================================================================== #

class TestTravelTime:
    def test_helicopter_faster_than_truck(self, solver, vehicles):
        heli  = next(v for v in vehicles if v.vehicle_type.type_id == "helicopter")
        truck = next(v for v in vehicles if v.vehicle_type.type_id == "heavy_truck")
        dist = 30.0
        assert (
            solver.calculate_travel_time(dist, heli.vehicle_type)
            < solver.calculate_travel_time(dist, truck.vehicle_type)
        )

    def test_travel_time_formula(self, solver, vehicles):
        truck = next(v for v in vehicles if v.vehicle_type.type_id == "heavy_truck")
        # 40 km at 40 km/h = 60 minutes
        t = solver.calculate_travel_time(40.0, truck.vehicle_type)
        assert t == pytest.approx(60.0)

    def test_zero_distance_zero_time(self, solver, vehicles):
        heli = next(v for v in vehicles if v.vehicle_type.type_id == "helicopter")
        assert solver.calculate_travel_time(0.0, heli.vehicle_type) == pytest.approx(0.0)


# ================================================================== #
#  Terrain accessibility tests                                         #
# ================================================================== #

class TestTerrainAccessibility:
    def test_helicopter_reaches_road_village(self, solver, vehicles, village_road):
        heli = next(v for v in vehicles if v.vehicle_type.type_id == "helicopter")
        assert solver.check_terrain_accessibility(village_road, heli.vehicle_type) is True

    def test_helicopter_reaches_dirt_road_village(self, solver, vehicles, village_dirt_road):
        heli = next(v for v in vehicles if v.vehicle_type.type_id == "helicopter")
        assert solver.check_terrain_accessibility(village_dirt_road, heli.vehicle_type) is True

    def test_helicopter_reaches_helicopter_only_village(self, solver, vehicles, village_helicopter_only):
        heli = next(v for v in vehicles if v.vehicle_type.type_id == "helicopter")
        assert solver.check_terrain_accessibility(village_helicopter_only, heli.vehicle_type) is True

    def test_heavy_truck_reaches_road_village(self, solver, vehicles, village_road):
        truck = next(v for v in vehicles if v.vehicle_type.type_id == "heavy_truck")
        assert solver.check_terrain_accessibility(village_road, truck.vehicle_type) is True

    def test_heavy_truck_cannot_reach_dirt_road(self, solver, vehicles, village_dirt_road):
        truck = next(v for v in vehicles if v.vehicle_type.type_id == "heavy_truck")
        assert solver.check_terrain_accessibility(village_dirt_road, truck.vehicle_type) is False

    def test_heavy_truck_cannot_reach_helicopter_only(self, solver, vehicles, village_helicopter_only):
        truck = next(v for v in vehicles if v.vehicle_type.type_id == "heavy_truck")
        assert solver.check_terrain_accessibility(village_helicopter_only, truck.vehicle_type) is False

    def test_van_reaches_road_and_dirt_road(self, solver, vehicles, village_road, village_dirt_road):
        van = next(v for v in vehicles if v.vehicle_type.type_id == "van_4x4")
        assert solver.check_terrain_accessibility(village_road, van.vehicle_type) is True
        assert solver.check_terrain_accessibility(village_dirt_road, van.vehicle_type) is True

    def test_van_cannot_reach_helicopter_only(self, solver, vehicles, village_helicopter_only):
        van = next(v for v in vehicles if v.vehicle_type.type_id == "van_4x4")
        assert solver.check_terrain_accessibility(village_helicopter_only, van.vehicle_type) is False

    def test_motorcycle_reaches_all_land_types(self, solver, vehicles, village_road, village_dirt_road, village_helicopter_only):
        moto = next(v for v in vehicles if v.vehicle_type.type_id == "motorcycle")
        assert solver.check_terrain_accessibility(village_road, moto.vehicle_type) is True
        assert solver.check_terrain_accessibility(village_dirt_road, moto.vehicle_type) is True
        assert solver.check_terrain_accessibility(village_helicopter_only, moto.vehicle_type) is True


# ================================================================== #
#  Resource preference tests                                           #
# ================================================================== #

class TestResourcePreferences:
    def test_helicopter_prefers_medical(self, solver, vehicles):
        heli = next(v for v in vehicles if v.vehicle_type.type_id == "helicopter")
        assert solver.can_vehicle_carry(heli, "medical_kit") is True
        assert solver.can_vehicle_carry(heli, "first_aid") is True

    def test_helicopter_does_not_prefer_food(self, solver, vehicles):
        heli = next(v for v in vehicles if v.vehicle_type.type_id == "helicopter")
        # helicopter has preferred_resources set → food is not listed
        assert solver.can_vehicle_carry(heli, "food") is False

    def test_truck_prefers_food_and_water(self, solver, vehicles):
        truck = next(v for v in vehicles if v.vehicle_type.type_id == "heavy_truck")
        assert solver.can_vehicle_carry(truck, "food") is True
        assert solver.can_vehicle_carry(truck, "water") is True

    def test_van_accepts_any_resource(self, solver, vehicles):
        van = next(v for v in vehicles if v.vehicle_type.type_id == "van_4x4")
        assert solver.can_vehicle_carry(van, "food") is True
        assert solver.can_vehicle_carry(van, "medical_kit") is True
        assert solver.can_vehicle_carry(van, "blanket") is True


# ================================================================== #
#  Greedy assignment tests                                             #
# ================================================================== #

class TestResourceAssignment:
    def test_assignment_returns_required_keys(self, solver, vehicles, village_road, available_resources):
        solver._village_map = {village_road.id: village_road}
        solver._vehicles = vehicles
        scores = make_scores(village_road.id)
        result = solver.assign_resources_to_vehicles(scores, vehicles, available_resources)
        assert "vehicle_cargo" in result
        assert "vehicle_village_assignments" in result
        assert "village_allocated" in result

    def test_village_receives_some_allocation(self, solver, vehicles, village_road, available_resources):
        solver._village_map = {village_road.id: village_road}
        solver._vehicles = vehicles
        scores = make_scores(village_road.id)
        result = solver.assign_resources_to_vehicles(scores, vehicles, available_resources)
        allocated = result["village_allocated"][village_road.id]
        assert sum(allocated.values()) > 0

    def test_vehicle_capacity_not_exceeded(self, solver, vehicles, village_road, available_resources):
        solver._village_map = {village_road.id: village_road}
        solver._vehicles = vehicles
        scores = make_scores(village_road.id)
        result = solver.assign_resources_to_vehicles(scores, vehicles, available_resources)
        for v in vehicles:
            total_loaded = sum(result["vehicle_cargo"][v.id].values())
            assert total_loaded <= v.vehicle_type.capacity_kg, (
                f"{v.id} overloaded: {total_loaded} > {v.vehicle_type.capacity_kg}"
            )

    def test_helicopter_only_village_served_by_helicopter(
        self, solver, vehicles, village_helicopter_only, available_resources
    ):
        solver._village_map = {village_helicopter_only.id: village_helicopter_only}
        solver._vehicles = vehicles
        scores = make_scores(village_helicopter_only.id)
        result = solver.assign_resources_to_vehicles(scores, vehicles, available_resources)
        # Only helicopters and motorcycles can reach helicopter_only
        aircraft_ids = {v.id for v in vehicles if v.vehicle_type.terrain_capability == TerrainCapability.ANY}
        moto_ids = {v.id for v in vehicles if v.vehicle_type.terrain_capability == TerrainCapability.DIRT_PATHS}
        capable_ids = aircraft_ids | moto_ids
        assigned = [
            v_id for v_id, vils in result["vehicle_village_assignments"].items()
            if village_helicopter_only.id in vils
        ]
        for v_id in assigned:
            assert v_id in capable_ids, f"{v_id} should not serve helicopter_only village"

    def test_higher_urgency_village_served_first(
        self, solver, vehicles, village_road, village_dirt_road, available_resources
    ):
        """High-urgency village should be allocated first (gets more when resources tight)."""
        solver._village_map = {
            village_road.id: village_road,
            village_dirt_road.id: village_dirt_road,
        }
        solver._vehicles = vehicles
        # village_road ranked first (higher urgency)
        scores = make_scores(village_road.id, village_dirt_road.id)
        result = solver.assign_resources_to_vehicles(
            scores, vehicles, {k: 100.0 for k in available_resources}  # tight supply
        )
        road_alloc = sum(result["village_allocated"][village_road.id].values())
        dirt_alloc = sum(result["village_allocated"][village_dirt_road.id].values())
        assert road_alloc >= dirt_alloc

    def test_medical_payload_prefers_fast_specialized_aircraft(
        self, solver, vehicles, village_road, available_resources
    ):
        solver._village_map = {village_road.id: village_road}
        solver._vehicles = vehicles
        result = solver.assign_resources_to_vehicles(
            make_scores(village_road.id),
            vehicles,
            {"medical_kit": available_resources["medical_kit"]},
        )
        selected = result["asset_selection"][village_road.id]["medical_kit"][0]
        assert selected["transport_mode"] == "air"
        assert selected["time_pressure"] > 0.5
        assert selected["estimated_one_way_minutes"] > 0

    def test_bulk_food_payload_prefers_ground_capacity(
        self, solver, vehicles, village_road, available_resources
    ):
        solver._village_map = {village_road.id: village_road}
        solver._vehicles = vehicles
        result = solver.assign_resources_to_vehicles(
            make_scores(village_road.id),
            vehicles,
            {"food": available_resources["food"]},
        )
        selected = result["asset_selection"][village_road.id]["food"][0]
        assert selected["transport_mode"] == "road"
        assert selected["payload_fit_score"] > 0.5
        assert selected["selection_score"] > 0


# ================================================================== #
#  Route construction tests                                            #
# ================================================================== #

class TestRouteConstruction:
    def _get_routes(self, solver, vehicles, villages, scores, resources):
        solver._village_map = {v.id: v for v in villages}
        solver._vehicles = vehicles
        assignment = solver.assign_resources_to_vehicles(scores, vehicles, resources)
        return solver.build_routes(assignment), assignment

    def test_routes_have_assigned_villages(self, solver, vehicles, village_road, available_resources):
        routes, _ = self._get_routes(
            solver, vehicles, [village_road],
            make_scores(village_road.id), available_resources
        )
        all_stops = [stop for r in routes for stop in r.stops]
        assert village_road.id in all_stops

    def test_route_distance_positive(self, solver, vehicles, village_road, available_resources):
        routes, _ = self._get_routes(
            solver, vehicles, [village_road],
            make_scores(village_road.id), available_resources
        )
        for r in routes:
            assert r.total_distance_km > 0, f"{r.vehicle_id} route has zero distance"

    def test_route_time_consistent_with_distance_and_speed(self, solver, vehicles, village_road, available_resources):
        routes, _ = self._get_routes(
            solver, vehicles, [village_road],
            make_scores(village_road.id), available_resources
        )
        for r in routes:
            v = next(veh for veh in vehicles if veh.id == r.vehicle_id)
            expected_min_time = (r.total_distance_km / v.vehicle_type.speed_kmh) * 60
            assert r.total_time_minutes == pytest.approx(expected_min_time, rel=0.01)

    def test_short_route_is_feasible(self, solver, vehicles, village_road, available_resources):
        routes, _ = self._get_routes(
            solver, vehicles, [village_road],
            make_scores(village_road.id), available_resources
        )
        # Depot to Banepa (~26km) is well within fuel range of all vehicles
        for r in routes:
            assert r.feasible is True, f"{r.vehicle_id} unexpectedly infeasible"

    def test_stop_details_include_eta(self, solver, vehicles, village_road, available_resources):
        routes, _ = self._get_routes(
            solver, vehicles, [village_road],
            make_scores(village_road.id), available_resources
        )
        for r in routes:
            for stop in r.stop_details:
                assert stop.eta_minutes > 0


# ================================================================== #
#  Full solve() integration tests                                      #
# ================================================================== #

class TestSolve:
    def test_solve_returns_vrpsolution(self, solver, vehicles, village_road, village_dirt_road, available_resources):
        solution = solver.solve(
            villages=[village_road, village_dirt_road],
            vehicles=vehicles,
            urgency_scores=make_scores(village_road.id, village_dirt_road.id),
            available_resources=available_resources,
        )
        assert isinstance(solution, VRPSolution)

    def test_solve_allocations_cover_all_villages(self, solver, vehicles, village_road, village_dirt_road, available_resources):
        solution = solver.solve(
            villages=[village_road, village_dirt_road],
            vehicles=vehicles,
            urgency_scores=make_scores(village_road.id, village_dirt_road.id),
            available_resources=available_resources,
        )
        allocation_ids = {a.village_id for a in solution.allocations}
        assert village_road.id in allocation_ids
        assert village_dirt_road.id in allocation_ids

    def test_solve_objective_between_0_and_1(self, solver, vehicles, village_road, available_resources):
        solution = solver.solve(
            villages=[village_road],
            vehicles=vehicles,
            urgency_scores=make_scores(village_road.id),
            available_resources=available_resources,
        )
        assert 0.0 <= solution.objective_value <= 1.0

    def test_solve_with_no_resources_all_unmet(self, solver, vehicles, village_road):
        solution = solver.solve(
            villages=[village_road],
            vehicles=vehicles,
            urgency_scores=make_scores(village_road.id),
            available_resources={},
        )
        assert village_road.id in solution.unmet_villages

    def test_solve_total_distance_positive(self, solver, vehicles, village_road, available_resources):
        solution = solver.solve(
            villages=[village_road],
            vehicles=vehicles,
            urgency_scores=make_scores(village_road.id),
            available_resources=available_resources,
        )
        assert solution.total_distance_km > 0

    def test_solve_convergence_iterations_is_one(self, solver, vehicles, village_road, available_resources):
        solution = solver.solve(
            villages=[village_road],
            vehicles=vehicles,
            urgency_scores=make_scores(village_road.id),
            available_resources=available_resources,
        )
        assert solution.convergence_iterations == 1

    def test_helicopter_only_village_handled(
        self, solver, vehicles, village_helicopter_only, available_resources
    ):
        solution = solver.solve(
            villages=[village_helicopter_only],
            vehicles=vehicles,
            urgency_scores=make_scores(village_helicopter_only.id),
            available_resources=available_resources,
        )
        alloc = next(a for a in solution.allocations if a.village_id == village_helicopter_only.id)
        assert sum(alloc.allocated_resources.values()) > 0
