#!/usr/bin/env python3
"""
Smoke test: VRP Solver -- Prompt 2.2
Loads 8 villages from nepal_villages.json, builds fleet from config.json,
ranks villages by urgency at t=0, then runs the greedy VRP solver.

Run from project root:
    python demo/test_vrp.py
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.resource import ResourceType, VillageResourceNeed
from backend.models.vehicle import Vehicle, VehicleType
from backend.models.village import Village
from backend.algorithms.urgency_calculator import UrgencyCalculator
from backend.algorithms.vrp_solver import VRPSolver, VRPSolution

SEP = "-" * 65


# ------------------------------------------------------------------ #
#  Loaders                                                             #
# ------------------------------------------------------------------ #

def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_resource_types(cfg: dict) -> dict:
    return {
        rid: ResourceType(**rdata)
        for rid, rdata in cfg["resource_types"].items()
    }


def build_fleet(cfg: dict) -> list:
    """Instantiate Vehicle objects from config fleet_composition."""
    vehicle_type_defs = cfg["vehicle_types"]
    composition = cfg["fleet_composition"]
    vehicles = []
    for type_id, count in composition.items():
        vt_data = vehicle_type_defs[type_id]
        vt = VehicleType(**vt_data)
        for i in range(1, count + 1):
            vehicles.append(
                Vehicle(
                    id=f"{type_id}_{i}",
                    name=f"{vt.name} #{i}",
                    vehicle_type=vt,
                )
            )
    return vehicles


def build_villages(data: dict) -> list:
    """Parse nepal_villages.json -> list of Village objects."""
    villages = []
    for v in data["villages"]:
        resource_needs = {
            rtype: VillageResourceNeed(**need_data)
            for rtype, need_data in v["resource_needs"].items()
        }
        villages.append(
            Village(
                id=v["id"],
                name=v["name"],
                lat=v["lat"],
                lng=v["lng"],
                population=v["population"],
                accessibility=v.get("accessibility", "road"),
                has_medical_facility=v.get("has_medical_facility", False),
                resource_needs=resource_needs,
            )
        )
    return villages


# ------------------------------------------------------------------ #
#  Display helpers (ASCII-only for Windows console compatibility)      #
# ------------------------------------------------------------------ #

def bar(ratio: float, width: int = 20) -> str:
    filled = int(round(ratio * width))
    return "#" * filled + "." * (width - filled)


def print_urgency_table(scores):
    print("\n" + SEP)
    print("  URGENCY RANKINGS (t = 0 hr)")
    print(SEP)
    print(f"  {'#':<3} {'Village':<15} {'Urgency':>8}  {'Critical':<9} Top Resource")
    print(f"  {'--':<3} {'-'*14:<15} {'-------':>8}  {'--------':<9} ------------")
    for s in scores:
        top = s.top_resource() or "-"
        crit = "YES (!)" if s.has_critical_shortage else "no"
        print(f"  #{s.ranking:<2} {s.village_id:<15} {s.total_urgency:>8.3f}  {crit:<9} {top}")


def print_routes(solution: VRPSolution, vehicles: list):
    v_map = {v.id: v for v in vehicles}
    print("\n" + SEP)
    print(f"  VEHICLE ROUTES  ({len(solution.routes)} active vehicles)")
    print(SEP)
    for route in solution.routes:
        v = v_map.get(route.vehicle_id)
        vtype = v.vehicle_type.type_id if v else "?"
        feasible_tag = "OK" if route.feasible else "INFEASIBLE"
        cargo_str = ", ".join(
            f"{rtype}: {amt:.0f}kg" for rtype, amt in route.cargo_manifest.items()
        )
        print(f"\n  [{feasible_tag}] {route.vehicle_id} ({vtype})")
        print(f"    Route   : depot -> {' -> '.join(route.stops)} -> depot")
        print(f"    Distance: {route.total_distance_km:.1f} km")
        print(f"    Time    : {route.total_time_minutes:.0f} min")
        print(f"    Cargo   : {cargo_str or '(empty)'}")
        for stop in route.stop_details:
            del_str = ", ".join(
                f"{rt}: {amt:.0f}kg" for rt, amt in stop.cargo_delivered.items()
            ) or "-"
            print(f"      * {stop.village_id:<15} ETA {stop.eta_minutes:>5.0f} min  [{del_str}]")
    if not solution.routes:
        print("  (no routes - all vehicles idle)")


def print_allocations(solution: VRPSolution, villages: list):
    print("\n" + SEP)
    print("  VILLAGE ALLOCATIONS")
    print(SEP)
    print(f"  {'Village':<15} {'Satisfied':<10} {'ETA':>6}  Resources allocated")
    print(f"  {'-'*14} {'-'*9} {'-'*6}  {'-'*28}")
    for alloc in solution.allocations:
        satisfied_tag = "YES" if alloc.satisfied else "NO"
        eta_str = f"{alloc.eta_minutes:.0f}m" if alloc.eta_minutes > 0 else "-"
        res_str = ", ".join(
            f"{rt}:{amt:.0f}" for rt, amt in alloc.allocated_resources.items()
        ) or "(none)"
        print(f"  {alloc.village_id:<15} {satisfied_tag:<10} {eta_str:>6}  {res_str}")


def print_summary(solution: VRPSolution):
    pct = solution.objective_value * 100
    print("\n" + SEP)
    print("  SOLUTION SUMMARY")
    print(SEP)
    print(f"  Total distance    : {solution.total_distance_km:.1f} km")
    print(f"  Active vehicles   : {len(solution.routes)}")
    print(f"  Objective value   : {solution.objective_value:.4f}  ({pct:.1f}% needs met)")
    print(f"  Satisfaction bar  : [{bar(solution.objective_value)}]")
    print(f"  Unmet villages    : {len(solution.unmet_villages)}")
    if solution.unmet_villages:
        print(f"    -> {', '.join(solution.unmet_villages)}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    config_path  = ROOT / "backend" / "data" / "config.json"
    village_path = ROOT / "backend" / "data" / "nepal_villages.json"

    for p in (config_path, village_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run from project root.")
            sys.exit(1)

    cfg          = load_config(config_path)
    village_data = load_config(village_path)

    resource_types = build_resource_types(cfg)
    vehicles       = build_fleet(cfg)
    villages       = build_villages(village_data)

    depot      = village_data["depot"]
    depot_loc  = (depot["lat"], depot["lng"])
    depot_stock: dict = {
        k: float(v) for k, v in depot["available_resources"].items()
    }

    fleet_summary = ", ".join(
        f"{tid}x{cnt}" for tid, cnt in cfg["fleet_composition"].items()
    )

    print("\n" + "=" * 65)
    print("  RAKSHYANET -- VRP SOLVER SMOKE TEST  (Prompt 2.2)")
    print("=" * 65)
    print(f"  Villages loaded : {len(villages)}")
    print(f"  Fleet size      : {len(vehicles)} vehicles  [{fleet_summary}]")
    print(f"  Depot stock     : {', '.join(f'{r}: {int(a)}' for r, a in depot_stock.items())}")

    # Step 1: Urgency ranking at t=0
    calc   = UrgencyCalculator(resource_types=resource_types)
    scores = calc.rank_villages(villages, timedelta(hours=0))
    print_urgency_table(scores)

    # Step 2: Solve VRP
    solver = VRPSolver(
        depot_location=depot_loc,
        terrain_graph={},
        resource_types=resource_types,
        config=cfg,
    )
    solution = solver.solve(
        villages=villages,
        vehicles=vehicles,
        urgency_scores=scores,
        available_resources=depot_stock,
    )

    # Step 3: Display results
    print_routes(solution, vehicles)
    print_allocations(solution, villages)
    print_summary(solution)

    # Step 4: Assertions
    assert isinstance(solution.objective_value, float), "objective_value must be float"
    assert 0.0 <= solution.objective_value <= 1.0, "objective_value out of [0,1]"
    assert solution.total_distance_km > 0, "total_distance_km must be > 0"
    assert len(solution.allocations) == len(villages), "allocation count mismatch"
    assert solution.convergence_iterations == 1, "greedy solver must be 1 pass"

    print("\n" + "=" * 65)
    print("  SMOKE TEST PASSED")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
