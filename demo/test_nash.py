#!/usr/bin/env python3
"""
Smoke test: Nash Equilibrium Solver -- Prompt 2.3
Runs VRP (greedy) then Nash (best-response) on the same 8 villages.
Prints a before/after comparison and convergence trace.

Run from project root:
    python demo/test_nash.py
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
from backend.algorithms.vrp_solver import VRPSolver
from backend.algorithms.nash_solver import NashSolver, NashEquilibrium

SEP = "-" * 68


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_resource_types(cfg: dict) -> dict:
    return {rid: ResourceType(**rdata) for rid, rdata in cfg["resource_types"].items()}


def build_fleet(cfg: dict) -> list:
    vehicles = []
    for type_id, count in cfg["fleet_composition"].items():
        vt = VehicleType(**cfg["vehicle_types"][type_id])
        for i in range(1, count + 1):
            vehicles.append(Vehicle(id=f"{type_id}_{i}", name=f"{vt.name} #{i}", vehicle_type=vt))
    return vehicles


def build_villages(data: dict) -> list:
    villages = []
    for v in data["villages"]:
        resource_needs = {
            rtype: VillageResourceNeed(**nd)
            for rtype, nd in v["resource_needs"].items()
        }
        villages.append(Village(
            id=v["id"], name=v["name"],
            lat=v["lat"], lng=v["lng"],
            population=v["population"],
            accessibility=v.get("accessibility", "road"),
            has_medical_facility=v.get("has_medical_facility", False),
            resource_needs=resource_needs,
        ))
    return villages


def bar(ratio: float, width: int = 24) -> str:
    filled = int(round(min(1.0, ratio) * width))
    return "#" * filled + "." * (width - filled)


def satisfaction_pct(strategy, village_map):
    v = village_map.get(strategy.village_id)
    if not v:
        return 0.0
    total_need = sum(n.current_need for n in v.resource_needs.values())
    total_got  = sum(strategy.allocated_resources.get(r, 0.0) for r in v.resource_needs)
    return (total_got / total_need * 100.0) if total_need > 0 else 0.0


def print_comparison(vrp_allocs, nash_eq: NashEquilibrium, village_map):
    vrp_map = {a.village_id: a for a in vrp_allocs}
    nash_map = {s.village_id: s for s in nash_eq.strategies}

    print("\n" + SEP)
    print("  VRP GREEDY  vs  NASH EQUILIBRIUM  -- per-village satisfaction %")
    print(SEP)
    print(f"  {'Village':<14} {'VRP':>6}  {'VRP bar':<26}  {'Nash':>6}  Nash bar")
    print(f"  {'-'*13} {'------':>6}  {'-'*24}  {'------':>6}  {'-'*24}")

    vrp_total_need = vrp_total_got = 0.0
    nash_total_need = nash_total_got = 0.0

    for vid, v in village_map.items():
        total_need = sum(n.current_need for n in v.resource_needs.values())

        vrp_got  = sum(vrp_map[vid].allocated_resources.get(r, 0.0) for r in v.resource_needs) if vid in vrp_map else 0.0
        nash_got = sum(nash_map[vid].allocated_resources.get(r, 0.0) for r in v.resource_needs) if vid in nash_map else 0.0

        vrp_pct  = vrp_got  / total_need * 100 if total_need > 0 else 0.0
        nash_pct = nash_got / total_need * 100 if total_need > 0 else 0.0

        vrp_total_need  += total_need;  vrp_total_got  += vrp_got
        nash_total_need += total_need;  nash_total_got += nash_got

        delta = nash_pct - vrp_pct
        delta_str = f"(+{delta:.0f}%)" if delta > 0 else f"({delta:.0f}%)" if delta < 0 else "(=)"
        print(f"  {vid:<14} {vrp_pct:>5.1f}%  [{bar(vrp_pct/100):<24}]  {nash_pct:>5.1f}%  [{bar(nash_pct/100):<24}] {delta_str}")

    vrp_overall  = vrp_total_got  / vrp_total_need  * 100 if vrp_total_need  > 0 else 0.0
    nash_overall = nash_total_got / nash_total_need * 100 if nash_total_need > 0 else 0.0
    print(f"  {'OVERALL':<14} {vrp_overall:>5.1f}%  [{bar(vrp_overall/100):<24}]  {nash_overall:>5.1f}%  [{bar(nash_overall/100):<24}]")


def print_convergence(nash_eq: NashEquilibrium):
    print("\n" + SEP)
    print(f"  CONVERGENCE TRACE  ({nash_eq.iterations} iterations, converged={nash_eq.converged})")
    print(SEP)
    print(f"  {'Iter':>4}  {'Max Change':>11}  {'Total Utility':>14}  Progress")
    print(f"  {'----':>4}  {'-'*11}  {'-'*14}  {'-'*20}")
    hist = nash_eq.convergence_history
    # Print first 3, last 3, and middle sample to keep output compact
    indices = sorted(set([0, 1, 2, len(hist)//2, len(hist)-3, len(hist)-2, len(hist)-1]))
    prev_idx = -2
    for idx in indices:
        if idx < 0 or idx >= len(hist):
            continue
        if idx > prev_idx + 1:
            print(f"  {'...':>4}")
        h = hist[idx]
        progress = bar(max(0.0, 1.0 - h.max_strategy_change / (hist[0].max_strategy_change + 1e-9)), 20)
        print(f"  {h.iteration:>4}  {h.max_strategy_change:>11.4f}  {h.total_utility:>14.4f}  [{progress}]")
        prev_idx = idx


def print_summary(vrp_obj, nash_eq: NashEquilibrium):
    print("\n" + SEP)
    print("  SUMMARY")
    print(SEP)
    print(f"  VRP objective value   : {vrp_obj:.4f}")
    print(f"  Nash total utility    : {nash_eq.total_utility:.4f}")
    print(f"  Welfare improvement   : {nash_eq.welfare_improvement_percent:+.1f}%")
    print(f"  Nash converged        : {nash_eq.converged}  (epsilon={nash_eq.epsilon_convergence:.6f})")
    print(f"  Iterations used       : {nash_eq.iterations}")


def main():
    config_path  = ROOT / "backend" / "data" / "config.json"
    village_path = ROOT / "backend" / "data" / "nepal_villages.json"
    for p in (config_path, village_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    cfg          = load_json(config_path)
    village_data = load_json(village_path)
    resource_types = build_resource_types(cfg)
    vehicles       = build_fleet(cfg)
    villages       = build_villages(village_data)
    village_map    = {v.id: v for v in villages}

    depot = village_data["depot"]
    depot_loc   = (depot["lat"], depot["lng"])
    depot_stock = {k: float(v) for k, v in depot["available_resources"].items()}

    print("\n" + "=" * 68)
    print("  RAKSHYANET -- NASH EQUILIBRIUM SMOKE TEST  (Prompt 2.3)")
    print("=" * 68)

    # --- Step 1: VRP greedy ---
    calc   = UrgencyCalculator(resource_types=resource_types)
    scores = calc.rank_villages(villages, timedelta(hours=0))

    vrp_solver = VRPSolver(depot_location=depot_loc, terrain_graph={},
                           resource_types=resource_types, config=cfg)
    vrp_sol = vrp_solver.solve(villages=villages, vehicles=vehicles,
                               urgency_scores=scores, available_resources=depot_stock)
    print(f"\n  [VRP]  objective={vrp_sol.objective_value:.4f}  "
          f"routes={len(vrp_sol.routes)}  "
          f"unmet={len(vrp_sol.unmet_villages)} villages")

    # --- Step 2: Nash equilibrium ---
    nash_solver = NashSolver(
        depot_resources=depot_stock,
        resource_types=resource_types,
        convergence_threshold=0.01,
        max_iterations=100,
        seed=42,
    )
    nash_eq = nash_solver.solve(villages=villages, vrp_solution=vrp_sol)
    print(f"  [Nash] utility={nash_eq.total_utility:.4f}  "
          f"iterations={nash_eq.iterations}  "
          f"converged={nash_eq.converged}")

    # --- Display ---
    print_comparison(vrp_sol.allocations, nash_eq, village_map)
    print_convergence(nash_eq)
    print_summary(vrp_sol.objective_value, nash_eq)

    # --- Assertions ---
    assert nash_eq.converged, "Nash solver did not converge"
    assert nash_eq.total_utility >= 0.0
    assert nash_eq.iterations <= 100
    assert len(nash_eq.strategies) == len(villages)
    assert len(nash_eq.convergence_history) >= 1

    print("\n" + "=" * 68)
    print("  SMOKE TEST PASSED")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
