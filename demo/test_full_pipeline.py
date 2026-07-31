#!/usr/bin/env python3
"""
Smoke test: Full Optimization Pipeline -- Prompt 2.5
Runs the complete Urgency -> VRP -> Nash -> KKT pipeline via StateManager.
Loads 8 Nepal villages and 13 vehicles from JSON data files.

Run from project root:
    python demo/test_full_pipeline.py
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
from backend.algorithms.state_manager import StateManager, OptimizationState

SEP  = "-" * 65
SEP2 = "=" * 65


# ------------------------------------------------------------------ #
#  Loaders (identical to prior smoke tests)                            #
# ------------------------------------------------------------------ #

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_resource_types(cfg: dict) -> dict:
    return {rid: ResourceType(**d) for rid, d in cfg["resource_types"].items()}


def build_fleet(cfg: dict) -> list:
    vehicles = []
    for tid, count in cfg["fleet_composition"].items():
        vt = VehicleType(**cfg["vehicle_types"][tid])
        for i in range(1, count + 1):
            vehicles.append(
                Vehicle(id=f"{tid}_{i}", name=f"{vt.name} #{i}", vehicle_type=vt)
            )
    return vehicles


def build_villages(data: dict) -> list:
    villages = []
    for v in data["villages"]:
        needs = {r: VillageResourceNeed(**nd) for r, nd in v["resource_needs"].items()}
        villages.append(
            Village(
                id=v["id"],
                name=v["name"],
                lat=v["lat"],
                lng=v["lng"],
                population=v["population"],
                accessibility=v.get("accessibility", "road"),
                has_medical_facility=v.get("has_medical_facility", False),
                resource_needs=needs,
            )
        )
    return villages


# ------------------------------------------------------------------ #
#  Print helpers                                                       #
# ------------------------------------------------------------------ #

def bar(fraction: float, width: int = 20) -> str:
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_urgency(scores):
    print("\n" + SEP)
    print("  URGENCY SCORES")
    print(SEP)
    print(f"  {'Village':<20} {'Score':>8}  {'Bar':<22}  Rank")
    print(f"  {'-'*19} {'-------':>8}  {'-'*22}  ----")
    max_score = max(s.total_urgency for s in scores) if scores else 1.0
    for rank, s in enumerate(scores, start=1):
        frac = s.total_urgency / max_score if max_score else 0.0
        print(f"  {s.village_id:<20} {s.total_urgency:>8.2f}  {bar(frac):<22}  #{rank}")


def print_vrp(vrp_sol):
    print("\n" + SEP)
    print("  VRP ROUTES")
    print(SEP)
    print(f"  Active routes : {len(vrp_sol.routes)}")
    print(f"  Total distance: {vrp_sol.total_distance_km:.1f} km")
    print(f"  Objective     : {vrp_sol.objective_value:.4f}")
    for route in vrp_sol.routes:
        stops = " -> ".join(route.stops)
        print(f"    [{route.vehicle_id}] {stops}  ({route.total_distance_km:.1f} km)")


def print_nash(nash_eq):
    print("\n" + SEP)
    print("  NASH EQUILIBRIUM")
    print(SEP)
    print(f"  Converged     : {nash_eq.converged}")
    print(f"  Iterations    : {nash_eq.iterations}")
    print(f"  Total utility : {nash_eq.total_utility:.4f}")
    print(f"  Welfare gain  : {nash_eq.welfare_improvement_percent:+.1f}%")
    print(f"\n  {'Village':<20} {'Food':>8} {'Water':>8} {'Medical':>8} {'Utility':>9}")
    print(f"  {'-'*19} {'------':>8} {'------':>8} {'-------':>8} {'-------':>9}")
    for strat in nash_eq.strategies:
        f = strat.allocated_resources.get("food", 0.0)
        w = strat.allocated_resources.get("water", 0.0)
        m = strat.allocated_resources.get("medical_kit", 0.0)
        print(f"  {strat.village_id:<20} {f:>8.1f} {w:>8.1f} {m:>8.1f} {strat.utility:>9.4f}")


def print_kkt(kkt_result):
    tick  = "[PASS]"
    cross = "[FAIL]"
    print("\n" + SEP)
    print("  KKT CONDITIONS")
    print(SEP)
    for cond in kkt_result.conditions:
        mark = tick if cond.satisfied else cross
        print(f"  {mark} {cond.condition_name}")
        print(f"         residual = {cond.constraint_value:.2e}  (tol {cond.tolerance:.0e})")

    print("\n  Lagrange Multipliers (shadow prices):")
    print(f"  {'Resource':<15} {'lambda':>10}  Status")
    print(f"  {'-'*14} {'------':>10}  ------")
    for rtype, lam in kkt_result.lagrange_multipliers.items():
        status = "scarce" if lam > 1e-9 else "slack"
        print(f"  {rtype:<15} {lam:>10.6f}  {status}")


def print_summary(result, elapsed_ms: float):
    badge = "CERTIFIED OPTIMAL" if result.kkt_verification.all_conditions_satisfied else "NOT CERTIFIED"
    print("\n" + SEP2)
    print(f"  PIPELINE RESULT: {badge}")
    print(SEP2)
    print(f"  Final state      : {result.state}")
    print(f"  Execution time   : {result.execution_time_seconds*1000:.1f} ms")
    print(f"  KKT conditions   : {'ALL PASS' if result.kkt_verification.all_conditions_satisfied else 'FAIL'}")
    print(f"  Objective value  : {result.nash_equilibrium.total_utility:.4f}")
    print(f"  Welfare vs VRP   : {result.nash_equilibrium.welfare_improvement_percent:+.1f}%")
    print(f"  CS violations    : {result.kkt_verification.complementary_slackness_violations}")
    print(f"  Timestamp        : {result.timestamp}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    cfg_path = ROOT / "backend" / "data" / "config.json"
    vil_path = ROOT / "backend" / "data" / "nepal_villages.json"
    ter_path = ROOT / "backend" / "data" / "terrain_graph.json"

    for p in (cfg_path, vil_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    cfg  = load_json(cfg_path)
    vil_data = load_json(vil_path)
    terrain  = load_json(ter_path) if ter_path.exists() else {}

    resource_types = build_resource_types(cfg)
    vehicles       = build_fleet(cfg)
    villages       = build_villages(vil_data)

    depot     = vil_data["depot"]
    depot_loc = (depot["lat"], depot["lng"])
    depot_stock = {k: float(v) for k, v in depot["available_resources"].items()}

    print("\n" + SEP2)
    print("  RAKSHYANET -- FULL PIPELINE SMOKE TEST  (Prompt 2.5)")
    print(SEP2)
    print(f"  Villages : {len(villages)}")
    print(f"  Vehicles : {len(vehicles)}")
    print(f"  Depot    : {depot_loc}")
    print(f"  Stock    : " + "  ".join(f"{k}={v}" for k, v in depot_stock.items()))

    # Build and run StateManager
    sm = StateManager(
        depot_location=depot_loc,
        depot_resources=depot_stock,
        terrain_graph=terrain,
        resource_types=resource_types,
        config=cfg,
    )

    print(f"\n  Initial state : {sm.get_state()}")
    print("  Running pipeline...")

    result = sm.run_full_optimization(
        villages=villages,
        vehicles=vehicles,
        time_elapsed=timedelta(hours=0),
    )

    print(f"  Final state   : {sm.get_state()}")

    # Print all outputs
    print_urgency(result.urgency_scores)
    print_vrp(result.vrp_solution)
    print_nash(result.nash_equilibrium)
    print_kkt(result.kkt_verification)
    print_summary(result, result.execution_time_seconds * 1000)

    # ---------------------------------------------------------------- #
    #  Assertions                                                        #
    # ---------------------------------------------------------------- #
    assert result.state == OptimizationState.COMPLETE, \
        f"Expected COMPLETE, got {result.state}: {result.error_message}"

    assert result.kkt_verification.all_conditions_satisfied, \
        "KKT conditions not all satisfied"

    assert len(result.urgency_scores) == len(villages), \
        f"Expected {len(villages)} urgency scores, got {len(result.urgency_scores)}"

    assert result.vrp_solution is not None
    assert result.nash_equilibrium is not None
    assert result.nash_equilibrium.converged is True
    assert result.execution_time_seconds > 0.0
    assert result.timestamp != ""
    assert result.error_message is None

    assert result.kkt_verification.all_conditions_satisfied is True
    for lam in result.kkt_verification.lagrange_multipliers.values():
        assert lam >= -1e-6, f"Negative lambda: {lam}"

    print("\n" + SEP2)
    print("  SMOKE TEST PASSED")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()
