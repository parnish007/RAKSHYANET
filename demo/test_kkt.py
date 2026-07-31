#!/usr/bin/env python3
"""
Smoke test: KKT Verifier -- Prompt 2.4
Runs VRP -> Nash -> KKT verification on the 8 Nepal villages.
Prints all four KKT conditions and shadow prices.

Run from project root:
    python demo/test_kkt.py
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
from backend.algorithms.nash_solver import NashSolver
from backend.algorithms.kkt_verifier import KKTVerifier, KKTVerificationResult

SEP = "-" * 62


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_resource_types(cfg):
    return {rid: ResourceType(**d) for rid, d in cfg["resource_types"].items()}


def build_fleet(cfg):
    vehicles = []
    for tid, count in cfg["fleet_composition"].items():
        vt = VehicleType(**cfg["vehicle_types"][tid])
        for i in range(1, count + 1):
            vehicles.append(Vehicle(id=f"{tid}_{i}", name=f"{vt.name} #{i}", vehicle_type=vt))
    return vehicles


def build_villages(data):
    villages = []
    for v in data["villages"]:
        needs = {r: VillageResourceNeed(**nd) for r, nd in v["resource_needs"].items()}
        villages.append(Village(
            id=v["id"], name=v["name"], lat=v["lat"], lng=v["lng"],
            population=v["population"],
            accessibility=v.get("accessibility", "road"),
            has_medical_facility=v.get("has_medical_facility", False),
            resource_needs=needs,
        ))
    return villages


def print_conditions(result: KKTVerificationResult):
    tick = "[PASS]"
    cross = "[FAIL]"
    print("\n" + SEP)
    print("  KKT CONDITIONS")
    print(SEP)
    for cond in result.conditions:
        mark = tick if cond.satisfied else cross
        print(f"  {mark} {cond.condition_name}")
        print(f"         residual = {cond.constraint_value:.2e}  (tol {cond.tolerance:.0e})")
        print(f"         {cond.description[:70]}")
        print()


def print_lambdas(result: KKTVerificationResult):
    print(SEP)
    print("  LAGRANGE MULTIPLIERS (shadow prices per resource)")
    print(SEP)
    print(f"  {'Resource':<15} {'lambda':>10}  Interpretation")
    print(f"  {'-'*14} {'-------':>10}  {'-'*28}")
    for rtype, lam in result.lagrange_multipliers.items():
        if lam > 1e-9:
            interp = "scarce -- has positive marginal value"
        else:
            interp = "slack  -- not fully consumed"
        print(f"  {rtype:<15} {lam:>10.6f}  {interp}")


def print_summary(result: KKTVerificationResult):
    badge = "CERTIFIED OPTIMAL" if result.all_conditions_satisfied else "NOT CERTIFIED"
    print("\n" + SEP)
    print(f"  VERIFICATION: {badge}")
    print(SEP)
    print(f"  All 4 KKT conditions : {'YES' if result.all_conditions_satisfied else 'NO'}")
    print(f"  Objective value      : {result.objective_value:.4f}")
    print(f"  CS violations        : {result.complementary_slackness_violations}")
    print(f"  Timestamp            : {result.verification_timestamp}")


def main():
    cfg_path  = ROOT / "backend" / "data" / "config.json"
    vil_path  = ROOT / "backend" / "data" / "nepal_villages.json"
    for p in (cfg_path, vil_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    cfg          = load_json(cfg_path)
    vil_data     = load_json(vil_path)
    resource_types = build_resource_types(cfg)
    vehicles       = build_fleet(cfg)
    villages       = build_villages(vil_data)

    depot     = vil_data["depot"]
    depot_loc = (depot["lat"], depot["lng"])
    depot_stock = {k: float(v) for k, v in depot["available_resources"].items()}

    print("\n" + "=" * 62)
    print("  RAKSHYANET -- KKT VERIFICATION SMOKE TEST  (Prompt 2.4)")
    print("=" * 62)

    # Step 1: VRP
    calc   = UrgencyCalculator(resource_types=resource_types)
    scores = calc.rank_villages(villages, timedelta(hours=0))
    vrp    = VRPSolver(depot_location=depot_loc, terrain_graph={},
                       resource_types=resource_types, config=cfg)
    vrp_sol = vrp.solve(villages=villages, vehicles=vehicles,
                        urgency_scores=scores, available_resources=depot_stock)
    print(f"\n  [VRP]  objective={vrp_sol.objective_value:.4f}")

    # Step 2: Nash
    nash = NashSolver(depot_resources=depot_stock, resource_types=resource_types,
                      convergence_threshold=0.01, max_iterations=100, seed=42)
    nash_eq = nash.solve(villages=villages, vrp_solution=vrp_sol)
    print(f"  [Nash] utility={nash_eq.total_utility:.4f}  "
          f"iterations={nash_eq.iterations}  welfare={nash_eq.welfare_improvement_percent:+.1f}%")

    # Step 3: KKT
    verifier = KKTVerifier(resource_types=resource_types, tolerance=1e-6)
    result   = verifier.verify(nash_eq, villages, depot_stock)
    print(f"  [KKT]  all_satisfied={result.all_conditions_satisfied}")

    print_conditions(result)
    print_lambdas(result)
    print_summary(result)

    # Assertions
    assert result.all_conditions_satisfied, "KKT conditions not all satisfied"
    assert len(result.conditions) == 4
    assert result.objective_value == nash_eq.total_utility
    assert result.verification_timestamp != ""
    for lam in result.lagrange_multipliers.values():
        assert lam >= -1e-6, f"Negative lambda: {lam}"

    print("\n" + "=" * 62)
    print("  SMOKE TEST PASSED")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
