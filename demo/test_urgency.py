#!/usr/bin/env python3
"""
Smoke test: Urgency Calculator — Prompt 2.1
Loads resource types from config.json, builds two villages, shows rankings at t=0 and t=4hr.

Run from project root:
    python demo/test_urgency.py
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

# Make sure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.resource import ResourceCategory, ResourceType, VillageResourceNeed
from backend.models.village import Village
from backend.algorithms.urgency_calculator import UrgencyCalculator


def load_resource_types(config_path: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    resource_types = {}
    for rid, rdata in cfg["resource_types"].items():
        resource_types[rid] = ResourceType(**rdata)
    return resource_types


def make_village_critical() -> Village:
    """Dhulikhel — landslide, medical clinic buried, all resources critical."""
    return Village(
        id="dhulikhel",
        name="Dhulikhel",
        lat=27.62, lng=85.55,
        population=5000,
        resource_needs={
            "food":        VillageResourceNeed(resource_type="food",        current_need=2500, min_need=1500, allocated=0),
            "water":       VillageResourceNeed(resource_type="water",       current_need=1500, min_need=1000, allocated=0),
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=50,   min_need=30,   allocated=0),
            "tarpaulin":   VillageResourceNeed(resource_type="tarpaulin",   current_need=200,  min_need=100,  allocated=0),
            "blanket":     VillageResourceNeed(resource_type="blanket",     current_need=300,  min_need=150,  allocated=0),
            "first_aid":   VillageResourceNeed(resource_type="first_aid",   current_need=80,   min_need=40,   allocated=0),
        },
    )


def make_village_moderate() -> Village:
    """Panauti — partially supplied, above survival threshold."""
    return Village(
        id="panauti",
        name="Panauti",
        lat=27.585, lng=85.517,
        population=4200,
        resource_needs={
            "food":        VillageResourceNeed(resource_type="food",        current_need=2100, min_need=1260, allocated=1400),
            "water":       VillageResourceNeed(resource_type="water",       current_need=1200, min_need=840,  allocated=900),
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=40,   min_need=24,   allocated=30),
            "tarpaulin":   VillageResourceNeed(resource_type="tarpaulin",   current_need=160,  min_need=80,   allocated=100),
            "blanket":     VillageResourceNeed(resource_type="blanket",     current_need=240,  min_need=120,  allocated=150),
            "first_aid":   VillageResourceNeed(resource_type="first_aid",   current_need=65,   min_need=32,   allocated=40),
        },
    )


def make_village_satisfied() -> Village:
    """Banepa — fully supplied."""
    return Village(
        id="banepa",
        name="Banepa",
        lat=27.632, lng=85.521,
        population=6500,
        resource_needs={
            "food":        VillageResourceNeed(resource_type="food",        current_need=3250, min_need=1950, allocated=3250),
            "water":       VillageResourceNeed(resource_type="water",       current_need=1950, min_need=1365, allocated=1950),
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=65,   min_need=39,   allocated=65),
            "tarpaulin":   VillageResourceNeed(resource_type="tarpaulin",   current_need=260,  min_need=130,  allocated=260),
            "blanket":     VillageResourceNeed(resource_type="blanket",     current_need=390,  min_need=195,  allocated=390),
            "first_aid":   VillageResourceNeed(resource_type="first_aid",   current_need=104,  min_need=52,   allocated=104),
        },
    )


def print_rankings(title: str, scores):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")
    print(f"  {'Rank':<5} {'Village':<18} {'Urgency':>8}  {'Critical':<9} {'Top Resource'}")
    print(f"  {'─'*4} {'─'*17} {'─'*8}  {'─'*8} {'─'*15}")
    for s in scores:
        top = s.top_resource() or "—"
        crit = "YES ⚠" if s.has_critical_shortage else "no"
        print(f"  #{s.ranking:<4} {s.village_id:<18} {s.total_urgency:>8.3f}  {crit:<9} {top}")


def main():
    config_path = ROOT / "backend" / "data" / "config.json"
    if not config_path.exists():
        print("ERROR: backend/data/config.json not found. Run from project root.")
        sys.exit(1)

    resource_types = load_resource_types(config_path)
    calc = UrgencyCalculator(resource_types=resource_types)

    villages = [
        make_village_critical(),
        make_village_moderate(),
        make_village_satisfied(),
    ]

    print("\n" + "="*55)
    print("  RAKSHYANET — URGENCY CALCULATOR SMOKE TEST")
    print("="*55)

    # Time factor preview
    print("\n  Time factor growth:")
    for h in [0, 1, 2, 4, 8]:
        f = calc.calculate_time_factor(float(h))
        print(f"    t={h}hr  →  {f:.4f}")

    # Rankings at t=0
    r0 = calc.rank_villages(villages, timedelta(hours=0))
    print_rankings("RANKINGS at t=0hr (initial state)", r0)

    # Rankings at t=4hr (urgency has escalated)
    r4 = calc.rank_villages(villages, timedelta(hours=4))
    print_rankings("RANKINGS at t=4hr (escalated urgency)", r4)

    # Re-optimization trigger check
    triggers = calc.detect_reoptimization_trigger(r0, r4, threshold=0.10)
    print(f"\n  Re-optimization triggers (ΔU > 0.10): {len(triggers)}")
    for t in triggers:
        print(f"    {t['village_id']:<18} Δ={t['delta']:.3f}  ({t['old_urgency']:.2f} → {t['new_urgency']:.2f})")

    print(f"\n{'='*55}")
    print(f"  SMOKE TEST PASSED")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
