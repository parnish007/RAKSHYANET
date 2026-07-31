#!/usr/bin/env python3
"""
Smoke test: Timeline Simulator + Re-optimization Trigger -- Prompts 5.1 / 5.2

Loads mock_news_timeline.json (6 events), runs simulation at 500x speed
with the ReoptimizationTrigger wired in, prints per-event results including
actual route changes, and asserts correct action routing.

Run from project root:
    python demo/test_timeline.py
"""
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.algorithms.state_manager import StateManager
from backend.demo.reoptimization_trigger import ReoptimizationConfig, ReoptimizationTrigger
from backend.demo.timeline_simulator import SimulatorConfig, TimelineSimulator
from backend.hitl.approval_queue import ApprovalQueue, ApprovalStatus
from backend.models.resource import ResourceType
from backend.models.vehicle import TerrainCapability, Vehicle, VehicleCategory, VehicleType
from backend.models.village import Village
from backend.rag.news_analyzer import (
    ACTION_AUTO_OPTIMIZE,
    ACTION_HITL_REQUIRED,
    ACTION_IGNORE,
    IntelligenceReport,
    NewsAnalyzer,
)

SEP  = "-" * 65
SEP2 = "=" * 65

TIMELINE_PATH = str(ROOT / "backend" / "demo" / "mock_news_timeline.json")


def bar(fraction: float, width: int = 16) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def action_tag(action: str) -> str:
    return {
        ACTION_AUTO_OPTIMIZE: "[AUTO  ]",
        ACTION_HITL_REQUIRED: "[HITL  ]",
        ACTION_IGNORE:        "[IGNORE]",
    }.get(action, f"[{action[:6]}]")


# ------------------------------------------------------------------ #
#  Callback: collect results for assertions                            #
# ------------------------------------------------------------------ #

results: list = []

def on_event(event, report: IntelligenceReport) -> None:
    results.append({
        "event_id":   event.event_id,
        "action":     report.recommended_action,
        "confidence": report.event.confidence,
        "severity":   report.event.severity,
        "expected":   event.expected_action,
    })


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print("\n" + SEP2)
    print("  RAKSHYANET -- TIMELINE SIMULATOR SMOKE TEST  (Prompts 5.1 / 5.2)")
    print(SEP2)

    # ---- Build infrastructure ------------------------------------ #
    print(f"\n  {SEP}")
    print(f"  BUILDING INFRASTRUCTURE")
    print(f"  {SEP}")

    config_data   = json.loads((ROOT / "backend" / "data" / "config.json").read_text())
    terrain_graph = json.loads((ROOT / "backend" / "data" / "terrain_graph.json").read_text())
    villages_data = json.loads((ROOT / "backend" / "data" / "nepal_villages.json").read_text())

    resource_types = {
        k: ResourceType(**v)
        for k, v in config_data.get("resource_types", {}).items()
    }

    state_manager = StateManager(
        depot_location=(27.7172, 85.3240),
        depot_resources={rt: 500.0 for rt in resource_types},
        terrain_graph=terrain_graph,
        resource_types=resource_types,
        config=config_data,
    )
    print(f"  StateManager : ready ({len(resource_types)} resource types)")

    villages = [
        Village(
            id=v["id"],
            name=v["name"],
            lat=v["lat"],
            lng=v["lng"],
            population=v["population"],
            terrain_difficulty=v["terrain_difficulty"],
            urgency_score=v["initial_urgency"],
            disaster_impact=v["disaster_impact"],
        )
        for v in villages_data["villages"]
    ]
    print(f"  Villages     : {len(villages)} loaded")

    vtype = VehicleType(
        type_id="van_4x4",
        name="4x4 Relief Van",
        category=VehicleCategory.GROUND_LIGHT,
        capacity_kg=800.0,
        speed_kmh=60.0,
        fuel_hours=8.0,
        terrain_capability=TerrainCapability.ALL_ROADS,
    )
    vehicles = [
        Vehicle(id=f"demo_van_{i}", name=f"Demo Van {i}", vehicle_type=vtype)
        for i in range(1, 4)
    ]
    print(f"  Vehicles     : {len(vehicles)} created")

    reopt_config = ReoptimizationConfig(
        urgency_change_threshold=0.10,
        enable_reoptimization=True,
        broadcast_via_p2p=False,
        log_optimization_changes=False,
    )
    trigger = ReoptimizationTrigger(
        config=reopt_config,
        state_manager=state_manager,
        villages=villages,
        vehicles=vehicles,
        time_elapsed=timedelta(hours=2),
    )
    print(f"  ReoptTrigger : ready (threshold={reopt_config.urgency_change_threshold})")

    queue    = ApprovalQueue(timeout_minutes=5)
    analyzer = NewsAnalyzer()
    config   = SimulatorConfig(
        timeline_path=TIMELINE_PATH,
        speed_multiplier=500.0,   # 1800 s timeline runs in ~3.6 s
        auto_approve_hitl=True,   # auto-approve for demo flow
        trigger_reoptimization=True,
        verbose_logging=False,
    )

    sim = TimelineSimulator(
        config, analyzer, queue,
        reoptimization_trigger=trigger,
        on_event_callback=on_event,
        villages=villages,
    )

    # ---- Load ---------------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  LOADING TIMELINE")
    print(f"  {SEP}")
    events = sim.load_timeline()
    print(f"  Timeline : {TIMELINE_PATH}")
    print(f"  Events   : {len(events)}")
    print(f"  Speed    : {config.speed_multiplier}x  "
          f"(1800 s -> ~{1800/config.speed_multiplier:.1f} s wall time)")
    print(f"  Auto-approve HITL : {config.auto_approve_hitl}")

    print(f"\n  {'#':<3} {'Event ID':<35} {'Offset':>7}  {'Expected Action'}")
    print(f"  {'-'*3} {'-'*34} {'-------':>7}  {'-'*15}")
    for i, e in enumerate(events, 1):
        print(f"  {i:<3} {e.event_id:<35} {e.timestamp_offset_seconds:>7.0f}s  "
              f"{e.expected_action}")

    # ---- Run simulation ------------------------------------------ #
    print(f"\n  {SEP}")
    print(f"  RUNNING SIMULATION")
    print(f"  {SEP}")

    t0 = time.time()
    sim.start_simulation()
    sim._thread.join(timeout=30.0)
    elapsed = time.time() - t0

    print(f"\n  Wall time: {elapsed:.2f} s")

    # ---- Print per-event results ---------------------------------- #
    print(f"\n  {SEP}")
    print(f"  EVENT RESULTS")
    print(f"  {SEP}")
    print(f"  {'Event ID':<35} {'Action':<10} {'Conf':>5}  {'Sev':>3}  "
          f"{'Exp.Action':<14} {'Match?'}")
    print(f"  {'-'*34} {'-'*9} {'-----':>5}  {'---':>3}  {'-'*13} {'------'}")

    for r in results:
        match = "YES" if r["action"] == r["expected"] else "NO "
        print(f"  {r['event_id']:<35} {action_tag(r['action']):<10} "
              f"{r['confidence']:>5.2f}  {r['severity']:>3}  "
              f"{r['expected']:<14} {match}")

    # ---- State summary ------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  SIMULATION STATE")
    print(f"  {SEP}")
    state = sim.get_state()
    print(f"  Events processed      : {state.events_processed}")
    print(f"  AUTO_OPTIMIZE count   : {state.auto_count}")
    print(f"  HITL count            : {state.hitl_count}")
    print(f"  IGNORE count          : {state.ignore_count}")
    print(f"  Reoptimizations       : {state.reoptimizations_triggered}")
    print(f"  Events pending HITL   : {state.events_pending_hitl}")
    print(f"  Is running            : {state.is_running}")

    # ---- HITL queue state ---------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  HITL QUEUE STATE")
    print(f"  {SEP}")
    history = queue.get_history()
    pending = queue.get_pending()
    print(f"  Pending : {len(pending)}")
    print(f"  History : {len(history)}")
    for req in history:
        print(f"    {req.request_id}  {req.event_id:<25} {req.status.value:<10} "
              f"by={req.reviewed_by or '-'}")

    # ---- Reoptimization history ---------------------------------- #
    print(f"\n  {SEP}")
    print(f"  REOPTIMIZATION HISTORY")
    print(f"  {SEP}")
    reopt_history = trigger.get_optimization_history()
    print(f"  Total reoptimizations : {len(reopt_history)}")
    print(f"  {'Event ID':<35} {'Routes':>6}  {'Welfare':>9}  {'Time(ms)':>8}")
    print(f"  {'-'*34} {'------':>6}  {'---------':>9}  {'--------':>8}")
    for ch in reopt_history:
        print(f"  {ch.trigger_event_id:<35} {ch.routes_changed:>6}  "
              f"{ch.welfare_improvement:>+9.4f}  {ch.execution_time_ms:>8.0f}")

    # ---- Assertions ---------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  ASSERTIONS")
    print(f"  {SEP}")

    assert state.events_processed == 6, \
        f"Expected 6 events processed, got {state.events_processed}"
    print("  [OK] 6 events processed")

    assert not state.is_running, "Simulation should have stopped"
    print("  [OK] simulation is not running")

    total = state.auto_count + state.hitl_count + state.ignore_count
    assert total == 6, f"Action counts don't sum to 6: {total}"
    print(f"  [OK] action counts sum to 6 (auto={state.auto_count}, "
          f"hitl={state.hitl_count}, ignore={state.ignore_count})")

    assert state.ignore_count >= 1, "Expected at least 1 IGNORE event"
    print(f"  [OK] at least 1 IGNORE event")

    assert state.reoptimizations_triggered > 0, "Expected some reoptimizations"
    print(f"  [OK] {state.reoptimizations_triggered} reoptimizations triggered")

    # All HITL events auto-approved (pending should be 0)
    assert state.events_pending_hitl == 0, \
        f"Expected 0 pending HITL (auto_approve=True), got {state.events_pending_hitl}"
    print("  [OK] 0 events pending HITL (all auto-approved)")

    # Callback fired for every event
    assert len(results) == 6, f"Callback should fire 6 times, fired {len(results)}"
    print("  [OK] callback fired for all 6 events")

    assert len(reopt_history) > 0, \
        f"ReoptimizationTrigger history should be non-empty, got {len(reopt_history)}"
    print(f"  [OK] {len(reopt_history)} reoptimization(s) in trigger history")

    for ch in reopt_history:
        assert ch.optimization_state == "complete", \
            f"Expected 'complete', got '{ch.optimization_state}'"
    print("  [OK] all reoptimizations completed successfully")

    # Test reset
    sim.reset()
    assert sim.state.events_processed == 0
    assert not sim.state.is_running
    print("  [OK] reset clears state")

    # Test stop mid-run
    sim2 = TimelineSimulator(
        SimulatorConfig(
            timeline_path=TIMELINE_PATH,
            speed_multiplier=1.0,   # real-time, so we can stop it
            auto_approve_hitl=False,
            verbose_logging=False,
        ),
        analyzer,
        ApprovalQueue(),
    )
    sim2.start_simulation()
    time.sleep(0.1)
    sim2.stop_simulation()
    assert not sim2.state.is_running
    print("  [OK] stop_simulation() works mid-run")

    # ---- Summary ------------------------------------------------- #
    print("\n" + SEP2)
    print("  SUMMARY")
    print(SEP2)
    print(f"  Timeline events     : 6")
    print(f"  Wall time           : {elapsed:.2f} s  (speed={config.speed_multiplier}x)")
    print(f"  AUTO_OPTIMIZE       : {state.auto_count}")
    print(f"  HITL (auto-approved): {state.hitl_count}")
    print(f"  IGNORED             : {state.ignore_count}")
    print(f"  Reoptimizations     : {state.reoptimizations_triggered}")
    print(f"  Trigger history     : {len(reopt_history)} entries")
    if reopt_history:
        total_welfare = sum(ch.welfare_improvement for ch in reopt_history)
        print(f"  Welfare delta sum   : {total_welfare:+.4f}")

    print("\n" + SEP2)
    print("  SMOKE TEST PASSED")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()