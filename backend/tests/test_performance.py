"""
Performance Tests -- Prompt 6.2

Verifies the system meets hackathon demo requirements:
  - Full optimization pipeline < 15 seconds
  - No memory leaks over repeated runs
  - Timeline simulation < 1s per event
  - No performance degradation under rapid re-optimization

Run: pytest backend/tests/test_performance.py -v -s
     (-s shows per-test timing output)
"""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from backend.algorithms.kkt_verifier import KKTVerifier
from backend.algorithms.nash_solver import NashSolver
from backend.algorithms.state_manager import OptimizationState, StateManager
from backend.algorithms.urgency_calculator import UrgencyCalculator
from backend.algorithms.vrp_solver import VRPSolver
from backend.demo.timeline_simulator import SimulatorConfig, TimelineSimulator
from backend.hitl.approval_queue import ApprovalQueue
from backend.models.resource import ResourceType
from backend.models.vehicle import TerrainCapability, Vehicle, VehicleCategory, VehicleType
from backend.models.village import Village
from backend.rag.news_analyzer import NewsAnalyzer

# ------------------------------------------------------------------ #
#  Performance targets                                                 #
# ------------------------------------------------------------------ #

TARGET_FULL_PIPELINE_S    = 15.0   # full Urgency→VRP→Nash→KKT
TARGET_PER_EVENT_S        =  1.0   # each RAG analysis
TARGET_COMPONENT_BENCH_S  =  5.0   # any single component
MEMORY_LEAK_THRESHOLD_MB  = 10.0   # growth over repeated runs

DATA = Path(__file__).parent.parent / "data"
TIMELINE_PATH = str(Path(__file__).parent.parent / "demo" / "mock_news_timeline.json")

# ------------------------------------------------------------------ #
#  psutil (optional)                                                   #
# ------------------------------------------------------------------ #

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def _memory_mb() -> float:
    if not PSUTIL_AVAILABLE:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 * 1024)


# ================================================================== #
#  PerformanceTimer                                                    #
# ================================================================== #

class PerformanceTimer:
    """Context manager for measuring elapsed wall time."""

    def __init__(self, label: str) -> None:
        self.label   = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "PerformanceTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._t0
        print(f"  [PERF] {self.label}: {self.elapsed:.3f}s")

    def assert_under(self, threshold: float, msg: str = "") -> None:
        assert self.elapsed < threshold, (
            f"{self.label} took {self.elapsed:.3f}s, expected <{threshold}s. {msg}"
        )


# ================================================================== #
#  Shared data-loading helpers                                         #
# ================================================================== #

def _load_config() -> Tuple[dict, dict, dict, Dict[str, ResourceType]]:
    config_data   = json.loads((DATA / "config.json").read_text())
    terrain_graph = json.loads((DATA / "terrain_graph.json").read_text())
    villages_raw  = json.loads((DATA / "nepal_villages.json").read_text())
    resource_types = {
        k: ResourceType(**v)
        for k, v in config_data.get("resource_types", {}).items()
    }
    return config_data, terrain_graph, villages_raw, resource_types


def _make_sm(config_data, terrain_graph, resource_types) -> StateManager:
    return StateManager(
        depot_location=(27.7172, 85.3240),
        depot_resources={rt: 500.0 for rt in resource_types},
        terrain_graph=terrain_graph,
        resource_types=resource_types,
        config=config_data,
    )


def _make_real_villages(villages_raw: dict) -> List[Village]:
    return [
        Village(
            id=v["id"], name=v["name"],
            lat=v["lat"], lng=v["lng"],
            population=v["population"],
            terrain_difficulty=v["terrain_difficulty"],
            urgency_score=v["initial_urgency"],
            disaster_impact=v["disaster_impact"],
        )
        for v in villages_raw["villages"]
    ]


def _make_synthetic_villages(n: int, base_urgency: float = 0.6) -> List[Village]:
    return [
        Village(
            id=f"syn_{i}", name=f"Village {i}",
            lat=27.60 + i * 0.05, lng=85.40 + i * 0.05,
            population=2000 + i * 200,
            terrain_difficulty=1.5,
            urgency_score=min(1.0, base_urgency + i * 0.03),
            disaster_impact=0.5,
        )
        for i in range(n)
    ]


def _make_fleet(n: int, category: str = "light") -> List[Vehicle]:
    if category == "heli":
        vtype = VehicleType(
            type_id="heli", name="Helicopter",
            category=VehicleCategory.AIRCRAFT,
            capacity_kg=500.0, speed_kmh=200.0, fuel_hours=2.5,
            terrain_capability=TerrainCapability.ANY,
        )
    else:
        vtype = VehicleType(
            type_id="van_4x4", name="4x4 Van",
            category=VehicleCategory.GROUND_LIGHT,
            capacity_kg=800.0, speed_kmh=60.0, fuel_hours=8.0,
            terrain_capability=TerrainCapability.ALL_ROADS,
        )
    return [Vehicle(id=f"v{i}", name=f"Van {i}", vehicle_type=vtype) for i in range(1, n + 1)]


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture(scope="module")
def perf_data():
    """Load all data once per module (expensive I/O)."""
    config_data, terrain_graph, villages_raw, resource_types = _load_config()
    sm       = _make_sm(config_data, terrain_graph, resource_types)
    villages = _make_real_villages(villages_raw)
    vehicles = _make_fleet(3)
    return {
        "sm":             sm,
        "villages":       villages,
        "vehicles":       vehicles,
        "config_data":    config_data,
        "terrain_graph":  terrain_graph,
        "resource_types": resource_types,
        "depot_resources": {rt: 500.0 for rt in resource_types},
    }


@pytest.fixture
def fresh_villages(perf_data):
    """Return a copy of villages with original urgency scores (test isolation)."""
    orig = perf_data["villages"]
    return [v.model_copy(update={}) for v in orig]


# ================================================================== #
#  1. Optimization Speed                                               #
# ================================================================== #

class TestOptimizationSpeed:
    def test_full_pipeline_under_15s(self, perf_data, fresh_villages):
        sm, vehicles = perf_data["sm"], perf_data["vehicles"]
        with PerformanceTimer("Full pipeline (8 villages, 3 vehicles)") as t:
            result = sm.run_full_optimization(
                villages=fresh_villages,
                vehicles=vehicles,
                time_elapsed=timedelta(hours=2),
            )
        assert result.state == OptimizationState.COMPLETE
        t.assert_under(TARGET_FULL_PIPELINE_S,
                       "Optimization too slow for live demo")

    def test_execution_time_recorded_in_result(self, perf_data, fresh_villages):
        result = perf_data["sm"].run_full_optimization(
            villages=fresh_villages,
            vehicles=perf_data["vehicles"],
            time_elapsed=timedelta(hours=2),
        )
        # execution_time_seconds uses datetime.now() which has ms precision;
        # sub-ms runs legitimately report 0.0 — assert non-negative, not > 0
        assert result.execution_time_seconds >= 0.0
        assert result.execution_time_seconds < TARGET_FULL_PIPELINE_S
        print(f"  [PERF] Reported: {result.execution_time_seconds:.3f}s")

    def test_result_is_kkt_certified_within_budget(self, perf_data, fresh_villages):
        with PerformanceTimer("Pipeline + KKT") as t:
            result = perf_data["sm"].run_full_optimization(
                villages=fresh_villages,
                vehicles=perf_data["vehicles"],
                time_elapsed=timedelta(hours=2),
            )
        assert result.kkt_verification is not None
        assert result.kkt_verification.all_conditions_satisfied
        t.assert_under(TARGET_FULL_PIPELINE_S)


# ================================================================== #
#  2. Scaling — village count                                          #
# ================================================================== #

class TestScalingVillages:
    @pytest.mark.parametrize("n,limit_s", [(4, 5.0), (8, 15.0)])
    def test_pipeline_scales_with_village_count(self, perf_data, n, limit_s):
        config_data   = perf_data["config_data"]
        terrain_graph = perf_data["terrain_graph"]
        resource_types = perf_data["resource_types"]

        sm       = _make_sm(config_data, terrain_graph, resource_types)
        villages = _make_synthetic_villages(n)
        vehicles = _make_fleet(3)

        with PerformanceTimer(f"Pipeline ({n} villages)") as t:
            result = sm.run_full_optimization(
                villages=villages,
                vehicles=vehicles,
                time_elapsed=timedelta(hours=2),
            )
        assert result.state == OptimizationState.COMPLETE
        t.assert_under(limit_s, f"{n} villages exceeded {limit_s}s budget")


# ================================================================== #
#  3. Scaling — vehicle count                                          #
# ================================================================== #

class TestScalingVehicles:
    @pytest.mark.parametrize("n,limit_s", [(3, 5.0), (9, 15.0), (13, 15.0)])
    def test_pipeline_scales_with_vehicle_count(self, perf_data, fresh_villages, n, limit_s):
        sm       = perf_data["sm"]
        vehicles = _make_fleet(n)

        with PerformanceTimer(f"Pipeline ({n} vehicles)") as t:
            result = sm.run_full_optimization(
                villages=fresh_villages,
                vehicles=vehicles,
                time_elapsed=timedelta(hours=2),
            )
        assert result.state == OptimizationState.COMPLETE
        t.assert_under(limit_s, f"{n} vehicles exceeded {limit_s}s budget")


# ================================================================== #
#  4. Memory — no leak over repeated runs                              #
# ================================================================== #

class TestMemory:
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_no_memory_leak_over_20_runs(self, perf_data):
        sm       = perf_data["sm"]
        vehicles = _make_fleet(3)
        villages = _make_synthetic_villages(4)  # small dataset — keeps test fast

        gc.collect()
        baseline = _memory_mb()
        print(f"\n  [MEM] Baseline: {baseline:.1f} MB")

        for i in range(20):
            result = sm.run_full_optimization(
                villages=[v.model_copy(update={}) for v in villages],
                vehicles=vehicles,
                time_elapsed=timedelta(hours=2),
            )
            assert result.state == OptimizationState.COMPLETE
            del result
            if i % 5 == 4:
                gc.collect()

        gc.collect()
        final   = _memory_mb()
        growth  = final - baseline
        print(f"  [MEM] Final: {final:.1f} MB  Growth: {growth:+.1f} MB over 20 runs")

        assert growth < MEMORY_LEAK_THRESHOLD_MB, (
            f"Memory grew {growth:.1f} MB over 20 runs (limit: {MEMORY_LEAK_THRESHOLD_MB} MB)"
        )


# ================================================================== #
#  5. Timeline simulation throughput                                   #
# ================================================================== #

class TestTimelineThroughput:
    def test_timeline_under_1s_per_event(self, perf_data):
        analyzer = NewsAnalyzer()
        queue    = ApprovalQueue()
        cfg      = SimulatorConfig(
            timeline_path=TIMELINE_PATH,
            speed_multiplier=1000.0,
            auto_approve_hitl=True,
            trigger_reoptimization=False,
            verbose_logging=False,
        )
        sim = TimelineSimulator(
            config=cfg,
            news_analyzer=analyzer,
            approval_queue=queue,
            villages=perf_data["villages"],
        )
        sim.load_timeline()
        n = len(sim.timeline)

        with PerformanceTimer(f"Timeline ({n} events)") as t:
            sim.start_simulation()
            sim._thread.join(timeout=30.0)

        assert sim.state.events_processed == n
        per_event = t.elapsed / n
        print(f"  [PERF] Per-event: {per_event:.3f}s")
        assert per_event < TARGET_PER_EVENT_S, (
            f"Event processing too slow: {per_event:.3f}s > {TARGET_PER_EVENT_S}s"
        )

    def test_news_analyzer_batch_speed(self, perf_data):
        analyzer = NewsAnalyzer()
        villages = perf_data["villages"]
        texts = [
            "Earthquake in Taplejung. Government relief requested.",
            "Landslide near Jumla. Road access cut off.",
            "Flooding in Janakpur. Families displaced.",
            "Hospital overwhelmed in Nepalgunj. Medical supplies needed.",
            "Bridge collapse confirmed near Jumla by authorities.",
        ]

        with PerformanceTimer("5 analyze_news calls") as t:
            for text in texts:
                report = analyzer.analyze_news(
                    raw_text=text,
                    villages=villages,
                    source="Nepal Police",
                    multi_source_confirmed=True,
                )
                assert report is not None

        per_call = t.elapsed / len(texts)
        print(f"  [PERF] Per analyze_news: {per_call:.3f}s")
        assert per_call < TARGET_PER_EVENT_S


# ================================================================== #
#  6. Stress — rapid re-optimizations                                  #
# ================================================================== #

class TestStress:
    def test_5_rapid_reoptimizations_no_degradation(self, perf_data):
        sm       = perf_data["sm"]
        vehicles = _make_fleet(3)
        timings  = []

        for i in range(5):
            # Slightly bump urgency each run to mimic live updates
            villages = _make_synthetic_villages(6, base_urgency=0.5 + i * 0.05)

            with PerformanceTimer(f"Reopt #{i + 1}") as t:
                result = sm.run_full_optimization(
                    villages=villages,
                    vehicles=vehicles,
                    time_elapsed=timedelta(hours=2 + i),
                )
            assert result.state == OptimizationState.COMPLETE
            timings.append(t.elapsed)

        avg_first = sum(timings[:2]) / 2
        avg_last  = sum(timings[-2:]) / 2
        delta     = avg_last - avg_first
        print(f"  [PERF] First-2 avg: {avg_first:.3f}s  Last-2 avg: {avg_last:.3f}s  "
              f"Delta: {delta:+.3f}s")

        # Allow 50% relative degradation OR 2s absolute, whichever is larger.
        # When avg_first rounds to 0.0 (sub-ms), the absolute 2s floor applies.
        tolerance = max(avg_first * 0.5, 2.0)
        assert delta < tolerance, (
            f"Performance degraded by {delta:.3f}s across 5 rapid reoptimizations"
        )

    def test_all_5_reoptimizations_complete(self, perf_data):
        sm       = perf_data["sm"]
        vehicles = _make_fleet(3)
        completed = 0

        for i in range(5):
            villages = _make_synthetic_villages(4, base_urgency=0.6)
            result   = sm.run_full_optimization(
                villages=villages,
                vehicles=vehicles,
                time_elapsed=timedelta(hours=1),
            )
            if result.state == OptimizationState.COMPLETE:
                completed += 1

        assert completed == 5


# ================================================================== #
#  7. Component benchmarks + benchmark.json                           #
# ================================================================== #

class TestComponentBenchmarks:
    def test_urgency_calculator_speed(self, perf_data):
        calc     = UrgencyCalculator(perf_data["resource_types"])
        villages = _make_synthetic_villages(8)

        with PerformanceTimer("UrgencyCalculator.rank_villages") as t:
            scores = calc.rank_villages(villages, timedelta(hours=2))

        assert len(scores) == 8
        t.assert_under(TARGET_COMPONENT_BENCH_S)

    def test_vrp_solver_speed(self, perf_data):
        config_data   = perf_data["config_data"]
        terrain_graph = perf_data["terrain_graph"]
        resource_types = perf_data["resource_types"]
        villages       = _make_synthetic_villages(8)
        vehicles       = _make_fleet(3)

        calc   = UrgencyCalculator(resource_types)
        scores = calc.rank_villages(villages, timedelta(hours=2))

        solver = VRPSolver(
            depot_location=(27.7172, 85.3240),
            terrain_graph=terrain_graph,
            resource_types=resource_types,
            config=config_data,
        )

        with PerformanceTimer("VRPSolver.solve") as t:
            vrp = solver.solve(
                villages=villages,
                vehicles=vehicles,
                urgency_scores=scores,
                available_resources=perf_data["depot_resources"],
            )

        assert vrp is not None
        t.assert_under(TARGET_COMPONENT_BENCH_S)

    def test_nash_solver_speed(self, perf_data):
        config_data    = perf_data["config_data"]
        terrain_graph  = perf_data["terrain_graph"]
        resource_types = perf_data["resource_types"]
        depot_res      = perf_data["depot_resources"]
        villages       = _make_synthetic_villages(8)
        vehicles       = _make_fleet(3)

        calc   = UrgencyCalculator(resource_types)
        scores = calc.rank_villages(villages, timedelta(hours=2))
        solver = VRPSolver((27.7172, 85.3240), terrain_graph, resource_types, config_data)
        vrp    = solver.solve(villages, vehicles, scores, depot_res)

        nash = NashSolver(depot_res, resource_types)

        with PerformanceTimer("NashSolver.solve") as t:
            eq = nash.solve(villages, vrp)

        assert eq is not None
        t.assert_under(TARGET_COMPONENT_BENCH_S)

    def test_benchmark_suite_saves_json(self, perf_data):
        config_data    = perf_data["config_data"]
        terrain_graph  = perf_data["terrain_graph"]
        resource_types = perf_data["resource_types"]
        depot_res      = perf_data["depot_resources"]
        villages       = _make_synthetic_villages(8)
        vehicles       = _make_fleet(3)

        ops: Dict[str, float] = {}

        # Stage 1: Urgency
        calc = UrgencyCalculator(resource_types)
        with PerformanceTimer("urgency_calculator") as t:
            scores = calc.rank_villages(villages, timedelta(hours=2))
        ops["urgency_calculator_ms"] = round(t.elapsed * 1000, 1)

        # Stage 2: VRP
        vrp_solver = VRPSolver((27.7172, 85.3240), terrain_graph, resource_types, config_data)
        with PerformanceTimer("vrp_solver") as t:
            vrp = vrp_solver.solve(villages, vehicles, scores, depot_res)
        ops["vrp_solver_ms"] = round(t.elapsed * 1000, 1)

        # Stage 3: Nash
        nash_solver = NashSolver(depot_res, resource_types)
        with PerformanceTimer("nash_solver") as t:
            eq = nash_solver.solve(villages, vrp)
        ops["nash_solver_ms"] = round(t.elapsed * 1000, 1)

        # Stage 4: KKT
        kkt_verifier = KKTVerifier(resource_types)
        with PerformanceTimer("kkt_verifier") as t:
            kkt = kkt_verifier.verify(eq, villages, depot_res)
        ops["kkt_verifier_ms"] = round(t.elapsed * 1000, 1)

        # Stage 5: Full pipeline
        sm = perf_data["sm"]
        fresh = _make_synthetic_villages(8)
        with PerformanceTimer("full_pipeline") as t:
            result = sm.run_full_optimization(fresh, vehicles, timedelta(hours=2))
        ops["full_pipeline_ms"] = round(t.elapsed * 1000, 1)

        benchmark = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset":   {"villages": 8, "vehicles": len(vehicles)},
            "targets":   {
                "full_pipeline_ms": TARGET_FULL_PIPELINE_S * 1000,
                "per_event_ms":     TARGET_PER_EVENT_S * 1000,
            },
            "operations": ops,
        }

        out = Path(__file__).parent / "benchmark.json"
        out.write_text(json.dumps(benchmark, indent=2))
        print(f"\n  [BENCH] Saved to {out}")
        print(f"  {json.dumps(ops, indent=4)}")

        assert ops["full_pipeline_ms"] < TARGET_FULL_PIPELINE_S * 1000
        assert out.exists()
