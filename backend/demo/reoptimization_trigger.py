"""
Re-optimization Trigger -- Prompt 5.2

Connects the Timeline Simulator to the State Manager pipeline so that
high-confidence news events and HITL approvals actually trigger route
re-calculation and update village urgency scores.

Typical usage
-------------
    trigger = ReoptimizationTrigger(
        config=ReoptimizationConfig(),
        state_manager=state_manager,
        villages=villages,
        vehicles=vehicles,
    )
    change = trigger.trigger_reoptimization(intelligence_report)
    print(f"Routes changed: {change.routes_changed}")
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from backend.algorithms.state_manager import StateManager
from backend.algorithms.vrp_solver import VRPSolution
from backend.models.vehicle import Vehicle
from backend.models.village import Village
from backend.p2p.gossip_protocol import GossipProtocol
from backend.rag.news_analyzer import IntelligenceReport


# ================================================================== #
#  Config and output models                                            #
# ================================================================== #

class ReoptimizationConfig(BaseModel):
    """Controls when and how re-optimization is triggered."""
    urgency_change_threshold:  float = Field(default=0.10, ge=0.0)
    enable_reoptimization:     bool  = True
    broadcast_via_p2p:         bool  = False
    log_optimization_changes:  bool  = True


class OptimizationChange(BaseModel):
    """Records what changed between two optimization runs."""
    trigger_event_id:    str
    triggered_at:        str
    urgency_changes:     Dict[str, float]       # village_id -> delta
    old_welfare:         float = 0.0
    new_welfare:         float = 0.0
    welfare_improvement: float = 0.0            # new_welfare - old_welfare
    routes_changed:      int   = 0
    execution_time_ms:   float = 0.0
    optimization_state:  str   = "complete"


# ================================================================== #
#  ReoptimizationTrigger                                               #
# ================================================================== #

class ReoptimizationTrigger:
    """
    Bridges IntelligenceReports to the StateManager optimization pipeline.

    Args:
        config:          Trigger configuration.
        state_manager:   StateManager that runs the 4-step pipeline.
        villages:        Live village list (mutated in-place for urgency updates).
        vehicles:        Fleet for the VRP solver.
        gossip_protocol: Optional — used for P2P broadcast after reopt.
        time_elapsed:    Time elapsed since disaster onset (fed to UrgencyCalculator).
    """

    def __init__(
        self,
        config:           ReoptimizationConfig,
        state_manager:    StateManager,
        villages:         List[Village],
        vehicles:         List[Vehicle],
        gossip_protocol:  Optional[GossipProtocol] = None,
        time_elapsed:     timedelta = timedelta(hours=2),
    ) -> None:
        self.config          = config
        self.state_manager   = state_manager
        self.villages        = villages
        self.vehicles        = vehicles
        self.gossip_protocol = gossip_protocol
        self.time_elapsed    = time_elapsed

        self.optimization_history: List[OptimizationChange] = []
        self._last_vrp_solution:   Optional[VRPSolution]    = None

    # -------------------------------------------------------------- #
    #  Public API                                                     #
    # -------------------------------------------------------------- #

    def should_trigger_reoptimization(
        self,
        urgency_changes: Dict[str, float],
    ) -> bool:
        """
        Return True if any urgency delta meets or exceeds the threshold.

        A zero or empty dict never triggers re-optimization.
        """
        if not self.config.enable_reoptimization or not urgency_changes:
            return False
        max_delta = max(abs(d) for d in urgency_changes.values())
        return max_delta >= self.config.urgency_change_threshold

    def trigger_reoptimization(
        self,
        report: IntelligenceReport,
    ) -> OptimizationChange:
        """
        Run the full 4-step pipeline in response to an IntelligenceReport.

        Steps:
          1. Apply urgency_change deltas to village scores.
          2. Run StateManager.run_full_optimization().
          3. Compare new VRP solution to the previous one.
          4. Optionally broadcast via P2P.
          5. Store in history and return OptimizationChange.

        Raises:
            ValueError: If the report has no urgency changes.
        """
        if not report.urgency_change:
            raise ValueError(
                f"Report for {report.event.event_id} has no urgency_change dict"
            )

        old_welfare = (
            self._last_vrp_solution.objective_value
            if self._last_vrp_solution is not None
            else 0.0
        )
        old_vrp = self._last_vrp_solution

        if self.config.log_optimization_changes:
            print(f"\n  [REOPT] Triggered by {report.event.event_id}")

        # Step 1: Apply urgency deltas
        self.apply_urgency_updates(report.urgency_change)

        # Step 2: Run pipeline
        t0     = time.time()
        result = self.state_manager.run_full_optimization(
            villages=self.villages,
            vehicles=self.vehicles,
            time_elapsed=self.time_elapsed,
        )
        exec_ms = (time.time() - t0) * 1000.0

        new_vrp = result.vrp_solution
        self._last_vrp_solution = new_vrp

        new_welfare = new_vrp.objective_value if new_vrp is not None else 0.0
        welfare_delta = new_welfare - old_welfare

        routes_changed = self._count_route_changes(old_vrp, new_vrp)

        if self.config.log_optimization_changes:
            print(f"  [REOPT] done in {exec_ms:.0f} ms  "
                  f"routes_changed={routes_changed}  "
                  f"welfare_delta={welfare_delta:+.4f}")

        change = OptimizationChange(
            trigger_event_id=report.event.event_id,
            triggered_at=datetime.now(timezone.utc).isoformat(),
            urgency_changes=dict(report.urgency_change),
            old_welfare=old_welfare,
            new_welfare=new_welfare,
            welfare_improvement=welfare_delta,
            routes_changed=routes_changed,
            execution_time_ms=exec_ms,
            optimization_state=result.state.value,
        )

        self.optimization_history.append(change)

        if self.config.broadcast_via_p2p and self.gossip_protocol and new_vrp:
            self._broadcast_solution(new_vrp, report.event.event_id)

        return change

    def apply_urgency_updates(self, urgency_changes: Dict[str, float]) -> None:
        """
        Apply urgency deltas to the live village list (in-place mutation).

        Values are clamped to [0.0, 1.0].
        Unknown village IDs are logged and skipped.
        """
        village_map = {v.id: v for v in self.villages}
        for village_id, delta in urgency_changes.items():
            village = village_map.get(village_id)
            if village is None:
                if self.config.log_optimization_changes:
                    print(f"  [REOPT] Warning: village '{village_id}' not found, skipping")
                continue
            old_score = village.urgency_score
            new_score = max(0.0, min(1.0, old_score + delta))
            village.urgency_score = new_score
            if self.config.log_optimization_changes:
                print(f"  [REOPT]   {village_id}: {old_score:.3f} -> {new_score:.3f} "
                      f"({delta:+.3f})")

    def get_optimization_history(self) -> List[OptimizationChange]:
        """Return all recorded optimizations, newest first."""
        return sorted(
            self.optimization_history,
            key=lambda c: c.triggered_at,
            reverse=True,
        )

    # -------------------------------------------------------------- #
    #  Internal helpers                                               #
    # -------------------------------------------------------------- #

    def _count_route_changes(
        self,
        old_sol: Optional[VRPSolution],
        new_sol: Optional[VRPSolution],
    ) -> int:
        """Count vehicles whose stop sequence changed between solutions."""
        if new_sol is None:
            return 0
        if old_sol is None:
            return len(new_sol.routes)

        old_routes = {r.vehicle_id: r.stops for r in old_sol.routes}
        changes = 0
        for route in new_sol.routes:
            old_stops = old_routes.get(route.vehicle_id)
            if old_stops is None or old_stops != route.stops:
                changes += 1
        return changes

    def _broadcast_solution(
        self,
        solution: VRPSolution,
        event_id: str,
    ) -> None:
        """Broadcast the new VRP solution over the P2P gossip network."""
        if not self.gossip_protocol:
            return
        payload = {
            "type":      "SOLUTION_UPDATE",
            "event_id":  event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "routes": [
                {
                    "vehicle_id": r.vehicle_id,
                    "stops":      r.stops,
                    "distance_km": r.total_distance_km,
                }
                for r in solution.routes
            ],
        }
        from backend.p2p.gossip_protocol import MSG_OPTIMIZATION_RESULT
        self.gossip_protocol.broadcast_message(payload, MSG_OPTIMIZATION_RESULT)
        if self.config.log_optimization_changes:
            print("  [REOPT] Broadcasted to P2P network")