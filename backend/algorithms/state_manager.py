"""
State Manager -- Prompt 2.5

Orchestrates the full optimization pipeline:
  Urgency -> greedy routing -> proportional allocation
  -> continuous social-welfare comparison -> allocation diagnostics

Maintains observable state so the frontend can poll progress.
Each run_full_optimization() call is independent (stateless between calls).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from backend.models.resource import ResourceType
from backend.models.vehicle import Vehicle
from backend.models.village import Village
from backend.algorithms.urgency_calculator import UrgencyCalculator, UrgencyScore
from backend.algorithms.vrp_solver import VRPSolver, VRPSolution
from backend.algorithms.nash_solver import NashSolver, NashEquilibrium
from backend.algorithms.kkt_verifier import KKTVerifier, KKTVerificationResult
from backend.algorithms.social_welfare_optimizer import (
    AllocationMethodComparison,
    SocialWelfareAllocationResult,
    SocialWelfareOptimizer,
)


# ------------------------------------------------------------------ #
#  State enum                                                          #
# ------------------------------------------------------------------ #

class OptimizationState(str, Enum):
    IDLE               = "idle"
    CALCULATING_URGENCY = "calculating_urgency"
    SOLVING_VRP        = "solving_vrp"
    SOLVING_NASH       = "solving_nash"
    OPTIMIZING_SOCIAL_WELFARE = "optimizing_social_welfare"
    VERIFYING_KKT      = "verifying_kkt"
    COMPLETE           = "complete"
    ERROR              = "error"


# ------------------------------------------------------------------ #
#  Result model                                                        #
# ------------------------------------------------------------------ #

class OptimizationResult(BaseModel):
    state: OptimizationState = OptimizationState.IDLE
    urgency_scores: List[UrgencyScore] = Field(default_factory=list)
    vrp_solution: Optional[VRPSolution] = None
    # Backward-compatible field name. The object is capped proportional allocation.
    nash_equilibrium: Optional[NashEquilibrium] = None
    social_welfare_allocation: Optional[SocialWelfareAllocationResult] = None
    allocation_comparison: Optional[AllocationMethodComparison] = None
    kkt_verification: Optional[KKTVerificationResult] = None
    execution_time_seconds: float = 0.0
    timestamp: str = ""
    error_message: Optional[str] = None
    gemma_signal: Dict[str, object] = Field(default_factory=dict)
    resource_snapshot: Dict[str, object] = Field(default_factory=dict)
    fleet_snapshot: List[Dict[str, object]] = Field(default_factory=list)


# ------------------------------------------------------------------ #
#  StateManager                                                        #
# ------------------------------------------------------------------ #

class StateManager:
    """
    Orchestrates urgency, routing, allocation, comparison, and diagnostics.

    Args:
        depot_location:  (lat, lng) of the central depot.
        depot_resources: {resource_type: amount_kg} available at depot.
        terrain_graph:   Raw terrain_graph.json dict (passed to VRPSolver).
        resource_types:  Dict[resource_id -> ResourceType].
        config:          Full config.json dict.
    """

    def __init__(
        self,
        depot_location: Tuple[float, float],
        depot_resources: Dict[str, float],
        terrain_graph: Dict,
        resource_types: Dict[str, ResourceType],
        config: Dict,
    ) -> None:
        self.depot_location  = depot_location
        self.depot_resources = dict(depot_resources)
        self.terrain_graph   = terrain_graph
        self.resource_types  = resource_types
        self.config          = config
        self._state          = OptimizationState.IDLE

    # ---------------------------------------------------------------- #
    #  State inspection                                                  #
    # ---------------------------------------------------------------- #

    def get_state(self) -> OptimizationState:
        """Return current pipeline state (observable by frontend)."""
        return self._state

    def reset(self) -> None:
        """Reset to IDLE so a new optimization can be started."""
        self._state = OptimizationState.IDLE

    # ---------------------------------------------------------------- #
    #  Full pipeline                                                     #
    # ---------------------------------------------------------------- #

    def run_full_optimization(
        self,
        villages: List[Village],
        vehicles: List[Vehicle],
        time_elapsed: timedelta,
        blocked_edge_ids: Optional[List[str]] = None,
        terrain_weighting: bool = True,
        honour_closures: bool = True,
    ) -> OptimizationResult:
        """
        Execute the optimization pipeline.

        Steps:
          1. UrgencyCalculator  -> urgency_scores
          2. VRPSolver          -> vrp_solution
          3. NashSolver         -> capped proportional allocation
          4. SocialWelfareOptimizer -> fairness-aware continuous candidate
          5. KKTVerifier        -> scoped diagnostics for the legacy candidate

        Returns OptimizationResult with all outputs.
        On any exception, state = ERROR and error_message is set.
        """
        start = datetime.now()

        # Partial result accumulated across steps
        urgency_scores: List[UrgencyScore]           = []
        vrp_solution:   Optional[VRPSolution]        = None
        nash_eq:        Optional[NashEquilibrium]     = None
        social_welfare: Optional[SocialWelfareAllocationResult] = None
        comparison:     Optional[AllocationMethodComparison] = None
        kkt_result:     Optional[KKTVerificationResult] = None

        try:
            # -------------------------------------------------------- #
            #  Step 1: Urgency calculation                              #
            # -------------------------------------------------------- #
            self._state = OptimizationState.CALCULATING_URGENCY
            calc = UrgencyCalculator(resource_types=self.resource_types)
            urgency_scores = calc.rank_villages(villages, time_elapsed)

            # -------------------------------------------------------- #
            #  Step 2: VRP routing                                      #
            # -------------------------------------------------------- #
            self._state = OptimizationState.SOLVING_VRP
            vrp_solver = VRPSolver(
                depot_location=self.depot_location,
                terrain_graph=self.terrain_graph,
                resource_types=self.resource_types,
                config=self.config,
                blocked_edge_ids=blocked_edge_ids,
                terrain_weighting=terrain_weighting,
                honour_closures=honour_closures,
            )
            vrp_solution = vrp_solver.solve(
                villages=villages,
                vehicles=vehicles,
                urgency_scores=urgency_scores,
                available_resources=self.depot_resources,
            )

            # -------------------------------------------------------- #
            #  Step 3: capped proportional allocation                   #
            # -------------------------------------------------------- #
            self._state = OptimizationState.SOLVING_NASH
            nash_solver = NashSolver(
                depot_resources=self.depot_resources,
                resource_types=self.resource_types,
            )
            nash_eq = nash_solver.solve(villages=villages, vrp_solution=vrp_solution)

            # -------------------------------------------------------- #
            #  Step 4: weighted Nash social-welfare comparison          #
            # -------------------------------------------------------- #
            self._state = OptimizationState.OPTIMIZING_SOCIAL_WELFARE
            welfare_optimizer = SocialWelfareOptimizer(
                depot_resources=self.depot_resources,
                resource_types=self.resource_types,
            )
            social_welfare, comparison = welfare_optimizer.solve(
                villages=villages,
                urgency_scores=urgency_scores,
                proportional=nash_eq,
            )

            # -------------------------------------------------------- #
            #  Step 5: allocation-only KKT consistency diagnostics      #
            # -------------------------------------------------------- #
            self._state = OptimizationState.VERIFYING_KKT
            verifier = KKTVerifier(resource_types=self.resource_types)
            kkt_result = verifier.verify(
                nash_solution=nash_eq,
                villages=villages,
                depot_resources=self.depot_resources,
            )

            self._state = OptimizationState.COMPLETE

        except Exception as exc:  # noqa: BLE001
            self._state = OptimizationState.ERROR
            elapsed = (datetime.now() - start).total_seconds()
            return OptimizationResult(
                state=self._state,
                urgency_scores=urgency_scores,
                vrp_solution=vrp_solution,
                nash_equilibrium=nash_eq,
                social_welfare_allocation=social_welfare,
                allocation_comparison=comparison,
                kkt_verification=kkt_result,
                execution_time_seconds=elapsed,
                timestamp=datetime.now(timezone.utc).isoformat(),
                error_message=str(exc),
                resource_snapshot=self._resource_snapshot(villages),
                fleet_snapshot=self._fleet_snapshot(vehicles, vrp_solution),
            )

        elapsed = (datetime.now() - start).total_seconds()
        return OptimizationResult(
            state=self._state,
            urgency_scores=urgency_scores,
            vrp_solution=vrp_solution,
            nash_equilibrium=nash_eq,
            social_welfare_allocation=social_welfare,
            allocation_comparison=comparison,
            kkt_verification=kkt_result,
            execution_time_seconds=elapsed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            resource_snapshot=self._resource_snapshot(villages),
            fleet_snapshot=self._fleet_snapshot(vehicles, vrp_solution),
        )

    def _resource_snapshot(self, villages: List[Village]) -> Dict[str, object]:
        demand = {resource_id: 0.0 for resource_id in self.resource_types}
        existing = {resource_id: 0.0 for resource_id in self.resource_types}
        minimum = {resource_id: 0.0 for resource_id in self.resource_types}
        for village in villages:
            for resource_id, need in village.resource_needs.items():
                demand[resource_id] = demand.get(resource_id, 0.0) + need.current_need
                existing[resource_id] = existing.get(resource_id, 0.0) + need.allocated
                minimum[resource_id] = minimum.get(resource_id, 0.0) + need.min_need

        return {
            "source_kind": "bundled_scenario_fixture",
            "source_label": "Mocked hackathon scenario data",
            "source_file": "backend/data/nepal_villages.json",
            "depot_available": dict(self.depot_resources),
            "reported_demand": demand,
            "existing_field_allocations": existing,
            "survival_thresholds": minimum,
            "resource_types": {
                resource_id: {
                    "name": resource.name,
                    "category": resource.category.value,
                    "unit": resource.unit,
                    "weight_per_unit_kg": resource.weight_per_unit,
                    "urgency_multiplier": resource.urgency_multiplier,
                }
                for resource_id, resource in self.resource_types.items()
            },
        }

    @staticmethod
    def _fleet_snapshot(
        vehicles: List[Vehicle],
        solution: Optional[VRPSolution],
    ) -> List[Dict[str, object]]:
        routes = {
            route.vehicle_id: route
            for route in (solution.routes if solution is not None else [])
        }
        return [
            {
                "vehicle_id": vehicle.id,
                "name": vehicle.name,
                "category": vehicle.vehicle_type.category.value,
                "capacity_kg": vehicle.vehicle_type.capacity_kg,
                "speed_kmh": vehicle.vehicle_type.speed_kmh,
                "fuel_hours": vehicle.vehicle_type.fuel_hours,
                "terrain_capability": vehicle.vehicle_type.terrain_capability.value,
                "status": "assigned" if vehicle.id in routes else "available",
                "assigned_route": (
                    {
                        "stops": routes[vehicle.id].stops,
                        "total_distance_km": routes[vehicle.id].total_distance_km,
                        "total_time_minutes": routes[vehicle.id].total_time_minutes,
                        "total_cargo_kg": routes[vehicle.id].total_cargo_kg,
                        "transport_mode": routes[vehicle.id].transport_mode,
                        "rerouted_due_to": routes[vehicle.id].rerouted_due_to,
                    }
                    if vehicle.id in routes
                    else None
                ),
                "source_kind": "bundled_scenario_fixture",
            }
            for vehicle in vehicles
        ]
