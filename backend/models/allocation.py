"""
AllocationResult — output of the capped proportional allocation solver.

Named `nash_solver` for backward compatibility only. It is not a Nash
equilibrium and does not model a strategic game; see MATH.md section 5.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RouteWaypoint(BaseModel):
    village_id: str
    lat: float
    lng: float
    eta_minutes: float
    cargo_kg: float


class VehicleRoute(BaseModel):
    vehicle_id: str
    waypoints: List[RouteWaypoint] = Field(default_factory=list)
    total_distance_km: float = 0.0
    fuel_required_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    total_cargo_kg: float = 0.0


class KKTConditions(BaseModel):
    """Reported KKT checks; these fields alone are not an optimality proof."""
    stationarity: bool = False          # ||∇L(x*)|| < 1e-6
    primal_feasibility: bool = False    # All constraints satisfied
    dual_feasibility: bool = False      # All multipliers ≥ 0
    complementary_slackness: bool = False  # λᵢ * gᵢ(x*) < 1e-6
    residual: float = float("inf")      # ||∇L|| value

    @property
    def all_satisfied(self) -> bool:
        return (
            self.stationarity
            and self.primal_feasibility
            and self.dual_feasibility
            and self.complementary_slackness
        )


class ConvergencePoint(BaseModel):
    iteration: int
    objective_value: float
    max_constraint_violation: float


class AllocationResult(BaseModel):
    # Core allocation: {vehicle_id: {village_id: amount_kg}}
    allocation: Dict[str, Dict[str, float]] = Field(default_factory=dict)

    # Legacy names: neither field independently proves global optimality.
    nash_equilibrium_reached: bool = False
    kkt_conditions: KKTConditions = Field(default_factory=KKTConditions)
    convergence_data: List[ConvergencePoint] = Field(default_factory=list)

    # Routes: {vehicle_id: VehicleRoute}
    routes: Dict[str, VehicleRoute] = Field(default_factory=dict)

    # Re-optimization metadata
    reoptimization_triggered: bool = False
    triggering_village_id: Optional[str] = None
    urgency_delta_that_triggered: Optional[float] = None

    # Performance
    solve_time_seconds: float = 0.0
    objective_value: Optional[float] = None
    solver_status: str = "NOT_RUN"  # OPTIMAL | INFEASIBLE | TIMEOUT | NOT_RUN
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def total_allocated_to(self, village_id: str) -> float:
        """Sum of all vehicle allocations to a village."""
        return sum(
            vehicle_alloc.get(village_id, 0.0)
            for vehicle_alloc in self.allocation.values()
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "nash_equilibrium_reached": self.nash_equilibrium_reached,
            "kkt_all_satisfied": self.kkt_conditions.all_satisfied,
            "kkt_residual": self.kkt_conditions.residual,
            "solve_time_seconds": self.solve_time_seconds,
            "solver_status": self.solver_status,
            "reoptimization_triggered": self.reoptimization_triggered,
            "vehicles_deployed": len(self.allocation),
            "total_cargo_kg": sum(
                sum(v.values()) for v in self.allocation.values()
            ),
        }
