"""Typed API contracts for optimization runs and operator decisions."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from backend.algorithms.state_manager import OptimizationResult
from backend.models.orchestration import OrchestrationRecord


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OptimizationRunStatus(str, Enum):
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"


class VehiclePosition(BaseModel):
    lat: float = Field(ge=-90.0, le=90.0)
    lng: float = Field(ge=-180.0, le=180.0)
    # Stops this asset has already served. They are excluded from re-planning,
    # because re-delivering to a location already served wastes the asset.
    served_stops: List[str] = Field(default_factory=list, max_length=50)


class OptimizationRunRequest(BaseModel):
    scenario_id: str = Field(default="nepal-national-demo", min_length=1, max_length=100)
    time_elapsed_hours: float = Field(default=2.0, ge=0.0, le=168.0)
    requested_by: str = Field(default="operator", min_length=1, max_length=100)
    analysis_id: Optional[str] = None
    blocked_edge_ids: List[str] = Field(default_factory=list, max_length=50)
    parent_run_id: Optional[str] = Field(default=None, max_length=100)
    trigger: str = Field(default="manual", min_length=1, max_length=100)
    disruption_reason: Optional[str] = Field(default=None, max_length=500)
    # Where each asset actually is when re-planning starts. Without this, every
    # re-optimization silently teleported the whole fleet back to its depot
    # position, so a plan computed mid-mission routed vehicles from where they
    # began rather than from where they are. Keyed by vehicle id.
    vehicle_positions: Dict[str, VehiclePosition] = Field(default_factory=dict)


class OptimizationDecisionRequest(BaseModel):
    reviewer: str = Field(default="operator", min_length=1, max_length=100)
    notes: Optional[str] = Field(default=None, max_length=2000)
    expected_updated_at: str = Field(min_length=1, max_length=100)
    expected_analysis_id: str = Field(min_length=1, max_length=100)


class OptimizationRunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: f"opt_{uuid4().hex[:12]}")
    scenario_id: str
    correlation_id: str = Field(default_factory=lambda: f"corr_{uuid4().hex[:12]}")
    analysis_id: Optional[str] = None
    requested_by: str
    blocked_edge_ids: List[str] = Field(default_factory=list)
    parent_run_id: Optional[str] = None
    trigger: str = "manual"
    disruption_reason: Optional[str] = None
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    status: OptimizationRunStatus = OptimizationRunStatus.RUNNING
    allocation_method: str = "proportional_allocation"
    comparison_allocation_method: str = "weighted_nash_social_welfare"
    routing_method: str = "greedy_urgency_nearest_neighbour"
    routing_engine: str = "multimodal_capability_constrained_road_graph_v2"
    diagnostic_scope: str = (
        "Feasibility and partial KKT consistency diagnostics; "
        "not an independent global-optimality proof."
    )
    route_feasible: Optional[bool] = None
    approval_blockers: List[str] = Field(default_factory=list)
    requires_human_approval: bool = True
    reviewed_by: Optional[str] = None
    review_notes: Optional[str] = None
    reviewed_at: Optional[str] = None
    result: Optional[OptimizationResult] = None
    error: Optional[str] = None
    # Present when this run was requested by Gemma through native function
    # calling rather than by an operator pressing a button. It carries the exact
    # arguments the model produced and the checks they passed.
    orchestration: Optional[OrchestrationRecord] = None
