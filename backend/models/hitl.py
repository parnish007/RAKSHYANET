"""
Human-in-the-Loop (HITL) decision model.
Coordinators approve or reject medium-confidence news events before re-optimization.
"""
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class HITLDecisionType(str, Enum):
    CONFIRM = "confirm"     # Approve → trigger re-optimization
    REJECT = "reject"       # Reject → log only, no optimization
    WAIT = "wait"           # Defer → extend timeout


class HITLStatus(str, Enum):
    PENDING = "PENDING"         # Awaiting coordinator action
    DECIDED = "DECIDED"         # Decision made
    TIMED_OUT = "TIMED_OUT"     # Timeout expired → auto-reject


HITL_TIMEOUT_SECONDS = 300  # 5 minutes (blueprint §1)


class HITLRequest(BaseModel):
    """Sent to the coordinator dashboard when an event needs approval."""
    event_id: str
    news_summary: str
    estimated_impact: str  # Human-readable impact preview
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    affected_village_id: Optional[str] = None
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(default=None)

    @model_validator(mode="after")
    def set_expiry(self) -> "HITLRequest":
        if self.expires_at is None:
            self.expires_at = self.requested_at + timedelta(seconds=HITL_TIMEOUT_SECONDS)
        return self

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at

    @property
    def seconds_remaining(self) -> float:
        delta = (self.expires_at - datetime.utcnow()).total_seconds()
        return max(0.0, delta)


class HITLDecision(BaseModel):
    """Coordinator's response to an HITL request."""
    event_id: str = Field(..., description="References HITLRequest.event_id")
    decision: HITLDecisionType
    coordinator_id: str = Field(..., description="ID of the emergency coordinator")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = Field(default=None, description="Optional coordinator comments")

    # Outcome tracking
    status: HITLStatus = HITLStatus.DECIDED
    optimization_triggered: bool = False

    @property
    def approved(self) -> bool:
        return self.decision == HITLDecisionType.CONFIRM

    @classmethod
    def auto_reject(cls, event_id: str) -> "HITLDecision":
        """Factory: create a system-generated rejection on timeout."""
        return cls(
            event_id=event_id,
            decision=HITLDecisionType.REJECT,
            coordinator_id="SYSTEM_TIMEOUT",
            notes=f"Auto-rejected after {HITL_TIMEOUT_SECONDS}s timeout",
            status=HITLStatus.TIMED_OUT,
        )

    def __repr__(self) -> str:
        return (
            f"HITLDecision(event={self.event_id!r}, decision={self.decision.value}, "
            f"coordinator={self.coordinator_id!r})"
        )
