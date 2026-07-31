"""
HITL API Routes -- Prompt 4.2

REST endpoints for the Human-in-the-Loop approval workflow.
All state lives in the module-level ApprovalQueue instance.
Production would use FastAPI dependency injection + a database-backed queue.

Endpoints:
    POST   /api/hitl/submit
    GET    /api/hitl/pending
    POST   /api/hitl/approve/{request_id}
    POST   /api/hitl/reject/{request_id}
    GET    /api/hitl/request/{request_id}
    GET    /api/hitl/history
    POST   /api/hitl/expire-old
    GET    /api/hitl/stats
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.hitl.approval_queue import (
    ApprovalQueue,
    ApprovalRequest,
    ApprovalStatus,
    ImpactPreview,
)
from backend.rag.news_analyzer import NewsEvent

router = APIRouter(prefix="/api/hitl", tags=["HITL"])

# Module-level queue (single-user demo; production: DI + persistent store)
approval_queue = ApprovalQueue(timeout_minutes=5)


# ================================================================== #
#  Request / Response models                                           #
# ================================================================== #

class SubmitEventRequest(BaseModel):
    event: NewsEvent
    preview_impact: bool = True


class SubmitEventResponse(BaseModel):
    request_id: str
    status: str
    expires_at: str
    impact_preview: Optional[ImpactPreview] = None


class ApproveRequestBody(BaseModel):
    reviewer: str
    notes: Optional[str] = None


class RejectRequestBody(BaseModel):
    reviewer: str
    reason: str


class QueueStats(BaseModel):
    pending_count: int
    approved_count: int
    rejected_count: int
    expired_count: int
    total_processed: int
    oldest_pending_age_seconds: Optional[float] = None


# ================================================================== #
#  Endpoints                                                           #
# ================================================================== #

@router.post("/submit", response_model=SubmitEventResponse, status_code=201)
async def submit_for_review(request: SubmitEventRequest):
    """
    Submit a medium-confidence event (0.5 ≤ confidence < 0.8) for human review.

    Returns a queued ApprovalRequest with a 5-minute countdown.
    """
    conf = request.event.confidence
    if not (0.5 <= conf < 0.8):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Event confidence {conf:.2f} not in HITL range [0.50, 0.80). "
                "Use AUTO_OPTIMIZE for ≥0.80 or IGNORE for <0.50."
            ),
        )

    approval_request = approval_queue.submit_for_review(request.event)

    # Impact preview omitted in this route — ImpactAnalyzer needs live
    # village/route state wired in Section 5. Frontend can call a separate
    # /api/hitl/impact endpoint once StateManager integration lands.
    return SubmitEventResponse(
        request_id=approval_request.request_id,
        status=approval_request.status.value,
        expires_at=approval_request.expires_at,
        impact_preview=None,
    )


@router.get("/pending", response_model=List[ApprovalRequest])
async def get_pending_requests():
    """
    Return all PENDING approval requests (oldest first).

    Auto-expires timed-out requests before responding.
    """
    expired_ids = approval_queue.expire_old_requests()
    if expired_ids:
        # Log for monitoring; not surfaced to caller
        pass
    return approval_queue.get_pending()


@router.post("/approve/{request_id}", response_model=ApprovalRequest)
async def approve_request(request_id: str, body: ApproveRequestBody):
    """
    Approve a pending request.

    After approval, Section 5 will wire StateManager re-optimisation here.
    """
    try:
        approved = approval_queue.approve(request_id, body.reviewer)
        # TODO (Section 5): trigger StateManager.run_full_optimization()
        return approved
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request '{request_id}' not found",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.post("/reject/{request_id}", response_model=ApprovalRequest)
async def reject_request(request_id: str, body: RejectRequestBody):
    """Reject a pending request. No re-optimisation triggered."""
    try:
        return approval_queue.reject(request_id, body.reviewer, body.reason)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request '{request_id}' not found",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.get("/request/{request_id}", response_model=ApprovalRequest)
async def get_request_details(request_id: str):
    """Return details for a single approval request."""
    req = approval_queue.get_request(request_id)
    if req is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request '{request_id}' not found",
        )
    return req


@router.get("/history", response_model=List[ApprovalRequest])
async def get_approval_history(
    limit: int = Query(default=20, ge=1, le=100),
    status_filter: Optional[ApprovalStatus] = Query(default=None),
):
    """
    Return resolved requests (APPROVED / REJECTED / EXPIRED), newest first.

    Args:
        limit: 1–100 results (default 20).
        status_filter: Restrict to one status value.
    """
    history = approval_queue.get_history(limit=limit)
    if status_filter is not None:
        history = [r for r in history if r.status == status_filter]
    return history


@router.post("/expire-old")
async def expire_old_requests():
    """Manually trigger expiration sweep. Useful for testing / scheduled jobs."""
    expired_ids = approval_queue.expire_old_requests()
    return {"expired_count": len(expired_ids), "expired_ids": expired_ids}


@router.get("/stats", response_model=QueueStats)
async def get_queue_stats():
    """Return queue statistics for the monitoring dashboard."""
    all_reqs = list(approval_queue.requests.values())
    pending  = approval_queue.get_pending()

    approved_count = sum(1 for r in all_reqs if r.status == ApprovalStatus.APPROVED)
    rejected_count = sum(1 for r in all_reqs if r.status == ApprovalStatus.REJECTED)
    expired_count  = sum(1 for r in all_reqs if r.status == ApprovalStatus.EXPIRED)

    oldest_age: Optional[float] = None
    if pending:
        now    = datetime.now(timezone.utc)
        oldest = min(pending, key=lambda r: r.submitted_at)
        oldest_age = (now - datetime.fromisoformat(oldest.submitted_at)).total_seconds()

    return QueueStats(
        pending_count=len(pending),
        approved_count=approved_count,
        rejected_count=rejected_count,
        expired_count=expired_count,
        total_processed=approved_count + rejected_count + expired_count,
        oldest_pending_age_seconds=oldest_age,
    )
