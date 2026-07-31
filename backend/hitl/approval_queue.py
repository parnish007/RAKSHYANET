"""
HITL Approval Queue -- Prompt 4.1

Manages the workflow for medium-confidence news events (0.5-0.8 confidence)
that require human review before triggering re-optimization.

Lifecycle:
    PENDING  →  APPROVED  (coordinator approves)
    PENDING  →  REJECTED  (coordinator rejects)
    PENDING  →  EXPIRED   (timeout reached — treated as safe-default reject)

Typical usage
-------------
    from backend.hitl.approval_queue import ApprovalQueue
    from backend.rag.news_analyzer import NewsEvent

    queue = ApprovalQueue(timeout_minutes=5)
    request = queue.submit_for_review(event)
    queue.approve(request.request_id, reviewer="coordinator_1")
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from backend.rag.news_analyzer import NewsEvent


# ================================================================== #
#  Enums and models                                                    #
# ================================================================== #

class ApprovalStatus(str, Enum):
    PENDING  = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED  = "EXPIRED"


class ApprovalRequest(BaseModel):
    """A queued review request for a medium-confidence news event."""
    model_config = {"frozen": False}

    request_id:       str
    event_id:         str
    news_event:       NewsEvent
    submitted_at:     str
    expires_at:       str
    status:           ApprovalStatus = ApprovalStatus.PENDING
    reviewed_by:      Optional[str]  = None
    reviewed_at:      Optional[str]  = None
    rejection_reason: Optional[str]  = None


class ImpactPreview(BaseModel):
    """
    Estimated impact of approving a news event on the current plan.

    All values are estimates — actual optimization runs AFTER approval.
    """
    urgency_changes:           Dict[str, float]          # village_id → delta urgency
    affected_villages:         List[str]
    resource_reallocation:     Dict[str, Dict[str, float]] # village_id → {resource → shift}
    eta_changes:               Dict[str, int]             # village_id → minutes Δ (+ = worse)
    welfare_improvement_estimate: float                   # expected % welfare gain if approved


# ================================================================== #
#  ApprovalQueue                                                       #
# ================================================================== #

class ApprovalQueue:
    """
    In-memory queue for HITL approval requests.

    Args:
        timeout_minutes: How long a request stays PENDING before auto-expiring.
    """

    def __init__(self, timeout_minutes: int = 5) -> None:
        self.timeout_minutes = timeout_minutes
        self.requests:      Dict[str, ApprovalRequest] = {}
        self.pending_queue: List[str] = []   # request_ids in submission order

    # -------------------------------------------------------------- #
    #  Submission                                                     #
    # -------------------------------------------------------------- #

    def submit_for_review(self, event: NewsEvent) -> ApprovalRequest:
        """
        Queue a medium-confidence event for human review.

        Returns the created ApprovalRequest (status=PENDING).
        """
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        now        = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.timeout_minutes)

        request = ApprovalRequest(
            request_id=request_id,
            event_id=event.event_id,
            news_event=event,
            submitted_at=now.isoformat(),
            expires_at=expires_at.isoformat(),
            status=ApprovalStatus.PENDING,
        )

        self.requests[request_id] = request
        self.pending_queue.append(request_id)
        return request

    # -------------------------------------------------------------- #
    #  Decisions                                                      #
    # -------------------------------------------------------------- #

    def approve(self, request_id: str, reviewer: str) -> ApprovalRequest:
        """
        Approve a pending request.

        Raises:
            KeyError:   Request not found.
            ValueError: Request already reviewed or expired.
        """
        request = self._get_or_raise(request_id)
        self._assert_pending(request)
        self._assert_not_expired(request)

        request.status      = ApprovalStatus.APPROVED
        request.reviewed_by = reviewer
        request.reviewed_at = datetime.now(timezone.utc).isoformat()
        self._remove_from_pending(request_id)
        return request

    def reject(
        self,
        request_id: str,
        reviewer: str,
        reason: str = "",
    ) -> ApprovalRequest:
        """
        Reject a pending request.

        Raises:
            KeyError:   Request not found.
            ValueError: Request already reviewed or expired.
        """
        request = self._get_or_raise(request_id)
        self._assert_pending(request)

        request.status           = ApprovalStatus.REJECTED
        request.reviewed_by      = reviewer
        request.reviewed_at      = datetime.now(timezone.utc).isoformat()
        request.rejection_reason = reason
        self._remove_from_pending(request_id)
        return request

    # -------------------------------------------------------------- #
    #  Expiration                                                     #
    # -------------------------------------------------------------- #

    def expire_old_requests(self) -> List[str]:
        """
        Scan pending requests and mark any past their timeout as EXPIRED.

        Returns:
            List of request_ids that were just expired.
        """
        now         = datetime.now(timezone.utc)
        expired_ids: List[str] = []

        for req_id in list(self.pending_queue):
            request    = self.requests[req_id]
            expires_at = datetime.fromisoformat(request.expires_at)
            if now > expires_at:
                request.status = ApprovalStatus.EXPIRED
                self._remove_from_pending(req_id)
                expired_ids.append(req_id)

        return expired_ids

    # -------------------------------------------------------------- #
    #  Queries                                                        #
    # -------------------------------------------------------------- #

    def get_pending(self) -> List[ApprovalRequest]:
        """Return all requests currently in PENDING status, in submission order."""
        return [
            self.requests[rid]
            for rid in self.pending_queue
            if self.requests[rid].status == ApprovalStatus.PENDING
        ]

    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Return the request with the given ID, or None."""
        return self.requests.get(request_id)

    def get_history(self, limit: int = 20) -> List[ApprovalRequest]:
        """
        Return the most recent non-PENDING requests (approved / rejected / expired).

        Ordered newest-first by reviewed_at / submitted_at.
        """
        resolved = [
            r for r in self.requests.values()
            if r.status != ApprovalStatus.PENDING
        ]
        # Sort: reviewed_at if available, else submitted_at (most recent first)
        resolved.sort(
            key=lambda r: r.reviewed_at or r.submitted_at,
            reverse=True,
        )
        return resolved[:limit]

    # -------------------------------------------------------------- #
    #  Internal helpers                                               #
    # -------------------------------------------------------------- #

    def _get_or_raise(self, request_id: str) -> ApprovalRequest:
        request = self.requests.get(request_id)
        if request is None:
            raise KeyError(f"Request {request_id!r} not found")
        return request

    def _assert_pending(self, request: ApprovalRequest) -> None:
        if request.status != ApprovalStatus.PENDING:
            raise ValueError(
                f"Request {request.request_id!r} is already {request.status.value}"
            )

    def _assert_not_expired(self, request: ApprovalRequest) -> None:
        if datetime.fromisoformat(request.expires_at) < datetime.now(timezone.utc):
            request.status = ApprovalStatus.EXPIRED
            self._remove_from_pending(request.request_id)
            raise ValueError(f"Request {request.request_id!r} has expired")

    def _remove_from_pending(self, request_id: str) -> None:
        try:
            self.pending_queue.remove(request_id)
        except ValueError:
            pass  # Already removed (safe no-op)
