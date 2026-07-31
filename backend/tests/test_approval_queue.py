"""
Tests for ApprovalQueue and related models -- Prompt 4.1
Run: pytest backend/tests/test_approval_queue.py -v
"""
import time
from datetime import datetime, timedelta, timezone

import pytest

from backend.hitl.approval_queue import (
    ApprovalQueue,
    ApprovalRequest,
    ApprovalStatus,
    ImpactPreview,
)
from backend.rag.news_analyzer import NewsEvent


# ================================================================== #
#  Helpers and fixtures                                                #
# ================================================================== #

def _make_event(
    event_id: str = "evt_test",
    severity: int = 7,
    confidence: float = 0.65,
    affected: list | None = None,
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        raw_text="Test flood event near Dhulikhel",
        location=["Dhulikhel"],
        severity=severity,
        confidence=confidence,
        affected_villages=affected or ["dhulikhel", "panauti"],
        resource_implications={"food": 20.0, "water": 15.0},
        requires_hitl=True,
    )


@pytest.fixture
def queue() -> ApprovalQueue:
    return ApprovalQueue(timeout_minutes=5)


@pytest.fixture
def sample_event() -> NewsEvent:
    return _make_event(event_id="evt_sample", confidence=0.65)


@pytest.fixture
def approved_request(queue, sample_event) -> ApprovalRequest:
    req = queue.submit_for_review(sample_event)
    queue.approve(req.request_id, reviewer="coord_1")
    return req


@pytest.fixture
def expired_request(queue) -> ApprovalRequest:
    """Submit a request then manually set its expires_at in the past."""
    event = _make_event(event_id="evt_stale", confidence=0.6)
    req   = queue.submit_for_review(event)
    # Force expiry by backdating expires_at
    past = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    req.expires_at = past
    return req


# ================================================================== #
#  Submission tests                                                    #
# ================================================================== #

class TestSubmission:
    def test_submit_returns_approval_request(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        assert isinstance(req, ApprovalRequest)

    def test_submit_status_is_pending(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        assert req.status == ApprovalStatus.PENDING

    def test_submit_request_id_is_unique(self, queue, sample_event):
        e2  = _make_event(event_id="evt_other", confidence=0.7)
        r1  = queue.submit_for_review(sample_event)
        r2  = queue.submit_for_review(e2)
        assert r1.request_id != r2.request_id

    def test_submit_request_id_starts_with_req(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        assert req.request_id.startswith("req_")

    def test_submit_expires_at_is_future(self, queue, sample_event):
        req  = queue.submit_for_review(sample_event)
        exp  = datetime.fromisoformat(req.expires_at)
        now  = datetime.now(timezone.utc)
        assert exp > now

    def test_submit_expires_at_is_5_minutes_from_now(self, queue, sample_event):
        req    = queue.submit_for_review(sample_event)
        exp    = datetime.fromisoformat(req.expires_at)
        sub    = datetime.fromisoformat(req.submitted_at)
        delta  = (exp - sub).total_seconds()
        assert 290 <= delta <= 310   # 5 min ± 10 s tolerance

    def test_submit_adds_to_pending_queue(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        assert req.request_id in queue.pending_queue

    def test_submit_event_id_stored(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        assert req.event_id == sample_event.event_id

    def test_submit_news_event_stored(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        assert req.news_event.raw_text == sample_event.raw_text


# ================================================================== #
#  Approval tests                                                      #
# ================================================================== #

class TestApproval:
    def test_approve_changes_status_to_approved(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.approve(req.request_id, reviewer="coord_1")
        assert req.status == ApprovalStatus.APPROVED

    def test_approve_sets_reviewed_by(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.approve(req.request_id, reviewer="coord_1")
        assert req.reviewed_by == "coord_1"

    def test_approve_sets_reviewed_at(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.approve(req.request_id, reviewer="coord_1")
        assert req.reviewed_at is not None

    def test_approve_removes_from_pending_queue(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.approve(req.request_id, reviewer="coord_1")
        assert req.request_id not in queue.pending_queue

    def test_approve_raises_key_error_for_unknown_id(self, queue):
        with pytest.raises(KeyError):
            queue.approve("req_nonexistent", reviewer="coord_1")

    def test_approve_raises_value_error_for_already_approved(
        self, queue, sample_event
    ):
        req = queue.submit_for_review(sample_event)
        queue.approve(req.request_id, reviewer="coord_1")
        with pytest.raises(ValueError):
            queue.approve(req.request_id, reviewer="coord_2")

    def test_approve_returns_updated_request(self, queue, sample_event):
        req    = queue.submit_for_review(sample_event)
        result = queue.approve(req.request_id, reviewer="coord_1")
        assert result.status == ApprovalStatus.APPROVED


# ================================================================== #
#  Rejection tests                                                     #
# ================================================================== #

class TestRejection:
    def test_reject_changes_status_to_rejected(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.reject(req.request_id, reviewer="coord_1", reason="unverified")
        assert req.status == ApprovalStatus.REJECTED

    def test_reject_stores_rejection_reason(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.reject(req.request_id, reviewer="coord_1", reason="false alarm")
        assert req.rejection_reason == "false alarm"

    def test_reject_sets_reviewed_by(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.reject(req.request_id, reviewer="coord_2", reason="")
        assert req.reviewed_by == "coord_2"

    def test_reject_sets_reviewed_at(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.reject(req.request_id, reviewer="coord_2", reason="")
        assert req.reviewed_at is not None

    def test_reject_removes_from_pending_queue(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.reject(req.request_id, reviewer="coord_2", reason="")
        assert req.request_id not in queue.pending_queue

    def test_reject_raises_value_error_for_already_approved(
        self, queue, approved_request
    ):
        with pytest.raises(ValueError):
            queue.reject(approved_request.request_id, reviewer="coord_3", reason="late")


# ================================================================== #
#  Expiration tests                                                    #
# ================================================================== #

class TestExpiration:
    def test_expire_old_returns_expired_ids(self, queue, expired_request):
        # expired_request is already in the queue with a past expires_at
        ids = queue.expire_old_requests()
        assert expired_request.request_id in ids

    def test_expired_request_status_becomes_expired(self, queue, expired_request):
        queue.expire_old_requests()
        assert expired_request.status == ApprovalStatus.EXPIRED

    def test_expired_request_removed_from_pending_queue(self, queue, expired_request):
        queue.expire_old_requests()
        assert expired_request.request_id not in queue.pending_queue

    def test_non_expired_request_stays_pending(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.expire_old_requests()
        assert req.status == ApprovalStatus.PENDING

    def test_approve_expired_request_raises_value_error(self, queue, expired_request):
        with pytest.raises(ValueError):
            queue.approve(expired_request.request_id, reviewer="coord_1")

    def test_expire_returns_empty_when_no_expired(self, queue, sample_event):
        queue.submit_for_review(sample_event)
        ids = queue.expire_old_requests()
        assert ids == []


# ================================================================== #
#  Query tests                                                         #
# ================================================================== #

class TestQueries:
    def test_get_pending_returns_only_pending(self, queue):
        e1 = _make_event("evt_a", confidence=0.6)
        e2 = _make_event("evt_b", confidence=0.7)
        r1 = queue.submit_for_review(e1)
        r2 = queue.submit_for_review(e2)
        queue.approve(r1.request_id, reviewer="coord_1")
        pending = queue.get_pending()
        ids = [r.request_id for r in pending]
        assert r1.request_id not in ids
        assert r2.request_id in ids

    def test_get_pending_empty_initially(self, queue):
        assert queue.get_pending() == []

    def test_get_request_returns_correct_request(self, queue, sample_event):
        req   = queue.submit_for_review(sample_event)
        found = queue.get_request(req.request_id)
        assert found is req

    def test_get_request_returns_none_for_unknown(self, queue):
        assert queue.get_request("req_ghost") is None

    def test_get_history_returns_resolved_requests(self, queue, approved_request):
        history = queue.get_history()
        ids = [r.request_id for r in history]
        assert approved_request.request_id in ids

    def test_get_history_excludes_pending(self, queue, sample_event):
        req     = queue.submit_for_review(sample_event)
        history = queue.get_history()
        ids     = [r.request_id for r in history]
        assert req.request_id not in ids

    def test_get_history_respects_limit(self, queue):
        for i in range(5):
            e = _make_event(f"evt_{i}", confidence=0.6)
            r = queue.submit_for_review(e)
            queue.approve(r.request_id, reviewer="coord_1")
        history = queue.get_history(limit=3)
        assert len(history) == 3

    def test_multiple_pending_maintained_in_order(self, queue):
        ids = []
        for i in range(4):
            e = _make_event(f"evt_{i}", confidence=0.6)
            r = queue.submit_for_review(e)
            ids.append(r.request_id)
        pending_ids = [r.request_id for r in queue.get_pending()]
        assert pending_ids == ids


# ================================================================== #
#  Edge-case tests                                                     #
# ================================================================== #

class TestEdgeCases:
    def test_empty_queue_get_pending_returns_empty_list(self, queue):
        assert queue.get_pending() == []

    def test_empty_queue_expire_returns_empty_list(self, queue):
        assert queue.expire_old_requests() == []

    def test_empty_queue_get_history_returns_empty_list(self, queue):
        assert queue.get_history() == []

    def test_reject_already_rejected_raises_value_error(self, queue, sample_event):
        req = queue.submit_for_review(sample_event)
        queue.reject(req.request_id, reviewer="c1", reason="first")
        with pytest.raises(ValueError):
            queue.reject(req.request_id, reviewer="c2", reason="second")

    def test_custom_timeout_reflected_in_expires_at(self):
        q   = ApprovalQueue(timeout_minutes=2)
        e   = _make_event("evt_short", confidence=0.6)
        req = q.submit_for_review(e)
        exp = datetime.fromisoformat(req.expires_at)
        sub = datetime.fromisoformat(req.submitted_at)
        delta = (exp - sub).total_seconds()
        assert 110 <= delta <= 130   # 2 min ± 10 s
