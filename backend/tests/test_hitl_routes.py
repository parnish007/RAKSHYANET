"""
Tests for HITL API Routes -- Prompt 4.2
Run: pytest backend/tests/test_hitl_routes.py -v
"""
import pytest
from fastapi.testclient import TestClient

# Import app from main.py and reset the queue before each test
from backend.api.main import app
import backend.api.hitl_routes as hitl_module
from backend.hitl.approval_queue import ApprovalQueue, ApprovalStatus


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture(autouse=True)
def reset_queue():
    """Fresh queue for every test to prevent cross-test contamination."""
    hitl_module.approval_queue = ApprovalQueue(timeout_minutes=5)
    yield
    hitl_module.approval_queue = ApprovalQueue(timeout_minutes=5)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def medium_event_payload():
    """A valid HITL-range event (confidence=0.65)."""
    return {
        "event": {
            "event_id": "evt_flood_01",
            "raw_text": "Flash flood near Dhulikhel. Road blocked. Families displaced.",
            "location": ["Dhulikhel"],
            "severity": 8,
            "confidence": 0.65,
            "affected_villages": ["dhulikhel", "panauti"],
            "resource_implications": {"food": 20.0, "medical_kit": 15.0},
            "requires_hitl": True,
        },
        "preview_impact": False,
    }


def _submit(client, confidence: float = 0.65, event_id: str = "evt_test"):
    """Helper: submit an event and return the response."""
    payload = {
        "event": {
            "event_id": event_id,
            "raw_text": "Test event",
            "location": ["Dhulikhel"],
            "severity": 6,
            "confidence": confidence,
            "affected_villages": ["dhulikhel"],
            "resource_implications": {},
            "requires_hitl": True,
        },
        "preview_impact": False,
    }
    return client.post("/api/hitl/submit", json=payload)


# ================================================================== #
#  Router / app setup                                                  #
# ================================================================== #

class TestSetup:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_stats_endpoint_reachable(self, client):
        r = client.get("/api/hitl/stats")
        assert r.status_code == 200

    def test_pending_endpoint_reachable(self, client):
        r = client.get("/api/hitl/pending")
        assert r.status_code == 200


# ================================================================== #
#  POST /api/hitl/submit                                               #
# ================================================================== #

class TestSubmit:
    def test_submit_returns_201(self, client, medium_event_payload):
        r = client.post("/api/hitl/submit", json=medium_event_payload)
        assert r.status_code == 201

    def test_submit_response_has_request_id(self, client, medium_event_payload):
        r = client.post("/api/hitl/submit", json=medium_event_payload)
        assert "request_id" in r.json()

    def test_submit_response_has_expires_at(self, client, medium_event_payload):
        r = client.post("/api/hitl/submit", json=medium_event_payload)
        assert "expires_at" in r.json()

    def test_submit_status_is_pending(self, client, medium_event_payload):
        r = client.post("/api/hitl/submit", json=medium_event_payload)
        assert r.json()["status"] == "PENDING"

    def test_submit_low_confidence_returns_400(self, client):
        r = _submit(client, confidence=0.3)
        assert r.status_code == 400

    def test_submit_high_confidence_returns_400(self, client):
        r = _submit(client, confidence=0.85)
        assert r.status_code == 400

    def test_submit_exactly_0_5_is_accepted(self, client):
        r = _submit(client, confidence=0.5)
        assert r.status_code == 201

    def test_submit_exactly_0_8_is_rejected(self, client):
        r = _submit(client, confidence=0.8)
        assert r.status_code == 400

    def test_submit_adds_to_pending(self, client, medium_event_payload):
        client.post("/api/hitl/submit", json=medium_event_payload)
        r = client.get("/api/hitl/pending")
        assert len(r.json()) == 1


# ================================================================== #
#  GET /api/hitl/pending                                               #
# ================================================================== #

class TestPending:
    def test_pending_returns_list(self, client):
        r = client.get("/api/hitl/pending")
        assert isinstance(r.json(), list)

    def test_pending_empty_initially(self, client):
        r = client.get("/api/hitl/pending")
        assert r.json() == []

    def test_pending_count_after_submit(self, client):
        _submit(client, event_id="evt_a")
        _submit(client, event_id="evt_b")
        r = client.get("/api/hitl/pending")
        assert len(r.json()) == 2

    def test_pending_auto_expires_old(self, client):
        from datetime import timedelta, timezone, datetime
        _submit(client, event_id="evt_old")
        # Manually backdate the only request
        req_id = list(hitl_module.approval_queue.requests.keys())[0]
        req = hitl_module.approval_queue.requests[req_id]
        req.expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

        r = client.get("/api/hitl/pending")
        assert r.json() == []  # expired, not returned


# ================================================================== #
#  POST /api/hitl/approve/{request_id}                                 #
# ================================================================== #

class TestApprove:
    def test_approve_returns_200(self, client):
        r_sub = _submit(client)
        req_id = r_sub.json()["request_id"]
        r = client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "coord_1"})
        assert r.status_code == 200

    def test_approve_status_becomes_approved(self, client):
        req_id = _submit(client).json()["request_id"]
        r = client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "coord_1"})
        assert r.json()["status"] == "APPROVED"

    def test_approve_sets_reviewed_by(self, client):
        req_id = _submit(client).json()["request_id"]
        r = client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "coord_maya"})
        assert r.json()["reviewed_by"] == "coord_maya"

    def test_approve_sets_reviewed_at(self, client):
        req_id = _submit(client).json()["request_id"]
        r = client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "coord_1"})
        assert r.json()["reviewed_at"] is not None

    def test_approve_unknown_id_returns_404(self, client):
        r = client.post("/api/hitl/approve/req_ghost", json={"reviewer": "c"})
        assert r.status_code == 404

    def test_approve_already_approved_returns_400(self, client):
        req_id = _submit(client).json()["request_id"]
        client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "c1"})
        r = client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "c2"})
        assert r.status_code == 400


# ================================================================== #
#  POST /api/hitl/reject/{request_id}                                  #
# ================================================================== #

class TestReject:
    def test_reject_returns_200(self, client):
        req_id = _submit(client).json()["request_id"]
        r = client.post(
            f"/api/hitl/reject/{req_id}",
            json={"reviewer": "coord_1", "reason": "false alarm"},
        )
        assert r.status_code == 200

    def test_reject_status_becomes_rejected(self, client):
        req_id = _submit(client).json()["request_id"]
        r = client.post(
            f"/api/hitl/reject/{req_id}",
            json={"reviewer": "coord_1", "reason": "unverified"},
        )
        assert r.json()["status"] == "REJECTED"

    def test_reject_stores_reason(self, client):
        req_id = _submit(client).json()["request_id"]
        r = client.post(
            f"/api/hitl/reject/{req_id}",
            json={"reviewer": "coord_1", "reason": "duplicate report"},
        )
        assert r.json()["rejection_reason"] == "duplicate report"

    def test_reject_unknown_id_returns_404(self, client):
        r = client.post(
            "/api/hitl/reject/req_ghost",
            json={"reviewer": "c", "reason": "x"},
        )
        assert r.status_code == 404

    def test_reject_already_approved_returns_400(self, client):
        req_id = _submit(client).json()["request_id"]
        client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "c1"})
        r = client.post(
            f"/api/hitl/reject/{req_id}",
            json={"reviewer": "c2", "reason": "late"},
        )
        assert r.status_code == 400


# ================================================================== #
#  GET /api/hitl/request/{request_id}                                  #
# ================================================================== #

class TestGetRequest:
    def test_get_request_returns_200(self, client):
        req_id = _submit(client).json()["request_id"]
        r = client.get(f"/api/hitl/request/{req_id}")
        assert r.status_code == 200

    def test_get_request_has_correct_event_id(self, client):
        _submit(client, event_id="evt_unique")
        req_id = list(hitl_module.approval_queue.requests.keys())[0]
        r = client.get(f"/api/hitl/request/{req_id}")
        assert r.json()["event_id"] == "evt_unique"

    def test_get_request_unknown_returns_404(self, client):
        r = client.get("/api/hitl/request/req_nope")
        assert r.status_code == 404


# ================================================================== #
#  GET /api/hitl/history                                               #
# ================================================================== #

class TestHistory:
    def test_history_empty_initially(self, client):
        r = client.get("/api/hitl/history")
        assert r.json() == []

    def test_history_contains_approved_request(self, client):
        req_id = _submit(client).json()["request_id"]
        client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "c"})
        r = client.get("/api/hitl/history")
        ids = [item["request_id"] for item in r.json()]
        assert req_id in ids

    def test_history_contains_rejected_request(self, client):
        req_id = _submit(client).json()["request_id"]
        client.post(f"/api/hitl/reject/{req_id}", json={"reviewer": "c", "reason": "x"})
        r = client.get("/api/hitl/history")
        assert any(item["request_id"] == req_id for item in r.json())

    def test_history_excludes_pending(self, client):
        req_id = _submit(client).json()["request_id"]
        r = client.get("/api/hitl/history")
        assert not any(item["request_id"] == req_id for item in r.json())

    def test_history_respects_limit(self, client):
        for i in range(5):
            rid = _submit(client, event_id=f"evt_{i}").json()["request_id"]
            client.post(f"/api/hitl/approve/{rid}", json={"reviewer": "c"})
        r = client.get("/api/hitl/history?limit=3")
        assert len(r.json()) == 3

    def test_history_status_filter_approved(self, client):
        rid_a = _submit(client, event_id="evt_a").json()["request_id"]
        rid_r = _submit(client, event_id="evt_b").json()["request_id"]
        client.post(f"/api/hitl/approve/{rid_a}", json={"reviewer": "c"})
        client.post(f"/api/hitl/reject/{rid_r}", json={"reviewer": "c", "reason": "x"})
        r = client.get("/api/hitl/history?status_filter=APPROVED")
        statuses = [item["status"] for item in r.json()]
        assert all(s == "APPROVED" for s in statuses)


# ================================================================== #
#  POST /api/hitl/expire-old                                           #
# ================================================================== #

class TestExpireOld:
    def test_expire_old_returns_200(self, client):
        r = client.post("/api/hitl/expire-old")
        assert r.status_code == 200

    def test_expire_old_returns_count_and_ids(self, client):
        r = client.post("/api/hitl/expire-old")
        data = r.json()
        assert "expired_count" in data
        assert "expired_ids" in data

    def test_expire_old_catches_backdated_request(self, client):
        from datetime import timedelta, timezone, datetime
        _submit(client, event_id="evt_stale")
        req = list(hitl_module.approval_queue.requests.values())[0]
        req.expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

        r = client.post("/api/hitl/expire-old")
        assert r.json()["expired_count"] == 1


# ================================================================== #
#  GET /api/hitl/stats                                                 #
# ================================================================== #

class TestStats:
    def test_stats_structure(self, client):
        r = client.get("/api/hitl/stats")
        data = r.json()
        assert "pending_count" in data
        assert "approved_count" in data
        assert "rejected_count" in data
        assert "expired_count" in data
        assert "total_processed" in data

    def test_stats_counts_after_submit(self, client):
        _submit(client)
        r = client.get("/api/hitl/stats")
        assert r.json()["pending_count"] == 1

    def test_stats_counts_after_approve(self, client):
        req_id = _submit(client).json()["request_id"]
        client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "c"})
        r = client.get("/api/hitl/stats")
        data = r.json()
        assert data["approved_count"] == 1
        assert data["pending_count"] == 0
        assert data["total_processed"] == 1

    def test_oldest_pending_age_is_none_when_empty(self, client):
        r = client.get("/api/hitl/stats")
        assert r.json()["oldest_pending_age_seconds"] is None

    def test_oldest_pending_age_is_float_when_pending(self, client):
        _submit(client)
        r = client.get("/api/hitl/stats")
        age = r.json()["oldest_pending_age_seconds"]
        assert age is not None
        assert age >= 0.0


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def test_submit_approve_appears_in_history(self, client):
        req_id = _submit(client, event_id="evt_int_a").json()["request_id"]
        client.post(f"/api/hitl/approve/{req_id}", json={"reviewer": "coord_maya"})
        history = client.get("/api/hitl/history").json()
        match = next((h for h in history if h["request_id"] == req_id), None)
        assert match is not None
        assert match["status"] == "APPROVED"
        assert match["reviewed_by"] == "coord_maya"

    def test_submit_reject_appears_in_history(self, client):
        req_id = _submit(client, event_id="evt_int_b").json()["request_id"]
        client.post(
            f"/api/hitl/reject/{req_id}",
            json={"reviewer": "coord_rama", "reason": "unconfirmed"},
        )
        history = client.get("/api/hitl/history").json()
        match = next((h for h in history if h["request_id"] == req_id), None)
        assert match is not None
        assert match["status"] == "REJECTED"
        assert match["rejection_reason"] == "unconfirmed"

    def test_full_lifecycle_stats(self, client):
        r_a = _submit(client, event_id="e_a").json()["request_id"]
        r_b = _submit(client, event_id="e_b").json()["request_id"]
        client.post(f"/api/hitl/approve/{r_a}", json={"reviewer": "c"})
        client.post(f"/api/hitl/reject/{r_b}", json={"reviewer": "c", "reason": "x"})
        stats = client.get("/api/hitl/stats").json()
        assert stats["approved_count"] == 1
        assert stats["rejected_count"] == 1
        assert stats["pending_count"] == 0
        assert stats["total_processed"] == 2