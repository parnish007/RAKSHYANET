#!/usr/bin/env python3
"""
Smoke test: HITL API Routes -- Prompt 4.2

Uses FastAPI TestClient (no live server required) to exercise all 8 endpoints:
  POST   /api/hitl/submit
  GET    /api/hitl/pending
  POST   /api/hitl/approve/{request_id}
  POST   /api/hitl/reject/{request_id}
  GET    /api/hitl/request/{request_id}
  GET    /api/hitl/history
  POST   /api/hitl/expire-old
  GET    /api/hitl/stats

Run from project root:
    python demo/test_hitl_api.py
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from backend.api.main import app
import backend.api.hitl_routes as hitl_module
from backend.hitl.approval_queue import ApprovalQueue

SEP  = "-" * 65
SEP2 = "=" * 65

client = TestClient(app)


def fresh_queue():
    hitl_module.approval_queue = ApprovalQueue(timeout_minutes=5)


def bar(fraction: float, width: int = 16) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def check(label: str, condition: bool) -> None:
    tag = "[OK]" if condition else "[FAIL]"
    print(f"  {tag} {label}")
    assert condition, f"Assertion failed: {label}"


def print_section(title: str) -> None:
    print(f"\n  {SEP}")
    print(f"  {title}")
    print(f"  {SEP}")


# ------------------------------------------------------------------ #
#  Event payloads                                                      #
# ------------------------------------------------------------------ #

def _payload(event_id: str, confidence: float, severity: int = 7) -> dict:
    return {
        "event": {
            "event_id": event_id,
            "raw_text": f"Test event {event_id} near Dhulikhel. Families displaced.",
            "location": ["Dhulikhel"],
            "severity": severity,
            "confidence": confidence,
            "affected_villages": ["dhulikhel", "panauti"],
            "resource_implications": {"food": 20.0, "medical_kit": 15.0},
            "requires_hitl": True,
        },
        "preview_impact": False,
    }


# ================================================================== #
#  Main smoke test                                                     #
# ================================================================== #

def main():
    fresh_queue()

    print("\n" + SEP2)
    print("  RAKSHYANET -- HITL API SMOKE TEST  (Prompt 4.2)")
    print(SEP2)

    # ---------------------------------------------------------------- #
    # 1. Health + root                                                  #
    # ---------------------------------------------------------------- #
    print_section("1. HEALTH CHECK")
    r = client.get("/health")
    check("GET /health returns 200", r.status_code == 200)
    check("status = ok", r.json()["status"] == "ok")
    print(f"  Response: {r.json()}")

    # ---------------------------------------------------------------- #
    # 2. Stats (empty)                                                   #
    # ---------------------------------------------------------------- #
    print_section("2. INITIAL STATS (empty queue)")
    r = client.get("/api/hitl/stats")
    check("GET /api/hitl/stats returns 200", r.status_code == 200)
    stats = r.json()
    check("pending_count == 0", stats["pending_count"] == 0)
    check("total_processed == 0", stats["total_processed"] == 0)
    check("oldest_pending_age_seconds is null", stats["oldest_pending_age_seconds"] is None)
    print(f"  Stats: {stats}")

    # ---------------------------------------------------------------- #
    # 3. Submit events                                                   #
    # ---------------------------------------------------------------- #
    print_section("3. SUBMIT 3 EVENTS")

    # Medium-confidence event 1 -- will APPROVE
    r1 = client.post("/api/hitl/submit", json=_payload("evt_flood_01", 0.72, severity=8))
    check("POST /submit (approve candidate) returns 201", r1.status_code == 201)
    req_id_approve = r1.json()["request_id"]
    check("request_id present", req_id_approve.startswith("req_"))
    check("status == PENDING", r1.json()["status"] == "PENDING")
    print(f"  [APPROVE] {req_id_approve}  conf=0.72  sev=8")

    # Medium-confidence event 2 -- will REJECT
    r2 = client.post("/api/hitl/submit", json=_payload("evt_rumor_02", 0.55, severity=5))
    check("POST /submit (reject candidate) returns 201", r2.status_code == 201)
    req_id_reject = r2.json()["request_id"]
    print(f"  [REJECT] {req_id_reject}  conf=0.55  sev=5")

    # Medium-confidence event 3 -- will EXPIRE
    r3 = client.post("/api/hitl/submit", json=_payload("evt_stale_03", 0.63, severity=6))
    check("POST /submit (expire candidate) returns 201", r3.status_code == 201)
    req_id_expire = r3.json()["request_id"]
    print(f"  [EXPIRE] {req_id_expire}  conf=0.63  sev=6")

    # ---------------------------------------------------------------- #
    # 4. Reject invalid confidence                                       #
    # ---------------------------------------------------------------- #
    print_section("4. CONFIDENCE VALIDATION")
    r_low  = client.post("/api/hitl/submit", json=_payload("evt_low",  0.3))
    r_high = client.post("/api/hitl/submit", json=_payload("evt_high", 0.9))
    check("conf=0.3 returns 400", r_low.status_code == 400)
    check("conf=0.9 returns 400", r_high.status_code == 400)
    print(f"  Low  confidence error: {r_low.json()['detail'][:60]}...")
    print(f"  High confidence error: {r_high.json()['detail'][:60]}...")

    # ---------------------------------------------------------------- #
    # 5. GET /pending                                                    #
    # ---------------------------------------------------------------- #
    print_section("5. GET PENDING REQUESTS")
    r = client.get("/api/hitl/pending")
    check("GET /api/hitl/pending returns 200", r.status_code == 200)
    pending = r.json()
    check("3 requests pending", len(pending) == 3)
    print(f"  Pending count: {len(pending)}")
    for p in pending:
        print(f"    {p['request_id']}  event={p['event_id']}  status={p['status']}")

    # ---------------------------------------------------------------- #
    # 6. GET /request/{id}                                               #
    # ---------------------------------------------------------------- #
    print_section("6. GET REQUEST BY ID")
    r = client.get(f"/api/hitl/request/{req_id_approve}")
    check("GET /api/hitl/request/{id} returns 200", r.status_code == 200)
    check("correct event_id returned", r.json()["event_id"] == "evt_flood_01")

    r_404 = client.get("/api/hitl/request/req_ghost")
    check("Unknown ID returns 404", r_404.status_code == 404)

    # ---------------------------------------------------------------- #
    # 7. Approve                                                         #
    # ---------------------------------------------------------------- #
    print_section("7. APPROVE evt_flood_01")
    r = client.post(
        f"/api/hitl/approve/{req_id_approve}",
        json={"reviewer": "coord_maya", "notes": "Confirmed via radio"},
    )
    check("POST /approve returns 200", r.status_code == 200)
    check("status == APPROVED", r.json()["status"] == "APPROVED")
    check("reviewed_by == coord_maya", r.json()["reviewed_by"] == "coord_maya")
    check("reviewed_at set", r.json()["reviewed_at"] is not None)
    print(f"  Approved by: {r.json()['reviewed_by']}  at: {r.json()['reviewed_at'][:19]}")

    # Double-approve must fail
    r_dup = client.post(f"/api/hitl/approve/{req_id_approve}", json={"reviewer": "c2"})
    check("Double-approve returns 400", r_dup.status_code == 400)

    # Unknown ID
    r_404 = client.post("/api/hitl/approve/req_ghost", json={"reviewer": "c"})
    check("Approve unknown ID returns 404", r_404.status_code == 404)

    # ---------------------------------------------------------------- #
    # 8. Reject                                                          #
    # ---------------------------------------------------------------- #
    print_section("8. REJECT evt_rumor_02")
    r = client.post(
        f"/api/hitl/reject/{req_id_reject}",
        json={"reviewer": "coord_maya", "reason": "Unverified social media source"},
    )
    check("POST /reject returns 200", r.status_code == 200)
    check("status == REJECTED", r.json()["status"] == "REJECTED")
    check("rejection_reason stored", r.json()["rejection_reason"] == "Unverified social media source")
    print(f"  Rejected by: {r.json()['reviewed_by']}  reason: {r.json()['rejection_reason']}")

    # Reject already-approved must fail
    r_bad = client.post(
        f"/api/hitl/reject/{req_id_approve}",
        json={"reviewer": "c", "reason": "late"},
    )
    check("Reject approved request returns 400", r_bad.status_code == 400)

    # ---------------------------------------------------------------- #
    # 9. Expire                                                          #
    # ---------------------------------------------------------------- #
    print_section("9. EXPIRE evt_stale_03")
    # Backdate the third request
    req = hitl_module.approval_queue.requests[req_id_expire]
    req.expires_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

    r = client.post("/api/hitl/expire-old")
    check("POST /expire-old returns 200", r.status_code == 200)
    check("expired_count == 1", r.json()["expired_count"] == 1)
    check("req_id_expire in expired_ids", req_id_expire in r.json()["expired_ids"])
    print(f"  Expired: {r.json()['expired_ids']}")

    # ---------------------------------------------------------------- #
    # 10. History                                                        #
    # ---------------------------------------------------------------- #
    print_section("10. GET HISTORY")
    r = client.get("/api/hitl/history")
    check("GET /api/hitl/history returns 200", r.status_code == 200)
    history = r.json()
    check("3 resolved requests in history", len(history) == 3)

    statuses = {h["request_id"]: h["status"] for h in history}
    check("evt_flood_01 APPROVED in history", statuses.get(req_id_approve) == "APPROVED")
    check("evt_rumor_02 REJECTED in history", statuses.get(req_id_reject) == "REJECTED")
    check("evt_stale_03 EXPIRED in history",  statuses.get(req_id_expire) == "EXPIRED")
    print(f"  History ({len(history)} entries):")
    for h in history:
        print(f"    {h['request_id']}  {h['event_id']:<15} {h['status']}")

    # Status filter
    r_fil = client.get("/api/hitl/history?status_filter=APPROVED")
    check("status_filter=APPROVED returns 1", len(r_fil.json()) == 1)

    # Limit
    r_lim = client.get("/api/hitl/history?limit=1")
    check("limit=1 returns 1", len(r_lim.json()) == 1)

    # ---------------------------------------------------------------- #
    # 11. Final stats                                                    #
    # ---------------------------------------------------------------- #
    print_section("11. FINAL STATS")
    r = client.get("/api/hitl/stats")
    stats = r.json()
    check("pending_count == 0",   stats["pending_count"] == 0)
    check("approved_count == 1",  stats["approved_count"] == 1)
    check("rejected_count == 1",  stats["rejected_count"] == 1)
    check("expired_count == 1",   stats["expired_count"] == 1)
    check("total_processed == 3", stats["total_processed"] == 3)
    check("oldest_pending_age is null (nothing pending)", stats["oldest_pending_age_seconds"] is None)
    print(f"  {stats}")

    # ---------------------------------------------------------------- #
    # Summary                                                            #
    # ---------------------------------------------------------------- #
    print("\n" + SEP2)
    print("  SUMMARY")
    print(SEP2)
    print(f"  Endpoints tested : 8")
    print(f"  Events submitted : 3  (+ 2 invalid confidence rejected)")
    print(f"  Approved         : 1  ({req_id_approve})")
    print(f"  Rejected         : 1  ({req_id_reject})")
    print(f"  Expired          : 1  ({req_id_expire})")
    print(f"  Assertions       : all passed")

    print("\n" + SEP2)
    print("  SMOKE TEST PASSED")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()