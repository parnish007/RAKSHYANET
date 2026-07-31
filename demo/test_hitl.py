#!/usr/bin/env python3
"""
Smoke test: HITL Approval System -- Prompt 4.1
Demonstrates the full Human-in-the-Loop workflow:
  - Submit 3 events (approve one, reject one, let one expire)
  - Calculate impact preview for each
  - Print approval history
  - Assert correct final statuses

Run from project root:
    python demo/test_hitl.py
"""
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.hitl.approval_queue import ApprovalQueue, ApprovalStatus
from backend.hitl.impact_analyzer import ImpactAnalyzer
from backend.models.village import Village
from backend.rag.news_analyzer import NewsEvent
from backend.algorithms.vrp_solver import VRPSolution, VillageAllocation, Route

SEP  = "-" * 65
SEP2 = "=" * 65


# ------------------------------------------------------------------ #
#  Test data builders                                                  #
# ------------------------------------------------------------------ #

def _make_event(
    event_id: str,
    severity: int,
    confidence: float,
    event_type: str,
    location: str,
    affected: list,
    raw_text: str,
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        raw_text=raw_text,
        location=[location],
        severity=severity,
        confidence=confidence,
        affected_villages=affected,
        resource_implications={"food": 20.0, "medical_kit": 15.0},
        requires_hitl=(0.5 <= confidence < 0.8),
    )


def _make_villages() -> list:
    data = [
        ("dhulikhel", 0.75, 8500),
        ("panauti",   0.60, 6200),
        ("banepa",    0.45, 7100),
        ("namobuddha", 0.55, 3400),
    ]
    return [
        Village(
            id=vid, name=vid.capitalize(),
            lat=27.6, lng=85.5,
            population=pop,
            terrain_difficulty=1.5,
            urgency_score=urg,
            disaster_impact=0.5,
        )
        for vid, urg, pop in data
    ]


def _make_vrp(villages: list) -> VRPSolution:
    allocations = [
        VillageAllocation(
            village_id=v.id,
            allocated_resources={"food": 50.0, "medical_kit": 30.0},
            vehicle_assignments=["v_heli_01"],
            eta_minutes=45.0,
            satisfied=True,
        )
        for v in villages
    ]
    route = Route(
        vehicle_id="v_heli_01",
        stops=[v.id for v in villages],
        total_distance_km=85.0,
        estimated_time_hours=1.5,
    )
    return VRPSolution(routes=[route], allocations=allocations, total_distance_km=85.0)


# ------------------------------------------------------------------ #
#  Print helpers                                                       #
# ------------------------------------------------------------------ #

def bar(fraction: float, width: int = 16) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def status_tag(status: ApprovalStatus) -> str:
    return {
        ApprovalStatus.PENDING:  "[PENDING ]",
        ApprovalStatus.APPROVED: "[APPROVED]",
        ApprovalStatus.REJECTED: "[REJECTED]",
        ApprovalStatus.EXPIRED:  "[EXPIRED ]",
    }[status]


def print_event_card(event: NewsEvent, label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'Event ID':<20}: {event.event_id}")
    print(f"  {'Raw text':<20}: {event.raw_text[:55]}...")
    print(f"  {'Confidence':<20}: {event.confidence:.2f}  {bar(event.confidence)}")
    print(f"  {'Severity':<20}: {event.severity}/10")
    print(f"  {'Affected villages':<20}: {', '.join(event.affected_villages)}")
    print(f"  {'Requires HITL':<20}: {event.requires_hitl}")


def print_impact(preview, label: str = "Impact Preview") -> None:
    print(f"\n  -- {label} --")
    for vid, delta in preview.urgency_changes.items():
        print(f"     {vid:<16} urgency delta={delta:+.4f}")
    print(f"     Welfare estimate : {preview.welfare_improvement_estimate:.4f}  "
          f"{bar(preview.welfare_improvement_estimate)}")
    for vid, eta_d in preview.eta_changes.items():
        sign = "+" if eta_d >= 0 else ""
        print(f"     {vid:<16} ETA shift={sign}{eta_d} min")


def print_history(history: list) -> None:
    print(f"\n  {'Request ID':<16} {'Event ID':<14} {'Status':<12} "
          f"{'Reviewer':<14} {'Reason'}")
    print(f"  {'-'*15} {'-'*13} {'-'*11} {'-'*13} {'-'*20}")
    for req in history:
        reason = req.rejection_reason or ""
        reviewer = req.reviewed_by or "-"
        print(f"  {req.request_id:<16} {req.event_id:<14} "
              f"{req.status.value:<12} {reviewer:<14} {reason}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    print("\n" + SEP2)
    print("  RAKSHYANET -- HITL APPROVAL SYSTEM SMOKE TEST  (Prompt 4.1)")
    print(SEP2)

    villages = _make_villages()
    vrp      = _make_vrp(villages)
    analyzer = ImpactAnalyzer(current_villages=villages, current_routes=vrp)
    queue    = ApprovalQueue(timeout_minutes=5)

    # ---- Define 3 test events ------------------------------------ #
    evt_approve = _make_event(
        event_id="evt_flood_01",
        severity=8,
        confidence=0.72,
        event_type="flood",
        location="Dhulikhel",
        affected=["dhulikhel", "panauti"],
        raw_text="Flash flood reported near Dhulikhel. Multiple families displaced. "
                 "Road to Panauti submerged. Relief teams requested.",
    )
    evt_reject = _make_event(
        event_id="evt_rumor_02",
        severity=5,
        confidence=0.55,
        event_type="landslide",
        location="Banepa",
        affected=["banepa"],
        raw_text="Unconfirmed reports of minor landslide near Banepa. Source unknown. "
                 "Not sure if road is blocked or not.",
    )
    evt_expire = _make_event(
        event_id="evt_stale_03",
        severity=6,
        confidence=0.63,
        event_type="earthquake",
        location="Namobuddha",
        affected=["namobuddha"],
        raw_text="Aftershock felt near Namobuddha. Some structural damage reported. "
                 "Awaiting official confirmation from district administration.",
    )

    # ---------------------------------------------------------------- #
    # STEP 1: Submit all 3 events                                       #
    # ---------------------------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  STEP 1: SUBMITTING 3 EVENTS TO APPROVAL QUEUE")
    print(f"  {SEP}")

    req_approve = queue.submit_for_review(evt_approve)
    req_reject  = queue.submit_for_review(evt_reject)
    req_expire  = queue.submit_for_review(evt_expire)

    print_event_card(evt_approve, f"[1] {evt_approve.event_id}  -- will APPROVE")
    print_event_card(evt_reject,  f"[2] {evt_reject.event_id}  -- will REJECT")
    print_event_card(evt_expire,  f"[3] {evt_expire.event_id}  -- will EXPIRE")

    print(f"\n  Pending queue size : {len(queue.get_pending())} (expected 3)")

    # ---------------------------------------------------------------- #
    # STEP 2: Impact preview for each pending request                   #
    # ---------------------------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  STEP 2: IMPACT PREVIEWS")
    print(f"  {SEP}")

    for req, lbl in [
        (req_approve, "evt_flood_01"),
        (req_reject,  "evt_rumor_02"),
        (req_expire,  "evt_stale_03"),
    ]:
        preview = analyzer.calculate_impact(req.news_event)
        print_impact(preview, label=lbl)

    # ---------------------------------------------------------------- #
    # STEP 3: Approve first event                                       #
    # ---------------------------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  STEP 3: COORDINATOR APPROVES evt_flood_01")
    print(f"  {SEP}")

    approved = queue.approve(req_approve.request_id, reviewer="coord_maya")
    print(f"  {status_tag(approved.status)}  {approved.request_id}")
    print(f"  Reviewed by : {approved.reviewed_by}")
    print(f"  Reviewed at : {approved.reviewed_at}")

    # ---------------------------------------------------------------- #
    # STEP 4: Reject second event                                       #
    # ---------------------------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  STEP 4: COORDINATOR REJECTS evt_rumor_02")
    print(f"  {SEP}")

    rejected = queue.reject(
        req_reject.request_id,
        reviewer="coord_maya",
        reason="Unverified source — no official confirmation",
    )
    print(f"  {status_tag(rejected.status)}  {rejected.request_id}")
    print(f"  Reviewed by : {rejected.reviewed_by}")
    print(f"  Reason      : {rejected.rejection_reason}")

    # ---------------------------------------------------------------- #
    # STEP 5: Simulate expiry of third event                            #
    # ---------------------------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  STEP 5: SIMULATING EXPIRY OF evt_stale_03")
    print(f"  {SEP}")

    # Backdate expires_at so expire_old_requests() catches it
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    req_expire.expires_at = past
    print(f"  Backdated expires_at to 1 minute ago.")

    expired_ids = queue.expire_old_requests()
    print(f"  Expired IDs : {expired_ids}")
    print(f"  {status_tag(req_expire.status)}  {req_expire.request_id}")

    # ---------------------------------------------------------------- #
    # STEP 6: Final queue state                                         #
    # ---------------------------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  STEP 6: FINAL QUEUE STATE")
    print(f"  {SEP}")

    pending = queue.get_pending()
    print(f"  Pending requests   : {len(pending)} (expected 0)")

    history = queue.get_history()
    print(f"  History entries    : {len(history)} (expected 3)")
    print_history(history)

    # ---------------------------------------------------------------- #
    # STEP 7: Assertions                                                #
    # ---------------------------------------------------------------- #
    print(f"\n  {SEP}")
    print(f"  STEP 7: ASSERTIONS")
    print(f"  {SEP}")

    assert req_approve.status == ApprovalStatus.APPROVED, \
        f"Expected APPROVED, got {req_approve.status}"
    print("  [OK] evt_flood_01 -> APPROVED")

    assert req_approve.reviewed_by == "coord_maya"
    print("  [OK] reviewed_by == 'coord_maya'")

    assert req_reject.status == ApprovalStatus.REJECTED
    print("  [OK] evt_rumor_02 -> REJECTED")

    assert req_reject.rejection_reason == "Unverified source — no official confirmation"
    print("  [OK] rejection_reason stored correctly")

    assert req_expire.status == ApprovalStatus.EXPIRED
    print("  [OK] evt_stale_03 -> EXPIRED")

    assert req_expire.request_id in expired_ids
    print("  [OK] expired ID returned by expire_old_requests()")

    assert len(queue.get_pending()) == 0
    print("  [OK] pending queue is empty after all decisions")

    assert len(queue.get_history()) == 3
    print("  [OK] history contains 3 resolved requests")

    # Verify history IDs
    history_ids = {r.request_id for r in history}
    assert req_approve.request_id in history_ids
    assert req_reject.request_id in history_ids
    assert req_expire.request_id in history_ids
    print("  [OK] all 3 requests present in history")

    # Re-approve already-approved should raise
    try:
        queue.approve(req_approve.request_id, reviewer="other")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  [OK] double-approve raises ValueError")

    # Approve non-existent should raise
    try:
        queue.approve("req_ghost", reviewer="ghost")
        assert False, "Should have raised KeyError"
    except KeyError:
        print("  [OK] unknown request_id raises KeyError")

    # Impact preview structure check
    preview = analyzer.calculate_impact(evt_approve)
    assert isinstance(preview.welfare_improvement_estimate, float)
    assert 0.0 <= preview.welfare_improvement_estimate <= 1.0
    assert set(preview.affected_villages) == set(evt_approve.affected_villages)
    print("  [OK] ImpactPreview structure valid")

    # ---------------------------------------------------------------- #
    # Summary                                                           #
    # ---------------------------------------------------------------- #
    print("\n" + SEP2)
    print("  SUMMARY")
    print(SEP2)
    print(f"  Events submitted   : 3")
    print(f"  Approved           : 1  ({req_approve.request_id})")
    print(f"  Rejected           : 1  ({req_reject.request_id})")
    print(f"  Expired            : 1  ({req_expire.request_id})")
    print(f"  Pending remaining  : 0")
    print(f"  History entries    : {len(queue.get_history())}")

    print("\n" + SEP2)
    print("  SMOKE TEST PASSED")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()