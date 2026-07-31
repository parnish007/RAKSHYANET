#!/usr/bin/env python3
"""
Smoke test: News Analyzer RAG Pipeline -- Prompt 3.1
Loads 5 events from demo/mock_news_timeline.json, runs the analyzer
on each, and asserts the expected action (AUTO_OPTIMIZE / HITL_REQUIRED / IGNORE).

Run from project root:
    python demo/test_news_analysis.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.models.resource import VillageResourceNeed
from backend.models.village import Village
from backend.rag.news_analyzer import (
    ACTION_AUTO_OPTIMIZE,
    ACTION_HITL_REQUIRED,
    ACTION_IGNORE,
    NewsAnalyzer,
    IntelligenceReport,
)

SEP  = "-" * 65
SEP2 = "=" * 65

# ------------------------------------------------------------------ #
#  Expected actions per event (ground truth for assertions)           #
# ------------------------------------------------------------------ #
# Derived from source_type:
#   verified_government / verified_ngo  → multi_source_confirmed=True
#   verified_news                       → multi_source_confirmed=False
#   unverified                          → multi_source_confirmed=False
MULTI_SOURCE_BY_TYPE = {
    "verified_government": True,
    "verified_ngo":        True,
    "verified_news":       False,
    "unverified":          False,
}

EXPECTED_ACTIONS = {
    "evt_001": ACTION_AUTO_OPTIMIZE,
    "evt_002": ACTION_HITL_REQUIRED,
    "evt_003": ACTION_AUTO_OPTIMIZE,
    "evt_004": ACTION_IGNORE,
    "evt_005": ACTION_AUTO_OPTIMIZE,
}


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_villages(vil_data: dict) -> list:
    villages = []
    for v in vil_data["villages"]:
        needs = {r: VillageResourceNeed(**nd) for r, nd in v["resource_needs"].items()}
        villages.append(Village(
            id=v["id"],
            name=v["name"],
            lat=v["lat"],
            lng=v["lng"],
            population=v["population"],
            accessibility=v.get("accessibility", "road"),
            has_medical_facility=v.get("has_medical_facility", False),
            resource_needs=needs,
        ))
    return villages


def action_badge(action: str) -> str:
    if action == ACTION_AUTO_OPTIMIZE:
        return "[AUTO]"
    if action == ACTION_HITL_REQUIRED:
        return "[HITL]"
    return "[IGNR]"


def conf_bar(conf: float, width: int = 20) -> str:
    filled = int(round(conf * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def print_report(idx: int, evt: dict, report: IntelligenceReport, expected: str) -> None:
    ok = "PASS" if report.recommended_action == expected else "FAIL"
    badge = action_badge(report.recommended_action)
    print(f"\n  [{ok}] Event {idx}: {evt['id']}")
    print(f"  {SEP}")
    print(f"  Source   : {evt['source']}")
    print(f"  Text     : {evt['text'][:72]}...")
    print(f"  Severity : {report.event.severity}/10")
    print(f"  Confidence: {report.event.confidence:.4f}  {conf_bar(report.event.confidence)}")
    print(f"  Action   : {badge} {report.recommended_action}  (expected: {expected})")
    print(f"  Affected : {report.event.affected_villages or '(none)'}")
    if report.event.resource_implications:
        impl_str = "  ".join(f"{k}=+{v:.0f}" for k, v in report.event.resource_implications.items())
        print(f"  Resources: {impl_str}")
    print(f"  Reasoning: {report.confidence_reasoning}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    news_path = ROOT / "demo" / "mock_news_timeline.json"
    vil_path  = ROOT / "backend" / "data" / "nepal_villages.json"

    for p in (news_path, vil_path):
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    news_data = load_json(news_path)
    vil_data  = load_json(vil_path)
    villages  = build_villages(vil_data)

    analyzer = NewsAnalyzer()

    print("\n" + SEP2)
    print("  RAKSHYANET -- NEWS ANALYZER SMOKE TEST  (Prompt 3.1)")
    print(SEP2)
    print(f"  Analyzer model   : {analyzer.model}")
    print(f"  High threshold   : {analyzer.confidence_thresholds['high']}")
    print(f"  Medium threshold : {analyzer.confidence_thresholds['medium']}")
    print(f"  Villages loaded  : {len(villages)}")
    print(f"  Events to analyze: {len(news_data['events'])}")

    results = []
    failures = []

    for idx, evt in enumerate(news_data["events"], start=1):
        multi = MULTI_SOURCE_BY_TYPE.get(evt.get("source_type", "unverified"), False)
        report = analyzer.analyze_news(
            raw_text=evt["text"],
            villages=villages,
            source=evt["source"],
            multi_source_confirmed=multi,
        )
        expected = EXPECTED_ACTIONS.get(evt["id"], ACTION_IGNORE)
        print_report(idx, evt, report, expected)
        results.append((evt["id"], report, expected))
        if report.recommended_action != expected:
            failures.append(evt["id"])

    # ---------------------------------------------------------------- #
    #  Summary table                                                    #
    # ---------------------------------------------------------------- #
    print("\n" + SEP2)
    print("  SUMMARY")
    print(SEP2)
    print(f"  {'Event':<12} {'Confidence':>10}  {'Severity':>8}  {'Action':<16}  {'Result'}")
    print(f"  {'-'*11} {'----------':>10}  {'--------':>8}  {'-'*16}  ------")
    for eid, report, expected in results:
        ok = "PASS" if report.recommended_action == expected else "FAIL ***"
        print(f"  {eid:<12} {report.event.confidence:>10.4f}  "
              f"{report.event.severity:>8}  "
              f"{report.recommended_action:<16}  {ok}")

    passed = len(results) - len(failures)
    print(f"\n  {passed}/{len(results)} events matched expected action.")

    # ---------------------------------------------------------------- #
    #  Assertions                                                       #
    # ---------------------------------------------------------------- #
    if failures:
        print(f"\n  FAILED events: {failures}")
        sys.exit(1)

    # Structural assertions
    for _, report, _ in results:
        assert isinstance(report, IntelligenceReport)
        assert 0.0 <= report.event.confidence <= 1.0
        assert 0 <= report.event.severity <= 10
        assert report.event.timestamp != ""
        assert report.recommended_action in (
            ACTION_AUTO_OPTIMIZE, ACTION_HITL_REQUIRED, ACTION_IGNORE
        )
        if report.recommended_action == ACTION_HITL_REQUIRED:
            assert report.event.requires_hitl is True
        else:
            assert report.event.requires_hitl is False

    print("\n" + SEP2)
    print("  SMOKE TEST PASSED")
    print(SEP2 + "\n")


if __name__ == "__main__":
    main()
