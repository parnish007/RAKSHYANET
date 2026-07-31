"""
Tests for NewsAnalyzer -- Prompt 3.1
Run: pytest backend/tests/test_news_analyzer.py -v
"""
import pytest
from typing import Dict, List

from backend.models.resource import VillageResourceNeed
from backend.models.village import Village
from backend.rag.news_analyzer import (
    ACTION_AUTO_OPTIMIZE,
    ACTION_HITL_REQUIRED,
    ACTION_IGNORE,
    IntelligenceReport,
    NewsAnalyzer,
    NewsEvent,
)


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture(scope="module")
def analyzer() -> NewsAnalyzer:
    return NewsAnalyzer()


def _make_village(vid: str, name: str, lat: float = 27.62, lng: float = 85.55) -> Village:
    needs = {
        "food": VillageResourceNeed(resource_type="food", current_need=500.0, min_need=300.0, allocated=0),
    }
    return Village(id=vid, name=name, lat=lat, lng=lng, population=1000,
                   accessibility="road", resource_needs=needs)


@pytest.fixture(scope="module")
def villages() -> List[Village]:
    return [
        _make_village("dhulikhel",  "Dhulikhel"),
        _make_village("panauti",    "Panauti"),
        _make_village("banepa",     "Banepa"),
        _make_village("namobuddha", "Namobuddha"),
        _make_village("panchkhal",  "Panchkhal"),
        _make_village("temal",      "Temal"),
        _make_village("bethanchowk","Bethanchowk"),
        _make_village("khopasi",    "Khopasi"),
    ]


@pytest.fixture(scope="module")
def high_confidence_news() -> str:
    """Official government source, specific village, multiple details."""
    return (
        "BREAKING: Major landslide in Dhulikhel. "
        "Medical clinic buried under debris. 15 people injured. "
        "Road to hospital blocked. Rescue operations ongoing. "
        "Source: Nepal Police Official."
    )


@pytest.fixture(scope="module")
def medium_confidence_news() -> str:
    """News media, district-level location, some details."""
    return (
        "Bridge collapse in Kavre District cuts off road access. "
        "500 families isolated. Relief supplies needed. "
        "Source: Kathmandu Post."
    )


@pytest.fixture(scope="module")
def low_confidence_news() -> str:
    """Unverified social media, vague location, no specifics."""
    return (
        "Heard there might be some issues somewhere near the hills. "
        "Not sure what exactly. @RandomUser123"
    )


# ================================================================== #
#  Extraction tests                                                    #
# ================================================================== #

class TestExtraction:
    def test_extract_location_from_text(self, analyzer):
        data = analyzer.extract_structured_data(
            "BREAKING: Landslide in Dhulikhel area. Road blocked."
        )
        # Dhulikhel should survive the stop-word filter
        assert any("Dhulikhel" in m or "dhulikhel" in m.lower()
                   for m in data["location"])

    def test_extract_event_type_landslide(self, analyzer):
        data = analyzer.extract_structured_data("Major landslide hit the valley.")
        assert data["event_type"] == "landslide"

    def test_extract_event_type_earthquake(self, analyzer):
        data = analyzer.extract_structured_data("Earthquake tremor felt in eastern hills.")
        assert data["event_type"] == "earthquake"

    def test_extract_event_type_flood(self, analyzer):
        data = analyzer.extract_structured_data("Flash flood warning issued for lowland areas.")
        assert data["event_type"] == "flood"

    def test_extract_event_type_bridge_collapse(self, analyzer):
        data = analyzer.extract_structured_data("Bridge collapse blocks main road to village.")
        assert data["event_type"] == "bridge_collapse"

    def test_extract_casualty_number_direct(self, analyzer):
        data = analyzer.extract_structured_data("15 people injured in the landslide.")
        assert data["casualties"] == 15

    def test_extract_casualty_number_killed(self, analyzer):
        data = analyzer.extract_structured_data("Report: 8 killed and 20 missing after flood.")
        assert data["casualties"] >= 8

    def test_extract_resource_needs_medical(self, analyzer):
        data = analyzer.extract_structured_data(
            "Hospital overwhelmed. Medical supplies urgently needed."
        )
        assert "medical_kit" in data["resources_needed"]

    def test_extract_resource_needs_rescue(self, analyzer):
        data = analyzer.extract_structured_data("People trapped under debris. Rescue teams needed.")
        assert "rescue_equipment" in data["resources_needed"]

    def test_extract_time_constraint_immediate(self, analyzer):
        data = analyzer.extract_structured_data("Immediate airlift of supplies required.")
        assert data["time_constraint"] == "immediate"

    def test_extract_time_constraint_default(self, analyzer):
        data = analyzer.extract_structured_data("Some relief may be needed eventually.")
        assert data["time_constraint"] == "72h"

    def test_extract_unknown_event_type(self, analyzer):
        data = analyzer.extract_structured_data("Some general report with no specific event.")
        assert data["event_type"] == "unknown"


# ================================================================== #
#  Confidence tests                                                    #
# ================================================================== #

class TestConfidence:
    def test_high_confidence_official_source(self, analyzer):
        data = {
            "location_specificity": 1.0,   # specific village
            "source_reliability":   1.0,   # government official
            "detail_completeness":  0.8,   # most fields filled
            "cross_validation":     1.0,   # multi-source confirmed
        }
        conf = analyzer.calculate_confidence(data)
        assert conf >= 0.80

    def test_medium_confidence_vague_location(self, analyzer):
        data = {
            "location_specificity": 0.5,   # district only
            "source_reliability":   0.8,   # news media
            "detail_completeness":  0.8,   # most fields
            "cross_validation":     0.0,   # single source
        }
        conf = analyzer.calculate_confidence(data)
        assert 0.50 <= conf < 0.80

    def test_low_confidence_social_media_rumor(self, analyzer):
        data = {
            "location_specificity": 0.2,   # vague
            "source_reliability":   0.4,   # social media
            "detail_completeness":  0.2,   # little data
            "cross_validation":     0.0,
        }
        conf = analyzer.calculate_confidence(data)
        assert conf < 0.50

    def test_confidence_decreases_with_missing_data(self, analyzer):
        full = {
            "location_specificity": 1.0,
            "source_reliability":   0.9,
            "detail_completeness":  1.0,
            "cross_validation":     0.0,
        }
        sparse = {
            "location_specificity": 1.0,
            "source_reliability":   0.9,
            "detail_completeness":  0.2,   # much less detail
            "cross_validation":     0.0,
        }
        assert analyzer.calculate_confidence(full) > analyzer.calculate_confidence(sparse)

    def test_confidence_clamped_to_one(self, analyzer):
        data = {
            "location_specificity": 1.0,
            "source_reliability":   1.0,
            "detail_completeness":  1.0,
            "cross_validation":     1.0,
        }
        assert analyzer.calculate_confidence(data) <= 1.0

    def test_confidence_nonnegative(self, analyzer):
        data = {
            "location_specificity": 0.0,
            "source_reliability":   0.0,
            "detail_completeness":  0.0,
            "cross_validation":     0.0,
        }
        assert analyzer.calculate_confidence(data) >= 0.0

    def test_cross_validation_increases_confidence(self, analyzer):
        base = {
            "location_specificity": 1.0,
            "source_reliability":   0.8,
            "detail_completeness":  0.6,
            "cross_validation":     0.0,
        }
        multi = dict(base, cross_validation=1.0)
        assert analyzer.calculate_confidence(multi) > analyzer.calculate_confidence(base)


# ================================================================== #
#  Village matching tests                                              #
# ================================================================== #

class TestVillageMatching:
    def test_exact_village_name_match(self, analyzer, villages):
        affected = analyzer.identify_affected_villages(["Dhulikhel"], villages)
        assert "dhulikhel" in affected

    def test_exact_village_id_match(self, analyzer, villages):
        affected = analyzer.identify_affected_villages(["panauti"], villages)
        assert "panauti" in affected

    def test_fuzzy_match_typo(self, analyzer, villages):
        # "Dhulkhel" is missing the 'i' — should still match "Dhulikhel"
        affected = analyzer.identify_affected_villages(["Dhulkhel"], villages)
        assert "dhulikhel" in affected

    def test_district_level_match(self, analyzer, villages):
        # "Kavre" should match all Kavre district villages present in the list
        affected = analyzer.identify_affected_villages(["Kavre"], villages)
        assert "dhulikhel" in affected
        assert "panauti"   in affected
        assert "banepa"    in affected

    def test_no_match_unrelated_location(self, analyzer, villages):
        affected = analyzer.identify_affected_villages(["London", "Paris"], villages)
        assert affected == []

    def test_multiple_villages_from_mentions(self, analyzer, villages):
        affected = analyzer.identify_affected_villages(["Dhulikhel", "Banepa"], villages)
        assert "dhulikhel" in affected
        assert "banepa"    in affected

    def test_no_duplicate_villages(self, analyzer, villages):
        # Same mention twice should not produce duplicates
        affected = analyzer.identify_affected_villages(
            ["Dhulikhel", "Dhulikhel"], villages
        )
        assert affected.count("dhulikhel") == 1

    def test_empty_mentions_returns_empty(self, analyzer, villages):
        assert analyzer.identify_affected_villages([], villages) == []


# ================================================================== #
#  Severity tests                                                      #
# ================================================================== #

class TestSeverity:
    def test_high_severity_over_50_casualties(self, analyzer):
        data = {"casualties": 55, "infrastructure_damage": "", "hospital_damage": False,
                "time_constraint": "72h", "resources_needed": []}
        assert analyzer.assess_severity(data) >= 4

    def test_medium_severity_10_to_50_casualties(self, analyzer):
        data = {"casualties": 20, "infrastructure_damage": "", "hospital_damage": False,
                "time_constraint": "72h", "resources_needed": []}
        sev = analyzer.assess_severity(data)
        assert 1 <= sev <= 4

    def test_infrastructure_damage_increases_severity(self, analyzer):
        base = {"casualties": 0, "infrastructure_damage": "", "hospital_damage": False,
                "time_constraint": "72h", "resources_needed": []}
        infra = dict(base, infrastructure_damage="road blocked")
        assert analyzer.assess_severity(infra) > analyzer.assess_severity(base)

    def test_infrastructure_adds_three_points(self, analyzer):
        no_infra = {"casualties": 0, "infrastructure_damage": "", "hospital_damage": False,
                    "time_constraint": "72h", "resources_needed": []}
        with_infra = dict(no_infra, infrastructure_damage="road blocked")
        diff = analyzer.assess_severity(with_infra) - analyzer.assess_severity(no_infra)
        assert diff == 3

    def test_immediate_time_constraint_adds_two(self, analyzer):
        base = {"casualties": 0, "infrastructure_damage": "", "hospital_damage": False,
                "time_constraint": "72h", "resources_needed": []}
        imm = dict(base, time_constraint="immediate")
        diff = analyzer.assess_severity(imm) - analyzer.assess_severity(base)
        assert diff == 2

    def test_severity_capped_at_10(self, analyzer):
        worst_case = {
            "casualties": 100,
            "infrastructure_damage": "road blocked",
            "hospital_damage": True,
            "time_constraint": "immediate",
            "resources_needed": ["medical_kit", "food"],
        }
        assert analyzer.assess_severity(worst_case) == 10

    def test_zero_severity_for_empty_data(self, analyzer):
        data = {"casualties": 0, "infrastructure_damage": "", "hospital_damage": False,
                "time_constraint": "72h", "resources_needed": []}
        assert analyzer.assess_severity(data) == 0


# ================================================================== #
#  Action determination tests                                          #
# ================================================================== #

class TestActionDetermination:
    def test_high_confidence_auto_optimize(self, analyzer):
        assert analyzer.determine_action(0.85, 8) == ACTION_AUTO_OPTIMIZE

    def test_high_confidence_boundary_auto_optimize(self, analyzer):
        assert analyzer.determine_action(0.80, 5) == ACTION_AUTO_OPTIMIZE

    def test_medium_confidence_hitl_required(self, analyzer):
        assert analyzer.determine_action(0.65, 6) == ACTION_HITL_REQUIRED

    def test_medium_confidence_boundary_hitl(self, analyzer):
        assert analyzer.determine_action(0.50, 3) == ACTION_HITL_REQUIRED

    def test_low_confidence_ignore(self, analyzer):
        assert analyzer.determine_action(0.30, 5) == ACTION_IGNORE

    def test_very_low_confidence_ignore(self, analyzer):
        assert analyzer.determine_action(0.10, 9) == ACTION_IGNORE

    def test_custom_thresholds(self):
        custom = NewsAnalyzer(confidence_thresholds={"high": 0.9, "medium": 0.6})
        # 0.85 would be AUTO with default thresholds but HITL with custom
        assert custom.determine_action(0.85, 8) == ACTION_HITL_REQUIRED
        assert custom.determine_action(0.92, 8) == ACTION_AUTO_OPTIMIZE


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def test_analyze_news_returns_intelligence_report(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages,
                                       source="Nepal Police", multi_source_confirmed=True)
        assert isinstance(report, IntelligenceReport)

    def test_high_confidence_news_auto_optimizes(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages,
                                       source="Nepal Police", multi_source_confirmed=True)
        assert report.recommended_action == ACTION_AUTO_OPTIMIZE

    def test_low_confidence_news_ignored(self, analyzer, villages, low_confidence_news):
        report = analyzer.analyze_news(low_confidence_news, villages)
        assert report.recommended_action == ACTION_IGNORE

    def test_report_event_has_raw_text(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages)
        assert report.event.raw_text == high_confidence_news

    def test_report_severity_positive_for_high_confidence(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages,
                                       source="Nepal Police", multi_source_confirmed=True)
        assert report.event.severity > 0

    def test_report_confidence_in_range(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages)
        assert 0.0 <= report.event.confidence <= 1.0

    def test_report_has_analysis_summary(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages)
        assert report.analysis_summary != ""

    def test_report_has_confidence_reasoning(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages)
        assert report.confidence_reasoning != ""

    def test_dhulikhel_news_identifies_village(self, analyzer, villages):
        text = "Landslide in Dhulikhel. Immediate medical help needed."
        report = analyzer.analyze_news(text, villages, source="Nepal Police",
                                       multi_source_confirmed=True)
        assert "dhulikhel" in report.event.affected_villages

    def test_requires_hitl_flag_set_correctly(self, analyzer, villages, medium_confidence_news):
        report = analyzer.analyze_news(medium_confidence_news, villages,
                                       source="Kathmandu Post", multi_source_confirmed=False)
        if report.recommended_action == ACTION_HITL_REQUIRED:
            assert report.event.requires_hitl is True
        else:
            assert report.event.requires_hitl is False

    def test_resource_implications_present_for_landslide(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages)
        assert len(report.event.resource_implications) > 0

    def test_sequential_events_independent(self, analyzer, villages,
                                           high_confidence_news, low_confidence_news):
        r1 = analyzer.analyze_news(high_confidence_news, villages,
                                   source="Nepal Police", multi_source_confirmed=True)
        r2 = analyzer.analyze_news(low_confidence_news, villages)
        # Results should be independent — first doesn't affect second
        assert r1.recommended_action == ACTION_AUTO_OPTIMIZE
        assert r2.recommended_action == ACTION_IGNORE

    def test_urgency_change_map_has_affected_villages(self, analyzer, villages):
        text = "Landslide in Banepa. Immediate medical help needed. 20 people injured."
        report = analyzer.analyze_news(text, villages, source="Nepal Police",
                                       multi_source_confirmed=True)
        for vid in report.event.affected_villages:
            assert vid in report.urgency_change

    def test_event_timestamp_present(self, analyzer, villages, high_confidence_news):
        report = analyzer.analyze_news(high_confidence_news, villages)
        assert report.event.timestamp != ""
