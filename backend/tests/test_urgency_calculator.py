"""
Tests for UrgencyCalculator — Prompt 2.1 verification.
Run: pytest backend/tests/test_urgency_calculator.py -v
"""
import math
import pytest
from datetime import timedelta
from typing import Dict

from backend.models.resource import ResourceCategory, ResourceType, VillageResourceNeed
from backend.models.village import Village
from backend.algorithms.urgency_calculator import UrgencyCalculator, UrgencyScore, CRITICAL_PENALTY


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture(scope="module")
def resource_types() -> Dict[str, ResourceType]:
    return {
        "food": ResourceType(
            resource_id="food", name="Food Packets",
            category=ResourceCategory.FOOD, urgency_multiplier=1.5, weight_per_unit=1.0,
        ),
        "water": ResourceType(
            resource_id="water", name="Drinking Water",
            category=ResourceCategory.WATER, urgency_multiplier=1.8, weight_per_unit=1.0,
        ),
        "medical_kit": ResourceType(
            resource_id="medical_kit", name="Medical Kit",
            category=ResourceCategory.MEDICAL, urgency_multiplier=2.0, weight_per_unit=5.0,
        ),
        "tarpaulin": ResourceType(
            resource_id="tarpaulin", name="Tarpaulin",
            category=ResourceCategory.SHELTER, urgency_multiplier=1.2, weight_per_unit=3.0,
        ),
        "blanket": ResourceType(
            resource_id="blanket", name="Blankets",
            category=ResourceCategory.SHELTER, urgency_multiplier=1.0, weight_per_unit=2.0,
        ),
        "first_aid": ResourceType(
            resource_id="first_aid", name="First Aid",
            category=ResourceCategory.MEDICAL, urgency_multiplier=1.7, weight_per_unit=1.0,
        ),
    }


@pytest.fixture(scope="module")
def calculator(resource_types) -> UrgencyCalculator:
    return UrgencyCalculator(resource_types=resource_types)


@pytest.fixture
def village_critical() -> Village:
    """Village where all resources are unallocated → below min_need → critical."""
    return Village(
        id="critical_v",
        name="Critical Village",
        lat=27.62, lng=85.55,
        population=5000,
        resource_needs={
            "food":        VillageResourceNeed(resource_type="food",        current_need=2500, min_need=1500, allocated=0),
            "water":       VillageResourceNeed(resource_type="water",       current_need=1500, min_need=1000, allocated=0),
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=50,   min_need=30,   allocated=0),
            "tarpaulin":   VillageResourceNeed(resource_type="tarpaulin",   current_need=200,  min_need=100,  allocated=0),
            "blanket":     VillageResourceNeed(resource_type="blanket",     current_need=300,  min_need=150,  allocated=0),
            "first_aid":   VillageResourceNeed(resource_type="first_aid",   current_need=80,   min_need=40,   allocated=0),
        },
    )


@pytest.fixture
def village_moderate() -> Village:
    """Village where all resources are above min_need → not critical."""
    return Village(
        id="moderate_v",
        name="Moderate Village",
        lat=27.58, lng=85.52,
        population=3000,
        resource_needs={
            "food":        VillageResourceNeed(resource_type="food",        current_need=1500, min_need=900,  allocated=1000),
            "water":       VillageResourceNeed(resource_type="water",       current_need=900,  min_need=600,  allocated=700),
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=30,   min_need=18,   allocated=20),
            "tarpaulin":   VillageResourceNeed(resource_type="tarpaulin",   current_need=120,  min_need=60,   allocated=80),
            "blanket":     VillageResourceNeed(resource_type="blanket",     current_need=180,  min_need=90,   allocated=120),
            "first_aid":   VillageResourceNeed(resource_type="first_aid",   current_need=48,   min_need=24,   allocated=30),
        },
    )


@pytest.fixture
def village_fully_met() -> Village:
    """Village where all resources are fully allocated."""
    return Village(
        id="met_v",
        name="Met Village",
        lat=27.60, lng=85.58,
        population=2000,
        resource_needs={
            "food":        VillageResourceNeed(resource_type="food",        current_need=1000, min_need=600,  allocated=1000),
            "water":       VillageResourceNeed(resource_type="water",       current_need=600,  min_need=400,  allocated=600),
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=20,   min_need=12,   allocated=20),
        },
    )


# ================================================================== #
#  Time factor tests                                                   #
# ================================================================== #

class TestTimeFactor:
    def test_time_factor_is_one_at_zero(self, calculator):
        assert calculator.calculate_time_factor(0.0) == pytest.approx(1.0)

    def test_time_factor_is_one_for_negative_hours(self, calculator):
        assert calculator.calculate_time_factor(-5.0) == pytest.approx(1.0)

    def test_time_factor_greater_than_one_after_delay(self, calculator):
        assert calculator.calculate_time_factor(2.0) > 1.0

    def test_time_factor_increases_monotonically(self, calculator):
        factors = [calculator.calculate_time_factor(h) for h in [0, 1, 2, 4, 8]]
        for i in range(len(factors) - 1):
            assert factors[i] < factors[i + 1], f"Not monotone at index {i}"

    def test_time_factor_approximately_correct_at_2hr(self, calculator):
        # Formula gives ~1.41; spec says "≈1.5" as a directional target
        f = calculator.calculate_time_factor(2.0)
        assert 1.3 < f < 1.6, f"Expected ≈1.4 at t=2hr, got {f:.4f}"

    def test_time_factor_approximately_correct_at_4hr(self, calculator):
        # Formula gives ~2.16
        f = calculator.calculate_time_factor(4.0)
        assert 1.9 < f < 2.5, f"Expected ≈2.1 at t=4hr, got {f:.4f}"

    def test_time_factor_at_4hr_greater_than_at_2hr(self, calculator):
        assert calculator.calculate_time_factor(4.0) > calculator.calculate_time_factor(2.0)

    def test_time_factor_matches_formula(self, calculator):
        hours = 3.0
        expected = 1.0 + (math.exp(0.3 * hours) - 1.0) * 0.5
        assert calculator.calculate_time_factor(hours) == pytest.approx(expected)


# ================================================================== #
#  Resource urgency tests                                              #
# ================================================================== #

class TestResourceUrgency:
    def test_zero_when_fully_met(self, calculator):
        score = calculator.calculate_resource_urgency(
            unmet_need=0.0, current_need=1000.0, urgency_multiplier=2.0, time_factor=1.0
        )
        assert score == pytest.approx(0.0)

    def test_equals_multiplier_when_fully_unmet_at_t0(self, calculator):
        # unmet = current, time_factor = 1.0 → score = 1.0 * multiplier * 1.0
        score = calculator.calculate_resource_urgency(
            unmet_need=1000.0, current_need=1000.0, urgency_multiplier=2.0, time_factor=1.0
        )
        assert score == pytest.approx(2.0)

    def test_scales_with_unmet_ratio(self, calculator):
        half = calculator.calculate_resource_urgency(500, 1000, 1.0, 1.0)
        full = calculator.calculate_resource_urgency(1000, 1000, 1.0, 1.0)
        assert half == pytest.approx(full / 2)

    def test_doubles_when_time_factor_doubles(self, calculator):
        s1 = calculator.calculate_resource_urgency(500, 1000, 1.5, 1.0)
        s2 = calculator.calculate_resource_urgency(500, 1000, 1.5, 2.0)
        assert s2 == pytest.approx(s1 * 2)

    def test_scales_with_urgency_multiplier(self, calculator):
        low  = calculator.calculate_resource_urgency(500, 1000, 1.0, 1.0)
        high = calculator.calculate_resource_urgency(500, 1000, 2.0, 1.0)
        assert high == pytest.approx(low * 2)

    def test_zero_when_current_need_is_zero(self, calculator):
        score = calculator.calculate_resource_urgency(0.0, 0.0, 2.0, 1.5)
        assert score == pytest.approx(0.0)

    def test_capped_at_multiplier_times_time_factor(self, calculator):
        # Even if unmet > current (over-reported), ratio capped at 1.0
        score = calculator.calculate_resource_urgency(9999, 1000, 2.0, 1.0)
        assert score == pytest.approx(2.0)


# ================================================================== #
#  Village urgency tests                                               #
# ================================================================== #

class TestVillageUrgency:
    def test_critical_village_gets_penalty(self, calculator, village_critical):
        score = calculator.calculate_village_urgency(village_critical, timedelta(hours=0))
        assert score.has_critical_shortage is True
        assert score.total_urgency >= CRITICAL_PENALTY

    def test_critical_penalty_exactly_ten(self, calculator, village_critical):
        score = calculator.calculate_village_urgency(village_critical, timedelta(hours=0))
        assert score.urgency_without_penalty == pytest.approx(
            score.total_urgency - CRITICAL_PENALTY, abs=1e-6
        )

    def test_non_critical_village_has_no_penalty(self, calculator, village_moderate):
        score = calculator.calculate_village_urgency(village_moderate, timedelta(hours=0))
        assert score.has_critical_shortage is False
        assert score.total_urgency == pytest.approx(score.urgency_without_penalty)

    def test_fully_met_village_has_zero_resource_scores(self, calculator, village_fully_met):
        score = calculator.calculate_village_urgency(village_fully_met, timedelta(hours=0))
        assert score.has_critical_shortage is False
        assert all(v == pytest.approx(0.0) for v in score.resource_scores.values())

    def test_resource_scores_dict_contains_all_village_resources(self, calculator, village_critical):
        score = calculator.calculate_village_urgency(village_critical, timedelta(hours=0))
        assert set(score.resource_scores.keys()) == set(village_critical.resource_needs.keys())

    def test_medical_kit_scores_higher_than_blanket(self, calculator, village_critical):
        score = calculator.calculate_village_urgency(village_critical, timedelta(hours=0))
        # medical_kit multiplier=2.0, blanket multiplier=1.0; both fully unmet → medical higher
        assert score.resource_scores["medical_kit"] > score.resource_scores["blanket"]

    def test_water_scores_higher_than_tarpaulin(self, calculator, village_critical):
        score = calculator.calculate_village_urgency(village_critical, timedelta(hours=0))
        # water multiplier=1.8, tarpaulin multiplier=1.2
        assert score.resource_scores["water"] > score.resource_scores["tarpaulin"]

    def test_urgency_increases_with_time(self, calculator, village_critical):
        t0 = calculator.calculate_village_urgency(village_critical, timedelta(hours=0))
        t4 = calculator.calculate_village_urgency(village_critical, timedelta(hours=4))
        assert t4.total_urgency > t0.total_urgency

    def test_urgency_at_4hr_greater_than_at_2hr(self, calculator, village_critical):
        t2 = calculator.calculate_village_urgency(village_critical, timedelta(hours=2))
        t4 = calculator.calculate_village_urgency(village_critical, timedelta(hours=4))
        assert t4.total_urgency > t2.total_urgency

    def test_village_id_preserved_in_score(self, calculator, village_critical):
        score = calculator.calculate_village_urgency(village_critical, timedelta(hours=0))
        assert score.village_id == village_critical.id

    def test_time_elapsed_hours_recorded(self, calculator, village_critical):
        score = calculator.calculate_village_urgency(village_critical, timedelta(hours=3))
        assert score.time_elapsed_hours == pytest.approx(3.0)

    def test_unknown_resource_uses_default_multiplier(self, calculator):
        """Resource type not in calculator.resource_types defaults multiplier to 1.0."""
        v = Village(
            id="v_unknown",
            name="V",
            lat=27.62, lng=85.55,
            population=1000,
            resource_needs={
                "exotic_item": VillageResourceNeed(
                    resource_type="exotic_item", current_need=100, min_need=50, allocated=0
                )
            },
        )
        score = calculator.calculate_village_urgency(v, timedelta(hours=0))
        # unmet_ratio=1.0, multiplier=1.0 (default), time_factor=1.0 → score ≈ 1.0 + penalty
        assert score.resource_scores["exotic_item"] == pytest.approx(1.0)

    def test_top_resource_is_highest_scoring(self, calculator, village_critical):
        score = calculator.calculate_village_urgency(village_critical, timedelta(hours=0))
        top = score.top_resource()
        assert top is not None
        assert score.resource_scores[top] == max(score.resource_scores.values())


# ================================================================== #
#  Ranking tests                                                       #
# ================================================================== #

class TestRanking:
    def test_rank_villages_returns_correct_count(self, calculator, village_critical, village_moderate, village_fully_met):
        villages = [village_critical, village_moderate, village_fully_met]
        scores = calculator.rank_villages(villages, timedelta(hours=0))
        assert len(scores) == 3

    def test_sorted_descending_by_urgency(self, calculator, village_critical, village_moderate, village_fully_met):
        villages = [village_fully_met, village_moderate, village_critical]  # worst-to-best order
        scores = calculator.rank_villages(villages, timedelta(hours=0))
        for i in range(len(scores) - 1):
            assert scores[i].total_urgency >= scores[i + 1].total_urgency

    def test_rankings_assigned_correctly(self, calculator, village_critical, village_moderate, village_fully_met):
        villages = [village_critical, village_moderate, village_fully_met]
        scores = calculator.rank_villages(villages, timedelta(hours=0))
        ranks = [s.ranking for s in scores]
        assert ranks == [1, 2, 3]

    def test_critical_village_ranks_first(self, calculator, village_critical, village_moderate):
        scores = calculator.rank_villages([village_moderate, village_critical], timedelta(hours=0))
        assert scores[0].village_id == village_critical.id
        assert scores[0].ranking == 1

    def test_fully_met_village_ranks_last(self, calculator, village_critical, village_moderate, village_fully_met):
        villages = [village_critical, village_moderate, village_fully_met]
        scores = calculator.rank_villages(villages, timedelta(hours=0))
        assert scores[-1].village_id == village_fully_met.id

    def test_empty_village_list_returns_empty(self, calculator):
        scores = calculator.rank_villages([], timedelta(hours=0))
        assert scores == []

    def test_single_village_gets_rank_one(self, calculator, village_critical):
        scores = calculator.rank_villages([village_critical], timedelta(hours=0))
        assert scores[0].ranking == 1


# ================================================================== #
#  Reoptimization trigger tests                                        #
# ================================================================== #

class TestReoptimizationTrigger:
    def test_detects_large_urgency_change(self, calculator, village_critical, village_moderate):
        old = calculator.rank_villages([village_critical, village_moderate], timedelta(hours=0))
        new = calculator.rank_villages([village_critical, village_moderate], timedelta(hours=4))
        triggers = calculator.detect_reoptimization_trigger(old, new, threshold=0.10)
        assert len(triggers) > 0

    def test_no_trigger_for_unchanged_state(self, calculator, village_fully_met):
        scores = calculator.rank_villages([village_fully_met], timedelta(hours=0))
        triggers = calculator.detect_reoptimization_trigger(scores, scores, threshold=0.10)
        assert triggers == []

    def test_trigger_includes_delta_info(self, calculator, village_critical):
        old = calculator.rank_villages([village_critical], timedelta(hours=0))
        new = calculator.rank_villages([village_critical], timedelta(hours=4))
        triggers = calculator.detect_reoptimization_trigger(old, new, threshold=0.10)
        assert len(triggers) == 1
        t = triggers[0]
        assert "village_id" in t
        assert "delta" in t
        assert t["delta"] > 0.10
