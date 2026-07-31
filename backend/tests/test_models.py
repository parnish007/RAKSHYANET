"""
Tests for all Pydantic data models — Prompt 1.2 verification.
Run: pytest backend/tests/test_models.py -v
"""
import math
import pytest
from datetime import datetime, timedelta

from backend.models import (
    Village,
    Helicopter, Truck, VehicleState, VehicleCategory, CargoItem,
    AllocationResult, VehicleRoute, RouteWaypoint, KKTConditions, ConvergencePoint,
    NewsEvent,
    HITLDecision, HITLRequest, HITLDecisionType, HITLStatus,
)


# ================================================================== #
#  Village                                                             #
# ================================================================== #

class TestVillage:
    def make_village(self, **overrides):
        defaults = dict(
            id="dhulikhel",
            name="Dhulikhel",
            lat=27.62,
            lng=85.55,
            population=5000,
            current_need=2500.0,
            min_need=1500.0,
            urgency_score=0.65,
            disaster_impact=0.70,
        )
        defaults.update(overrides)
        return Village(**defaults)

    def test_creates_with_valid_data(self):
        v = self.make_village()
        assert v.id == "dhulikhel"
        assert v.population == 5000

    def test_urgency_score_bounds(self):
        with pytest.raises(Exception):
            self.make_village(urgency_score=1.5)
        with pytest.raises(Exception):
            self.make_village(urgency_score=-0.1)

    def test_min_need_cannot_exceed_current_need(self):
        with pytest.raises(Exception):
            self.make_village(current_need=500.0, min_need=600.0)

    def test_calculate_distance_from_depot(self, depot_location):
        v = self.make_village()
        dist = v.calculate_distance_from(*depot_location)
        assert dist > 0
        # Dhulikhel is ~30km from Kathmandu
        assert 20.0 < dist < 45.0

    def test_calculate_distance_same_point(self):
        v = self.make_village(lat=27.62, lng=85.55)
        assert v.calculate_distance_from(27.62, 85.55) == pytest.approx(0.0, abs=0.01)

    def test_update_urgency_saves_previous(self):
        v = self.make_village(urgency_score=0.5)
        v.update_urgency(0.9)
        assert v.urgency_score == pytest.approx(0.9)
        assert v.previous_urgency == pytest.approx(0.5)

    def test_update_urgency_clamps_to_bounds(self):
        v = self.make_village(urgency_score=0.5)
        v.update_urgency(1.5)
        assert v.urgency_score == pytest.approx(1.0)
        v.update_urgency(-0.5)
        assert v.urgency_score == pytest.approx(0.0)

    def test_urgency_delta(self):
        v = self.make_village(urgency_score=0.5, previous_urgency=0.5)
        v.update_urgency(0.8)
        assert v.urgency_delta() == pytest.approx(0.3, abs=0.001)

    def test_unmet_need_computed(self):
        v = self.make_village(current_need=2500.0, allocated=1000.0)
        assert v.unmet_need == pytest.approx(1500.0)

    def test_unmet_need_never_negative(self):
        v = self.make_village(current_need=1000.0, min_need=600.0, allocated=2000.0)
        assert v.unmet_need == pytest.approx(0.0)

    def test_higher_population_higher_fairness_weight(self):
        high = self.make_village(population=10000)
        low = self.make_village(population=500)
        assert high.fairness_weight > low.fairness_weight


# ================================================================== #
#  Vehicle                                                             #
# ================================================================== #

class TestHelicopter:
    def make_heli(self, **overrides):
        defaults = dict(id="heli_1")
        defaults.update(overrides)
        return Helicopter(**defaults)

    def test_defaults(self):
        h = self.make_heli()
        assert h.capacity_kg == 500.0
        assert h.speed_kmh == 200.0
        assert h.fuel_hours == 2.0
        assert h.terrain == "any"
        assert h.vehicle_type.category == VehicleCategory.AIRCRAFT

    def test_fuel_range(self):
        h = self.make_heli()
        assert h.fuel_range_km == pytest.approx(400.0)

    def test_initial_state_available(self):
        h = self.make_heli()
        assert h.state == VehicleState.AVAILABLE
        assert h.is_available_for_assignment

    def test_can_deliver_to_any_terrain(self):
        h = self.make_heli()
        assert h.can_deliver_to("road") is True
        assert h.can_deliver_to("mountain") is True
        assert h.can_deliver_to("any") is True

    def test_load_cargo_reduces_remaining_capacity(self):
        h = self.make_heli()
        loaded = h.load_cargo("dhulikhel", 300.0)
        assert loaded == pytest.approx(300.0)
        assert h.remaining_capacity == pytest.approx(200.0)

    def test_load_cargo_capped_at_capacity(self):
        h = self.make_heli()
        loaded = h.load_cargo("dhulikhel", 9999.0)
        assert loaded == pytest.approx(500.0)
        assert h.remaining_capacity == pytest.approx(0.0)

    def test_load_cargo_raises_if_deployed(self):
        h = self.make_heli()
        h.state = VehicleState.DEPLOYED
        with pytest.raises(ValueError, match="DEPLOYED"):
            h.load_cargo("dhulikhel", 100.0)

    def test_deploy_changes_state(self):
        h = self.make_heli()
        h.deploy("dhulikhel")
        assert h.state == VehicleState.IN_TRANSIT
        assert h.destination == "dhulikhel"

    def test_return_to_depot_resets(self):
        h = self.make_heli()
        h.deploy("dhulikhel")
        h.load_cargo("dhulikhel", 100.0)
        depot = (27.7172, 85.3240)
        h.return_to_depot(depot)
        assert h.state == VehicleState.AVAILABLE
        assert h.cargo_manifest == {}
        assert h.current_location == depot


class TestTruck:
    def make_truck(self, **overrides):
        defaults = dict(id="truck_1")
        defaults.update(overrides)
        return Truck(**defaults)

    def test_defaults(self):
        t = self.make_truck()
        assert t.capacity_kg == 2000.0
        assert t.speed_kmh == 40.0
        assert t.terrain == "roads_only"
        assert t.vehicle_type.category == VehicleCategory.GROUND_HEAVY

    def test_can_only_deliver_on_road(self):
        t = self.make_truck()
        assert t.can_deliver_to("road") is True
        assert t.can_deliver_to("mountain") is False
        assert t.can_deliver_to("any") is False


# ================================================================== #
#  AllocationResult                                                    #
# ================================================================== #

class TestAllocationResult:
    def test_empty_default(self):
        r = AllocationResult()
        assert r.nash_equilibrium_reached is False
        assert r.solver_status == "NOT_RUN"
        assert r.allocation == {}

    def test_total_allocated_to_village(self):
        r = AllocationResult(
            allocation={
                "heli_1": {"dhulikhel": 300.0, "panauti": 100.0},
                "truck_1": {"dhulikhel": 500.0},
            }
        )
        assert r.total_allocated_to("dhulikhel") == pytest.approx(800.0)
        assert r.total_allocated_to("panauti") == pytest.approx(100.0)
        assert r.total_allocated_to("unknown") == pytest.approx(0.0)

    def test_kkt_all_satisfied(self):
        kkt = KKTConditions(
            stationarity=True,
            primal_feasibility=True,
            dual_feasibility=True,
            complementary_slackness=True,
            residual=5e-7,
        )
        assert kkt.all_satisfied is True

    def test_kkt_not_satisfied_if_any_false(self):
        kkt = KKTConditions(
            stationarity=True,
            primal_feasibility=False,
            dual_feasibility=True,
            complementary_slackness=True,
        )
        assert kkt.all_satisfied is False

    def test_summary_keys(self):
        r = AllocationResult(nash_equilibrium_reached=True, solver_status="OPTIMAL")
        s = r.summary()
        assert "nash_equilibrium_reached" in s
        assert "kkt_all_satisfied" in s
        assert "solve_time_seconds" in s


# ================================================================== #
#  NewsEvent                                                           #
# ================================================================== #

class TestNewsEvent:
    def make_event(self, **overrides):
        defaults = dict(
            id="evt_001",
            source="@NepalPolice",
            text="BREAKING: Major landslide in Dhulikhel. Medical clinic buried. Casualties reported.",
        )
        defaults.update(overrides)
        return NewsEvent(**defaults)

    def test_is_trusted_source(self):
        e = self.make_event(source="@NepalPolice")
        assert e.is_trusted_source() is True

    def test_untrusted_source(self):
        e = self.make_event(source="@randomuser123")
        assert e.is_trusted_source() is False

    def test_extract_severity_keywords(self):
        e = self.make_event()
        keywords = e.extract_severity_keywords()
        assert "landslide" in keywords
        assert "buried" in keywords
        assert "medical" in keywords
        assert "casualties" in keywords

    def test_computed_severity_capped_at_1(self):
        # Pile on many keywords to ensure cap works
        e = self.make_event(
            text="buried collapse medical landslide critical casualties dead injured trapped flood"
        )
        assert e.computed_severity() <= 1.0

    def test_computed_severity_zero_for_no_keywords(self):
        e = self.make_event(text="Everything is fine today in Kathmandu.")
        assert e.computed_severity() == pytest.approx(0.0)

    def test_confidence_high_for_trusted_source(self):
        e = self.make_event(source="@NepalPolice")
        conf = e.compute_confidence(multi_source_confirmed=True, keyword_count=4)
        assert conf >= 0.8

    def test_auto_optimize_flag(self):
        e = self.make_event(confidence_score=0.9)
        assert e.auto_optimize is True

    def test_requires_hitl_flag(self):
        e = self.make_event(confidence_score=0.65)
        assert e.requires_hitl is True

    def test_severity_score_bounded(self):
        with pytest.raises(Exception):
            self.make_event(severity_score=1.5)


# ================================================================== #
#  HITLDecision                                                        #
# ================================================================== #

class TestHITL:
    def test_decision_confirm(self):
        d = HITLDecision(
            event_id="evt_001",
            decision=HITLDecisionType.CONFIRM,
            coordinator_id="coord_01",
        )
        assert d.approved is True

    def test_decision_reject(self):
        d = HITLDecision(
            event_id="evt_001",
            decision=HITLDecisionType.REJECT,
            coordinator_id="coord_01",
        )
        assert d.approved is False

    def test_auto_reject_factory(self):
        d = HITLDecision.auto_reject("evt_999")
        assert d.decision == HITLDecisionType.REJECT
        assert d.coordinator_id == "SYSTEM_TIMEOUT"
        assert d.status == HITLStatus.TIMED_OUT

    def test_hitl_request_expiry(self):
        req = HITLRequest(
            event_id="evt_001",
            news_summary="Landslide in Dhulikhel",
            estimated_impact="3 helicopters rerouted, ETA -12 min",
            confidence_score=0.72,
        )
        assert req.seconds_remaining > 0
        assert req.is_expired is False

    def test_hitl_request_expired(self):
        past = datetime.utcnow() - timedelta(seconds=400)
        req = HITLRequest(
            event_id="evt_001",
            news_summary="Old event",
            estimated_impact="None",
            confidence_score=0.72,
            requested_at=past,
            expires_at=past + timedelta(seconds=300),
        )
        assert req.is_expired is True
        assert req.seconds_remaining == pytest.approx(0.0)
