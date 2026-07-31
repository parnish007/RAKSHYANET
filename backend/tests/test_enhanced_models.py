"""
Tests for Prompt 1.4 enhanced models:
  - ResourceType, VillageResourceNeed
  - VehicleType (config template), Vehicle (unified class)
  - Village with multi-resource needs
  - config.json structure
Run: pytest backend/tests/test_enhanced_models.py -v
"""
import json
import pytest
from pathlib import Path

from backend.models import (
    ResourceCategory, ResourceType, VillageResourceNeed,
    VehicleCategory, TerrainCapability, VehicleType, Vehicle,
    Helicopter, Truck, VehicleState,
    Village,
)

ROOT = Path(__file__).parents[2]
CONFIG_FILE = ROOT / "backend" / "data" / "config.json"


# ================================================================== #
#  ResourceType                                                        #
# ================================================================== #

class TestResourceType:
    def make_resource(self, **overrides):
        defaults = dict(
            resource_id="food",
            name="Food Packets",
            category=ResourceCategory.FOOD,
            unit="kg",
            urgency_multiplier=1.5,
            weight_per_unit=1.0,
        )
        defaults.update(overrides)
        return ResourceType(**defaults)

    def test_creates_valid_resource(self):
        r = self.make_resource()
        assert r.resource_id == "food"
        assert r.category == ResourceCategory.FOOD

    def test_urgency_multiplier_bounds(self):
        with pytest.raises(Exception):
            self.make_resource(urgency_multiplier=2.5)  # max is 2.0
        with pytest.raises(Exception):
            self.make_resource(urgency_multiplier=-0.1)

    def test_is_perishable_with_shelf_life(self):
        r = self.make_resource(shelf_life_hours=72.0)
        assert r.is_perishable is True

    def test_is_not_perishable_without_shelf_life(self):
        r = self.make_resource(shelf_life_hours=None)
        assert r.is_perishable is False

    def test_all_resource_categories(self):
        for cat in ResourceCategory:
            r = self.make_resource(category=cat)
            assert r.category == cat

    def test_medical_kit_high_urgency(self):
        kit = ResourceType(
            resource_id="medical_kit",
            name="Medical Kit",
            category=ResourceCategory.MEDICAL,
            urgency_multiplier=2.0,
            weight_per_unit=5.0,
        )
        food = self.make_resource(urgency_multiplier=1.5)
        assert kit.urgency_multiplier > food.urgency_multiplier


# ================================================================== #
#  VillageResourceNeed                                                 #
# ================================================================== #

class TestVillageResourceNeed:
    def make_need(self, **overrides):
        defaults = dict(
            resource_type="food",
            current_need=1000.0,
            min_need=600.0,
            allocated=0.0,
        )
        defaults.update(overrides)
        return VillageResourceNeed(**defaults)

    def test_unmet_need_zero_when_fully_allocated(self):
        n = self.make_need(current_need=1000.0, allocated=1000.0)
        assert n.unmet_need == pytest.approx(0.0)

    def test_unmet_need_never_negative(self):
        n = self.make_need(current_need=500.0, allocated=800.0)
        assert n.unmet_need == pytest.approx(0.0)

    def test_unmet_need_correct(self):
        n = self.make_need(current_need=1000.0, allocated=300.0)
        assert n.unmet_need == pytest.approx(700.0)

    def test_critical_when_below_min(self):
        n = self.make_need(min_need=600.0, allocated=400.0)
        assert n.critical is True

    def test_not_critical_when_above_min(self):
        n = self.make_need(min_need=600.0, allocated=700.0)
        assert n.critical is False

    def test_satisfaction_ratio_zero_at_start(self):
        n = self.make_need(current_need=1000.0, allocated=0.0)
        assert n.satisfaction_ratio == pytest.approx(0.0)

    def test_satisfaction_ratio_one_when_fully_met(self):
        n = self.make_need(current_need=1000.0, allocated=1000.0)
        assert n.satisfaction_ratio == pytest.approx(1.0)

    def test_satisfaction_ratio_capped_at_one(self):
        n = self.make_need(current_need=500.0, allocated=9999.0)
        assert n.satisfaction_ratio == pytest.approx(1.0)

    def test_satisfaction_ratio_partial(self):
        n = self.make_need(current_need=1000.0, allocated=400.0)
        assert n.satisfaction_ratio == pytest.approx(0.4)


# ================================================================== #
#  VehicleType (config template)                                       #
# ================================================================== #

class TestVehicleTypeConfig:
    def make_type(self, **overrides):
        defaults = dict(
            type_id="test_heli",
            name="Test Helicopter",
            category=VehicleCategory.AIRCRAFT,
            capacity_kg=600.0,
            speed_kmh=180.0,
            fuel_hours=3.0,
            terrain_capability=TerrainCapability.ANY,
            cost_per_km=4.0,
        )
        defaults.update(overrides)
        return VehicleType(**defaults)

    def test_creates_vehicle_type(self):
        vt = self.make_type()
        assert vt.type_id == "test_heli"
        assert vt.category == VehicleCategory.AIRCRAFT

    def test_fuel_range_km(self):
        vt = self.make_type(speed_kmh=200.0, fuel_hours=2.0)
        assert vt.fuel_range_km == pytest.approx(400.0)

    def test_preferred_resources_defaults_empty(self):
        vt = self.make_type()
        assert vt.preferred_resources == []

    def test_preferred_resources_set(self):
        vt = self.make_type(preferred_resources=["medical_kit", "first_aid"])
        assert "medical_kit" in vt.preferred_resources

    def test_all_terrain_capabilities(self):
        for tc in TerrainCapability:
            vt = self.make_type(terrain_capability=tc)
            assert vt.terrain_capability == tc

    def test_all_vehicle_categories(self):
        for cat in VehicleCategory:
            vt = self.make_type(category=cat)
            assert vt.category == cat


# ================================================================== #
#  Vehicle (unified instance)                                          #
# ================================================================== #

class TestVehicle:
    def make_vehicle(self, terrain=TerrainCapability.ANY, **overrides):
        vtype = VehicleType(
            type_id="test",
            name="Test Vehicle",
            category=VehicleCategory.AIRCRAFT,
            capacity_kg=500.0,
            speed_kmh=200.0,
            fuel_hours=2.0,
            terrain_capability=terrain,
            cost_per_km=3.0,
        )
        defaults = dict(id="v1", name="V1", vehicle_type=vtype)
        defaults.update(overrides)
        return Vehicle(**defaults)

    def test_remaining_capacity_full_at_start(self):
        v = self.make_vehicle()
        assert v.remaining_capacity == pytest.approx(500.0)

    def test_load_cargo_by_resource_type(self):
        v = self.make_vehicle()
        loaded = v.load_cargo("food", 200.0)
        assert loaded == pytest.approx(200.0)
        assert "food" in v.cargo_manifest
        assert v.cargo_manifest["food"] == pytest.approx(200.0)

    def test_load_multiple_resource_types(self):
        v = self.make_vehicle()
        v.load_cargo("food", 200.0)
        v.load_cargo("water", 150.0)
        assert v.remaining_capacity == pytest.approx(150.0)
        assert len(v.cargo_manifest) == 2

    def test_load_same_resource_type_accumulates(self):
        v = self.make_vehicle()
        v.load_cargo("food", 100.0)
        v.load_cargo("food", 150.0)
        assert v.cargo_manifest["food"] == pytest.approx(250.0)

    def test_load_capped_at_remaining_capacity(self):
        v = self.make_vehicle()
        v.load_cargo("food", 400.0)
        loaded = v.load_cargo("water", 9999.0)  # only 100kg left
        assert loaded == pytest.approx(100.0)

    def test_load_raises_when_deployed(self):
        v = self.make_vehicle()
        v.state = VehicleState.DEPLOYED
        with pytest.raises(ValueError, match="DEPLOYED"):
            v.load_cargo("food", 100.0)

    def test_return_to_depot_clears_cargo_dict(self):
        v = self.make_vehicle()
        v.load_cargo("food", 200.0)
        v.return_to_depot((27.7172, 85.3240))
        assert v.cargo_manifest == {}

    def test_can_carry_resource_no_preference(self):
        v = self.make_vehicle()  # no preferred_resources
        assert v.can_carry_resource("food") is True
        assert v.can_carry_resource("medical_kit") is True

    def test_can_carry_resource_with_preference(self):
        vtype = VehicleType(
            type_id="medical_heli",
            name="Medical Helicopter",
            category=VehicleCategory.AIRCRAFT,
            capacity_kg=500.0,
            speed_kmh=200.0,
            fuel_hours=2.0,
            terrain_capability=TerrainCapability.ANY,
            preferred_resources=["medical_kit", "first_aid"],
        )
        v = Vehicle(id="mh1", name="Medical Heli 1", vehicle_type=vtype)
        assert v.can_carry_resource("medical_kit") is True
        assert v.can_carry_resource("food") is False


# ================================================================== #
#  Terrain routing                                                     #
# ================================================================== #

class TestTerrainCapability:
    def make_vehicle_with_terrain(self, terrain: TerrainCapability) -> Vehicle:
        vtype = VehicleType(
            type_id="test",
            name="Test",
            category=VehicleCategory.GROUND_HEAVY,
            capacity_kg=2000.0,
            speed_kmh=40.0,
            fuel_hours=8.0,
            terrain_capability=terrain,
        )
        return Vehicle(id="v1", name="V1", vehicle_type=vtype)

    def test_any_reaches_road(self):
        v = self.make_vehicle_with_terrain(TerrainCapability.ANY)
        assert v.can_deliver_to("road") is True

    def test_any_reaches_dirt_road(self):
        v = self.make_vehicle_with_terrain(TerrainCapability.ANY)
        assert v.can_deliver_to("dirt_road") is True

    def test_paved_roads_reaches_road(self):
        v = self.make_vehicle_with_terrain(TerrainCapability.PAVED_ROADS)
        assert v.can_deliver_to("road") is True

    def test_paved_roads_cannot_reach_dirt_road(self):
        v = self.make_vehicle_with_terrain(TerrainCapability.PAVED_ROADS)
        assert v.can_deliver_to("dirt_road") is False

    def test_all_roads_reaches_both(self):
        v = self.make_vehicle_with_terrain(TerrainCapability.ALL_ROADS)
        assert v.can_deliver_to("road") is True
        assert v.can_deliver_to("dirt_road") is True

    def test_dirt_paths_reaches_dirt_road(self):
        v = self.make_vehicle_with_terrain(TerrainCapability.DIRT_PATHS)
        assert v.can_deliver_to("dirt_road") is True


# ================================================================== #
#  Helicopter / Truck factories                                        #
# ================================================================== #

class TestFactories:
    def test_helicopter_factory_defaults(self):
        h = Helicopter(id="heli_1")
        assert h.capacity_kg == 500.0
        assert h.speed_kmh == 200.0
        assert h.terrain == "any"
        assert h.vehicle_type.category == VehicleCategory.AIRCRAFT

    def test_helicopter_factory_custom_capacity(self):
        h = Helicopter(id="heli_big", capacity_kg=800.0)
        assert h.capacity_kg == 800.0

    def test_truck_factory_defaults(self):
        t = Truck(id="truck_1")
        assert t.capacity_kg == 2000.0
        assert t.speed_kmh == 40.0
        assert t.terrain == "roads_only"
        assert t.vehicle_type.category == VehicleCategory.GROUND_HEAVY

    def test_truck_cannot_reach_dirt_road(self):
        t = Truck(id="truck_1")
        assert t.can_deliver_to("dirt_road") is False

    def test_helicopter_can_reach_dirt_road(self):
        h = Helicopter(id="heli_1")
        assert h.can_deliver_to("dirt_road") is True


# ================================================================== #
#  Village with multi-resource needs                                   #
# ================================================================== #

class TestVillageResourceNeeds:
    def make_village(self, **overrides):
        needs = {
            "food":        VillageResourceNeed(resource_type="food",        current_need=2500, min_need=1500),
            "water":       VillageResourceNeed(resource_type="water",       current_need=1500, min_need=1000),
            "medical_kit": VillageResourceNeed(resource_type="medical_kit", current_need=50,   min_need=30),
        }
        defaults = dict(
            id="dhulikhel",
            name="Dhulikhel",
            lat=27.62,
            lng=85.55,
            population=5000,
            resource_needs=needs,
        )
        defaults.update(overrides)
        return Village(**defaults)

    def test_total_unmet_need_sums_all_resources_in_mixed_units(self):
        v = self.make_village()
        expected = 2500 + 1500 + 50  # all allocated=0
        assert v.total_unmet_need_mixed_units == pytest.approx(expected)

    def test_unmet_need_property_matches(self):
        v = self.make_village()
        assert v.unmet_need == v.total_unmet_need_mixed_units

    def test_has_critical_shortage_when_all_below_min(self):
        v = self.make_village()  # allocated=0, min_need>0 → all critical
        assert v.has_critical_shortage is True

    def test_no_critical_shortage_when_all_above_min(self):
        needs = {
            "food": VillageResourceNeed(resource_type="food", current_need=1000, min_need=500, allocated=600),
        }
        v = self.make_village(resource_needs=needs)
        assert v.has_critical_shortage is False

    def test_get_resource_need_returns_correct(self):
        v = self.make_village()
        n = v.get_resource_need("food")
        assert n is not None
        assert n.current_need == pytest.approx(2500)

    def test_get_resource_need_returns_none_for_unknown(self):
        v = self.make_village()
        assert v.get_resource_need("unknown_resource") is None

    def test_legacy_current_need_still_works(self):
        v = Village(
            id="test", name="Test", lat=27.62, lng=85.55,
            population=1000,
            current_need=2000.0, min_need=1000.0, allocated=500.0,
        )
        assert v.unmet_need == pytest.approx(1500.0)

    def test_resource_needs_takes_priority_over_legacy(self):
        needs = {
            "food": VillageResourceNeed(resource_type="food", current_need=1000, min_need=500, allocated=0),
        }
        v = Village(
            id="test", name="Test", lat=27.62, lng=85.55,
            population=1000,
            current_need=9999.0, min_need=5000.0,  # legacy fields present but ignored
            resource_needs=needs,
        )
        # Should use resource_needs, not current_need
        assert v.unmet_need == pytest.approx(1000.0)


# ================================================================== #
#  config.json                                                         #
# ================================================================== #

class TestConfigJson:
    @pytest.fixture(scope="class")
    def config(self):
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

    def test_file_is_valid_json(self):
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_has_required_top_level_keys(self, config):
        for key in ("scenario_name", "vehicle_types", "fleet_composition", "resource_types"):
            assert key in config, f"config.json missing top-level key '{key}'"

    def test_vehicle_types_parseable_as_models(self, config):
        for type_id, vtype_data in config["vehicle_types"].items():
            vt = VehicleType(**vtype_data)
            assert vt.type_id == type_id

    def test_fleet_composition_references_valid_types(self, config):
        defined_types = set(config["vehicle_types"].keys())
        for vtype_id in config["fleet_composition"]:
            assert vtype_id in defined_types, (
                f"fleet_composition references unknown type '{vtype_id}'"
            )

    def test_fleet_composition_counts_positive(self, config):
        for vtype_id, count in config["fleet_composition"].items():
            assert count > 0, f"fleet_composition count for '{vtype_id}' must be > 0"

    def test_total_fleet_size(self, config):
        total = sum(config["fleet_composition"].values())
        assert total >= 9, f"Fleet should have at least 9 vehicles, got {total}"

    def test_resource_types_parseable_as_models(self, config):
        for rtype_id, rtype_data in config["resource_types"].items():
            rt = ResourceType(**rtype_data)
            assert rt.resource_id == rtype_id

    def test_all_six_resource_types_present(self, config):
        expected = {"food", "water", "medical_kit", "tarpaulin", "blanket", "first_aid"}
        present = set(config["resource_types"].keys())
        missing = expected - present
        assert not missing, f"config.json missing resource types: {missing}"

    def test_medical_resources_have_highest_urgency(self, config):
        medical_types = [
            rt for rt in config["resource_types"].values()
            if rt["category"] == "medical"
        ]
        non_medical = [
            rt for rt in config["resource_types"].values()
            if rt["category"] not in ("medical",)
        ]
        max_medical = max(r["urgency_multiplier"] for r in medical_types)
        max_non_medical = max(r["urgency_multiplier"] for r in non_medical)
        assert max_medical >= max_non_medical, "Medical resources should have highest urgency multipliers"
