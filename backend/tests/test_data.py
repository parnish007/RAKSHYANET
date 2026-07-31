"""
Tests for mock data files — Prompt 1.3 verification.
Run: pytest backend/tests/test_data.py -v
"""
import json
import math
from datetime import datetime
from pathlib import Path

import networkx as nx
import pytest

# ------------------------------------------------------------------ #
#  Paths                                                               #
# ------------------------------------------------------------------ #

ROOT = Path(__file__).parents[2]  # rakshyanet/
VILLAGES_FILE  = ROOT / "backend" / "data" / "nepal_villages.json"
FLEET_FILE     = ROOT / "backend" / "data" / "fleet_config.json"
GRAPH_FILE     = ROOT / "backend" / "data" / "terrain_graph.json"
TIMELINE_FILE  = ROOT / "demo"    / "mock_news_timeline.json"

VILLAGE_REQUIRED_FIELDS = {
    "id", "name", "lat", "lng", "population",
    "initial_urgency", "disaster_impact",
    "resource_needs",
    "terrain_difficulty", "has_medical_facility", "accessibility",
}

RESOURCE_NEED_FIELDS = {"resource_type", "current_need", "min_need", "allocated"}
EXPECTED_RESOURCE_TYPES = {"food", "water", "medical_kit", "tarpaulin", "blanket", "first_aid"}

VEHICLE_REQUIRED_FIELDS = {
    "id", "name", "capacity_kg", "speed_kmh",
    "fuel_hours", "terrain_capability", "current_location",
}

NEWS_REQUIRED_FIELDS = {
    "id", "timestamp", "source", "source_type",
    "text", "village_id", "severity_score", "confidence",
    "keywords", "multi_source_confirmation", "verified",
}


# ------------------------------------------------------------------ #
#  Fixtures — load JSON once per test session                          #
# ------------------------------------------------------------------ #

@pytest.fixture(scope="session")
def villages_data():
    return json.loads(VILLAGES_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def fleet_data():
    return json.loads(FLEET_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def graph_data():
    return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def timeline_data():
    return json.loads(TIMELINE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def nx_graph(graph_data):
    G = nx.Graph()
    for node in graph_data["nodes"]:
        G.add_node(node["id"])
    for edge in graph_data["edges"]:
        G.add_edge(
            edge["from"], edge["to"],
            distance_km=edge["distance_km"],
            has_road=edge["has_road"],
        )
    return G


# ================================================================== #
#  nepal_villages.json                                                 #
# ================================================================== #

class TestVillagesData:
    def test_file_is_valid_json(self):
        data = json.loads(VILLAGES_FILE.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_has_depot(self, villages_data):
        assert "depot" in villages_data
        depot = villages_data["depot"]
        assert "lat" in depot and "lng" in depot

    def test_exactly_eight_villages(self, villages_data):
        assert len(villages_data["villages"]) == 8

    def test_all_villages_have_required_fields(self, villages_data):
        for village in villages_data["villages"]:
            missing = VILLAGE_REQUIRED_FIELDS - village.keys()
            assert not missing, f"Village {village.get('id')} missing fields: {missing}"

    def test_village_ids_are_unique(self, villages_data):
        ids = [v["id"] for v in villages_data["villages"]]
        assert len(ids) == len(set(ids)), "Duplicate village IDs found"

    def test_coordinates_in_nepal_bounds(self, villages_data):
        # Nepal bounding box: lat 26.3–30.5, lng 80.0–88.2
        for v in villages_data["villages"]:
            assert 26.3 <= v["lat"] <= 30.5, f"{v['id']} lat out of Nepal bounds"
            assert 80.0 <= v["lng"] <= 88.2, f"{v['id']} lng out of Nepal bounds"

    def test_urgency_scores_in_bounds(self, villages_data):
        for v in villages_data["villages"]:
            assert 0.0 <= v["initial_urgency"] <= 1.0, f"{v['id']} urgency out of [0,1]"
            assert 0.0 <= v["disaster_impact"] <= 1.0, f"{v['id']} disaster_impact out of [0,1]"

    def test_resource_needs_structure(self, villages_data):
        for v in villages_data["villages"]:
            rn = v["resource_needs"]
            assert isinstance(rn, dict), f"{v['id']} resource_needs must be a dict"
            assert rn, f"{v['id']} resource_needs is empty"
            for rtype, need in rn.items():
                missing = RESOURCE_NEED_FIELDS - need.keys()
                assert not missing, f"{v['id']}.{rtype} missing fields: {missing}"
                assert need["min_need"] <= need["current_need"], (
                    f"{v['id']}.{rtype}: min_need > current_need"
                )

    def test_all_six_resource_types_present_in_each_village(self, villages_data):
        for v in villages_data["villages"]:
            present = set(v["resource_needs"].keys())
            missing = EXPECTED_RESOURCE_TYPES - present
            assert not missing, f"{v['id']} missing resource types: {missing}"

    def test_population_positive(self, villages_data):
        for v in villages_data["villages"]:
            assert v["population"] > 0, f"{v['id']} has non-positive population"

    def test_expected_village_ids_present(self, villages_data):
        ids = {v["id"] for v in villages_data["villages"]}
        expected = {
            "mahendranagar", "jumla", "pokhara", "bharatpur",
            "janakpur", "dharan", "taplejung", "nepalgunj",
        }
        assert ids == expected

    def test_fixture_spans_all_seven_provinces(self, villages_data):
        ids = {v["id"] for v in villages_data["villages"]}
        province_representatives = {
            "Koshi": "taplejung",
            "Madhesh": "janakpur",
            "Bagmati": "bharatpur",
            "Gandaki": "pokhara",
            "Lumbini": "nepalgunj",
            "Karnali": "jumla",
            "Sudurpashchim": "mahendranagar",
        }
        assert set(province_representatives.values()) <= ids

    def test_depot_has_available_resources(self, villages_data):
        depot = villages_data["depot"]
        assert "available_resources" in depot
        for rtype in EXPECTED_RESOURCE_TYPES:
            assert rtype in depot["available_resources"], (
                f"Depot missing resource '{rtype}'"
            )
            assert depot["available_resources"][rtype] > 0

    def test_total_food_need_exceeds_any_single_vehicle(self, villages_data):
        total_food = sum(
            v["resource_needs"]["food"]["current_need"]
            for v in villages_data["villages"]
        )
        assert total_food > 500, "Total food need must exceed single helicopter capacity"


# ================================================================== #
#  fleet_config.json                                                   #
# ================================================================== #

class TestFleetData:
    def test_file_is_valid_json(self):
        data = json.loads(FLEET_FILE.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_has_helicopters_and_trucks_keys(self, fleet_data):
        assert "helicopters" in fleet_data
        assert "trucks" in fleet_data

    def test_exactly_four_helicopters(self, fleet_data):
        assert len(fleet_data["helicopters"]) == 4

    def test_exactly_five_trucks(self, fleet_data):
        assert len(fleet_data["trucks"]) == 5

    def test_all_vehicles_have_required_fields(self, fleet_data):
        all_vehicles = fleet_data["helicopters"] + fleet_data["trucks"]
        for v in all_vehicles:
            missing = VEHICLE_REQUIRED_FIELDS - v.keys()
            assert not missing, f"Vehicle {v.get('id')} missing fields: {missing}"

    def test_vehicle_ids_unique_across_fleet(self, fleet_data):
        ids = [v["id"] for v in fleet_data["helicopters"] + fleet_data["trucks"]]
        assert len(ids) == len(set(ids)), "Duplicate vehicle IDs found"

    def test_helicopter_specs(self, fleet_data):
        for h in fleet_data["helicopters"]:
            assert h["capacity_kg"] == 500
            assert h["speed_kmh"] == 200
            assert h["fuel_hours"] == 4.5
            assert h["terrain_capability"] == "any"

    def test_truck_specs(self, fleet_data):
        for t in fleet_data["trucks"]:
            assert t["capacity_kg"] == 2000
            assert t["speed_kmh"] == 40
            assert t["fuel_hours"] == 50.0
            assert t["terrain_capability"] == "roads_only"

    def test_all_vehicles_start_at_depot(self, fleet_data):
        depot_lat, depot_lng = 27.7172, 85.3240
        for v in fleet_data["helicopters"] + fleet_data["trucks"]:
            loc = v["current_location"]
            assert loc["lat"] == pytest.approx(depot_lat, abs=0.001)
            assert loc["lng"] == pytest.approx(depot_lng, abs=0.001)

    def test_total_helicopter_capacity(self, fleet_data):
        total = sum(h["capacity_kg"] for h in fleet_data["helicopters"])
        assert total == pytest.approx(2000.0)  # 4 × 500kg

    def test_total_truck_capacity(self, fleet_data):
        total = sum(t["capacity_kg"] for t in fleet_data["trucks"])
        assert total == pytest.approx(10000.0)  # 5 × 2000kg


# ================================================================== #
#  terrain_graph.json                                                  #
# ================================================================== #

class TestTerrainGraph:
    def test_file_is_valid_json(self):
        data = json.loads(GRAPH_FILE.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_has_nodes_and_edges(self, graph_data):
        assert "nodes" in graph_data
        assert "edges" in graph_data

    def test_nine_nodes(self, graph_data):
        # depot + 8 villages
        assert len(graph_data["nodes"]) == 9

    def test_depot_node_present(self, graph_data):
        node_ids = [n["id"] for n in graph_data["nodes"]]
        assert "depot" in node_ids

    def test_all_village_nodes_present(self, graph_data):
        node_ids = {n["id"] for n in graph_data["nodes"]}
        expected = {
            "depot", "mahendranagar", "jumla", "pokhara", "bharatpur",
            "janakpur", "dharan", "taplejung", "nepalgunj",
        }
        assert node_ids == expected

    def test_graph_coordinates_match_operational_locations(
        self,
        graph_data,
        villages_data,
    ):
        expected = {
            "depot": villages_data["depot"],
            **{v["id"]: v for v in villages_data["villages"]},
        }
        for node in graph_data["nodes"]:
            location = expected[node["id"]]
            assert node["lat"] == pytest.approx(location["lat"], abs=1e-6)
            assert node["lng"] == pytest.approx(location["lng"], abs=1e-6)

    def test_graph_distances_are_plausible_road_distances(self, graph_data):
        nodes = {
            node["id"]: (node["lat"], node["lng"])
            for node in graph_data["nodes"]
        }

        def haversine_km(left, right):
            lat1, lon1 = map(math.radians, left)
            lat2, lon2 = map(math.radians, right)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                math.sin(dlat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            )
            return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        for edge in graph_data["edges"]:
            direct = haversine_km(nodes[edge["from"]], nodes[edge["to"]])
            assert edge["distance_km"] >= direct * 0.98
            assert edge["distance_km"] <= direct * 2.5

    def test_graph_is_connected(self, nx_graph):
        assert nx.is_connected(nx_graph), (
            f"Graph is not connected. Components: {list(nx.connected_components(nx_graph))}"
        )

    def test_all_nodes_reachable_from_depot(self, nx_graph):
        reachable = nx.node_connected_component(nx_graph, "depot")
        all_nodes = set(nx_graph.nodes)
        assert reachable == all_nodes

    def test_edge_distances_positive(self, graph_data):
        for edge in graph_data["edges"]:
            assert edge["distance_km"] > 0, f"Edge {edge['from']}→{edge['to']} has zero distance"

    def test_edge_node_refs_exist(self, graph_data):
        node_ids = {n["id"] for n in graph_data["nodes"]}
        for edge in graph_data["edges"]:
            assert edge["from"] in node_ids, f"Unknown node '{edge['from']}' in edge"
            assert edge["to"] in node_ids,   f"Unknown node '{edge['to']}' in edge"

    def test_all_edges_have_road(self, graph_data):
        for edge in graph_data["edges"]:
            assert edge["has_road"] is True, (
                f"Edge {edge['from']}→{edge['to']} has_road=False (truck routing will fail)"
            )

    def test_shortest_path_depot_to_all_villages(self, nx_graph):
        villages = [n for n in nx_graph.nodes if n != "depot"]
        for v in villages:
            path = nx.shortest_path(nx_graph, "depot", v)
            assert len(path) >= 2, f"No path found from depot to {v}"


# ================================================================== #
#  mock_news_timeline.json                                             #
# ================================================================== #

class TestNewsTimeline:
    def test_file_is_valid_json(self):
        data = json.loads(TIMELINE_FILE.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_has_events_key(self, timeline_data):
        assert "events" in timeline_data

    def test_at_least_four_events(self, timeline_data):
        assert len(timeline_data["events"]) >= 4

    def test_all_events_have_required_fields(self, timeline_data):
        for event in timeline_data["events"]:
            missing = NEWS_REQUIRED_FIELDS - event.keys()
            assert not missing, f"Event {event.get('id')} missing fields: {missing}"

    def test_timestamps_are_valid_iso(self, timeline_data):
        for event in timeline_data["events"]:
            try:
                datetime.fromisoformat(event["timestamp"])
            except ValueError:
                pytest.fail(f"Event {event['id']} has invalid timestamp: {event['timestamp']}")

    def test_timestamps_are_chronological(self, timeline_data):
        events = timeline_data["events"]
        timestamps = [datetime.fromisoformat(e["timestamp"]) for e in events]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], (
                f"Events out of order at index {i}: {timestamps[i]} > {timestamps[i+1]}"
            )

    def test_event_ids_unique(self, timeline_data):
        ids = [e["id"] for e in timeline_data["events"]]
        assert len(ids) == len(set(ids)), "Duplicate event IDs"

    def test_severity_and_confidence_in_bounds(self, timeline_data):
        for event in timeline_data["events"]:
            assert 0.0 <= event["severity_score"] <= 1.0, (
                f"Event {event['id']} severity_score out of [0,1]"
            )
            assert 0.0 <= event["confidence"] <= 1.0, (
                f"Event {event['id']} confidence out of [0,1]"
            )

    def test_village_ids_reference_known_villages(self, timeline_data):
        known_villages = {
            "mahendranagar", "jumla", "pokhara", "bharatpur",
            "janakpur", "dharan", "taplejung", "nepalgunj",
        }
        for event in timeline_data["events"]:
            vid = event.get("village_id")
            if vid is not None:
                assert vid in known_villages, (
                    f"Event {event['id']} references unknown village '{vid}'"
                )

    def test_verified_events_have_trusted_sources(self, timeline_data):
        # Verified events should come from known source types
        trusted_types = {"verified_government", "verified_news", "verified_ngo"}
        for event in timeline_data["events"]:
            if event["verified"]:
                assert event["source_type"] in trusted_types, (
                    f"Event {event['id']} is verified but source_type is '{event['source_type']}'"
                )

    def test_unverified_events_have_low_confidence(self, timeline_data):
        for event in timeline_data["events"]:
            if not event["verified"]:
                assert event["confidence"] < 0.5, (
                    f"Unverified event {event['id']} has suspiciously high confidence {event['confidence']}"
                )

    def test_at_least_one_high_confidence_event(self, timeline_data):
        high_conf = [e for e in timeline_data["events"] if e["confidence"] >= 0.8]
        assert len(high_conf) >= 1, "Need at least one auto-optimize event for demo"

    def test_at_least_one_medium_confidence_hitl_event(self, timeline_data):
        hitl_events = [e for e in timeline_data["events"] if 0.5 <= e["confidence"] < 0.8]
        assert len(hitl_events) >= 1, "Need at least one HITL event for demo"
