"""Integration coverage for the real optimization API and event contract."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.websocket_manager import ws_manager
from backend.services.optimization_service import optimization_service


@pytest.fixture(autouse=True)
def reset_runtime_state():
    optimization_service.runs.clear()
    optimization_service.run_order.clear()
    ws_manager.message_history.clear()
    yield
    optimization_service.runs.clear()
    optimization_service.run_order.clear()
    ws_manager.message_history.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_real_optimization_endpoint_runs_state_manager(client):
    response = client.post(
        "/api/optimization/run",
        json={
            "scenario_id": "nepal-national-demo",
            "time_elapsed_hours": 2,
            "requested_by": "test-operator",
        },
    )

    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "awaiting_approval"
    assert run["allocation_method"] == "proportional_allocation"
    assert run["routing_method"] == "greedy_urgency_nearest_neighbour"
    assert run["requires_human_approval"] is True
    assert run["route_feasible"] is True
    assert run["approval_blockers"] == []
    assert run["analysis_snapshot"]["analysis_id"] == run["analysis_id"]
    assert run["analysis_snapshot"]["output"]
    assert run["result"]["state"] == "complete"
    assert len(run["result"]["urgency_scores"]) == 8
    assert run["result"]["vrp_solution"]["routes"]
    assert run["result"]["nash_equilibrium"]["strategies"]
    assert run["result"]["nash_equilibrium"]["convergence_history"]
    assert run["result"]["nash_equilibrium"]["convergence_threshold"] == 0.01
    assert run["result"]["nash_equilibrium"]["convergence_metric"] == "max_normalized_allocation_change"
    assert run["result"]["nash_equilibrium"]["convergence_history"][0]["max_normalized_change"] >= 0
    assert run["result"]["kkt_verification"]["conditions"]
    assert run["result"]["resource_snapshot"]["source_kind"] == "bundled_scenario_fixture"
    assert run["result"]["resource_snapshot"]["depot_available"]["food"] == 5000
    assert run["result"]["fleet_snapshot"]
    assert run["result"]["urgency_scores"][0]["components"]
    assert "critical_penalty" in run["result"]["urgency_scores"][0]
    assert run["result"]["vrp_solution"]["allocations"][0]["resource_decisions"]
    gemma_signal = run["result"]["gemma_signal"]
    assert len(gemma_signal["input_scores"]) == 3
    assert sum(
        bool(item["selected_for_max"])
        for item in gemma_signal["input_scores"]
    ) >= 1
    assert gemma_signal["calculation"]["resulting_boost"] == gemma_signal["boost"]

    recovered = client.get(f"/api/optimization/runs/{run['run_id']}").json()
    assert recovered["analysis_snapshot"] == run["analysis_snapshot"]


def test_real_run_emits_versioned_reproducible_events(client):
    response = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    )
    assert response.status_code == 200
    run = response.json()

    event_types = [message.type for message in ws_manager.message_history]
    assert event_types[0] == "optimization_started"
    assert "urgency_updated" in event_types
    assert "route_generated" in event_types
    assert "allocation_generated" in event_types
    assert "validation_completed" in event_types
    assert "optimization_completed" in event_types
    assert event_types[-1] == "hitl_review_required"

    for message in ws_manager.message_history:
        dumped = message.model_dump(mode="json")
        assert dumped["event_id"].startswith("evt_")
        assert dumped["scenario_id"] == "nepal-national-demo"
        assert dumped["schema_version"] == "1.0"
        assert dumped["event_type"] == dumped["type"]
        assert dumped["correlation_id"] == run["correlation_id"]


def test_national_fixture_routes_respect_vehicle_fuel_autonomy(client):
    response = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    )
    assert response.status_code == 200
    routes = response.json()["result"]["vrp_solution"]["routes"]

    root = Path(__file__).resolve().parents[2]
    fleet = json.loads(
        (root / "backend" / "data" / "fleet_config.json").read_text(
            encoding="utf-8"
        )
    )
    vehicles = {
        vehicle["id"]: vehicle
        for vehicle in fleet["helicopters"] + fleet["trucks"]
    }
    assert routes
    for route in routes:
        vehicle = vehicles[route["vehicle_id"]]
        fuel_range_km = vehicle["speed_kmh"] * vehicle["fuel_hours"]
        assert route["feasible"] is True
        assert route["infeasibility_reason"] is None
        assert route["total_distance_km"] <= fuel_range_km


def test_ground_routes_follow_graph_geometry_and_aircraft_remain_direct(client):
    run = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    ).json()
    routes = run["result"]["vrp_solution"]["routes"]

    ground = [route for route in routes if route["transport_mode"] == "road"]
    aircraft = [route for route in routes if route["transport_mode"] == "air"]
    assert ground and aircraft
    assert all(route["road_edge_ids"] for route in ground)
    assert all(len(route["path_coordinates"]) > len(route["stops"]) + 1 for route in ground)
    assert all(not route["road_edge_ids"] for route in aircraft)


def test_sudden_road_block_creates_new_run_and_reroutes_ground_fleet(client):
    baseline = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    ).json()
    edge_id = "prithvi_kathmandu_bharatpur"
    rerun = client.post(
        "/api/optimization/run",
        json={
            "scenario_id": "nepal-national-demo",
            "analysis_id": baseline["analysis_id"],
            "parent_run_id": baseline["run_id"],
            "trigger": "road_closure",
            "disruption_reason": "Operator exercise: sudden landslide closure",
            "blocked_edge_ids": [edge_id],
        },
    )

    assert rerun.status_code == 200
    body = rerun.json()
    assert body["run_id"] != baseline["run_id"]
    assert body["parent_run_id"] == baseline["run_id"]
    assert body["blocked_edge_ids"] == [edge_id]
    assert body["status"] == "awaiting_approval"
    ground_routes = [
        route for route in body["result"]["vrp_solution"]["routes"]
        if route["transport_mode"] == "road"
    ]
    assert ground_routes
    assert all(edge_id not in route["road_edge_ids"] for route in ground_routes)
    assert any(edge_id in route["rerouted_due_to"] for route in ground_routes)


def test_plan_requires_single_human_decision(client):
    run = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    ).json()
    run_id = run["run_id"]

    approved = client.post(
        f"/api/optimization/runs/{run_id}/approve",
        json={
            "reviewer": "commander",
            "notes": "Verified inventory and routes",
            "expected_updated_at": run["updated_at"],
            "expected_analysis_id": run["analysis_id"],
        },
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["reviewed_by"] == "commander"

    duplicate = client.post(
        f"/api/optimization/runs/{run_id}/reject",
        json={
            "reviewer": "second-reviewer",
            "notes": "Too late",
            "expected_updated_at": approved.json()["updated_at"],
            "expected_analysis_id": run["analysis_id"],
        },
    )
    assert duplicate.status_code == 409


def test_review_rejects_a_stale_snapshot_timestamp(client):
    run = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    ).json()

    stale = client.post(
        f"/api/optimization/runs/{run['run_id']}/approve",
        json={
            "reviewer": "commander",
            "notes": "Reviewing an obsolete snapshot",
            "expected_updated_at": "2000-01-01T00:00:00+00:00",
            "expected_analysis_id": run["analysis_id"],
        },
    )
    assert stale.status_code == 409
    assert "changed after the review snapshot" in stale.json()["detail"]

    current = client.post(
        f"/api/optimization/runs/{run['run_id']}/approve",
        json={
            "reviewer": "commander",
            "notes": "Verified current immutable snapshot",
            "expected_updated_at": run["updated_at"],
            "expected_analysis_id": run["analysis_id"],
        },
    )
    assert current.status_code == 200


def test_review_rejects_a_run_when_a_newer_run_exists(client):
    first = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    ).json()
    second = client.post(
        "/api/optimization/run",
        json={
            "scenario_id": "nepal-national-demo",
            "analysis_id": first["analysis_id"],
            "parent_run_id": first["run_id"],
            "trigger": "manual_recompute",
        },
    ).json()

    stale = client.post(
        f"/api/optimization/runs/{first['run_id']}/approve",
        json={
            "reviewer": "commander",
            "notes": "Attempting to approve an obsolete plan",
            "expected_updated_at": first["updated_at"],
            "expected_analysis_id": first["analysis_id"],
        },
    )
    assert stale.status_code == 409
    assert second["run_id"] in stale.json()["detail"]


@pytest.mark.parametrize("failure_mode", ["infeasible", "empty"])
def test_review_rejects_infeasible_or_empty_route_set(client, failure_mode):
    run = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    ).json()
    record = optimization_service.get(run["run_id"])
    assert record is not None

    if failure_mode == "infeasible":
        route = record.result.vrp_solution.routes[0]
        route.feasible = False
        route.infeasibility_reason = "Injected route-feasibility regression"
        record.approval_blockers = [
            f"Infeasible assigned route set for assets: {route.vehicle_id}."
        ]
    else:
        record.result.vrp_solution.routes = []
        record.approval_blockers = ["No assigned routes were generated."]
    record.route_feasible = False

    blocked = client.post(
        f"/api/optimization/runs/{run['run_id']}/approve",
        json={
            "reviewer": "commander",
            "notes": "This snapshot must not pass the authorization boundary",
            "expected_updated_at": run["updated_at"],
            "expected_analysis_id": run["analysis_id"],
        },
    )

    assert blocked.status_code == 409
    assert "cannot be approved" in blocked.json()["detail"]
    assert optimization_service.get(run["run_id"]).status.value == "awaiting_approval"


def test_review_requires_both_immutable_snapshot_tokens(client):
    run = client.post(
        "/api/optimization/run",
        json={"scenario_id": "nepal-national-demo"},
    ).json()

    missing = client.post(
        f"/api/optimization/runs/{run['run_id']}/approve",
        json={"reviewer": "commander", "notes": "Missing snapshot identity"},
    )

    assert missing.status_code == 422


def test_no_frontend_source_fabricates_math_or_claims_optimality():
    """No rendered surface may invent numbers or claim optimality.

    This used to read one component that has since been deleted. Scanning the
    whole live frontend enforces the same rule everywhere instead of in one file,
    so a future component cannot reintroduce a fabricated convergence animation
    or an optimality claim the engine does not make.
    """
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "frontend" / "src").rglob("*.jsx"))
    assert sources, "frontend sources must be present"

    offences = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for needle in ("math.random", "certified optimal"):
            if needle in lowered:
                offences.append(f"{path.relative_to(root)}: {needle}")
        # "global optimum" is permitted only inside an explicit denial, which is
        # how the convergence panel disclaims what it does not prove.
        for line in lowered.splitlines():
            if "global optimum" in line and not any(
                marker in line for marker in ("not ", "never", "no ")
            ):
                offences.append(f"{path.relative_to(root)}: unqualified global optimum")

    assert not offences, "fabricated or overclaiming frontend source: " + "; ".join(offences)
