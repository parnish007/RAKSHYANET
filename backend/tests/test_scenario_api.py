"""API coverage for operator-facing mocked timeline switching."""
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app
from backend.api.websocket_manager import ws_manager
from backend.services.gemma_service import gemma_service
from backend.services.optimization_service import optimization_service


@pytest.fixture(autouse=True)
def reset_runtime_state():
    previous_key = gemma_service.online_provider.api_key
    gemma_service.online_provider.api_key = ""
    gemma_service.analyses.clear()
    gemma_service.analysis_order.clear()
    optimization_service.runs.clear()
    optimization_service.run_order.clear()
    ws_manager.message_history.clear()
    yield
    gemma_service.online_provider.api_key = previous_key
    gemma_service.analyses.clear()
    gemma_service.analysis_order.clear()
    optimization_service.runs.clear()
    optimization_service.run_order.clear()
    ws_manager.message_history.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_lists_five_simulated_scenarios_with_visible_timelines(client):
    response = client.get("/api/demo/scenarios")

    assert response.status_code == 200
    scenarios = response.json()["scenarios"]
    assert len(scenarios) == 5
    assert all(item["simulated"] is True for item in scenarios)
    assert all(len(item["timeline"]) == 5 for item in scenarios)
    assert all(item["closure"]["blocked_edge_ids"] for item in scenarios)


def test_activates_baseline_scenario_as_reviewable_run(client):
    response = client.post(
        "/api/demo/scenarios/taplejung-landslide-mechi-closure/activate",
        json={"stage": "baseline", "requested_by": "scenario-api-test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stage"] == "baseline"
    assert body["run"]["status"] == "awaiting_approval"
    assert body["run"]["parent_run_id"] is None
    assert body["run"]["analysis_id"] == body["analysis"]["analysis_id"]
    assert all(item["simulated"] is True for item in body["analysis"]["evidence"])


def test_activates_disrupted_scenario_as_closure_filtered_child_run(client):
    response = client.post(
        "/api/demo/scenarios/jumla-bridge-karnali-closure/activate",
        json={"stage": "disrupted", "requested_by": "scenario-api-test"},
    )

    assert response.status_code == 200
    body = response.json()
    run = body["run"]
    blocked = set(run["blocked_edge_ids"])
    assert body["stage"] == "disrupted"
    assert run["parent_run_id"] == body["baseline_run_id"]
    assert blocked == {"karnali_pokhara_jumla"}
    assert run["route_feasible"] is True
    for route in run["result"]["vrp_solution"]["routes"]:
        if route["transport_mode"] == "road" and route["feasible"]:
            assert not blocked.intersection(route["road_edge_ids"])


def test_unknown_scenario_is_rejected(client):
    response = client.post(
        "/api/demo/scenarios/not-a-fixture/activate",
        json={"stage": "baseline"},
    )
    assert response.status_code == 404
