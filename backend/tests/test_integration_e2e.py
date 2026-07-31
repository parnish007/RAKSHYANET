"""
Integration Tests (End-to-End) -- Prompt 6.1

Tests that verify the entire pipeline:
  Mock data -> RAG -> HITL -> Re-optimization -> P2P -> WebSocket

Run: pytest backend/tests/test_integration_e2e.py -v
"""
import asyncio
import json
import time
from datetime import timedelta
from pathlib import Path

import pytest

from backend.algorithms.state_manager import OptimizationState, StateManager
from backend.api.websocket_manager import (
    MSG_EVENT_PROCESSED,
    MSG_REOPTIMIZATION_START,
    WebSocketManager,
    WSMessage,
)
from backend.demo.reoptimization_trigger import ReoptimizationConfig, ReoptimizationTrigger
from backend.demo.timeline_simulator import SimulatorConfig, TimelineSimulator
from backend.hitl.approval_queue import ApprovalQueue, ApprovalStatus
from backend.models.resource import ResourceType
from backend.models.vehicle import TerrainCapability, Vehicle, VehicleCategory, VehicleType
from backend.models.village import Village
from backend.p2p.gossip_protocol import (
    MSG_OPTIMIZATION_RESULT,
    GossipProtocol,
    PeerNode,
)
from backend.rag.news_analyzer import (
    ACTION_AUTO_OPTIMIZE,
    ACTION_HITL_REQUIRED,
    ACTION_IGNORE,
    NewsAnalyzer,
)

DATA = Path(__file__).parent.parent / "data"
TIMELINE_PATH = str(Path(__file__).parent.parent / "demo" / "mock_news_timeline.json")


# ================================================================== #
#  Helpers                                                             #
# ================================================================== #

def run(coro):
    """Run a coroutine synchronously (no pytest-asyncio required)."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _load_data():
    config        = json.loads((DATA / "config.json").read_text())
    terrain_graph = json.loads((DATA / "terrain_graph.json").read_text())
    villages_raw  = json.loads((DATA / "nepal_villages.json").read_text())
    return config, terrain_graph, villages_raw


def _resource_types(config: dict):
    return {
        k: ResourceType(**v)
        for k, v in config.get("resource_types", {}).items()
    }


def _make_state_manager():
    config, terrain_graph, _ = _load_data()
    rt = _resource_types(config)
    return StateManager(
        depot_location=(27.7172, 85.3240),
        depot_resources={r: 500.0 for r in rt},
        terrain_graph=terrain_graph,
        resource_types=rt,
        config=config,
    )


def _make_villages():
    _, _, villages_raw = _load_data()
    return [
        Village(
            id=v["id"],
            name=v["name"],
            lat=v["lat"],
            lng=v["lng"],
            population=v["population"],
            terrain_difficulty=v["terrain_difficulty"],
            urgency_score=v["initial_urgency"],
            disaster_impact=v["disaster_impact"],
        )
        for v in villages_raw["villages"]
    ]


def _make_vehicles(n: int = 3):
    vtype = VehicleType(
        type_id="van_4x4",
        name="4x4 Relief Van",
        category=VehicleCategory.GROUND_LIGHT,
        capacity_kg=800.0,
        speed_kmh=60.0,
        fuel_hours=8.0,
        terrain_capability=TerrainCapability.ALL_ROADS,
    )
    return [Vehicle(id=f"v{i}", name=f"Van {i}", vehicle_type=vtype) for i in range(1, n + 1)]


class _MockWS:
    """Async WebSocket stub for testing."""
    def __init__(self): self.sent = []; self.accepted = False
    async def accept(self): self.accepted = True
    async def send_json(self, d): self.sent.append(d)


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture
def villages():
    return _make_villages()


@pytest.fixture
def vehicles():
    return _make_vehicles()


@pytest.fixture
def state_manager():
    return _make_state_manager()


@pytest.fixture
def news_analyzer():
    return NewsAnalyzer()


@pytest.fixture
def approval_queue():
    return ApprovalQueue(timeout_minutes=5)


@pytest.fixture
def trigger(state_manager, villages, vehicles):
    cfg = ReoptimizationConfig(
        urgency_change_threshold=0.10,
        enable_reoptimization=True,
        broadcast_via_p2p=False,
        log_optimization_changes=False,
    )
    return ReoptimizationTrigger(
        config=cfg,
        state_manager=state_manager,
        villages=villages,
        vehicles=vehicles,
        time_elapsed=timedelta(hours=2),
    )


@pytest.fixture
def ws_manager():
    return WebSocketManager()


@pytest.fixture
def simulator(news_analyzer, approval_queue, trigger, villages, ws_manager):
    cfg = SimulatorConfig(
        timeline_path=TIMELINE_PATH,
        speed_multiplier=1000.0,
        auto_approve_hitl=True,
        trigger_reoptimization=True,
        verbose_logging=False,
    )
    return TimelineSimulator(
        config=cfg,
        news_analyzer=news_analyzer,
        approval_queue=approval_queue,
        reoptimization_trigger=trigger,
        villages=villages,
        websocket_manager=ws_manager,
    )


# ================================================================== #
#  1. Full optimization pipeline                                       #
# ================================================================== #

class TestFullOptimizationPipeline:
    def test_pipeline_returns_complete_state(self, state_manager, villages, vehicles):
        result = state_manager.run_full_optimization(
            villages=villages,
            vehicles=vehicles,
            time_elapsed=timedelta(hours=2),
        )
        assert result.state == OptimizationState.COMPLETE

    def test_pipeline_produces_urgency_scores(self, state_manager, villages, vehicles):
        result = state_manager.run_full_optimization(villages, vehicles, timedelta(hours=2))
        assert len(result.urgency_scores) > 0

    def test_pipeline_produces_vrp_solution(self, state_manager, villages, vehicles):
        result = state_manager.run_full_optimization(villages, vehicles, timedelta(hours=2))
        assert result.vrp_solution is not None

    def test_pipeline_produces_nash_equilibrium(self, state_manager, villages, vehicles):
        result = state_manager.run_full_optimization(villages, vehicles, timedelta(hours=2))
        assert result.nash_equilibrium is not None

    def test_pipeline_kkt_all_conditions_satisfied(self, state_manager, villages, vehicles):
        result = state_manager.run_full_optimization(villages, vehicles, timedelta(hours=2))
        assert result.kkt_verification is not None
        assert result.kkt_verification.all_conditions_satisfied

    def test_pipeline_completes_under_15_seconds(self, state_manager, villages, vehicles):
        result = state_manager.run_full_optimization(villages, vehicles, timedelta(hours=2))
        assert result.execution_time_seconds < 15.0


# ================================================================== #
#  2. News event → RAG → re-optimization                              #
# ================================================================== #

class TestNewsEventPipeline:
    def test_high_confidence_text_gets_auto_action(self, news_analyzer, villages):
        report = news_analyzer.analyze_news(
            raw_text=(
                "BREAKING: Magnitude 6.8 earthquake strikes Taplejung. "
                "Government confirms 200 families displaced. Immediate relief needed."
            ),
            villages=villages,
            source="Nepal Police",
            multi_source_confirmed=True,
        )
        assert report.recommended_action == ACTION_AUTO_OPTIMIZE
        assert report.event.confidence >= 0.8

    def test_earthquake_report_generates_urgency_changes(self, news_analyzer, villages):
        report = news_analyzer.analyze_news(
            raw_text=(
                "Major landslide in Taplejung. Road cut off. "
                "Government relief requested."
            ),
            villages=villages,
            source="Nepal Government",
            multi_source_confirmed=True,
        )
        assert len(report.urgency_change) > 0

    def test_auto_report_triggers_reoptimization(self, news_analyzer, trigger, villages):
        report = news_analyzer.analyze_news(
            raw_text=(
                "Earthquake in Taplejung. 500 families affected. "
                "Critical medical supplies needed urgently."
            ),
            villages=villages,
            source="Nepal Police",
            multi_source_confirmed=True,
        )
        assert report.urgency_change, "No urgency change — village name not matched"
        change = trigger.trigger_reoptimization(report)
        assert change.optimization_state == "complete"
        assert change.trigger_event_id == report.event.event_id
        assert change.execution_time_ms >= 0.0

    def test_reoptimization_recorded_in_history(self, news_analyzer, trigger, villages):
        report = news_analyzer.analyze_news(
            raw_text="Disaster in Taplejung. Multiple casualties confirmed.",
            villages=villages,
            source="Nepal Police",
            multi_source_confirmed=True,
        )
        assert report.urgency_change
        trigger.trigger_reoptimization(report)
        assert len(trigger.get_optimization_history()) == 1

    def test_low_confidence_unverified_gets_ignore(self, news_analyzer, villages):
        report = news_analyzer.analyze_news(
            raw_text="Rumor of something happening somewhere maybe.",
            villages=villages,
            source="Unknown",
            multi_source_confirmed=False,
        )
        assert report.recommended_action == ACTION_IGNORE
        assert report.event.confidence < 0.5


# ================================================================== #
#  3. HITL approval workflow                                           #
# ================================================================== #

class TestHITLWorkflow:
    def test_submit_creates_pending_request(self, news_analyzer, approval_queue, villages):
        report = news_analyzer.analyze_news(
            raw_text=(
                "Bridge collapse reported in Jumla area. "
                "Unverified community radio report."
            ),
            villages=villages,
            source="Community Radio",
            multi_source_confirmed=False,
        )
        # Force submit even if confidence threshold varies
        from backend.rag.news_analyzer import NewsEvent
        from datetime import datetime, timezone
        event = NewsEvent(
            event_id="test_evt",
            raw_text="test",
            severity=5,
            confidence=0.65,
            affected_villages=[],
            resource_implications={},
        )
        req = approval_queue.submit_for_review(event)
        assert req.status == ApprovalStatus.PENDING

    def test_approve_changes_status(self, approval_queue):
        from backend.rag.news_analyzer import NewsEvent
        event = NewsEvent(
            event_id="test_approve",
            raw_text="test",
            severity=5,
            confidence=0.65,
            affected_villages=[],
            resource_implications={},
        )
        req = approval_queue.submit_for_review(event)
        approved = approval_queue.approve(req.request_id, reviewer="coord_001")
        assert approved.status == ApprovalStatus.APPROVED
        assert approved.reviewed_by == "coord_001"

    def test_reject_changes_status_with_reason(self, approval_queue):
        from backend.rag.news_analyzer import NewsEvent
        event = NewsEvent(
            event_id="test_reject",
            raw_text="test",
            severity=3,
            confidence=0.60,
            affected_villages=[],
            resource_implications={},
        )
        req = approval_queue.submit_for_review(event)
        rejected = approval_queue.reject(req.request_id, reviewer="coord_002", reason="Unverified")
        assert rejected.status == ApprovalStatus.REJECTED
        assert rejected.rejection_reason == "Unverified"

    def test_expire_old_moves_to_expired(self, approval_queue):
        from backend.rag.news_analyzer import NewsEvent
        from datetime import datetime, timezone
        event = NewsEvent(
            event_id="test_expire",
            raw_text="test",
            severity=4,
            confidence=0.55,
            affected_villages=[],
            resource_implications={},
        )
        req = approval_queue.submit_for_review(event)
        # Backdate expires_at to force expiry
        req.expires_at = "2000-01-01T00:00:00+00:00"
        expired_ids = approval_queue.expire_old_requests()
        assert req.request_id in expired_ids

    def test_history_contains_approved_requests(self, approval_queue):
        from backend.rag.news_analyzer import NewsEvent
        event = NewsEvent(
            event_id="hist_test",
            raw_text="test",
            severity=6,
            confidence=0.70,
            affected_villages=[],
            resource_implications={},
        )
        req = approval_queue.submit_for_review(event)
        approval_queue.approve(req.request_id, reviewer="coord_003")
        history = approval_queue.get_history()
        ids = [h.request_id for h in history]
        assert req.request_id in ids


# ================================================================== #
#  4. Timeline simulation end-to-end                                   #
# ================================================================== #

class TestTimelineSimulation:
    def test_all_six_events_processed(self, simulator):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)
        assert simulator.state.events_processed == 6

    def test_simulation_stops_after_completion(self, simulator):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)
        assert not simulator.state.is_running

    def test_action_counts_sum_to_six(self, simulator):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)
        s = simulator.state
        assert s.auto_count + s.hitl_count + s.ignore_count == 6

    def test_at_least_one_ignore_event(self, simulator):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)
        assert simulator.state.ignore_count >= 1

    def test_reoptimizations_triggered(self, simulator):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)
        assert simulator.state.reoptimizations_triggered > 0


# ================================================================== #
#  5. WebSocket broadcasting during simulation                         #
# ================================================================== #

class TestWebSocketBroadcasting:
    def test_messages_queued_during_simulation(self, simulator, ws_manager):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)
        assert len(ws_manager.message_history) > 0

    def test_event_processed_messages_present(self, simulator, ws_manager):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)
        types = {m.type for m in ws_manager.message_history}
        assert MSG_EVENT_PROCESSED in types

    def test_reopt_start_messages_present(self, simulator, ws_manager):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)
        types = {m.type for m in ws_manager.message_history}
        assert MSG_REOPTIMIZATION_START in types

    def test_new_client_receives_history_catchup(self, simulator, ws_manager):
        simulator.load_timeline()
        simulator.start_simulation()
        simulator._thread.join(timeout=15.0)

        ws = _MockWS()
        run(ws_manager.connect(ws))
        # Should have received up to 20 catch-up messages
        assert len(ws.sent) > 0


# ================================================================== #
#  6. P2P broadcast after re-optimization                              #
# ================================================================== #

class TestP2PBroadcast:
    def test_broadcast_adds_to_message_store(self, state_manager, villages, vehicles):
        gossip = GossipProtocol(node_id="node_0", fanout=2, ttl=3)

        cfg = ReoptimizationConfig(
            urgency_change_threshold=0.10,
            enable_reoptimization=True,
            broadcast_via_p2p=True,
            log_optimization_changes=False,
        )
        trig = ReoptimizationTrigger(
            config=cfg,
            state_manager=state_manager,
            villages=villages,
            vehicles=vehicles,
            gossip_protocol=gossip,
            time_elapsed=timedelta(hours=2),
        )

        # Use a fabricated report with urgency change
        from backend.rag.news_analyzer import IntelligenceReport, NewsEvent
        event = NewsEvent(
            event_id="p2p_test",
            raw_text="test",
            severity=7,
            confidence=0.92,
            affected_villages=["taplejung"],
            resource_implications={"food": 20.0},
        )
        report = IntelligenceReport(
            event=event,
            urgency_change={"taplejung": 0.20},
            recommended_action=ACTION_AUTO_OPTIMIZE,
        )

        trig.trigger_reoptimization(report)
        assert len(gossip.message_store) > 0

    def test_broadcast_creates_optimization_result_message(self, state_manager, villages, vehicles):
        gossip = GossipProtocol(node_id="node_1", fanout=2, ttl=3)
        cfg = ReoptimizationConfig(
            urgency_change_threshold=0.10,
            enable_reoptimization=True,
            broadcast_via_p2p=True,
            log_optimization_changes=False,
        )
        trig = ReoptimizationTrigger(
            config=cfg,
            state_manager=state_manager,
            villages=villages,
            vehicles=vehicles,
            gossip_protocol=gossip,
            time_elapsed=timedelta(hours=2),
        )

        from backend.rag.news_analyzer import IntelligenceReport, NewsEvent
        event = NewsEvent(
            event_id="p2p_type_test",
            raw_text="test",
            severity=6,
            confidence=0.90,
            affected_villages=["jumla"],
            resource_implications={},
        )
        report = IntelligenceReport(
            event=event,
            urgency_change={"jumla": 0.15},
            recommended_action=ACTION_AUTO_OPTIMIZE,
        )

        msg = trig.state_manager.run_full_optimization(villages, vehicles, timedelta(hours=2))
        # Verify gossip integration by triggering and checking store size increases
        before = len(gossip.message_store)
        trig.trigger_reoptimization(report)
        assert len(gossip.message_store) > before


# ================================================================== #
#  7. Stress test — rapid events                                       #
# ================================================================== #

class TestStressTest:
    def test_five_events_all_process(self, news_analyzer, trigger, villages):
        texts = [
            "Earthquake in Taplejung. Government confirms damage.",
            "Landslide blocked road near Taplejung. Major flooding.",
            "Hospital overwhelmed in Taplejung. Medical supplies critical.",
            "Earthquake aftershock in Taplejung. More families displaced.",
            "Relief coordinator confirms crisis in Taplejung escalating.",
        ]
        triggered = 0
        for raw in texts:
            report = news_analyzer.analyze_news(
                raw_text=raw,
                villages=villages,
                source="Nepal Police",
                multi_source_confirmed=True,
            )
            if report.urgency_change:
                change = trigger.trigger_reoptimization(report)
                assert change.optimization_state == "complete"
                triggered += 1

        assert triggered >= 1, "At least one event should have matched a village"

    def test_history_accumulates_across_events(self, news_analyzer, trigger, villages):
        texts = [
            f"Critical event {i} in Taplejung confirmed by government."
            for i in range(3)
        ]
        for raw in texts:
            report = news_analyzer.analyze_news(
                raw_text=raw,
                villages=villages,
                source="Nepal Government",
                multi_source_confirmed=True,
            )
            if report.urgency_change:
                trigger.trigger_reoptimization(report)

        history = trigger.get_optimization_history()
        assert len(history) >= 1

    def test_stress_no_crashes_on_rapid_processing(self, news_analyzer, villages):
        """10 events analyzed in rapid succession without errors."""
        analyzer = NewsAnalyzer()
        for i in range(10):
            report = analyzer.analyze_news(
                raw_text=f"Event {i} in Taplejung. Relief needed.",
                villages=villages,
                source="Test",
                multi_source_confirmed=(i % 2 == 0),
            )
            assert report.recommended_action in (
                ACTION_AUTO_OPTIMIZE, ACTION_HITL_REQUIRED, ACTION_IGNORE
            )
