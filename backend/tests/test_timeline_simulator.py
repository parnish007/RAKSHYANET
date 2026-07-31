"""
Tests for TimelineSimulator -- Prompt 5.1
Run: pytest backend/tests/test_timeline_simulator.py -v
"""
import json
import time
from pathlib import Path

import pytest

from backend.demo.timeline_simulator import (
    SimulationState,
    SimulatorConfig,
    TimelineEvent,
    TimelineSimulator,
)
from backend.hitl.approval_queue import ApprovalQueue, ApprovalStatus
from backend.rag.news_analyzer import (
    ACTION_AUTO_OPTIMIZE,
    ACTION_HITL_REQUIRED,
    ACTION_IGNORE,
    NewsAnalyzer,
)

TIMELINE_PATH = str(
    Path(__file__).parent.parent / "demo" / "mock_news_timeline.json"
)


# ================================================================== #
#  Fixtures                                                            #
# ================================================================== #

@pytest.fixture
def analyzer():
    return NewsAnalyzer()


@pytest.fixture
def queue():
    return ApprovalQueue(timeout_minutes=5)


def _make_config(
    speed: float = 99999.0,   # effectively instant for tests
    auto_approve: bool = False,
    verbose: bool = False,
) -> SimulatorConfig:
    return SimulatorConfig(
        timeline_path=TIMELINE_PATH,
        speed_multiplier=speed,
        auto_approve_hitl=auto_approve,
        trigger_reoptimization=True,
        verbose_logging=verbose,
    )


@pytest.fixture
def simulator(analyzer, queue):
    return TimelineSimulator(_make_config(), analyzer, queue)


@pytest.fixture
def loaded_simulator(simulator):
    simulator.load_timeline()
    return simulator


# Inline test-only timeline (2 events: 1 AUTO, 1 IGNORE)
SMALL_TIMELINE = {
    "timeline_name": "test",
    "description": "test",
    "events": [
        {
            "timestamp_offset_seconds": 0,
            "event_id": "evt_quake",
            "raw_text": "6.2 earthquake Dhulikhel. Multiple casualties. Nepal Police confirms damage.",
            "source": "Nepal Police",
            "source_type": "verified_government",
            "expected_confidence": 0.90,
            "expected_action": "AUTO_OPTIMIZE",
        },
        {
            "timestamp_offset_seconds": 1,
            "event_id": "evt_rumor",
            "raw_text": "Heard something might have happened somewhere. @user123",
            "source": "Anonymous",
            "source_type": "unverified",
            "expected_confidence": 0.20,
            "expected_action": "IGNORE",
        },
    ],
}


@pytest.fixture
def small_timeline_path(tmp_path):
    p = tmp_path / "small_timeline.json"
    p.write_text(json.dumps(SMALL_TIMELINE), encoding="utf-8")
    return str(p)


@pytest.fixture
def small_simulator(analyzer, queue, small_timeline_path):
    cfg = SimulatorConfig(
        timeline_path=small_timeline_path,
        # Keep the two-event run observable long enough to test start/stop state.
        speed_multiplier=20.0,
        auto_approve_hitl=False,
        trigger_reoptimization=True,
        verbose_logging=False,
    )
    return TimelineSimulator(cfg, analyzer, queue)


# ================================================================== #
#  Initialization tests                                                #
# ================================================================== #

class TestInitialization:
    def test_simulator_creates_with_valid_config(self, analyzer, queue):
        sim = TimelineSimulator(_make_config(), analyzer, queue)
        assert sim is not None

    def test_state_starts_not_running(self, simulator):
        assert simulator.state.is_running is False

    def test_state_starts_with_zero_counts(self, simulator):
        s = simulator.state
        assert s.events_processed == 0
        assert s.events_pending_hitl == 0
        assert s.reoptimizations_triggered == 0

    def test_timeline_empty_before_load(self, simulator):
        assert simulator.timeline == []

    def test_config_speed_multiplier_stored(self, simulator):
        assert simulator.config.speed_multiplier == pytest.approx(99999.0)


# ================================================================== #
#  Timeline loading tests                                              #
# ================================================================== #

class TestTimelineLoading:
    def test_load_returns_list_of_timeline_events(self, simulator):
        events = simulator.load_timeline()
        assert isinstance(events, list)
        assert all(isinstance(e, TimelineEvent) for e in events)

    def test_load_returns_6_events_for_main_timeline(self, simulator):
        events = simulator.load_timeline()
        assert len(events) == 6

    def test_events_sorted_by_timestamp(self, simulator):
        events = simulator.load_timeline()
        offsets = [e.timestamp_offset_seconds for e in events]
        assert offsets == sorted(offsets)

    def test_first_event_is_national_taplejung_incident(self, simulator):
        events = simulator.load_timeline()
        assert events[0].event_id == "evt_landslide_taplejung"

    def test_missing_file_raises_file_not_found(self, analyzer, queue):
        cfg = SimulatorConfig(
            timeline_path="/nonexistent/path/timeline.json",
            speed_multiplier=1.0,
            verbose_logging=False,
        )
        sim = TimelineSimulator(cfg, analyzer, queue)
        with pytest.raises(FileNotFoundError):
            sim.load_timeline()

    def test_timeline_stored_on_simulator(self, simulator):
        simulator.load_timeline()
        assert len(simulator.timeline) == 6


# ================================================================== #
#  Event processing tests                                              #
# ================================================================== #

class TestEventProcessing:
    def _make_event(self, event_id, raw_text, source, source_type,
                    expected_confidence=0.5, expected_action="IGNORE",
                    offset=0) -> TimelineEvent:
        return TimelineEvent(
            timestamp_offset_seconds=offset,
            event_id=event_id,
            raw_text=raw_text,
            source=source,
            source_type=source_type,
            expected_confidence=expected_confidence,
            expected_action=expected_action,
        )

    def test_process_event_returns_intelligence_report(self, loaded_simulator):
        event = loaded_simulator.timeline[0]
        report = loaded_simulator.process_event(event)
        from backend.rag.news_analyzer import IntelligenceReport
        assert isinstance(report, IntelligenceReport)

    def test_process_event_increments_events_processed(self, loaded_simulator):
        event = loaded_simulator.timeline[0]
        loaded_simulator.process_event(event)
        assert loaded_simulator.state.events_processed == 1

    def test_process_event_updates_current_time(self, loaded_simulator):
        event = loaded_simulator.timeline[2]   # offset=600
        loaded_simulator.process_event(event)
        assert loaded_simulator.state.current_time_seconds == 600

    def test_ignore_event_increments_ignore_count(self, loaded_simulator):
        ignore_event = next(
            e for e in loaded_simulator.timeline if e.expected_action == "IGNORE"
        )
        loaded_simulator.process_event(ignore_event)
        assert loaded_simulator.state.ignore_count == 1

    def test_auto_optimize_event_triggers_reopt(self, loaded_simulator):
        auto_event = next(
            e for e in loaded_simulator.timeline if e.expected_action == "AUTO_OPTIMIZE"
        )
        before = loaded_simulator.state.reoptimizations_triggered
        report = loaded_simulator.process_event(auto_event)
        if report.recommended_action == ACTION_AUTO_OPTIMIZE:
            assert loaded_simulator.state.reoptimizations_triggered > before

    def test_callback_called_on_process(self, loaded_simulator):
        called = []
        loaded_simulator.on_event_callback = lambda e, r: called.append(e.event_id)
        loaded_simulator.process_event(loaded_simulator.timeline[0])
        assert len(called) == 1


# ================================================================== #
#  HITL routing tests                                                  #
# ================================================================== #

class TestHITLRouting:
    def test_hitl_event_added_to_queue(self, analyzer, queue):
        cfg = SimulatorConfig(
            timeline_path=TIMELINE_PATH,
            speed_multiplier=99999.0,
            auto_approve_hitl=False,
            verbose_logging=False,
        )
        sim = TimelineSimulator(cfg, analyzer, queue)
        sim.load_timeline()
        hitl_event = next(
            e for e in sim.timeline if e.expected_action == "HITL_REQUIRED"
        )
        report = sim.process_event(hitl_event)
        if report.recommended_action == ACTION_HITL_REQUIRED:
            assert len(queue.get_pending()) >= 1

    def test_auto_approve_hitl_approves_immediately(self, analyzer, queue):
        cfg = SimulatorConfig(
            timeline_path=TIMELINE_PATH,
            speed_multiplier=99999.0,
            auto_approve_hitl=True,
            verbose_logging=False,
        )
        sim = TimelineSimulator(cfg, analyzer, queue)
        sim.load_timeline()
        hitl_event = next(
            e for e in sim.timeline if e.expected_action == "HITL_REQUIRED"
        )
        report = sim.process_event(hitl_event)
        if report.recommended_action == ACTION_HITL_REQUIRED:
            history = queue.get_history()
            assert any(r.status == ApprovalStatus.APPROVED for r in history)
            assert sim.state.events_pending_hitl == 0

    def test_no_auto_approve_leaves_pending(self, analyzer, queue):
        cfg = SimulatorConfig(
            timeline_path=TIMELINE_PATH,
            speed_multiplier=99999.0,
            auto_approve_hitl=False,
            verbose_logging=False,
        )
        sim = TimelineSimulator(cfg, analyzer, queue)
        sim.load_timeline()
        hitl_event = next(
            e for e in sim.timeline if e.expected_action == "HITL_REQUIRED"
        )
        report = sim.process_event(hitl_event)
        if report.recommended_action == ACTION_HITL_REQUIRED:
            assert len(queue.get_pending()) >= 1


# ================================================================== #
#  Simulation control tests                                            #
# ================================================================== #

class TestSimulationControl:
    def test_start_sets_is_running(self, small_simulator):
        small_simulator.start_simulation()
        assert small_simulator.state.is_running is True
        if small_simulator._thread:
            small_simulator._thread.join(timeout=3.0)

    def test_start_sets_started_at(self, small_simulator):
        small_simulator.start_simulation()
        if small_simulator._thread:
            small_simulator._thread.join(timeout=3.0)
        assert small_simulator.state.started_at is not None

    def test_double_start_raises_runtime_error(self, small_simulator):
        small_simulator.start_simulation()
        try:
            with pytest.raises(RuntimeError):
                small_simulator.start_simulation()
        finally:
            small_simulator.stop_simulation()

    def test_stop_sets_is_running_false(self, small_simulator):
        small_simulator.start_simulation()
        small_simulator.stop_simulation()
        assert small_simulator.state.is_running is False

    def test_get_state_returns_simulation_state(self, simulator):
        state = simulator.get_state()
        assert isinstance(state, SimulationState)

    def test_reset_clears_state(self, small_simulator):
        small_simulator.start_simulation()
        if small_simulator._thread:
            small_simulator._thread.join(timeout=3.0)
        small_simulator.reset()
        assert small_simulator.state.events_processed == 0
        assert small_simulator.state.is_running is False

    def test_reset_while_running_stops_first(self, small_simulator):
        small_simulator.start_simulation()
        small_simulator.reset()
        assert small_simulator.state.is_running is False


# ================================================================== #
#  Full simulation integration tests                                   #
# ================================================================== #

class TestFullSimulation:
    def test_all_events_processed(self, small_simulator):
        small_simulator.start_simulation()
        small_simulator._thread.join(timeout=5.0)
        assert small_simulator.state.events_processed == 2

    def test_full_main_timeline_processes_6_events(self, analyzer, queue):
        cfg = SimulatorConfig(
            timeline_path=TIMELINE_PATH,
            speed_multiplier=99999.0,
            auto_approve_hitl=True,
            verbose_logging=False,
        )
        sim = TimelineSimulator(cfg, analyzer, queue)
        sim.start_simulation()
        sim._thread.join(timeout=10.0)
        assert sim.state.events_processed == 6

    def test_action_counts_sum_to_events_processed(self, analyzer, queue):
        cfg = SimulatorConfig(
            timeline_path=TIMELINE_PATH,
            speed_multiplier=99999.0,
            auto_approve_hitl=True,
            verbose_logging=False,
        )
        sim = TimelineSimulator(cfg, analyzer, queue)
        sim.start_simulation()
        sim._thread.join(timeout=10.0)
        total = sim.state.auto_count + sim.state.hitl_count + sim.state.ignore_count
        assert total == sim.state.events_processed

    def test_callback_called_for_each_event(self, small_simulator):
        received = []
        small_simulator.on_event_callback = lambda e, r: received.append(e.event_id)
        small_simulator.start_simulation()
        small_simulator._thread.join(timeout=5.0)
        assert len(received) == 2
