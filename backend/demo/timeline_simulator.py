"""
Timeline Simulator -- Prompt 5.1

Orchestrates the RAG -> HITL -> Re-optimization demo workflow by replaying
a scripted news-event timeline at configurable speed.

Each event passes through:
  1. NewsAnalyzer  (confidence scoring + action routing)
  2. ApprovalQueue (if HITL_REQUIRED)
  3. StateManager  (re-optimization stub -- wired in Prompt 5.2)

Typical usage
-------------
    config   = SimulatorConfig(timeline_path="backend/demo/mock_news_timeline.json",
                               speed_multiplier=100.0, auto_approve_hitl=True)
    analyzer = NewsAnalyzer()
    queue    = ApprovalQueue(timeout_minutes=5)
    sim      = TimelineSimulator(config, analyzer, queue)
    sim.load_timeline()
    sim.start_simulation()
    sim._thread.join()
    print(sim.get_state())
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from backend.hitl.approval_queue import ApprovalQueue
from backend.rag.news_analyzer import (
    ACTION_AUTO_OPTIMIZE,
    ACTION_HITL_REQUIRED,
    ACTION_IGNORE,
    IntelligenceReport,
    NewsAnalyzer,
)
from backend.api.websocket_manager import (
    WebSocketManager,
    WSMessage,
    MSG_EVENT_PROCESSED,
    MSG_REOPTIMIZATION_START,
    MSG_REOPTIMIZATION_DONE,
    MSG_HITL_SUBMITTED,
    MSG_HITL_APPROVED,
)

# Forward reference — imported lazily to avoid circular imports
_ReoptimizationTrigger = None
def _get_trigger_class():
    global _ReoptimizationTrigger
    if _ReoptimizationTrigger is None:
        from backend.demo.reoptimization_trigger import ReoptimizationTrigger
        _ReoptimizationTrigger = ReoptimizationTrigger
    return _ReoptimizationTrigger


# ================================================================== #
#  Data models                                                         #
# ================================================================== #

class TimelineEvent(BaseModel):
    """A single scripted event in the demo timeline."""
    timestamp_offset_seconds: float
    event_id:                 str
    raw_text:                 str
    source:                   str
    source_type:              str   # verified_government | verified_ngo | verified_news | unverified
    expected_confidence:      float = Field(ge=0.0, le=1.0)
    expected_action:          str   # AUTO_OPTIMIZE | HITL_REQUIRED | IGNORE


class SimulatorConfig(BaseModel):
    """Configuration for a simulation run."""
    timeline_path:          str
    speed_multiplier:       float = Field(default=1.0, gt=0.0)
    auto_approve_hitl:      bool  = False
    trigger_reoptimization: bool  = True
    verbose_logging:        bool  = True


class SimulationState(BaseModel):
    """Live state exposed to monitoring/UI."""
    model_config = {"frozen": False}

    current_time_seconds:      float = 0.0
    events_processed:          int   = 0
    events_pending_hitl:       int   = 0
    reoptimizations_triggered: int   = 0
    is_running:                bool  = False
    started_at:                Optional[str] = None
    auto_count:                int   = 0
    hitl_count:                int   = 0
    ignore_count:              int   = 0


# ================================================================== #
#  TimelineSimulator                                                   #
# ================================================================== #

class TimelineSimulator:
    """
    Replays a JSON news-event timeline through the full RAG -> HITL pipeline.

    Args:
        config:                  Runtime configuration.
        news_analyzer:           NewsAnalyzer instance for RAG analysis.
        approval_queue:          ApprovalQueue for HITL routing.
        reoptimization_trigger:  Optional trigger that calls StateManager on AUTO events.
        on_event_callback:       Optional fn(event, report) called after each event.
    """

    def __init__(
        self,
        config:                  SimulatorConfig,
        news_analyzer:           NewsAnalyzer,
        approval_queue:          ApprovalQueue,
        reoptimization_trigger:  Optional[object]          = None,
        on_event_callback:       Optional[Callable]        = None,
        villages:                Optional[list]            = None,
        websocket_manager:       Optional[WebSocketManager] = None,
    ) -> None:
        self.config                  = config
        self.news_analyzer           = news_analyzer
        self.approval_queue          = approval_queue
        self.reoptimization_trigger  = reoptimization_trigger
        self.on_event_callback       = on_event_callback
        self.villages                = villages or []
        self.websocket_manager       = websocket_manager

        self.timeline:  List[TimelineEvent] = []
        self.state      = SimulationState()

        self._stop_flag  = threading.Event()
        self._thread:    Optional[threading.Thread] = None
        self._start_wall: float = 0.0

    # -------------------------------------------------------------- #
    #  Public API                                                     #
    # -------------------------------------------------------------- #

    def load_timeline(self) -> List[TimelineEvent]:
        """
        Load and sort timeline events from the configured JSON file.

        Raises:
            FileNotFoundError: if the path doesn't exist.
        """
        path = Path(self.config.timeline_path)
        if not path.exists():
            raise FileNotFoundError(f"Timeline not found: {path}")

        data   = json.loads(path.read_text(encoding="utf-8"))
        events = [TimelineEvent(**evt) for evt in data["events"]]
        events.sort(key=lambda e: e.timestamp_offset_seconds)
        self.timeline = events
        return events

    def start_simulation(self) -> None:
        """
        Start processing events in a background daemon thread.

        Raises:
            RuntimeError: if simulation is already running.
        """
        if self.state.is_running:
            raise RuntimeError("Simulation already running")

        if not self.timeline:
            self.load_timeline()

        self.state.is_running = True
        self.state.started_at = datetime.now(timezone.utc).isoformat()
        self._start_wall      = time.time()
        self._stop_flag.clear()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        if self.config.verbose_logging:
            print(f"[SIM] Started -- {len(self.timeline)} events, "
                  f"speed={self.config.speed_multiplier}x")

    def stop_simulation(self) -> None:
        """Signal the background thread to stop; wait up to 5 s."""
        if not self.state.is_running:
            return
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.state.is_running = False
        if self.config.verbose_logging:
            print("[SIM] Stopped")

    def process_event(self, event: TimelineEvent) -> IntelligenceReport:
        """
        Route a single TimelineEvent through RAG -> HITL -> reopt stub.

        Updates self.state and calls on_event_callback if set.
        Returns the IntelligenceReport from NewsAnalyzer.
        """
        multi_source = event.source_type in ("verified_government", "verified_ngo")

        report = self.news_analyzer.analyze_news(
            raw_text=event.raw_text,
            villages=self.villages,
            source=event.source,
            multi_source_confirmed=multi_source,
        )

        action     = report.recommended_action
        confidence = report.event.confidence

        if self.config.verbose_logging:
            elapsed = time.time() - self._start_wall if self._start_wall else 0.0
            print(f"[T+{elapsed:5.1f}s] {event.event_id:<35} "
                  f"conf={confidence:.2f}  action={action}")

        # Broadcast EVENT_PROCESSED to all WebSocket clients
        self._ws_broadcast(WSMessage(
            type=MSG_EVENT_PROCESSED,
            payload={
                "event_id":   event.event_id,
                "action":     action,
                "confidence": confidence,
                "severity":   report.event.severity,
            },
        ))

        if action == ACTION_IGNORE:
            self.state.ignore_count += 1

        elif action == ACTION_HITL_REQUIRED:
            req = self.approval_queue.submit_for_review(report.event)
            self.state.hitl_count          += 1
            self.state.events_pending_hitl += 1

            self._ws_broadcast(WSMessage(
                type=MSG_HITL_SUBMITTED,
                payload={"request_id": req.request_id, "event_id": event.event_id},
            ))

            if self.config.verbose_logging:
                print(f"         -> HITL queued: {req.request_id}")

            if self.config.auto_approve_hitl:
                self.approval_queue.approve(req.request_id, reviewer="auto_simulator")
                self.state.events_pending_hitl -= 1

                self._ws_broadcast(WSMessage(
                    type=MSG_HITL_APPROVED,
                    payload={"request_id": req.request_id, "event_id": event.event_id},
                ))

                if self.config.trigger_reoptimization:
                    self._ws_broadcast(WSMessage(
                        type=MSG_REOPTIMIZATION_START,
                        payload={"event_id": event.event_id},
                    ))
                    if (
                        self.reoptimization_trigger is not None
                        and self.reoptimization_trigger.should_trigger_reoptimization(
                            report.urgency_change
                        )
                    ):
                        change = self.reoptimization_trigger.trigger_reoptimization(report)
                        self.state.reoptimizations_triggered += 1
                        self._ws_broadcast(WSMessage(
                            type=MSG_REOPTIMIZATION_DONE,
                            payload={
                                "event_id":          event.event_id,
                                "routes_changed":    change.routes_changed,
                                "welfare_delta":     change.welfare_improvement,
                                "execution_time_ms": change.execution_time_ms,
                            },
                        ))
                        if self.config.verbose_logging:
                            print(f"         -> Auto-approved + reopt "
                                  f"#{self.state.reoptimizations_triggered}  "
                                  f"routes_changed={change.routes_changed}")
                    else:
                        self.state.reoptimizations_triggered += 1
                        if self.config.verbose_logging:
                            print(f"         -> Auto-approved + reopt "
                                  f"#{self.state.reoptimizations_triggered}")

        elif action == ACTION_AUTO_OPTIMIZE:
            self.state.auto_count += 1
            if self.config.trigger_reoptimization:
                self._ws_broadcast(WSMessage(
                    type=MSG_REOPTIMIZATION_START,
                    payload={"event_id": event.event_id},
                ))
                if (
                    self.reoptimization_trigger is not None
                    and self.reoptimization_trigger.should_trigger_reoptimization(
                        report.urgency_change
                    )
                ):
                    change = self.reoptimization_trigger.trigger_reoptimization(report)
                    self.state.reoptimizations_triggered += 1
                    self._ws_broadcast(WSMessage(
                        type=MSG_REOPTIMIZATION_DONE,
                        payload={
                            "event_id":          event.event_id,
                            "routes_changed":    change.routes_changed,
                            "welfare_delta":     change.welfare_improvement,
                            "execution_time_ms": change.execution_time_ms,
                        },
                    ))
                    if self.config.verbose_logging:
                        print(f"         -> Reopt #{self.state.reoptimizations_triggered}  "
                              f"routes_changed={change.routes_changed}  "
                              f"welfare={change.welfare_improvement:+.4f}")
                else:
                    self.state.reoptimizations_triggered += 1
                    if self.config.verbose_logging:
                        print(f"         -> Auto-optimize reopt "
                              f"#{self.state.reoptimizations_triggered}")

        self.state.events_processed     += 1
        self.state.current_time_seconds  = event.timestamp_offset_seconds

        if self.on_event_callback:
            self.on_event_callback(event, report)

        return report

    def _ws_broadcast(self, message: WSMessage) -> None:
        """Thread-safe fire-and-forget WebSocket broadcast."""
        if self.websocket_manager:
            self.websocket_manager.broadcast_sync(message)

    def get_state(self) -> SimulationState:
        """Return a snapshot of the current simulation state."""
        return self.state

    def reset(self) -> None:
        """Stop simulation (if running) and clear all state."""
        if self.state.is_running:
            self.stop_simulation()
        self.state       = SimulationState()
        self._start_wall = 0.0

    # -------------------------------------------------------------- #
    #  Internal                                                       #
    # -------------------------------------------------------------- #

    def _run_loop(self) -> None:
        """Background thread: sleep between events then process each."""
        start = time.time()

        for event in self.timeline:
            if self._stop_flag.is_set():
                break

            target_wall = event.timestamp_offset_seconds / self.config.speed_multiplier
            sleep_for   = target_wall - (time.time() - start)

            if sleep_for > 0:
                # Interruptible sleep: poll stop_flag every 0.05 s
                deadline = time.time() + sleep_for
                while time.time() < deadline:
                    if self._stop_flag.is_set():
                        break
                    time.sleep(min(0.05, deadline - time.time()))

            if self._stop_flag.is_set():
                break

            try:
                self.process_event(event)
            except Exception as exc:
                print(f"[SIM] Error processing {event.event_id}: {exc}")

        self.state.is_running = False
        if self.config.verbose_logging:
            print(f"[SIM] Complete -- {self.state.events_processed} events processed")
