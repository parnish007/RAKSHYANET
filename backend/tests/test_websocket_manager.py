"""
Tests for WebSocketManager -- Prompt 5.3
Run: pytest backend/tests/test_websocket_manager.py -v
"""
import asyncio
from typing import List

import pytest

from backend.api.websocket_manager import (
    MSG_EVENT_PROCESSED,
    MSG_HITL_APPROVED,
    MSG_REOPTIMIZATION_DONE,
    WSMessage,
    WebSocketManager,
)


# ================================================================== #
#  Mock WebSocket                                                      #
# ================================================================== #

class MockWebSocket:
    """Minimal async WebSocket stand-in for testing."""

    def __init__(self, fail_on_send: bool = False) -> None:
        self.sent: List[dict] = []
        self.accepted: bool   = False
        self.fail_on_send     = fail_on_send

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, data: dict) -> None:
        if self.fail_on_send:
            raise RuntimeError("Simulated send failure")
        self.sent.append(data)

    async def receive_text(self) -> str:
        raise Exception("No data")


def run(coro):
    """Run a coroutine synchronously (no pytest-asyncio required)."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _msg(type_: str = MSG_EVENT_PROCESSED, **kw) -> WSMessage:
    return WSMessage(type=type_, payload=kw)


# ================================================================== #
#  Connection tests                                                    #
# ================================================================== #

class TestConnection:
    def test_connect_accepts_websocket(self):
        async def _t():
            mgr = WebSocketManager()
            ws  = MockWebSocket()
            await mgr.connect(ws)
            assert ws.accepted is True
        run(_t())

    def test_connect_adds_to_active_connections(self):
        async def _t():
            mgr = WebSocketManager()
            ws  = MockWebSocket()
            await mgr.connect(ws)
            assert ws in mgr.active_connections
        run(_t())

    def test_get_active_connections_count(self):
        async def _t():
            mgr = WebSocketManager()
            ws1, ws2 = MockWebSocket(), MockWebSocket()
            await mgr.connect(ws1)
            await mgr.connect(ws2)
            assert mgr.get_active_connections() == 2
        run(_t())

    def test_disconnect_removes_from_pool(self):
        async def _t():
            mgr = WebSocketManager()
            ws  = MockWebSocket()
            await mgr.connect(ws)
            mgr.disconnect(ws)
            assert ws not in mgr.active_connections
        run(_t())

    def test_disconnect_unknown_socket_is_noop(self):
        mgr = WebSocketManager()
        ws  = MockWebSocket()
        mgr.disconnect(ws)  # should not raise

    def test_disconnect_reduces_count(self):
        async def _t():
            mgr = WebSocketManager()
            ws1, ws2 = MockWebSocket(), MockWebSocket()
            await mgr.connect(ws1)
            await mgr.connect(ws2)
            mgr.disconnect(ws1)
            assert mgr.get_active_connections() == 1
        run(_t())


# ================================================================== #
#  Broadcast tests                                                     #
# ================================================================== #

class TestBroadcast:
    def test_broadcast_sends_to_all_clients(self):
        async def _t():
            mgr      = WebSocketManager()
            ws1, ws2 = MockWebSocket(), MockWebSocket()
            await mgr.connect(ws1)
            await mgr.connect(ws2)
            await mgr.broadcast(_msg(event_id="e1"))
            assert len(ws1.sent) == 1
            assert len(ws2.sent) == 1
        run(_t())

    def test_broadcast_payload_matches(self):
        async def _t():
            mgr = WebSocketManager()
            ws  = MockWebSocket()
            await mgr.connect(ws)
            await mgr.broadcast(_msg(foo="bar"))
            assert ws.sent[0]["payload"]["foo"] == "bar"
        run(_t())

    def test_broadcast_removes_disconnected_clients(self):
        async def _t():
            mgr      = WebSocketManager()
            good_ws  = MockWebSocket()
            bad_ws   = MockWebSocket(fail_on_send=True)
            await mgr.connect(good_ws)
            await mgr.connect(bad_ws)
            await mgr.broadcast(_msg())
            assert bad_ws not in mgr.active_connections
            assert good_ws in mgr.active_connections
        run(_t())

    def test_broadcast_message_type_preserved(self):
        async def _t():
            mgr = WebSocketManager()
            ws  = MockWebSocket()
            await mgr.connect(ws)
            await mgr.broadcast(WSMessage(type=MSG_REOPTIMIZATION_DONE, payload={"routes_changed": 3}))
            assert ws.sent[0]["type"] == MSG_REOPTIMIZATION_DONE
        run(_t())

    def test_send_to_client_serializes_correctly(self):
        async def _t():
            mgr = WebSocketManager()
            ws  = MockWebSocket()
            msg = WSMessage(type=MSG_HITL_APPROVED, payload={"request_id": "req_abc"})
            await mgr.send_to_client(ws, msg)
            assert ws.sent[0]["type"] == MSG_HITL_APPROVED
            assert ws.sent[0]["payload"]["request_id"] == "req_abc"
        run(_t())


# ================================================================== #
#  History tests                                                       #
# ================================================================== #

class TestHistory:
    def test_broadcast_adds_to_history(self):
        async def _t():
            mgr = WebSocketManager()
            await mgr.broadcast(_msg())
            assert len(mgr.message_history) == 1
        run(_t())

    def test_history_capped_at_max(self):
        async def _t():
            mgr = WebSocketManager()
            mgr.max_history = 5
            for i in range(10):
                await mgr.broadcast(_msg(i=i))
            assert len(mgr.message_history) == 5
        run(_t())

    def test_new_client_receives_history_on_connect(self):
        async def _t():
            mgr = WebSocketManager()
            # Pre-populate history without any clients
            for i in range(3):
                await mgr.broadcast(_msg(i=i))
            # New client should receive catch-up messages
            ws = MockWebSocket()
            await mgr.connect(ws)
            assert len(ws.sent) == 3
        run(_t())

    def test_new_client_receives_at_most_20_history(self):
        async def _t():
            mgr = WebSocketManager()
            for i in range(30):
                await mgr.broadcast(_msg(i=i))
            ws = MockWebSocket()
            await mgr.connect(ws)
            assert len(ws.sent) == 20
        run(_t())

    def test_history_oldest_dropped_when_full(self):
        async def _t():
            mgr = WebSocketManager()
            mgr.max_history = 3
            for i in range(4):
                await mgr.broadcast(_msg(seq=i))
            payloads = [m.payload.get("seq") for m in mgr.message_history]
            assert 0 not in payloads      # oldest dropped
            assert 3 in payloads          # newest kept
        run(_t())


# ================================================================== #
#  Integration tests                                                   #
# ================================================================== #

class TestIntegration:
    def test_multiple_clients_get_same_message(self):
        async def _t():
            mgr = WebSocketManager()
            clients = [MockWebSocket() for _ in range(5)]
            for ws in clients:
                await mgr.connect(ws)
            await mgr.broadcast(_msg(event_id="shared"))
            for ws in clients:
                assert ws.sent[-1]["payload"]["event_id"] == "shared"
        run(_t())

    def test_one_disconnect_does_not_affect_others(self):
        async def _t():
            mgr      = WebSocketManager()
            ws1, ws2 = MockWebSocket(), MockWebSocket()
            await mgr.connect(ws1)
            await mgr.connect(ws2)
            mgr.disconnect(ws1)
            await mgr.broadcast(_msg())
            assert len(ws2.sent) == 1
            assert len(ws1.sent) == 0
        run(_t())

    def test_broadcast_sync_noop_when_no_connections(self):
        mgr = WebSocketManager()
        msg = _msg()
        mgr.broadcast_sync(msg)  # should not raise; no-op because no connections

    def test_empty_manager_has_zero_connections(self):
        mgr = WebSocketManager()
        assert mgr.get_active_connections() == 0

    def test_ws_message_has_timestamp(self):
        msg = _msg()
        assert isinstance(msg.timestamp, str)
        assert len(msg.timestamp) > 10   # ISO-format string