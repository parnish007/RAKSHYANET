"""
WebSocket Manager -- Prompt 5.3

Manages real-time WebSocket connections and broadcasts optimization events,
news updates, and HITL status changes to all connected frontend clients.

Typical usage (FastAPI)
-----------------------
    from backend.api.websocket_manager import ws_manager, WSMessage, MSG_EVENT_PROCESSED

    msg = WSMessage(type=MSG_EVENT_PROCESSED, payload={"event_id": "evt_001"})
    await ws_manager.broadcast(msg)

From a background thread (simulator)
-------------------------------------
    ws_manager.broadcast_sync(msg)   # thread-safe
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import WebSocket
from pydantic import BaseModel, Field


# ================================================================== #
#  Message type constants                                              #
# ================================================================== #

MSG_EVENT_PROCESSED      = "EVENT_PROCESSED"
MSG_REOPTIMIZATION_START = "REOPTIMIZATION_START"
MSG_REOPTIMIZATION_DONE  = "REOPTIMIZATION_COMPLETE"
MSG_VEHICLE_UPDATE       = "VEHICLE_UPDATE"
MSG_HITL_SUBMITTED       = "HITL_SUBMITTED"
MSG_HITL_APPROVED        = "HITL_APPROVED"
MSG_HITL_REJECTED        = "HITL_REJECTED"


# ================================================================== #
#  Message model                                                       #
# ================================================================== #

class WSMessage(BaseModel):
    """Versioned event envelope; ``type`` remains for legacy clients."""
    type: str
    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    scenario_id: str = "nepal-national-demo"
    timestamp: str                 = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: str = "1.0"
    correlation_id: Optional[str] = None
    event_type: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.event_type is None:
            self.event_type = self.type


# ================================================================== #
#  WebSocketManager                                                    #
# ================================================================== #

class WebSocketManager:
    """
    Manages a pool of active WebSocket connections.

    Thread safety
    -------------
    ``broadcast`` and ``connect`` are async — call from async context.
    ``broadcast_sync`` posts to the running event loop via
    ``run_coroutine_threadsafe``, safe to call from a background thread.
    """

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self.message_history:   List[WSMessage]  = []
        self.max_history:       int               = 100

    # -------------------------------------------------------------- #
    #  Connection lifecycle                                           #
    # -------------------------------------------------------------- #

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket and replay the last 20 messages."""
        await websocket.accept()
        self.active_connections.append(websocket)

        for msg in self.message_history[-20:]:
            await self.send_to_client(websocket, msg)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket from the active pool."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    def get_active_connections(self) -> int:
        """Return the number of currently connected clients."""
        return len(self.active_connections)

    # -------------------------------------------------------------- #
    #  Broadcasting                                                   #
    # -------------------------------------------------------------- #

    async def broadcast(self, message: WSMessage) -> None:
        """
        Send a message to every connected client.

        Appends to history and auto-removes clients that have disconnected.
        """
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        await self._send_to_clients(message)

    async def _send_to_clients(self, message: WSMessage) -> None:
        """Send to all active connections; remove those that error."""
        disconnected: List[WebSocket] = []
        for websocket in self.active_connections:
            try:
                await self.send_to_client(websocket, message)
            except Exception:
                disconnected.append(websocket)
        for ws in disconnected:
            self.disconnect(ws)

    async def send_to_client(self, websocket: WebSocket, message: WSMessage) -> None:
        """Serialize and send a message to one client."""
        await websocket.send_json(message.model_dump())

    def broadcast_sync(self, message: WSMessage) -> None:
        """
        Thread-safe broadcast for use from background (non-async) threads.

        Always appends to message_history so history is consistent even
        with no active connections.  Skips async send if no clients connected.
        """
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)

        if not self.active_connections:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)
            else:
                loop.run_until_complete(self.broadcast(message))
        except Exception:
            pass


# ================================================================== #
#  Module-level singleton (shared across routes)                       #
# ================================================================== #

ws_manager = WebSocketManager()
