"""
WebSocket Routes -- Prompt 5.3

Exposes a single /ws endpoint for real-time event streaming to the frontend.

Connection:  ws://localhost:8000/ws
Protocol:    server -> client only (push)
Keep-alive:  client may send "ping"; server replies "pong"
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.websocket_manager import ws_manager

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Real-time event stream.

    On connect: replays last 20 messages (catch-up for reconnects).
    While alive: receives optional ping, replies pong.
    On disconnect: removed from active pool.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)