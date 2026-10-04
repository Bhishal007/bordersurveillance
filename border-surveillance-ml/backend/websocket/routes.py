"""WebSocket endpoint -- control-room frontend connects here to receive
events the instant they're created (via POST /api/events), instead of
polling the REST API.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.websocket.manager import manager

router = APIRouter()


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect client->server messages, but need to keep
            # the receive loop alive to detect disconnects promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
