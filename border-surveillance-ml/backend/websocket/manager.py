"""WebSocket connection manager -- broadcasts new events to every
connected control-room client in real time. Per the architecture
review: needs reconnect tolerance and shouldn't crash the broadcast
loop if one client's socket has died without cleanly disconnecting.
"""

import json
import logging
from typing import List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket connected. Total clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("WebSocket disconnected. Total clients: %d", len(self.active_connections))

    async def broadcast_event(self, event_dict: dict):
        message = json.dumps(event_dict, default=str)
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead_connections.append(connection)
        for dead in dead_connections:
            self.disconnect(dead)


manager = ConnectionManager()
