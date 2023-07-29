import json
from fastapi import WebSocket


# WebSocket manager to manage multiple WebSocket connections
class WebSocketManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, data: dict):
        for connection in self.active_connections:
            await connection.send_text(json.dumps(data))