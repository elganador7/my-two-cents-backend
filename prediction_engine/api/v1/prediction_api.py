from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from prediction_engine.classes.connection_manager import ConnectionManager

from datetime import datetime
import asyncio

import json
import random

router = APIRouter()

# WebSocket manager to keep track of connected clients
manager = ConnectionManager()

# Function to send data to all connected websockets
async def send_data():
    while True:
        # Your task that generates data (random number in this case)
        data = random.randint(1, 100)

        # Send data to all connected websockets
        manager.broadcast(data)

        # Wait for 2 seconds before sending data again
        await asyncio.sleep(2)

@router.websocket("/ws/calculate")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Perform your calculation here (replace with your actual calculation)
            timestamp = int(datetime.now().timestamp() * 1000)

            result = {
                "value": 42,
                "timestamp" : timestamp
            }

            # Convert the result to JSON and send it to the connected client
            await websocket.send_json(result)

            # Optional: You can add a delay here if needed
            await asyncio.sleep(1)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)