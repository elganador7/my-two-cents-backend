from fastapi import APIRouter, Depends, Header, HTTPException, WebSocket
from prediction_engine.web import app
from pydantic import BaseModel

from prediction_engine.config_files.logger_config import logger
from prediction_engine.classes.web_socket_manager import WebSocketManager

router = APIRouter(
    prefix="/prediction",
)

# WebSocket manager to keep track of connected clients
manager = WebSocketManager()

@app.websocket("/calculate/")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Perform your calculation here (replace with your actual calculation)
            result = {"value": 42}

            # Convert the result to JSON and send it to the connected client
            await websocket.send_json(result)

            # Optional: You can add a delay here if needed
            # await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await manager.disconnect(websocket)


