## External Libraries
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from datetime import datetime
import time
import asyncio

## Local Imports
from prediction_engine.classes.connection_manager import ConnectionManager
from prediction_engine.helpers.trader import get_stock_price
from prediction_engine.config_files.logger_config import logger


prediction_router = APIRouter()

# WebSocket manager to keep track of connected clients
manager = ConnectionManager()

# Function to send data to all connected websockets
async def send_data(ticker_name):
    while True:
        # Your task that generates data (random number in this case)

        try:
            price, timestamp, volume = get_stock_price(ticker_name)
            if (price > (last_long_action_price + 0.07*(7-(timestamp-last_long_action_timestamp)/60000)) and long_position > 0):
                balance += price*long_position
                print(f"Sold {long_position} shares at ${price} at a profit of ${price-last_long_action_price}/share")
                long_position = 0
        except Exception as e:
            logger.error(e)
            time.sleep(1.8)
            continue
        
        # Send data to all connected websockets
        manager.broadcast(1)

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