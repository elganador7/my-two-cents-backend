## External Libraries
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from datetime import datetime
import time
import asyncio

import pandas as pd
import numpy as np

import tensorflow as tf

from prediction_engine.classes.connection_manager import ConnectionManager

data_df = pd.read_pickle("./assets/sample_data/july_apple_raw_28-31.pkl")

testing_router = APIRouter()

# WebSocket manager to keep track of connected clients
testing_manager = ConnectionManager()


@testing_router.websocket("/ws/testws")
async def websocket_endpoint(websocket: WebSocket):
    await testing_manager.connect(websocket)
    try:
        # Perform your calculation here (replace with your actual calculation)

        for _, row in data_df.iterrows():
            data_to_send = row.to_dict()
            await websocket.send_json(data_to_send)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await testing_manager.disconnect(websocket)
