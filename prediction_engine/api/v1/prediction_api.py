## External Libraries
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from datetime import datetime
import time
import asyncio

import pandas as pd
import numpy as np

import tensorflow as tf

## Local Imports
from prediction_engine.classes.connection_manager import ConnectionManager
from prediction_engine.helpers.trader import get_stock_price, group_dataframe, generate_recommendation_max
from prediction_engine.config_files.logger_config import logger

prediction_router = APIRouter()

# WebSocket manager to keep track of connected clients
manager = ConnectionManager()

positions = {
    "AAPL" : {
        "long_position" : 0,
        "last_long_action_price" : 0.0,
        "last_long_action_timestamp" : 0,
        "last_short_action_price" : 0.0,
        "last_short_action_timestamp" : 0,
        "curr_interval" : []
    }
}

pred_dist = 5
min_models = [tf.keras.models.load_model(f'./prediction_engine/models/{i+1}_min_bc_model') for i in range(pred_dist)]
max_models = [tf.keras.models.load_model(f'./prediction_engine/models/{j+1}_max_bc_model') for j in range(pred_dist)]

# Function to send data to all connected websockets
async def send_data(ticker_name):
    # Your task that generates data (random number in this case)
    last_long_action_price = positions[ticker_name]["last_long_action_price"]
    last_long_action_timestamp = positions[ticker_name]["last_long_action_timestamp"]
    curr_interval = positions[ticker_name]["curr_interval"]
    long_position = positions[ticker_name]["long_position"]
    while True:
        try:
            price, timestamp, volume = get_stock_price(ticker_name)
            if (price > (last_long_action_price + 0.05*(3-(timestamp-last_long_action_timestamp)/60000)) and long_position > 0):
                balance += price*long_position
                print(f"Sold {long_position} shares at ${price} at a profit of ${price-last_long_action_price}/share")
                long_position = 0
        except Exception as e:
            logger.error(e)
            time.sleep(1.8)
            continue

        curr_interval.append({'price': price, 'timestamp': timestamp, "volume" : volume})

        df = pd.DataFrame.from_records(curr_interval)

        stat_array, curr_interval = group_dataframe(df = df, column = "timestamp", timestamp=timestamp)

        if stat_array is not None:
            stat_array_expanded = np.expand_dims(stat_array, axis=0)

            predicted_values_min = [min_models[i].predict(stat_array_expanded, verbose=0) for i in range(pred_dist)]
            predicted_values_max = [max_models[j].predict(stat_array_expanded, verbose=0) for j in range(pred_dist)]

            logger.info(predicted_values_max)

            result = {
                "predicted_values_min" : predicted_values_min,
                "predicted_values_max" : predicted_values_max,
                "price" : price,
                "curr_min" : {stat_array[:-1][1][0]}, 
                "curr_max" : {stat_array[:-1][0][1]},
            }
            
            # Send data to all connected websockets
            await manager.broadcast(result)

        # Wait for 2 seconds before sending data again
        await asyncio.sleep(2)

@prediction_router.websocket("/ws/calculate")
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