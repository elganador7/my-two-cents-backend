## External Libraries
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from datetime import datetime
import time
import asyncio

import pandas as pd
import numpy as np

import tensorflow as tf

from pydantic import BaseModel

## Local Imports
from prediction_engine.classes.connection_manager import ConnectionManager
from prediction_engine.helpers.trainer import create_stepped_sequences, create_sequences, generate_recommendation_max
from prediction_engine.config_files.logger_config import logger

training_router = APIRouter()

class TrainerModel(BaseModel):
    # Required Inputs
    ticker_symbol: str
    training_start: int
    training_end: int
    training_interval_size: int
    training_interval_unit : str
    lookback_intervals : int
    prediction_intervals : int
    file_path: str
    repair_plan_name: str

# @training_router.post(
#     "/train_model",
#     responses={
#         200: {"model": str},
#     },
# )
# async def train_model(
#     model_training_data: PredictorModel,
# ):
#     ## Load Data
#     ## For now, this is all done locally with pre-loaded data, this, #TODO to fix this

#     sequence_length = model_training_data.lookback_intervals
#     pred_distance = model_training_data.prediction_intervals

#     X, y_min_sequences, y_max_sequences = create_stepped_sequences("./train_data.pkl", sequence_length, pred_dist=pred_distance)
#     X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
#     test_X, test_y_min_sequences, test_y_max_sequences = create_stepped_sequences("./test_data.pkl", sequence_length, pred_dist = pred_distance)


#     return None