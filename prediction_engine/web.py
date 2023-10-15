from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import asyncio

from prediction_engine.config_files.logger_config import logger
from prediction_engine.api.v1 import prediction_api, testing_api #trainer_api

app = FastAPI()

origins = [
    "http://localhost:3000",  # Update this with your React app's URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(prediction_api.prediction_router, prefix="/api")
app.include_router(testing_api.testing_router, prefix="/testing")
# app.include_router(trainer_api.training_router))

# Start the data sending task
# asyncio.create_task(send_data(ticker_name="AAPL"))