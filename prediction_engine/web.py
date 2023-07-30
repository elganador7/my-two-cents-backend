from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from prediction_engine.config_files.logger_config import logger
from prediction_engine.api.v1.prediction_api import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can alter with time
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
