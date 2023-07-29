from prediction_engine.api.v1 import prediction_api
from fastapi import FastAPI

app = FastAPI(
    description="train and test models for various stocks"
)

app = FastAPI()

app.include_router(prediction_api, prefix="/prediction")