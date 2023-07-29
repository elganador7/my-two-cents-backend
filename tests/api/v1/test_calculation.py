from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import Mock

from prediction_engine.api.v1 import prediction_api


app = FastAPI()
app.include_router(prediction_api.router)
client = TestClient(app)


def test_200_repair_generation():
    body = {
                "column" : "nominal_deg_min",
                "threshold" : -15,
                "data_entry_id" : "dod-forecast-20221125-ddbd2a",
                "inspection_id" : "dod-forecast-20221125-ddbd2a",
                "file_path" : "data.json",
                "repair_plan_name" : "testing_gen_api",
                "test" : True
            }
    result = client.post(
        url="/repair_plan/generate_plan", 
        headers={"accept": "application/json", "Authorization": "Bearer some_token"},
        json=body
    )
    assert result.status_code == 200


def test_400_bad_slug():
    body = {
                "column" : "nominal_deg_min",
                "threshold" : -15,
                "data_entry_id" : "",
                "inspection_id" : "",
                "file_path" : "data.json",
                "repair_plan_name" : "testing_gen_api",
                "test" : True
            }

    result = client.post(
        url="/repair_plan/generate_plan", 
        headers={"accept": "application/json", "Authorization": "Bearer some_token"},
        json=body
    )
    assert result.status_code == 400