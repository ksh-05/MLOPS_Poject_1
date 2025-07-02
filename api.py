import numpy as np 
import joblib
from config.path_config import *
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import pandas as pd 

app = FastAPI()

feature_names = [
    "lead_time", "no_of_special_requests", "avg_price_per_room",
    "arrival_month", "arrival_date", "market_segment_type",
    "no_of_week_nights", "no_of_weekend_nights",
    "type_of_meal_plan", "room_type_reserved"
]

class DataModel(BaseModel):
    lead_time: int
    no_of_special_requests: int
    avg_price_per_room: float
    arrival_month: int
    arrival_date: int
    market_segment_type: int
    no_of_week_nights: int
    no_of_weekend_nights: int
    type_of_meal_plan: int
    room_type_reserved: int

loaded_model = joblib.load(MODEL_OUTPUT_PATH)

@app.post("/predict")
async def predict(features : DataModel):
    feature = np.array(list(features.dict().values()))
    feature = pd.DataFrame([feature],columns=feature_names)
    inference = loaded_model.predict(feature)
    return {"output": int(inference[0])}

