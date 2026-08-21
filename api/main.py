"""
FastAPI app serving the trained flight price prediction model.

Loads the saved model artifact once at startup (not per-request), and
exposes a single /predict endpoint. Deployed on Render, not SageMaker,
to avoid AWS billing risk (see README for reasoning).
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Flight Price Prediction API")

model = joblib.load("model_artifact.joblib")

FEATURE_ORDER = [
    "startingAirport",
    "destinationAirport",
    "primaryAirline",
    "flightDayOfWeek",
    "flightMonth",
    "departureHour",
    "numStops",
    "totalTravelDistance",
    "daysUntilDeparture",
    "isBasicEconomy",
]


class FlightFeatures(BaseModel):
    startingAirport: str
    destinationAirport: str
    primaryAirline: str
    flightDayOfWeek: int
    flightMonth: int
    departureHour: int
    numStops: int
    totalTravelDistance: float
    daysUntilDeparture: int
    isBasicEconomy: bool


class PredictionResponse(BaseModel):
    predictedFare: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: FlightFeatures):
    row = features.model_dump()
    row["isBasicEconomy"] = int(row["isBasicEconomy"])
    df = pd.DataFrame([row])[FEATURE_ORDER]
    prediction = model.predict(df)[0]
    return PredictionResponse(predictedFare=float(prediction))
