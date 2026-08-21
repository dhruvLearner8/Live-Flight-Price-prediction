"""
Retrain the final locked XGBoost model (n_estimators=300, max_depth=6,
learning_rate=0.1) and predict a single dummy example, so we can
compare it against a real current price found via web search.
"""
import sys
sys.path.insert(0, "scripts")

from datetime import date

import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

CATEGORICAL_FEATURES = [
    "startingAirport",
    "destinationAirport",
    "primaryAirline",
    "flightDayOfWeek",
    "flightMonth",
    "departureHour",
]
NUMERIC_FEATURES = [
    "numStops",
    "totalTravelDistance",
    "daysUntilDeparture",
    "isBasicEconomy",
]
TARGET = "totalFare"

N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.1
RANDOM_STATE = 42

print("Loading features.csv and training final XGBoost model...")
df = pl.read_csv("features.csv").to_pandas()
df["isBasicEconomy"] = df["isBasicEconomy"].astype(int)
train_df = df[df["split"] == "train"]

X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y_train = train_df[TARGET]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
], remainder="passthrough")

model = Pipeline([
    ("preprocess", preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])
model.fit(X_train, y_train)

# --- build one example itinerary ---
today = date(2026, 8, 20)
travel_date = date(2026, 9, 24)
days_until_departure = (travel_date - today).days
flight_day_of_week = travel_date.isoweekday()  # 1=Mon .. 7=Sun
flight_month = travel_date.month

example = pd.DataFrame([{
    "startingAirport": "LAX",
    "destinationAirport": "BOS",
    "numStops": 0,
    "primaryAirline": "Delta",
    "totalTravelDistance": 2611,  # real LAX-BOS great-circle distance
    "isBasicEconomy": 0,
    "daysUntilDeparture": days_until_departure,
    "flightDayOfWeek": flight_day_of_week,
    "flightMonth": flight_month,
    "departureHour": 8,
}])

print("\nExample itinerary:")
print(f"  LAX -> BOS, nonstop, Delta, main cabin economy")
print(f"  Travel date: {travel_date} ({travel_date.strftime('%A')})")
print(f"  Days until departure: {days_until_departure}")
print(f"  Departure hour: 8am")

pred = model.predict(example[CATEGORICAL_FEATURES + NUMERIC_FEATURES])[0]
print(f"\nModel predicted price: ${pred:.2f}")
