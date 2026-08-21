"""
Train the final locked XGBoost model and save it to disk as a single
file, so the API server can just load it instantly instead of
retraining from scratch on every deploy/restart.
"""
import joblib
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
OUTPUT_PATH = "model_artifact.joblib"

print("Loading features.csv...")
df = pl.read_csv("features.csv").to_pandas()
df["isBasicEconomy"] = df["isBasicEconomy"].astype(int)

# train on train+validation combined - test stays held out from this
# artifact too, but for the deployed model we want it to learn from as
# much data as reasonably available up to the point we locked the design
train_df = df[df["split"].isin(["train", "validation"])]
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

print("Training final model on train+validation...")
model.fit(X_train, y_train)

joblib.dump(model, OUTPUT_PATH)
print(f"Saved {OUTPUT_PATH}")
