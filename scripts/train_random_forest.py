"""
Train a Random Forest baseline to predict totalFare.

Trains on the chronological "train" split, evaluates on "validation"
(the "test" split stays untouched until we pick a final winner across
all three models). Logs params/metrics/model to MLflow so this run is
directly comparable to the XGBoost and LightGBM runs we'll add later.
"""
import sys
sys.path.insert(0, "scripts")

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mlflow_setup import init_mlflow

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

N_ESTIMATORS = 100
MAX_DEPTH = 20
RANDOM_STATE = 42

print("Loading features.csv...")
df = pl.read_csv("features.csv").to_pandas()

# isBasicEconomy comes in as a Python bool column - cast to int (0/1) so
# it's unambiguous as a numeric feature
df["isBasicEconomy"] = df["isBasicEconomy"].astype(int)

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]
print(f"Train rows: {len(train_df):,}   Validation rows: {len(val_df):,}   Test rows: {len(test_df):,}")

X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y_train = train_df[TARGET]
X_val = val_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y_val = val_df[TARGET]
X_test = test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y_test = test_df[TARGET]

# one-hot encode categoricals, pass numeric columns through unchanged
preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
], remainder="passthrough")

model = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

init_mlflow()

with mlflow.start_run(run_name="random_forest_baseline"):
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("categorical_features", CATEGORICAL_FEATURES)
    mlflow.log_param("numeric_features", NUMERIC_FEATURES)
    mlflow.log_param("train_rows", len(train_df))
    mlflow.log_param("validation_rows", len(val_df))

    print("Training...")
    model.fit(X_train, y_train)

    print("Predicting on validation set...")
    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print("Predicting on test set (final, one-time evaluation)...")
    test_preds = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_r2", test_r2)

    mlflow.sklearn.log_model(model, name="model")

    print(f"VALIDATION MAE: ${mae:.2f}   RMSE: ${rmse:.2f}   R2: {r2:.4f}")
    print(f"TEST       MAE: ${test_mae:.2f}   RMSE: ${test_rmse:.2f}   R2: {test_r2:.4f}")
