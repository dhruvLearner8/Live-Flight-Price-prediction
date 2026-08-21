"""
Train a LightGBM baseline to predict totalFare.

Same train/validation split, same feature treatment (one-hot encoding
for categoricals) as train_random_forest.py and train_xgboost.py, so
all three runs are a fair, apples-to-apples comparison in MLflow.
"""
import sys
sys.path.insert(0, "scripts")

import mlflow
import mlflow.sklearn
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
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

N_ESTIMATORS = 300
MAX_DEPTH = 6
LEARNING_RATE = 0.1
RANDOM_STATE = 42

print("Loading features.csv...")
df = pl.read_csv("features.csv").to_pandas()
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

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
], remainder="passthrough")

model = Pipeline([
    ("preprocess", preprocessor),
    ("lgbm", LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )),
])

init_mlflow()

with mlflow.start_run(run_name="lightgbm_baseline"):
    mlflow.log_param("model_type", "LGBMRegressor")
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth", MAX_DEPTH)
    mlflow.log_param("learning_rate", LEARNING_RATE)
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

    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=[
            "lightgbm.sklearn.LGBMRegressor",
            "lightgbm.basic.Booster",
            "collections.OrderedDict",
        ],
    )

    print(f"VALIDATION MAE: ${mae:.2f}   RMSE: ${rmse:.2f}   R2: {r2:.4f}")
    print(f"TEST       MAE: ${test_mae:.2f}   RMSE: ${test_rmse:.2f}   R2: {test_r2:.4f}")
