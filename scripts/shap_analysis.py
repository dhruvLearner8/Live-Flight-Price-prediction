"""
SHAP interpretability for our final model: XGBoost baseline
(n_estimators=300, max_depth=6, learning_rate=0.1), the winner of the
3-model comparison.

Retrains the exact same pipeline (deterministic given random_state=42,
so this reproduces the winning model rather than needing to reload it
from MLflow's artifact store) then explains its predictions on a
sample of the validation set using SHAP's TreeExplainer.

Produces two plots:
- shap_summary_bar.png: global feature importance (mean |SHAP value|)
- shap_summary_beeswarm.png: importance + direction of effect per row
"""
import sys
sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import polars as pl
import shap
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
SHAP_SAMPLE_SIZE = 5000

print("Loading features.csv...")
df = pl.read_csv("features.csv").to_pandas()
df["isBasicEconomy"] = df["isBasicEconomy"].astype(int)

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]

X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
y_train = train_df[TARGET]
X_val = val_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
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

print("Training final XGBoost model...")
model.fit(X_train, y_train)

print(f"Sampling {SHAP_SAMPLE_SIZE:,} rows from validation set for SHAP...")
X_val_sample = X_val.sample(n=SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE)

# transform through the fitted preprocessor to get the actual matrix
# XGBoost saw (one-hot encoded columns + passthrough numeric columns).
# OneHotEncoder returns a sparse matrix by default - convert to dense so
# SHAP can read actual feature values for the beeswarm plot's coloring.
X_transformed = model.named_steps["preprocess"].transform(X_val_sample)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()
feature_names = model.named_steps["preprocess"].get_feature_names_out()

print("Computing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(model.named_steps["xgb"])
shap_values = explainer.shap_values(X_transformed)

# bar plot: mean |SHAP value| per feature, ranked - "how much did each
# feature matter, on average, across all predictions"
plt.figure()
shap.summary_plot(
    shap_values, X_transformed, feature_names=feature_names,
    plot_type="bar", show=False, max_display=20,
)
plt.tight_layout()
plt.savefig("shap_summary_bar.png", dpi=150)
plt.close()
print("Saved shap_summary_bar.png")

# beeswarm plot: same ranking, but also shows whether high/low feature
# values push price up or down for each individual row
plt.figure()
shap.summary_plot(
    shap_values, X_transformed, feature_names=feature_names,
    show=False, max_display=20,
)
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=150)
plt.close()
print("Saved shap_summary_beeswarm.png")

# print ranked mean |SHAP| per feature as well, for a plain-text summary
import numpy as np
mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
print("\nFeature importance ranking (mean |SHAP value|, in dollars):")
for name, val in ranking[:20]:
    print(f"  {name:40s} ${val:.2f}")
