"""
Shared MLflow config, imported by every model training script.

Keeping this in one place (rather than repeating it in each training
script) means every run lands in the same experiment, so all Random
Forest / XGBoost / LightGBM runs are comparable side by side in the
MLflow UI. A typo'd experiment name in one script would otherwise
silently split its runs into a separate, invisible experiment.

Tracking backend is a local SQLite database (./mlflow.db) - no server
needed for a local portfolio project. MLflow 3.x deprecated the plain
filesystem store in favor of a database backend. mlflow.db and the
mlartifacts/ folder it writes to are both gitignored.
"""
import mlflow

EXPERIMENT_NAME = "flight-price-prediction"


def init_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
