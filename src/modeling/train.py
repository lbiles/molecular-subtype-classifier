from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR
from src.data.preprocess import PreprocessArtifacts


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> Dict[str, object]:
    # Simple baseline linear classifier
    lr = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )
    lr.fit(X_train, y_train)

    # Nonlinear model
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    return {"logistic_regression": lr, "random_forest": rf}


def save_models(
    models: Dict[str, object],
    artifacts: PreprocessArtifacts,
    models_dir: Path = MODELS_DIR,
) -> Dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)

    lr_path = models_dir / "logistic_regression.joblib"
    rf_path = models_dir / "random_forest.joblib"
    prep_path = models_dir / "preprocess_artifacts.joblib"

    joblib.dump(models["logistic_regression"], lr_path)
    joblib.dump(models["random_forest"], rf_path)

    joblib.dump(
        {
            "feature_columns": artifacts.feature_columns,
            "label_encoder": artifacts.label_encoder,
            "scaler": artifacts.scaler,
            "medians": artifacts.medians,
        },
        prep_path,
    )

    return {
        "logistic_regression": lr_path,
        "random_forest": rf_path,
        "preprocess_artifacts": prep_path,
    }


def load_models(models_dir: Path = MODELS_DIR) -> Tuple[Dict[str, object], dict]:
    lr = joblib.load(models_dir / "logistic_regression.joblib")
    rf = joblib.load(models_dir / "random_forest.joblib")
    prep = joblib.load(models_dir / "preprocess_artifacts.joblib")
    return {"logistic_regression": lr, "random_forest": rf}, prep
