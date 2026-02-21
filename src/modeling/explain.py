import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.modeling.train import load_models


def save_random_forest_feature_importance(
    top_n: int = 20, out_dir=REPORTS_DIR, model_name: str = "random_forest"
):
    """
    Save top-N feature importances for Random Forest as CSV.
    """
    models, prep = load_models()
    if model_name not in models:
        raise ValueError(
            f"Unknown model_name='{model_name}'. Available: {list(models.keys())}"
        )

    model = models[model_name]
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model '{model_name}' does not expose feature_importances_.")

    feature_columns = prep["feature_columns"]
    importances = np.asarray(model.feature_importances_, dtype=float)

    if len(importances) != len(feature_columns):
        raise ValueError(
            f"Mismatch: {len(importances)} importances vs {len(feature_columns)} feature columns."
        )

    df = (
        pd.DataFrame({"feature": feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feature_importance_{model_name}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_logistic_regression_coefficients(
    top_n: int = 20, out_dir=REPORTS_DIR, model_name: str = "logistic_regression"
):
    """
    Save a simple LR coefficient summary.

    For multinomial LR, we compute mean absolute coefficient across classes
    to get an overall 'importance-like' ranking.
    """
    models, prep = load_models()
    if model_name not in models:
        raise ValueError(
            f"Unknown model_name='{model_name}'. Available: {list(models.keys())}"
        )

    model = models[model_name]
    if not hasattr(model, "coef_"):
        raise ValueError(f"Model '{model_name}' does not expose coef_.")

    feature_columns = prep["feature_columns"]
    coef = np.asarray(
        model.coef_, dtype=float
    )  # shape: (n_classes, n_features) OR (1, n_features)

    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    if coef.shape[1] != len(feature_columns):
        raise ValueError(
            f"Mismatch: coef has {coef.shape[1]} features vs {len(feature_columns)} feature columns."
        )

    # mean absolute coefficient across classes
    scores = np.mean(np.abs(coef), axis=0)

    df = (
        pd.DataFrame({"feature": feature_columns, "mean_abs_coef": scores})
        .sort_values("mean_abs_coef", ascending=False)
        .head(top_n)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feature_importance_{model_name}.csv"
    df.to_csv(out_path, index=False)
    return out_path
