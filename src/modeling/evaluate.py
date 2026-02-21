from pathlib import Path
import json
import pandas as pd

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import REPORTS_DIR


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, class_names=None):
    """
    Evaluate trained classifier and return metrics.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
    }

    # Add labeled versions if class names provided
    if class_names is not None:
        metrics["class_names"] = class_names
        metrics["classification_report_labeled"] = classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        )
        metrics["confusion_matrix_labeled"] = {
            "labels": class_names,
            "matrix": cm.tolist(),
        }

    return metrics


def evaluate_models(
    models: dict, X_test: np.ndarray, y_test: np.ndarray, class_names=None
):
    """
    Evaluate all trained models and return comparison dict
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, class_names=class_names)
    return results


def save_metrics(results: dict, reports_dir: Path = REPORTS_DIR):
    """
    Save evaluation results to JSON and (when available) labeled confusion matrices to CSV.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics.json"

    # Save full metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save labeled confusion matrices as CSV (human-readable)
    for model_name, model_metrics in results.items():
        labeled = model_metrics.get("confusion_matrix_labeled")
        if not labeled:
            continue  # class_names weren't provided

        labels = labeled["labels"]
        matrix = labeled["matrix"]

        df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
        df_cm.index.name = "true_label"

        cm_path = reports_dir / f"confusion_matrix_{model_name}.csv"
        df_cm.to_csv(cm_path)

    return metrics_path
