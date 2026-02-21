import pandas as pd

from src.data.preprocess import PreprocessArtifacts, transform_features
from src.modeling.train import load_models


def predict_from_dataframe(
    df_features: pd.DataFrame,
    model_name: str = "random_forest",
    return_proba: bool = True,
    include_label_id: bool = True,
):
    """
    Predict PAM50 subtype from a feature dataframe.
    """
    models, prep = load_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model_name='{model_name}'. Available: {list(models.keys())}"
        )

    model = models[model_name]

    artifacts = PreprocessArtifacts(
        feature_columns=prep["feature_columns"],
        label_encoder=prep["label_encoder"],
        scaler=prep["scaler"],
        medians=prep.get("medians", {}),
    )

    X = transform_features(df_features, artifacts)
    y_pred = model.predict(X)

    out = pd.DataFrame(index=df_features.index)

    if include_label_id:
        out["predicted_label_id"] = y_pred

    out["predicted_label"] = artifacts.label_encoder.inverse_transform(y_pred)

    if return_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        class_names = list(artifacts.label_encoder.classes_)
        for i, cname in enumerate(class_names):
            out[f"proba_{cname}"] = proba[:, i]

    return out
