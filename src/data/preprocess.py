from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import TARGET_COLUMN


@dataclass
class PreprocessArtifacts:
    feature_columns: List[str]
    label_encoder: LabelEncoder
    scaler: StandardScaler
    medians: Dict[str, float]  # for handling missing values


def clean_metabric_df(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_classes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply label-level cleaning only (no feature fitting):
      - drop rows with missing target
      - drop specified classes (default: ["NC"])
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    drop_classes = drop_classes or ["NC"]

    df = df.dropna(subset=[target_column]).copy()
    df = df[~df[target_column].isin(drop_classes)].copy()
    return df


def get_feature_columns(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> List[str]:
    """
    Select numeric feature columns. Excludes target if numeric.
    """
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in feature_columns:
        feature_columns.remove(target_column)
    return feature_columns


def fit_preprocess(
    df_train: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[np.ndarray, np.ndarray, PreprocessArtifacts]:
    """
    Fit preprocessing artifacts using TRAIN ONLY (prevents leakage).
    Returns:
      - X_train (scaled)
      - y_train (encoded)
      - artifacts (feature_columns, medians, scaler, label_encoder)
    """
    if target_column not in df_train.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data.")

    feature_columns = get_feature_columns(df_train, target_column=target_column)

    X_df = df_train[feature_columns].copy()
    y_raw = df_train[target_column].astype(str)

    # Compute train medians and impute TRAIN using train medians
    medians_series = X_df.median(numeric_only=True)
    X_df = X_df.fillna(medians_series)

    # Fit label encoder on TRAIN labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Fit scaler on TRAIN features only
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    artifacts = PreprocessArtifacts(
        feature_columns=feature_columns,
        label_encoder=label_encoder,
        scaler=scaler,
        medians=medians_series.to_dict(),
    )
    return X, y, artifacts


def transform_features(
    df: pd.DataFrame,
    artifacts: PreprocessArtifacts,
) -> np.ndarray:
    """
    Transform features using previously fit artifacts
    - aligns columns
    - imputes missing using TRAIN medians
    - scales using TRAIN-fitted scaler
    """
    missing = [c for c in artifacts.feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing[:10]}... (total {len(missing)})"
        )

    X_df = df[artifacts.feature_columns].copy()

    # Impute using train medians
    X_df = X_df.fillna(pd.Series(artifacts.medians))

    # Scale
    X = artifacts.scaler.transform(X_df.values)
    return X


def transform_labels(
    df: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    target_column: str = TARGET_COLUMN,
) -> np.ndarray:
    """
    Encode labels using train-fitted label encoder.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    y_raw = df[target_column].astype(str)
    return artifacts.label_encoder.transform(y_raw)
