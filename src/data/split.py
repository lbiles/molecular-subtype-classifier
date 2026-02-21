from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_df_train_test(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/test sets."""
    stratify_y = df[target_column] if stratify else None

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )
    return df_train.copy(), df_test.copy()
