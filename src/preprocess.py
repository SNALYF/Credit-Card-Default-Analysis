from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_COLS


def get_drop_cols(columns: List[str]) -> list:
    """Columns containing 'AMT' are dropped (to reduce multicollinearity)."""
    return [c for c in columns if "AMT" in c]


def build_tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocessor for tree-based models (e.g., XGBoost):
    - passthrough numeric features
    - one-hot encode categoricals
    - drop *_AMT* columns
    """
    all_cols = list(X.columns)
    drop_cols = get_drop_cols(all_cols)

    categorical_cols = [c for c in CATEGORICAL_COLS if c in all_cols]
    passthrough_cols = [
        c for c in all_cols if c not in categorical_cols + drop_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("passthrough", "passthrough", passthrough_cols),
            (
                "onehotencoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("drop", "drop", drop_cols),
        ]
    )

    return preprocessor


def build_linear_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocessor for linear/logistic models:
    - standardize numeric features
    - one-hot encode categorical
    - drop *_AMT* columns
    """
    all_cols = list(X.columns)
    drop_cols = get_drop_cols(all_cols)

    categorical_cols = [c for c in CATEGORICAL_COLS if c in all_cols]
    numeric_cols = [
        c for c in all_cols if c not in categorical_cols + drop_cols
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("standardscaler", StandardScaler(), numeric_cols),
            (
                "onehotencoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("drop", "drop", drop_cols),
        ]
    )

    return preprocessor
