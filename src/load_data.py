from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, TARGET_COL, RANDOM_STATE, TEST_SIZE


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw UCI credit card dataset and perform basic cleaning:
    - rename target column to 'target'
    - drop ID column
    - cast categorical columns
    - merge EDUCATION values 0, 5, 6 into 5 (as in the notebook)
    """
    df = pd.read_csv(path)

    # Rename target & drop ID
    df = df.rename({"default.payment.next.month": TARGET_COL}, axis=1)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Categorical columns
    cat_cols = [
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        TARGET_COL,
    ]

    # Fix EDUCATION 0/5/6 -> 5 (same logic as notebook)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 5 if x in [0, 5, 6] else x)

    df[cat_cols] = df[cat_cols].astype("category")

    return df


def train_test_split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test, returning X_train, X_test, y_train, y_test.
    """
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=1 - test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
