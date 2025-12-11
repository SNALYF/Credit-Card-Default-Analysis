import pandas as pd

# Categorical columns used in preprocessing
CATEGORICAL_COLS = [
    "SEX",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "EDUCATION",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering consistent with the notebook:

    - usage_rate: (sum(BILL_AMT) - sum(PAY_AMT)) / LIMIT_BAL
    - repayment_rate: PAY_AMT1 / BILL_AMT2
    - growth: BILL_AMT1 - BILL_AMT6
    - avg_bill: average of BILL_AMT1..6
    - avg_pay: average of PAY_AMT1..6

    Operates row-wise and does not use the target, so can be safely applied
    to both train and test sets separately.
    """
    df = df.copy()

    bill_cols = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ]
    pay_cols = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    # usage_rate (avoid division by zero)
    df["usage_rate"] = (df[bill_cols].sum(axis=1) - df[pay_cols].sum(axis=1)) / df[
        "LIMIT_BAL"
    ].replace(0, 1)

    # repayment_rate
    df["repayment_rate"] = df["PAY_AMT1"] / df["BILL_AMT2"].replace(0, 1)

    # growth: recent vs oldest bill
    df["growth"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    # avg bill & avg pay
    df["avg_bill"] = df[bill_cols].mean(axis=1)
    df["avg_pay"] = df[pay_cols].mean(axis=1)

    return df
