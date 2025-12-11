from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

from .config import MODELS_DIR, RESULTS_DIR, RANDOM_STATE
from .load_data import load_raw_data, train_test_split_data
from .features import engineer_features
from .preprocess import build_tree_preprocessor
from .utils import ensure_dir, save_or_update_json


def build_xgb_pipeline(X_train_eng: pd.DataFrame, y_train) -> RandomizedSearchCV:
    """
    Build an XGBoost + preprocessing pipeline and wrap it in RandomizedSearchCV
    to optimize multiple hyperparameters (same idea as in the notebook).
    """
    # Preprocessor
    preprocessor = build_tree_preprocessor(X_train_eng)

    # Class imbalance weight
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    scale_pos_weight = neg / pos

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("xgbclassifier", xgb),
        ]
    )

    # Hyperparameter search space (mirrors notebook)
    param_distributions = {
        "xgbclassifier__n_estimators": randint(100, 1000),
        "xgbclassifier__learning_rate": loguniform(0.001, 0.3),
        "xgbclassifier__max_depth": randint(3, 10),
        "xgbclassifier__subsample": uniform(0.6, 0.4),
        "xgbclassifier__colsample_bytree": uniform(0.6, 0.4),
        "xgbclassifier__min_child_weight": randint(1, 10),
        "xgbclassifier__gamma": uniform(0.0, 5.0),
        "xgbclassifier__reg_lambda": loguniform(1e-3, 10.0),
        "xgbclassifier__reg_alpha": loguniform(1e-3, 10.0),
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_jobs=-1,
        scoring="f1",
        return_train_score=True,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    return search


def main() -> None:
    # ---------- Load & split ----------
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # ---------- Feature engineering ----------
    X_train_eng = engineer_features(X_train)
    # NOTE: we do NOT touch X_test here, keeping test data for later evaluation.

    # ---------- Build & tune model ----------
    search = build_xgb_pipeline(X_train_eng, y_train)
    search.fit(X_train_eng, y_train)

    best_estimator = search.best_estimator_
    best_row = (
        pd.DataFrame(search.cv_results_)[
            ["mean_test_score", "mean_train_score"]
        ]
        .sort_values("mean_test_score", ascending=False)
        .iloc[0]
    )

    # ---------- Save model ----------
    ensure_dir(MODELS_DIR)
    model_path = MODELS_DIR / "xgb_best_model.pkl"
    joblib.dump(best_estimator, model_path)
    print(f"Saved best model to {model_path}")

    # ---------- Save validation metrics ----------
    ensure_dir(RESULTS_DIR)
    metrics_path = RESULTS_DIR / "metrics.json"
    val_metrics = {
        "validation": {
            "mean_f1_val": float(best_row["mean_test_score"]),
            "mean_f1_train": float(best_row["mean_train_score"]),
            "best_params": search.best_params_,
        }
    }
    save_or_update_json(val_metrics, metrics_path)
    print(f"Validation metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
