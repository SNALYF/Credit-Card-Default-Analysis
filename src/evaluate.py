from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import MODELS_DIR, RESULTS_DIR
from .load_data import load_raw_data, train_test_split_data
from .features import engineer_features
from .utils import ensure_dir, save_or_update_json


def main() -> None:
    # ---------- Load model ----------
    model_path = MODELS_DIR / "xgb_best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )
    best_pipe = joblib.load(model_path)

    # ---------- Load & split data ----------
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # ---------- Feature engineering on test ----------
    X_test_eng = engineer_features(X_test)

    # ---------- Predict ----------
    y_pred = best_pipe.predict(X_test_eng)
    if hasattr(best_pipe, "predict_proba"):
        y_proba = best_pipe.predict_proba(X_test_eng)[:, 1]
    else:
        # fallback for models without predict_proba
        y_proba = best_pipe.decision_function(X_test_eng)

    # ---------- Metrics ----------
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Test metrics:")
    print(f"  F1        : {f1:.6f}")
    print(f"  Precision : {precision:.6f}")
    print(f"  Recall    : {recall:.6f}")
    print(f"  Accuracy  : {acc:.6f}")

    # ---------- Save metrics ----------
    ensure_dir(RESULTS_DIR)
    metrics_path = RESULTS_DIR / "metrics.json"
    test_metrics = {
        "test": {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(acc),
        }
    }
    save_or_update_json(test_metrics, metrics_path)
    print(f"Updated metrics saved to {metrics_path}")

    # ---------- Plots ----------
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=np.unique(y_test)
    )
    disp.plot()
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.tight_layout()
    roc_path = RESULTS_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {roc_path}")

    # Precision-Recall curve
    PrecisionRecallDisplay.from_predictions(y_test, y_proba)
    plt.tight_layout()
    pr_path = RESULTS_DIR / "pr_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print(f"Precision-Recall curve saved to {pr_path}")


if __name__ == "__main__":
    main()
