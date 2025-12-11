from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import MODELS_DIR, SHAP_DIR
from .load_data import load_raw_data, train_test_split_data
from .features import engineer_features
from .utils import ensure_dir


def main() -> None:
    # ---------- Load model ----------
    model_path = MODELS_DIR / "xgb_best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train_model.py first."
        )
    pipe = joblib.load(model_path)

    # ---------- Load & split data ----------
    df = load_raw_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    # ---------- Feature engineering on test ----------
    X_test_eng = engineer_features(X_test)

    # ---------- Get encoded features ----------
    preprocessor = pipe.named_steps["preprocessor"]
    X_test_enc = preprocessor.transform(X_test_eng)
    feature_names = preprocessor.get_feature_names_out()

    # ---------- SHAP explainer ----------
    xgb_model = pipe.named_steps["xgbclassifier"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_enc)

    ensure_dir(SHAP_DIR)

    # ---------- Global importance (bar) ----------
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    bar_path = SHAP_DIR / "shap_bar.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Global SHAP bar plot saved to {bar_path}")

    # ---------- Choose example indices (one 0, one 1) ----------
    y_test_reset = y_test.reset_index(drop=True)
    zero_indices = y_test_reset[y_test_reset == 0].index.tolist()
    one_indices = y_test_reset[y_test_reset == 1].index.tolist()

    if zero_indices and one_indices:
        ex_0 = zero_indices[10 if len(zero_indices) > 10 else 0]
        ex_1 = one_indices[10 if len(one_indices) > 10 else 0]

        # Waterfall for negative example
        shap.plots.waterfall(shap_values[ex_0], show=False, max_display=15)
        plt.tight_layout()
        wf0_path = SHAP_DIR / "waterfall_example_0.png"
        plt.savefig(wf0_path, dpi=150)
        plt.close()
        print(f"Waterfall plot (target=0) saved to {wf0_path}")

        # Waterfall for positive example
        shap.plots.waterfall(shap_values[ex_1], show=False, max_display=15)
        plt.tight_layout()
        wf1_path = SHAP_DIR / "waterfall_example_1.png"
        plt.savefig(wf1_path, dpi=150)
        plt.close()
        print(f"Waterfall plot (target=1) saved to {wf1_path}")
    else:
        print("Could not find both class 0 and class 1 examples in test set.")


if __name__ == "__main__":
    main()
