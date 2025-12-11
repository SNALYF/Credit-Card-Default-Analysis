from pathlib import Path

# --------- Paths ---------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SHAP_DIR = RESULTS_DIR / "shap_plots"

# --------- General settings ---------
TARGET_COL = "target"
RANDOM_STATE = 573
TEST_SIZE = 0.2
