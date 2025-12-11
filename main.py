"""
main.py
python main.py
python main.py --step train
python main.py --step eval
python main.py --step shap
python main.py --step all
"""

import argparse

from src.train_model import main as train_main
from src.evaluate import main as eval_main
from src.shap_explain import main as shap_main


def run_all():
    print("\n====== Step 1: Training & hyperparameter tuning ======\n")
    train_main()

    print("\n====== Step 2: Evaluating on test set ======\n")
    eval_main()

    print("\n====== Step 3: Generating SHAP explanations ======\n")
    shap_main()

    print("\n Pipeline finished. Check 'models/' and 'results/' folders.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ML pipeline.")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "train", "eval", "shap"],
        help="Choose which step to run (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.step == "all":
        run_all()
    elif args.step == "train":
        train_main()
    elif args.step == "eval":
        eval_main()
    elif args.step == "shap":
        shap_main()
