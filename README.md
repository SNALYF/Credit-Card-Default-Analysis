## Lab 4: Credit Card Default Prediction — Reproducible ML Pipeline

This project implements a complete, reproducible machine-learning pipeline for predicting credit card default using the Default of Credit Card Clients dataset.
The pipeline follows the full ML workflow taught in DSCI 573, including preprocessing, feature engineering, model selection, hyperparameter optimization, and interpretation using SHAP.

# This repository contains:

A modularized Python pipeline (src/)

Scripts for training, evaluation, and generating SHAP explanations

A reproducible environment file

A detailed report inside the Jupyter notebook

This README documenting how to run the pipeline end-to-end

# Repository Structure:
```
project-root/
│
├── data/
│   └── (dataset not included—ignored by .gitignore)
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── shap_explain.py
│   └── utils.py
│
├── models/
│   └── xgb_best_model.pkl
│
├── results/
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── shap_plots/
│
├── environment.yml
├── README.md
└── notebook/
    └── lab4.ipynb
```
# Dataset

Default of Credit Card Clients Dataset (UCI Machine Learning Repository)

30,000 samples

24 features

Target: default.payment.next.month (renamed to target)

This dataset must be manually downloaded due to GitHub storage and privacy reasons.
Place the downloaded CSV as:

data/UCI_Credit_Card.csv

# Installation
1. Create the environment
conda env create -f environment.yml
conda activate 573

2. Install optional packages (SHAP, xgboost)

If not included already:

pip install shap xgboost altair_ally

# How to Run the Pipeline
1. Feature engineering + preprocessing + training
python src/train_model.py


This will:
✔ Load and split the dataset
✔ Engineer new features
✔ Apply transformations
✔ Train the tuned XGBoost model
✔ Save the model into models/xgb_best_model.pkl
✔ Save validation metrics into results/metrics.json

2. Evaluate on the test set
python src/evaluate.py


This script outputs:

F1, precision, recall, accuracy
ROC curve
Precision–recall curve
Confusion matrix

All saved in: results/

3. Generate SHAP explanations
python src/shap_explain.py


# Outputs:

Global feature importance (bar/summary)

Waterfall plots for chosen observations
Saved under:

results/shap_plots/

#  Model Summary

The pipeline tests several models but ultimately selects XGBoost (with tuned hyperparameters) due to highest validation F1.

Final Test Performance:

Metric	Score
F1	0.550
Precision	0.496
Recall	0.617
Accuracy	0.772

# Interpretation Summary

Using SHAP:

PAY_0, PAY_2, usage patterns, and credit limit are the most impactful features.

Repayment behaviors have clear monotonic influence: worse repayment history → higher default probability.

SHAP force plots provide local explanations for individual predictions.

Reproducibility Notes

No test data leakage (Golden Rule respected)

All feature engineering performed inside pipeline functions

Random seeds fixed for all models

Dataset excluded via .gitignore

# Tools & Technologies

Python 3.12

NumPy, pandas

scikit-learn

XGBoost

SHAP

Altair / Matplotlib

Otter (for MDS autograding)

# Possible Future Improvements

Try LightGBM, which may outperform XGBoost on tabular data

Create stacked models (meta modeling)

Improve feature engineering for time-series characteristics

Try focal loss or SMOTE to address class imbalance

Use Bayesian optimization for more efficient hyperparameter search

# Acknowledgements

This project follows the MDS 2025-26 curriculum (DSCI 571 + DSCI 573).
Dataset originally from: Yeh et al. (2009) – UCI Machine Learning Repository.