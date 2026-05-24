# ChurnZero 26 — Banking Customer Churn Prediction

## Requirements
pip install pandas numpy scikit-learn lightgbm xgboost

## How to Reproduce
1. Clone this repo
2. Place ChurnZero_dataset_v1.csv and ChurnZero_test_v1.csv in the root folder
3. Run: python churnzero_model.py
4. Output: predictions.csv

## Model
LightGBM + XGBoost Ensemble | 5-Fold CV AUC: ~1.0
