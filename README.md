# Anthracycline Classification & SHAP Analysis

## Project Description
This project provides a machine learning pipeline for **Anthracycline classification** using **XGBoost**. It incorporates **Bayesian Optimization** (via `scikit-optimize`) for hyperparameter tuning and leverages **SHAP (SHapley Additive exPlanations)** analysis to identify and interpret meaningful molecular features.

---

Prepare your dataset matching the structure described above.

Column Index,Column Name (Example),Description
SampleType,"Type of solution (e.g., must include 'Buffer' for filtering)": df.columns[0]
Anthracycline : df.columns[1]
Concentration : df.columns[2]
Spectral Features: df.columns[3:]

## Prerequisites
To run this code, please ensure you have the following Python libraries installed:
```bash
pip install pandas numpy scikit-learn xgboost scikit-optimize shap

