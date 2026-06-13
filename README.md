# Optical Spectral Fingerprinting for Anthracycline Detection

This repository contains the core machine learning and feature importance pipeline for the paper:
> **"Optical Spectral Fingerprinting Enables Sensitive Detection of Anthracycline Chemotherapeutics in Synthetic Clinical Biofluids"**

## Project Description
1. This project provides a machine learning pipeline for **Anthracycline classification** using **XGBoost**. It incorporates **Bayesian Optimization** (via `scikit-optimize`) for hyperparameter tuning and leverages **SHAP (SHapley Additive exPlanations)** analysis to identify and interpret meaningful molecular features.

2. Features are standardized and decomposed via Principal Component Analysis (PCA) based purely on training concentrations in a controlled buffer environment. A SVM classifier is then trained and tested. To evaluate translational viability, the pre-trained PCA space and SVM decision boundaries are directly applied to classify targets within complex synthetic clinical biofluids.

---
## Demo Dataset
A dummy dataset (`sample_data.xlsx`) is provided in this repository to demonstrate the code structure and verify that the pipeline executes correctly.

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
