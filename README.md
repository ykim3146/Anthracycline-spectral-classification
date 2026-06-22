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

## Dataset
A dataset (`Anthracycline_dataset.xlsx`) is provided to verify that the data processing and analysis pipeline executes correctly. 

## Prerequisites
To run this code, please ensure you have the following Python libraries installed:
```bash
pip install pandas numpy scikit-learn xgboost scikit-optimize shap
