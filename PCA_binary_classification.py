

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def model_pca_svm(df, n_components=10):
    sol_col = df.columns[0]  # SampleType (Buffer, Urine, Sweat)
    anthra_col = df.columns[1]  # Anthracycline (Target)
    conc_col = df.columns[2]  # Concentration
    signal_cols = df.columns[3:]  # Spectral Features

    # Concentration splits defined in the protocol
    train_concs = [0.1, 1, 10, 100]
    test_concs = [0.5, 5, 50]

    # ==============================================================================
    # 1. Baseline Training & Testing on "Buffer" Matrix
    # ==============================================================================
    df_buffer = df[df[sol_col] == 'Buffer'].copy()

    train_buffer_df = df_buffer[df_buffer[conc_col].isin(train_concs)]
    test_buffer_df = df_buffer[df_buffer[conc_col].isin(test_concs)]

    X_train_buff = train_buffer_df[signal_cols].values
    y_train_buff = train_buffer_df[anthra_col].values

    X_test_buff = test_buffer_df[signal_cols].values
    y_test_buff = test_buffer_df[anthra_col].values

    # Fit Scaler and PCA on Buffer Training Data only (To avoid data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_buff)

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Train SVM Classifier using Buffer PCA features
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_pca, y_train_buff)

    # Evaluate on Buffer Test Data
    X_test_buff_pca = pca.transform(scaler.transform(X_test_buff))
    y_pred_buff = svm_model.predict(X_test_buff_pca)
    buffer_acc = accuracy_score(y_test_buff, y_pred_buff)

    results = {"Buffer (Internal Test)": buffer_acc}

    # ==============================================================================
    # 2. Generalization Evaluation on "Urine" and "Sweat" Matrices
    # ==============================================================================
    # Evaluate other biofluids using the pre-trained Buffer models
    target_biofluids = ['Urine', 'Sweat']

    for fluid in target_biofluids:
        df_fluid = df[df[sol_col] == fluid].copy()

        # Evaluate using specific test concentrations
        test_fluid_df = df_fluid[df_fluid[conc_col].isin(test_concs)]

        if test_fluid_df.empty:
            print(f"Warning: No valid test samples found for {fluid}.")
            continue

        X_eval = test_fluid_df[signal_cols].values
        y_eval = test_fluid_df[anthra_col].values

        # Project biofluid data into the established Buffer PCA-SVM space
        X_eval_pca = pca.transform(scaler.transform(X_eval))
        y_pred_eval = svm_model.predict(X_eval_pca)
        eval_acc = accuracy_score(y_eval, y_pred_eval)

        results[f"{fluid} (Clinical Evaluation)"] = eval_acc

