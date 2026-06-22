

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
except ImportError:
    raise ImportError("Please install scikit-optimize: pip install scikit-optimize")

# SHAP for model interpretation
try:
    import shap
except ImportError:
    raise ImportError("Please install shap: pip install shap")


def main(df):
    """
    Args:
        df (pd.DataFrame): The input dataframe containing the dataset.
    """
    # ==============================================================================
    # 1. Column Definitions
    # ==============================================================================
    sol_col = df.columns[0]       # SampleType
    anthra_col = df.columns[1]     # Anthracycline (Target)
    conc_col = df.columns[2]       # Concentration
    group_col = df.columns[3]     # Replicate
    signal_cols = df.columns[4:]   # Features

    # Define concentrations for training and testing
    train_concs = [0.1, 1, 10, 100]
    test_concs = [0.5, 5, 50]

    # Target Encoding
    le = LabelEncoder()
    df['target'] = le.fit_transform(df[anthra_col].astype(str))
    class_names = list(le.classes_)

    # ==============================================================================
    # 2. Data Filtering (Buffer Selection) & Train/Test Split
    # ==============================================================================
    # Filter dataset where SampleType is 'Buffer'
    df_buffer = df[df[sol_col] == 'Buffer']

    # Split into train and test sets based on defined concentrations
    train_df = df_buffer[df_buffer[conc_col].isin(train_concs)]
    test_df = df_buffer[df_buffer[conc_col].isin(test_concs)]

    X_train = train_df[signal_cols].values
    y_train = train_df['target'].values

    groups_train = train_df[group_col].values
    
    X_test = test_df[signal_cols].values
    y_test = test_df['target'].values

    # Preprocessing Pipeline (Imputation + Scaling)
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    X_train_prep = preprocessing_pipeline.fit_transform(X_train)
    X_test_prep = preprocessing_pipeline.transform(X_test)

    feature_names = signal_cols.tolist()
    X_train_shap_df = pd.DataFrame(X_train_prep, columns=feature_names)

    # ==============================================================================
    # 3. XGBoost Model & Hyperparameter Optimization
    # ==============================================================================
    xgb_model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        base_score=0.5,
        n_jobs=-1
    )

    # Define the search space for Bayesian Optimization
    search_space = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(3, 10),
        'n_estimators': Integer(50, 300),
        'subsample': Real(0.6, 1.0),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Bayesian Optimization using BayesSearchCV
    opt = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=search_space,
        n_iter=100,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    # Execute training and hyperparameter tuning
    opt.fit(X_train_prep, y_train, groups=groups_train)

    # Model Evaluation
    best_model = opt.best_estimator_
    y_pred = best_model.predict(X_test_prep)
    test_acc = accuracy_score(y_test, y_pred)


    # ==============================================================================
    # 4. SHAP Analysis
    # ==============================================================================
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_shap_df)

    return best_model, shap_values

