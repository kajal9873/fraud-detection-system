"""
model.py — Model training, saving, and loading
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from config import (
    MODEL_DIR, MODEL_PATH, SCALER_PATH,
    XGB_PARAMS, RANDOM_STATE,
)


def train_logistic_regression(
    X_train, y_train,
    C: float = 0.01,
    max_iter: int = 1000,
) -> LogisticRegression:
    """
    Train Logistic Regression as baseline model.

    Args:
        X_train  : Training features (post-SMOTE)
        y_train  : Training labels (post-SMOTE)
        C        : Inverse regularization strength
        max_iter : Max solver iterations

    Returns:
        Trained LogisticRegression model
    """
    print("Training Logistic Regression (baseline)...")
    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    print("✅ Logistic Regression trained.")
    return lr


def train_xgboost(
    X_train, y_train,
    X_val=None, y_val=None,
    params: dict = None,
    verbose: int = 100,
) -> XGBClassifier:
    """
    Train XGBoost classifier (main model).

    Args:
        X_train  : Training features (post-SMOTE)
        y_train  : Training labels (post-SMOTE)
        X_val    : Validation features for early stopping eval (optional)
        y_val    : Validation labels (optional)
        params   : XGBoost hyperparameters (defaults from config.py)
        verbose  : Print progress every N rounds (0 = silent)

    Returns:
        Trained XGBClassifier
    """
    print("Training XGBoost (main model)...")
    p = params or XGB_PARAMS

    xgb = XGBClassifier(**p)

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"]  = verbose

    xgb.fit(X_train, y_train, **fit_kwargs)
    print("✅ XGBoost trained.")
    return xgb


def save_model(model, path: str = MODEL_PATH) -> None:
    """
    Save trained model to disk using joblib.

    Args:
        model : Trained sklearn-compatible model
        path  : Output path for .pkl file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"✅ Model saved → {path} ({size_mb:.1f} MB)")


def load_model(path: str = MODEL_PATH):
    """
    Load saved model from disk.

    Args:
        path: Path to .pkl file

    Returns:
        Loaded model object
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Run train.py or 02_model_training.ipynb first."
        )
    model = joblib.load(path)
    print(f"✅ Model loaded from {path}")
    return model


def load_scaler(path: str = SCALER_PATH):
    """
    Load saved StandardScaler from disk.

    Args:
        path: Path to scaler .pkl file

    Returns:
        Loaded StandardScaler
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scaler not found at '{path}'.\n"
            "Run train.py or 02_model_training.ipynb first."
        )
    scaler = joblib.load(path)
    print(f"✅ Scaler loaded from {path}")
    return scaler


def predict(model, X) -> tuple[np.ndarray, np.ndarray]:
    """
    Run prediction on input features.

    Args:
        model : Trained model
        X     : Feature array or DataFrame

    Returns:
        (y_pred, y_prob) — binary predictions and fraud probabilities
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    return y_pred, y_prob


if __name__ == "__main__":
    from preprocess import run_preprocessing
    from evaluate import print_metrics

    X_train, X_test, y_train, y_test = run_preprocessing()

    # Train both models
    lr  = train_logistic_regression(X_train, y_train)
    xgb = train_xgboost(X_train, y_train, X_val=X_test, y_val=y_test)

    # Evaluate
    y_pred_lr,  y_prob_lr  = predict(lr,  X_test)
    y_pred_xgb, y_prob_xgb = predict(xgb, X_test)

    print("\n── Logistic Regression ──")
    print_metrics(y_test, y_pred_lr, y_prob_lr)

    print("\n── XGBoost ──────────────")
    print_metrics(y_test, y_pred_xgb, y_prob_xgb)

    # Save XGBoost
    save_model(xgb)
