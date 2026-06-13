"""
config.py — Central configuration for Fraud Detection System
All paths, hyperparameters, and constants in one place.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

DATA_PATH        = os.path.join(DATA_DIR,  "creditcard.csv")
MODEL_PATH       = os.path.join(MODEL_DIR, "xgb_model.pkl")
SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.pkl")

# ─── Features ─────────────────────────────────────────────────────────────────

V_FEATURES    = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS  = V_FEATURES + ["Amount_scaled", "Time_scaled"]
TARGET_COL    = "Class"

# ─── Preprocessing ────────────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Time column normalization constants (computed from full dataset)
TIME_MEAN = 94813.0
TIME_STD  = 47488.0

# ─── SMOTE ────────────────────────────────────────────────────────────────────

SMOTE_K_NEIGHBORS = 5

# ─── XGBoost ──────────────────────────────────────────────────────────────────

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

# ─── SHAP ─────────────────────────────────────────────────────────────────────

SHAP_SAMPLE_SIZE = 2000   # Number of samples for SHAP computation (speed vs accuracy)
SHAP_TOP_N       = 5      # Top N features to return in API response

# ─── Thresholds ───────────────────────────────────────────────────────────────

FRAUD_THRESHOLD  = 0.5    # Probability above which transaction is flagged as fraud
RISK_HIGH        = 0.7    # >= HIGH risk
RISK_MEDIUM      = 0.3    # >= MEDIUM risk, < HIGH
