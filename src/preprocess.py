"""
preprocess.py — Data loading, scaling, SMOTE, train-test split
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from config import (
    DATA_PATH, MODEL_DIR, SCALER_PATH,
    FEATURE_COLS, TARGET_COL,
    TEST_SIZE, RANDOM_STATE,
    SMOTE_K_NEIGHBORS,
)


def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load creditcard.csv and perform basic validation.

    Args:
        data_path: Path to creditcard.csv

    Returns:
        Raw DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            "Download from:\n"
            "  Kaggle  : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "  Zenodo  : https://zenodo.org/records/7395559"
        )

    df = pd.read_csv(data_path)

    # Validate expected columns
    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    if df.isnull().sum().sum() > 0:
        print(f"⚠️  Warning: {df.isnull().sum().sum()} NaN values found — dropping rows.")
        df = df.dropna()

    print(f"✅ Data loaded: {df.shape[0]:,} rows | "
          f"Fraud: {df[TARGET_COL].sum():,} ({df[TARGET_COL].mean()*100:.4f}%)")
    return df


def scale_features(df: pd.DataFrame, save_scaler: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale Amount and Time columns (V1-V28 already PCA-scaled).
    Uses separate scalers to avoid overwriting.

    Args:
        df          : Raw DataFrame
        save_scaler : Whether to save scaler.pkl to models/

    Returns:
        (df_scaled, scaler_amount)
    """
    df = df.copy()

    scaler_amount = StandardScaler()
    scaler_time   = StandardScaler()

    df["Amount_scaled"] = scaler_amount.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler_time.fit_transform(df[["Time"]])

    if save_scaler:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler_amount, SCALER_PATH)
        print(f"✅ Scaler saved → {SCALER_PATH}")

    return df, scaler_amount


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target from scaled DataFrame.

    Args:
        df: Scaled DataFrame (must have Amount_scaled, Time_scaled)

    Returns:
        (X, y)
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}. "
            "Run scale_features() before get_X_y()."
        )

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Stratified train-test split — preserves fraud ratio in both sets.

    Args:
        X            : Feature DataFrame
        y            : Target Series
        test_size    : Fraction for test set (default 0.2)
        random_state : Seed for reproducibility

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"✅ Split done:")
    print(f"   Train → {X_train.shape} | Fraud: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
    print(f"   Test  → {X_test.shape}  | Fraud: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to training set ONLY.

    ⚠️  Critical: Never apply SMOTE to test set — causes data leakage.
        Synthetic samples in test set would inflate metrics artificially.

    Args:
        X_train      : Training features
        y_train      : Training labels
        k_neighbors  : SMOTE k_neighbors parameter
        random_state : Seed

    Returns:
        (X_train_resampled, y_train_resampled)
    """
    print(f"   Before SMOTE → Legit: {(y_train==0).sum():,} | Fraud: {(y_train==1).sum():,}")

    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"   After  SMOTE → Legit: {(y_res==0).sum():,} | Fraud: {(y_res==1).sum():,} "
          f"| Total: {len(X_res):,}")
    return X_res, y_res


def run_preprocessing(data_path: str = DATA_PATH) -> tuple:
    """
    Full preprocessing pipeline — load → scale → split → SMOTE.

    Args:
        data_path: Path to creditcard.csv

    Returns:
        (X_train_sm, X_test, y_train_sm, y_test)
    """
    print("\n── Preprocessing Pipeline ──────────────────────────")
    df = load_data(data_path)
    df, _ = scale_features(df, save_scaler=True)
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\n── Applying SMOTE ──────────────────────────────────")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    print("────────────────────────────────────────────────────\n")
    return X_train_sm, X_test, y_train_sm, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_preprocessing()
    print(f"Ready for training: X_train={X_train.shape}, X_test={X_test.shape}")
