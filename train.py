"""
train.py — Standalone training script for Fraud Detection System
Run this to train the model and save artifacts to models/ folder.

Usage:
    python train.py
    python train.py --data data/creditcard.csv --output models/
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_DATA_PATH = "data/creditcard.csv"
DEFAULT_MODEL_DIR = "models"
DEFAULT_REPORT_DIR = "reports"

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}


# ─── CLI args ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Fraud Detection Model")
    parser.add_argument("--data",   default=DEFAULT_DATA_PATH, help="Path to creditcard.csv")
    parser.add_argument("--output", default=DEFAULT_MODEL_DIR, help="Directory to save model artifacts")
    parser.add_argument("--report", default=DEFAULT_REPORT_DIR, help="Directory to save plots")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    return parser.parse_args()


# ─── Steps ────────────────────────────────────────────────────────────────────

def load_data(data_path: str) -> pd.DataFrame:
    print(f"\n[1/6] Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "Or: https://zenodo.org/records/7395559"
        )
    df = pd.read_csv(data_path)
    print(f"      Shape : {df.shape}")
    print(f"      Fraud : {df['Class'].sum():,} ({df['Class'].mean()*100:.4f}%)")
    print(f"      NaNs  : {df.isnull().sum().sum()}")
    return df


def preprocess(df: pd.DataFrame, model_dir: str):
    print("\n[2/6] Preprocessing...")

    # Separate scalers for Amount and Time
    scaler_amount = StandardScaler()
    scaler_time   = StandardScaler()

    df = df.copy()
    df["Amount_scaled"] = scaler_amount.fit_transform(df[["Amount"]])
    df["Time_scaled"]   = scaler_time.fit_transform(df[["Time"]])

    # Save scaler (Amount scaler — needed at inference time)
    joblib.dump(scaler_amount, os.path.join(model_dir, "scaler.pkl"))
    print(f"      Scaler saved → {model_dir}/scaler.pkl")

    X = df[FEATURE_COLS]
    y = df["Class"]
    return X, y


def split_and_smote(X, y, test_size: float):
    print("\n[3/6] Train/test split + SMOTE...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"      Train : {X_train.shape} | Fraud: {y_train.sum()}")
    print(f"      Test  : {X_test.shape}  | Fraud: {y_test.sum()}")

    # SMOTE only on training set — no data leakage
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"      After SMOTE — Train: {X_train_sm.shape} | Fraud: {y_train_sm.sum():,}")

    return X_train_sm, X_test, y_train_sm, y_test


def train_models(X_train, y_train, X_test, y_test):
    print("\n[4/6] Training models...")

    # Baseline — Logistic Regression
    print("      Training Logistic Regression (baseline)...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=0.01)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    print(f"      LR  → ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f} | "
          f"F1: {f1_score(y_test, y_pred_lr):.4f} | "
          f"Recall: {recall_score(y_test, y_pred_lr):.4f}")

    # Main model — XGBoost
    print("      Training XGBoost (main model)...")
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    print(f"      XGB → ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f} | "
          f"F1: {f1_score(y_test, y_pred_xgb):.4f} | "
          f"Recall: {recall_score(y_test, y_pred_xgb):.4f}")

    return xgb, lr, y_pred_xgb, y_prob_xgb, y_pred_lr, y_prob_lr


def evaluate(xgb, lr, X_test, y_test, y_pred_xgb, y_prob_xgb, y_pred_lr, y_prob_lr, report_dir):
    print("\n[5/6] Evaluation & saving plots...")

    results = {
        "Model":     ["Logistic Regression", "XGBoost"],
        "Precision": [round(precision_score(y_test, y_pred_lr), 4),
                      round(precision_score(y_test, y_pred_xgb), 4)],
        "Recall":    [round(recall_score(y_test, y_pred_lr), 4),
                      round(recall_score(y_test, y_pred_xgb), 4)],
        "F1-Score":  [round(f1_score(y_test, y_pred_lr), 4),
                      round(f1_score(y_test, y_pred_xgb), 4)],
        "ROC-AUC":   [round(roc_auc_score(y_test, y_prob_lr), 4),
                      round(roc_auc_score(y_test, y_prob_xgb), 4)],
        "AUPRC":     [round(average_precision_score(y_test, y_prob_lr), 4),
                      round(average_precision_score(y_test, y_prob_xgb), 4)],
    }
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)

    # Save confusion matrix plot
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    import seaborn as sns
    for ax, y_pred, title in zip(
        axes,
        [y_pred_lr, y_pred_xgb],
        ["Logistic Regression", "XGBoost"]
    ):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, ax=ax, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
        )
        ax.set_title(
            f"{title}\nP={precision_score(y_test,y_pred):.3f} "
            f"R={recall_score(y_test,y_pred):.3f} "
            f"F1={f1_score(y_test,y_pred):.3f}"
        )
    plt.suptitle("Confusion Matrices", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(report_dir, "confusion_matrices.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Plot saved → {plot_path}")

    return results_df


def save_model(xgb, model_dir: str):
    print("\n[6/6] Saving model...")
    model_path = os.path.join(model_dir, "xgb_model.pkl")
    joblib.dump(xgb, model_path)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"      Model saved → {model_path} ({size_mb:.1f} MB)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Create output dirs
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)

    print("=" * 60)
    print("  FRAUD DETECTION — MODEL TRAINING")
    print("=" * 60)
    print(f"  Data   : {args.data}")
    print(f"  Output : {args.output}")
    print(f"  Reports: {args.report}")

    # Pipeline
    df = load_data(args.data)
    X, y = preprocess(df, args.output)
    X_train, X_test, y_train, y_test = split_and_smote(X, y, args.test_size)
    xgb, lr, y_pred_xgb, y_prob_xgb, y_pred_lr, y_prob_lr = train_models(
        X_train, y_train, X_test, y_test
    )
    evaluate(xgb, lr, X_test, y_test, y_pred_xgb, y_prob_xgb, y_pred_lr, y_prob_lr, args.report)
    save_model(xgb, args.output)

    print("\n✅ Training complete!")
    print(f"   → Run API  : cd api && uvicorn main:app --reload")
    print(f"   → Run App  : cd app && streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
