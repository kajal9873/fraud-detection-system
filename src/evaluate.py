"""
evaluate.py — Model evaluation: metrics, confusion matrix, ROC/PR curves
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from config import REPORT_DIR, FRAUD_THRESHOLD


def print_metrics(y_true, y_pred, y_prob, model_name: str = "Model") -> dict:
    """
    Print and return key evaluation metrics.

    Args:
        y_true      : True labels
        y_pred      : Binary predictions
        y_prob      : Fraud probabilities
        model_name  : Label for display

    Returns:
        dict of metric name → value
    """
    metrics = {
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "auprc":     round(average_precision_score(y_true, y_prob), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"  ROC-AUC : {metrics['roc_auc']}")
    print(f"  AUPRC   : {metrics['auprc']}")
    print(f"{'='*50}")

    return metrics


def compare_models(
    y_true,
    predictions: dict,          # {"Model Name": (y_pred, y_prob)}
) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        y_true      : True labels
        predictions : Dict of model_name → (y_pred, y_prob)

    Returns:
        DataFrame with one row per model
    """
    rows = []
    for name, (y_pred, y_prob) in predictions.items():
        rows.append({
            "Model":     name,
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall":    round(recall_score(y_true, y_pred), 4),
            "F1-Score":  round(f1_score(y_true, y_pred), 4),
            "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
            "AUPRC":     round(average_precision_score(y_true, y_prob), 4),
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    return df


def plot_confusion_matrices(
    y_true,
    predictions: dict,          # {"Model Name": y_pred}
    save: bool = True,
    filename: str = "confusion_matrices.png",
) -> None:
    """
    Plot side-by-side confusion matrices for multiple models.

    Args:
        y_true      : True labels
        predictions : Dict of model_name → y_pred
        save        : Whether to save to reports/
        filename    : Output filename
    """
    plt.style.use("dark_background")
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, ax=ax,
            annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["Predicted: Legit", "Predicted: Fraud"],
            yticklabels=["Actual: Legit",    "Actual: Fraud"],
        )
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)
        ax.set_title(f"{name}\nP={p:.3f} | R={r:.3f} | F1={f:.3f}")

    plt.suptitle("Confusion Matrices — Fraud = Positive Class", fontsize=13, y=1.02)
    plt.tight_layout()

    if save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✅ Confusion matrix saved → {path}")

    plt.show()
    plt.close()


def plot_roc_pr_curves(
    y_true,
    probabilities: dict,        # {"Model Name": y_prob}
    save: bool = True,
    filename: str = "roc_pr_curves.png",
) -> None:
    """
    Plot ROC and Precision-Recall curves for multiple models.

    Args:
        y_true        : True labels
        probabilities : Dict of model_name → y_prob
        save          : Whether to save to reports/
        filename      : Output filename
    """
    plt.style.use("dark_background")
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for (name, y_prob), color in zip(probabilities.items(), colors):
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)

    # ROC — random baseline
    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # PR
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve\n(Better metric for imbalanced data)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    plt.tight_layout()

    if save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✅ ROC/PR curves saved → {path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    from preprocess import run_preprocessing
    from model import train_logistic_regression, train_xgboost, predict

    X_train, X_test, y_train, y_test = run_preprocessing()

    lr  = train_logistic_regression(X_train, y_train)
    xgb = train_xgboost(X_train, y_train, X_val=X_test, y_val=y_test)

    y_pred_lr,  y_prob_lr  = predict(lr,  X_test)
    y_pred_xgb, y_prob_xgb = predict(xgb, X_test)

    compare_models(y_test, {
        "Logistic Regression": (y_pred_lr,  y_prob_lr),
        "XGBoost":             (y_pred_xgb, y_prob_xgb),
    })

    plot_confusion_matrices(y_test, {
        "Logistic Regression": y_pred_lr,
        "XGBoost":             y_pred_xgb,
    })

    plot_roc_pr_curves(y_test, {
        "Logistic Regression": y_prob_lr,
        "XGBoost":             y_prob_xgb,
    })
