"""
explainer.py — SHAP-based model explainability
Wraps shap.TreeExplainer for XGBoost with helper functions
used by both the Streamlit app and FastAPI backend.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import REPORT_DIR, FEATURE_COLS, SHAP_SAMPLE_SIZE, SHAP_TOP_N


def get_explainer(model):
    """
    Create a SHAP TreeExplainer for a tree-based model.

    Args:
        model : Trained XGBoost / LightGBM / RandomForest model

    Returns:
        shap.TreeExplainer instance
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP not installed. Run: pip install shap"
        )

    explainer = shap.TreeExplainer(model)
    print(f"✅ SHAP explainer ready | Base value: {explainer.expected_value:.4f}")
    return explainer


def compute_shap_values(explainer, X_sample: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP values for a sample of data.

    Args:
        explainer : shap.TreeExplainer
        X_sample  : DataFrame of shape (n_samples, n_features)

    Returns:
        numpy array of SHAP values — shape (n_samples, n_features)
    """
    print(f"Computing SHAP values for {len(X_sample)} samples...")
    shap_values = explainer.shap_values(X_sample)
    print(f"✅ SHAP values computed — shape: {shap_values.shape}")
    return shap_values


def get_top_features(
    explainer,
    input_df: pd.DataFrame,
    top_n: int = SHAP_TOP_N,
    feature_names: list = None,
) -> pd.DataFrame:
    """
    Get top N features driving a single prediction.
    Used by FastAPI /predict endpoint.

    Args:
        explainer     : shap.TreeExplainer
        input_df      : Single-row DataFrame (1, n_features)
        top_n         : Number of top features to return
        feature_names : Column names (defaults to FEATURE_COLS)

    Returns:
        DataFrame with columns: feature, shap_value, abs_shap, direction
    """
    features = feature_names or FEATURE_COLS
    sv = explainer.shap_values(input_df)
    shap_vals = sv[0] if sv.ndim == 2 else sv

    result = (
        pd.DataFrame({
            "feature":   features,
            "shap_value": shap_vals,
            "abs_shap":   np.abs(shap_vals),
        })
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result["direction"] = result["shap_value"].apply(
        lambda v: "towards fraud" if v > 0 else "towards legitimate"
    )
    return result


def plot_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    plot_type: str = "bar",
    save: bool = True,
    filename: str = None,
) -> None:
    """
    Plot global SHAP summary (bar or beeswarm).

    Args:
        shap_values : SHAP values array (n_samples, n_features)
        X_sample    : Feature DataFrame used to compute SHAP values
        plot_type   : 'bar' (mean importance) or 'dot' (beeswarm)
        save        : Whether to save plot to reports/
        filename    : Output filename (auto-named if None)
    """
    import shap

    fname = filename or f"shap_summary_{plot_type}.png"

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type=plot_type,
        show=False,
        **({"color": "#FF9800"} if plot_type == "bar" else {}),
    )
    title = ("SHAP Feature Importance — Mean |SHAP Value|"
             if plot_type == "bar"
             else "SHAP Beeswarm — Feature Impact Direction & Magnitude")
    plt.title(title, fontsize=13)
    plt.tight_layout()

    if save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✅ SHAP summary plot saved → {path}")

    plt.show()
    plt.close()


def plot_waterfall(
    explainer,
    X_sample: pd.DataFrame,
    idx: int = 0,
    max_display: int = 15,
    save: bool = True,
    filename: str = "shap_waterfall.png",
) -> None:
    """
    Plot SHAP waterfall chart for a single prediction.
    Shows how each feature pushed the prediction up or down.

    Args:
        explainer   : shap.TreeExplainer
        X_sample    : Feature DataFrame
        idx         : Row index to explain
        max_display : Max features to show
        save        : Whether to save plot
        filename    : Output filename
    """
    import shap

    sv = explainer.shap_values(X_sample)
    explanation = shap.Explanation(
        values=sv[idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[idx].values,
        feature_names=X_sample.columns.tolist(),
    )

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(explanation, show=False, max_display=max_display)
    plt.tight_layout()

    if save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✅ Waterfall plot saved → {path}")

    plt.show()
    plt.close()


def plot_dependence(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    top_n: int = 4,
    save: bool = True,
    filename: str = "shap_dependence.png",
) -> None:
    """
    Plot SHAP dependence plots for top N features.

    Args:
        shap_values : SHAP values array
        X_sample    : Feature DataFrame
        top_n       : Number of top features to plot
        save        : Whether to save plot
        filename    : Output filename
    """
    import shap

    top_features = (
        pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns)
        .nlargest(top_n)
        .index.tolist()
    )

    cols = 2
    rows = (top_n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        shap.dependence_plot(feat, shap_values, X_sample, ax=axes[i], show=False)
        axes[i].set_title(f"SHAP Dependence: {feat}")

    # Hide unused subplots
    for j in range(len(top_features), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("How Top Features Influence Fraud Predictions", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        path = os.path.join(REPORT_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✅ Dependence plots saved → {path}")

    plt.show()
    plt.close()


if __name__ == "__main__":
    from preprocess import run_preprocessing
    from model import train_xgboost, load_model, predict
    import random

    X_train, X_test, y_train, y_test = run_preprocessing()
    xgb = load_model()

    # Sample for SHAP
    X_sample = X_test.sample(n=min(SHAP_SAMPLE_SIZE, len(X_test)), random_state=42)

    explainer   = get_explainer(xgb)
    shap_values = compute_shap_values(explainer, X_sample)

    # Global plots
    plot_summary(shap_values, X_sample, plot_type="bar")
    plot_summary(shap_values, X_sample, plot_type="dot")
    plot_dependence(shap_values, X_sample)

    # Local explanation for first sample
    plot_waterfall(explainer, X_sample, idx=0)

    # Top features for one prediction
    top = get_top_features(explainer, X_sample.iloc[[0]])
    print("\nTop SHAP features for transaction[0]:")
    print(top.to_string(index=False))
