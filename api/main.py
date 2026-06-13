import os
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    TransactionInput,
    PredictionResponse,
    SHAPFeature,
    HealthResponse,
)

# ─── Model loading ─────────────────────────────────────────────────────────────

MODEL: object = None
SCALER: object = None
EXPLAINER: object = None

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]

# Time normalization constants (from training data statistics)
TIME_MEAN = 94813.0
TIME_STD  = 47488.0


def _resolve_paths():
    """Auto-detect model paths across environments."""
    candidates = [
        ("/content/xgb_model.pkl",        "/content/scaler.pkl"),
        ("/content/models/xgb_model.pkl", "/content/models/scaler.pkl"),
        ("../models/xgb_model.pkl",       "../models/scaler.pkl"),
        ("models/xgb_model.pkl",          "models/scaler.pkl"),
    ]
    for model_path, scaler_path in candidates:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            return model_path, scaler_path
    raise FileNotFoundError(
        "Model files not found. Run 02_model_training.ipynb first to generate "
        "xgb_model.pkl and scaler.pkl."
    )


def _load_shap_explainer(model):
    """Try to load SHAP explainer (optional dependency)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        print("✅ SHAP explainer loaded")
        return explainer
    except ImportError:
        print("⚠️  SHAP not installed — feature importance will use XGBoost built-in scores")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global MODEL, SCALER, EXPLAINER
    print("🚀 Loading model...")
    model_path, scaler_path = _resolve_paths()
    MODEL  = joblib.load(model_path)
    SCALER = joblib.load(scaler_path)
    EXPLAINER = _load_shap_explainer(MODEL)
    print(f"✅ Model loaded from {model_path}")
    yield
    print("🛑 Shutting down")


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time credit card fraud detection using XGBoost + SHAP.\n\n"
        "**Dataset:** ULB Credit Card Fraud (284,807 transactions, 0.172% fraud)\n\n"
        "**Model:** XGBoost trained with SMOTE oversampling\n\n"
        "**Features:** V1–V28 (PCA-transformed) + Amount + Time"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper functions ─────────────────────────────────────────────────────────

def preprocess(tx: TransactionInput) -> pd.DataFrame:
    """Scale Amount & Time, build feature DataFrame."""
    amount_scaled = SCALER.transform([[tx.Amount]])[0][0]
    time_scaled   = (tx.Time - TIME_MEAN) / TIME_STD

    v_vals = [getattr(tx, f"V{i}") for i in range(1, 29)]
    row    = v_vals + [amount_scaled, time_scaled]
    return pd.DataFrame([row], columns=FEATURE_COLS)


def get_shap_features(input_df: pd.DataFrame, n: int = 5):
    """Return top N SHAP feature contributions."""
    if EXPLAINER is not None:
        sv = EXPLAINER.shap_values(input_df)
        shap_vals = sv[0]
    else:
        # Fallback: use XGBoost feature importances as proxy
        shap_vals = MODEL.feature_importances_

    shap_df = (
        pd.DataFrame({
            "feature":    FEATURE_COLS,
            "shap_value": shap_vals,
            "abs_shap":   np.abs(shap_vals),
        })
        .sort_values("abs_shap", ascending=False)
        .head(n)
    )

    return [
        SHAPFeature(
            feature=row["feature"],
            shap_value=round(float(row["shap_value"]), 4),
            direction="towards fraud" if row["shap_value"] > 0 else "towards legitimate",
        )
        for _, row in shap_df.iterrows()
    ]


def risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.3:
        return "MEDIUM"
    return "LOW"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "🛡️ Fraud Detection API is running",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Check if the API and model are ready."""
    return HealthResponse(
        status="ok" if MODEL is not None else "model_not_loaded",
        model_loaded=MODEL is not None,
        model_type=type(MODEL).__name__ if MODEL else "None",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: TransactionInput):
    """
    Analyze a transaction and return fraud probability + SHAP explanation.

    - **is_fraud**: True if fraud_probability > 0.5
    - **fraud_probability**: Model's confidence that this is fraud (0–1)
    - **risk_level**: LOW / MEDIUM / HIGH
    - **top_shap_features**: Which features drove this prediction and how
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    try:
        input_df = preprocess(transaction)
        prob     = float(MODEL.predict_proba(input_df)[0][1])
        is_fraud = prob > 0.5
        shap_features = get_shap_features(input_df)

        return PredictionResponse(
            is_fraud=is_fraud,
            fraud_probability=round(prob, 4),
            risk_level=risk_level(prob),
            confidence=round(max(prob, 1 - prob), 4),
            top_shap_features=shap_features,
            amount=transaction.Amount,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[TransactionInput]):
    """
    Analyze multiple transactions at once (max 100).
    Returns list of predictions in the same order.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Max 100 transactions per batch.")

    results = []
    for tx in transactions:
        try:
            input_df = preprocess(tx)
            prob     = float(MODEL.predict_proba(input_df)[0][1])
            results.append({
                "is_fraud":          prob > 0.5,
                "fraud_probability": round(prob, 4),
                "risk_level":        risk_level(prob),
                "amount":            tx.Amount,
            })
        except Exception as e:
            results.append({"error": str(e)})

    return {"predictions": results, "total": len(results)}
