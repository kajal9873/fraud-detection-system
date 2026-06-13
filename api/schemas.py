from pydantic import BaseModel, Field
from typing import List, Optional


class TransactionInput(BaseModel):
    """Input schema for a single transaction prediction."""

    V1: float = Field(default=0.0, description="PCA feature V1")
    V2: float = Field(default=0.0, description="PCA feature V2")
    V3: float = Field(default=0.0, description="PCA feature V3")
    V4: float = Field(default=0.0, description="PCA feature V4")
    V5: float = Field(default=0.0, description="PCA feature V5")
    V6: float = Field(default=0.0, description="PCA feature V6")
    V7: float = Field(default=0.0, description="PCA feature V7")
    V8: float = Field(default=0.0, description="PCA feature V8")
    V9: float = Field(default=0.0, description="PCA feature V9")
    V10: float = Field(default=0.0, description="PCA feature V10")
    V11: float = Field(default=0.0, description="PCA feature V11")
    V12: float = Field(default=0.0, description="PCA feature V12")
    V13: float = Field(default=0.0, description="PCA feature V13")
    V14: float = Field(default=0.0, description="PCA feature V14")
    V15: float = Field(default=0.0, description="PCA feature V15")
    V16: float = Field(default=0.0, description="PCA feature V16")
    V17: float = Field(default=0.0, description="PCA feature V17")
    V18: float = Field(default=0.0, description="PCA feature V18")
    V19: float = Field(default=0.0, description="PCA feature V19")
    V20: float = Field(default=0.0, description="PCA feature V20")
    V21: float = Field(default=0.0, description="PCA feature V21")
    V22: float = Field(default=0.0, description="PCA feature V22")
    V23: float = Field(default=0.0, description="PCA feature V23")
    V24: float = Field(default=0.0, description="PCA feature V24")
    V25: float = Field(default=0.0, description="PCA feature V25")
    V26: float = Field(default=0.0, description="PCA feature V26")
    V27: float = Field(default=0.0, description="PCA feature V27")
    V28: float = Field(default=0.0, description="PCA feature V28")
    Amount: float = Field(default=100.0, ge=0.0, description="Transaction amount in USD")
    Time: float = Field(default=50000.0, ge=0.0, description="Seconds elapsed since first transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.35,  "V2": -0.07, "V3": 2.53,  "V4": 1.37,
                "V5": -0.33,  "V6": 0.46,  "V7": 0.23,  "V8": 0.09,
                "V9": 0.36,   "V10": 0.09, "V11": -0.55, "V12": -0.61,
                "V13": -0.99, "V14": -0.31, "V15": 1.46, "V16": -0.47,
                "V17": 0.20,  "V18": 0.02, "V19": 0.40,  "V20": 0.25,
                "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
                "V25": 0.12,  "V26": -0.19,"V27": 0.13,  "V28": -0.02,
                "Amount": 149.62,
                "Time": 0.0
            }
        }


class SHAPFeature(BaseModel):
    """A single SHAP feature contribution."""
    feature: str
    shap_value: float
    direction: str   # 'towards fraud' or 'towards legitimate'


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""
    is_fraud: bool
    fraud_probability: float = Field(description="Probability of fraud (0.0 - 1.0)")
    risk_level: str = Field(description="LOW / MEDIUM / HIGH")
    confidence: float = Field(description="Model confidence = max(prob, 1-prob)")
    top_shap_features: List[SHAPFeature] = Field(description="Top 5 features driving this prediction")
    amount: float
    model_version: str = "xgboost-v1"


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: str
    model_loaded: bool
    model_type: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
