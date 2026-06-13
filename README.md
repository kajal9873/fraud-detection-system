# 🛡️ Credit Card Fraud Detection System

> Real-time fraud detection using **XGBoost + SMOTE + SHAP** on 284,807 transactions.  
> Full-stack ML project with FastAPI backend and Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)

---

## 📌 Problem Statement

Credit card fraud detection is a classic **severely imbalanced classification problem** — only **0.172%** of transactions are fraudulent (492 out of 284,807). A naive model predicting everything as legitimate achieves 99.83% accuracy but catches **zero frauds**.

This project addresses the problem using:
- **SMOTE** to handle class imbalance without data leakage
- **XGBoost** for high-performance gradient boosting
- **SHAP** for model explainability (why was a transaction flagged?)
- **Precision-Recall AUC** as the primary evaluation metric (not accuracy)

---

## 📊 Results

| Model | Precision | Recall | F1-Score | ROC-AUC | AUPRC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | ~0.87 | ~0.62 | ~0.72 | ~0.97 | ~0.71 |
| **XGBoost + SMOTE** | **~0.93** | **~0.91** | **~0.92** | **~0.98** | **~0.86** |

> *Metrics on held-out 20% test set. SMOTE applied only on training data.*

---

## 🗂️ Project Structure

```
fraud-detection/
│
├── data/
│   └── creditcard.csv          ← Download from Kaggle (not in repo)
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory Data Analysis
│   ├── 02_model_training.ipynb ← SMOTE + XGBoost + evaluation
│   └── 03_shap_explainability.ipynb ← SHAP feature importance
│
├── src/
│   ├── preprocess.py           ← Scaling, SMOTE, train-test split
│   ├── model.py                ← Model training & saving
│   ├── explainer.py            ← SHAP TreeExplainer wrapper
│   ├── evaluate.py             ← Metrics, confusion matrix, ROC curve
│   └── config.py               ← Constants and paths
│
├── api/
│   ├── main.py                 ← FastAPI app (/predict, /health, /predict/batch)
│   ├── schemas.py              ← Pydantic input/output models
│   └── requirements.txt
│
├── app/
│   ├── streamlit_app.py        ← Streamlit dashboard (dark cyberpunk theme)
│   └── requirements.txt
│
├── models/                     ← Saved artifacts (gitignored)
│   ├── xgb_model.pkl
│   └── scaler.pkl
│
├── reports/                    ← Generated plots (gitignored)
│
├── train.py                    ← Standalone training script
├── requirements.txt            ← All dependencies
└── .gitignore
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download `creditcard.csv` from Kaggle and place it in `data/`:
- **Kaggle:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🚀 Usage

### Train the model
```bash
python train.py
# or with custom paths:
python train.py --data data/creditcard.csv --output models/ --report reports/
```

### Run the API
```bash
cd api
uvicorn main:app --reload --port 8000
```
API docs available at: `http://localhost:8000/docs`

### Run the Streamlit app
```bash
cd app
streamlit run streamlit_app.py
```

---

## 🔌 API Reference

### `POST /predict`
Analyze a single transaction.

**Request:**
```json
{
  "V1": -1.35, "V2": -0.07, "V3": 2.53,
  "V14": -0.31, "V17": 0.20,
  "Amount": 149.62,
  "Time": 0.0
}
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0312,
  "risk_level": "LOW",
  "confidence": 0.9688,
  "top_shap_features": [
    { "feature": "V14", "shap_value": -0.4821, "direction": "towards legitimate" },
    { "feature": "V12", "shap_value": -0.3104, "direction": "towards legitimate" }
  ],
  "amount": 149.62
}
```

### `GET /health`
Check model status.

### `POST /predict/batch`
Analyze up to 100 transactions at once.

---

## 🧠 Key Technical Decisions

### Why SMOTE instead of class_weight?
SMOTE generates synthetic minority samples in feature space, giving the model more diverse fraud examples to learn from. Applied **only on training data** to prevent data leakage.

### Why Precision-Recall AUC over ROC-AUC?
ROC-AUC can be misleadingly optimistic on imbalanced datasets. PR-AUC focuses only on the minority (fraud) class and is a stricter, more meaningful metric.

### Why SHAP?
XGBoost is a black-box model. SHAP (SHapley Additive exPlanations) uses game theory to attribute each feature's contribution to a specific prediction — critical for fintech applications where regulators may require explanations for blocked transactions.

---

## 📈 EDA Insights

- **Class imbalance:** 492 fraud vs 284,315 legitimate (0.172%)
- **Fraud amounts:** Fraudulent transactions tend to have lower amounts (mean ~$122 vs ~$88)
- **Time patterns:** Fraud spikes during nighttime low-activity hours
- **Top predictive features:** V14, V12, V10, V17, V11 (PCA components from original card data)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost |
| Baseline | Logistic Regression |
| Imbalance | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Backend | FastAPI + Pydantic |
| Frontend | Streamlit |
| Serialization | Joblib |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Dataset

**Credit Card Fraud Detection** — Anonymized credit card transactions from European cardholders (September 2013).

- 284,807 transactions over 2 days
- 28 PCA-transformed features (V1–V28) + Amount + Time
- Features are anonymized due to confidentiality

> Dataset: ULB Machine Learning Group — [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---
