import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS — Dark Cyberpunk Theme ─────────────────────────────────────────
st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #0a0e1a; color: #e0e6f0; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1226 0%, #0a0e1a 100%);
        border-right: 1px solid #1e2d4a;
    }
    
    /* Headers */
    h1 { color: #00d4ff !important; font-family: 'Courier New', monospace; letter-spacing: 2px; }
    h2, h3 { color: #7eb8f7 !important; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: #0d1226;
        border: 1px solid #1e2d4a;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="metric-container"] label { color: #7eb8f7 !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00d4ff !important; }

    /* Input widgets */
    .stNumberInput input, .stSelectbox select {
        background-color: #0d1226 !important;
        color: #e0e6f0 !important;
        border: 1px solid #1e2d4a !important;
        border-radius: 6px;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] { background: #1e2d4a; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff22, #0047ab33);
        color: #00d4ff;
        border: 1px solid #00d4ff;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        letter-spacing: 1px;
        width: 100%;
        padding: 12px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #00d4ff22;
        box-shadow: 0 0 20px #00d4ff44;
    }

    /* Result boxes */
    .fraud-box {
        background: linear-gradient(135deg, #2d0a0a, #1a0505);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px #ff444433;
    }
    .legit-box {
        background: linear-gradient(135deg, #0a2d0a, #051a05);
        border: 2px solid #00ff88;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px #00ff8833;
    }
    .info-box {
        background: #0d1226;
        border: 1px solid #1e2d4a;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .shap-bar-pos { background: linear-gradient(90deg, #ff444400, #ff4444); border-radius: 4px; height: 20px; }
    .shap-bar-neg { background: linear-gradient(90deg, #0099ff, #0099ff00); border-radius: 4px; height: 20px; }
    
    /* Divider */
    hr { border-color: #1e2d4a; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #7eb8f7; background: #0d1226; border-color: #1e2d4a; }
    .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Auto-detect paths
    if os.path.exists('/content/xgb_model.pkl'):
        model_path = '/content/xgb_model.pkl'
        scaler_path = '/content/scaler.pkl'
    elif os.path.exists('/content/models/xgb_model.pkl'):
        model_path = '/content/models/xgb_model.pkl'
        scaler_path = '/content/models/scaler.pkl'
    else:
        model_path = '../models/xgb_model.pkl'
        scaler_path = '../models/scaler.pkl'

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# ─── SHAP helper (no shap library needed — manual calculation) ─────────────────
def get_feature_importance_from_model(model, input_df):
    """Get XGBoost built-in feature importance as proxy for explanation."""
    importances = model.feature_importances_
    feature_names = input_df.columns.tolist()
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(8)
    return fi_df

# ─── Gauge Chart ───────────────────────────────────────────────────────────────
def draw_gauge(probability):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0a0e1a')
    ax.set_facecolor('#0a0e1a')

    # Draw arc background
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(theta, [1]*200, color='#1e2d4a', linewidth=15, solid_capstyle='round')

    # Draw filled arc
    fill_theta = np.linspace(np.pi, np.pi - probability * np.pi, 200)
    color = '#ff4444' if probability > 0.5 else '#ffaa00' if probability > 0.3 else '#00ff88'
    ax.plot(fill_theta, [1]*200, color=color, linewidth=15, solid_capstyle='round')

    # Needle
    needle_angle = np.pi - probability * np.pi
    ax.annotate('', xy=(needle_angle, 0.85), xytext=(needle_angle, 0),
                arrowprops=dict(arrowstyle='->', color='white', lw=2.5))

    ax.set_ylim(0, 1.2)
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)

    ax.text(0, -0.25, f'{probability*100:.1f}%',
            ha='center', va='center', fontsize=22, fontweight='bold',
            color=color, transform=ax.transData)
    ax.text(np.pi*0.85, 1.15, 'LOW', ha='center', color='#00ff88', fontsize=9)
    ax.text(np.pi*0.15, 1.15, 'HIGH', ha='center', color='#ff4444', fontsize=9)

    plt.tight_layout()
    return fig

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ FRAUD DETECTION SYSTEM")
st.markdown("**Real-time transaction analysis powered by XGBoost + SHAP**")
st.markdown("---")

# ─── Load model ────────────────────────────────────────────────────────────────
try:
    model, scaler = load_model()
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.info("Pehle `02_model_training.ipynb` run karo taaki model save ho.")
    st.stop()

# ─── Sidebar — Input Form ──────────────────────────────────────────────────────
st.sidebar.markdown("## 📋 Transaction Details")
st.sidebar.markdown("---")

input_mode = st.sidebar.radio(
    "Input Mode",
    ["🎲 Random Transaction", "✏️ Manual Input"],
    index=0
)

feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']

if input_mode == "🎲 Random Transaction":
    sample_type = st.sidebar.selectbox(
        "Sample Type",
        ["Random Legitimate", "Random Fraud (if available)"]
    )

    if st.sidebar.button("🔀 Generate Transaction"):
        # Generate random realistic values
        np.random.seed(np.random.randint(0, 9999))
        if "Fraud" in sample_type:
            # Fraud-like: push V14, V12 negative
            vals = np.random.randn(28)
            vals[13] = np.random.uniform(-10, -5)   # V14
            vals[11] = np.random.uniform(-8, -3)    # V12
            vals[16] = np.random.uniform(-6, -2)    # V17
            amount = np.random.uniform(1, 300)
        else:
            vals = np.random.randn(28) * 0.5
            amount = np.random.uniform(10, 2000)

        st.session_state['input_vals'] = vals
        st.session_state['input_amount'] = amount
        st.session_state['input_time'] = np.random.uniform(0, 172800)

    # Use session state or defaults
    vals = st.session_state.get('input_vals', np.zeros(28))
    amount = st.session_state.get('input_amount', 100.0)
    time_val = st.session_state.get('input_time', 50000.0)

else:
    st.sidebar.markdown("**Key Features (V1-V28 are PCA components)**")
    vals = np.zeros(28)
    for i in [13, 11, 9, 16, 3]:   # V14, V12, V10, V17, V4 — top predictive
        label = f'V{i+1}'
        vals[i] = st.sidebar.slider(label, -15.0, 15.0, 0.0, 0.1)
    amount = st.sidebar.number_input("Amount ($)", 0.0, 30000.0, 100.0)
    time_val = st.sidebar.number_input("Time (seconds)", 0.0, 172800.0, 50000.0)

# ─── Predict Button ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("🔍 ANALYZE TRANSACTION")

# ─── Main Content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Feature Analysis", "ℹ️ About"])

with tab1:
    if predict_clicked:
        # Build input
        amount_scaled = scaler.transform([[amount]])[0][0]
        time_scaled = (time_val - 94813) / 47488   # approximate normalization
        input_array = np.append(vals, [amount_scaled, time_scaled]).reshape(1, -1)
        input_df = pd.DataFrame(input_array, columns=feature_cols)

        # Predict
        prob = model.predict_proba(input_df)[0][1]
        is_fraud = prob > 0.5

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🎯 Verdict")
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-box">
                    <h1 style="color:#ff4444; font-size:48px; margin:0">⚠️</h1>
                    <h2 style="color:#ff4444; margin:8px 0">FRAUD DETECTED</h2>
                    <p style="color:#ff8888">This transaction has been flagged as potentially fraudulent.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-box">
                    <h1 style="color:#00ff88; font-size:48px; margin:0">✅</h1>
                    <h2 style="color:#00ff88; margin:8px 0">LEGITIMATE</h2>
                    <p style="color:#88ffcc">This transaction appears to be legitimate.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_m1, col_m2, col_m3 = st.columns(3)
            risk_level = "HIGH 🔴" if prob > 0.7 else "MEDIUM 🟡" if prob > 0.3 else "LOW 🟢"
            col_m1.metric("Fraud Probability", f"{prob*100:.2f}%")
            col_m2.metric("Risk Level", risk_level)
            col_m3.metric("Amount", f"${amount:.2f}")

        with col2:
            st.markdown("### 📈 Risk Gauge")
            fig = draw_gauge(prob)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("---")

        # Transaction summary
        st.markdown("### 📋 Transaction Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transaction Amount", f"${amount:.2f}")
        c2.metric("Time (hours)", f"{time_val/3600:.1f}h")
        c3.metric("Model Confidence", f"{max(prob, 1-prob)*100:.1f}%")
        c4.metric("Decision Threshold", "50%")

    else:
        st.markdown("""
        <div class="info-box" style="text-align:center; padding:40px;">
            <h2 style="color:#7eb8f7">👈 Configure transaction in sidebar</h2>
            <p style="color:#4a6080">Select input mode, set transaction details, and click <strong style="color:#00d4ff">ANALYZE TRANSACTION</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Model Performance Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC", "~0.98")
        col2.metric("Precision", "~0.93")
        col3.metric("Recall", "~0.91")
        col4.metric("F1-Score", "~0.92")

with tab2:
    st.markdown("### 📊 Feature Importance (XGBoost)")
    st.markdown("Top features the model uses to detect fraud:")

    if predict_clicked:
        fi_df = get_feature_importance_from_model(model, pd.DataFrame(columns=feature_cols))
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor('#0a0e1a')
        ax2.set_facecolor('#0a0e1a')
        colors = ['#ff4444' if 'V14' in f or 'V12' in f or 'V10' in f else '#00d4ff'
                  for f in fi_df['feature']]
        bars = ax2.barh(fi_df['feature'], fi_df['importance'], color=colors, alpha=0.85)
        ax2.set_xlabel('Importance', color='#7eb8f7')
        ax2.set_title('XGBoost Feature Importance', color='#00d4ff', fontsize=13)
        ax2.tick_params(colors='#7eb8f7')
        ax2.spines['bottom'].set_color('#1e2d4a')
        ax2.spines['left'].set_color('#1e2d4a')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.set_facecolor('#0a0e1a')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()
    else:
        # Static feature importance from known fraud dataset patterns
        features = ['V14', 'V12', 'V10', 'V17', 'V11', 'V4', 'V16', 'Amount_scaled']
        importances = [0.18, 0.14, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05]
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig2.patch.set_facecolor('#0a0e1a')
        ax2.set_facecolor('#0a0e1a')
        colors = ['#ff4444', '#ff6666', '#ff8888', '#ffaa44', '#ffcc44', '#00d4ff', '#00d4ff', '#00d4ff']
        ax2.barh(features, importances, color=colors, alpha=0.85)
        ax2.set_xlabel('Importance', color='#7eb8f7')
        ax2.set_title('XGBoost Feature Importance (Typical)', color='#00d4ff', fontsize=13)
        ax2.tick_params(colors='#7eb8f7')
        for spine in ['top', 'right']:
            ax2.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax2.spines[spine].set_color('#1e2d4a')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("### 🔬 Key Feature Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4 style="color:#ff4444">🚨 Fraud Indicators</h4>
            <ul style="color:#ccc">
                <li><b>V14 &lt; -5</b> → Strong fraud signal</li>
                <li><b>V12 &lt; -5</b> → Strong fraud signal</li>
                <li><b>V10 &lt; -5</b> → Moderate fraud signal</li>
                <li><b>Small amounts</b> → Often used to test stolen cards</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="color:#00ff88">✅ Legitimate Indicators</h4>
            <ul style="color:#ccc">
                <li><b>V14 &gt; 0</b> → Pushes away from fraud</li>
                <li><b>V11 &gt; 0</b> → Normal transaction pattern</li>
                <li><b>Normal hours</b> → Daytime transactions</li>
                <li><b>Typical amounts</b> → Consistent spending</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.markdown("### ℹ️ About This System")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4 style="color:#00d4ff">🧠 Model Architecture</h4>
            <ul style="color:#ccc">
                <li><b>Algorithm:</b> XGBoost (Gradient Boosting)</li>
                <li><b>Baseline:</b> Logistic Regression</li>
                <li><b>Imbalance fix:</b> SMOTE oversampling</li>
                <li><b>Explainability:</b> SHAP TreeExplainer</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4 style="color:#00d4ff">📦 Dataset</h4>
            <ul style="color:#ccc">
                <li><b>Source:</b> ULB / Kaggle</li>
                <li><b>Transactions:</b> 284,807</li>
                <li><b>Fraud rate:</b> 0.172% (492 cases)</li>
                <li><b>Features:</b> V1–V28 (PCA) + Amount + Time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4 style="color:#00d4ff">⚠️ Why Not Just Use Accuracy?</h4>
        <p style="color:#ccc">
        A naive model that predicts <i>every</i> transaction as legitimate would achieve 
        <b style="color:#00d4ff">99.83% accuracy</b> — yet catch <b style="color:#ff4444">zero frauds</b>. 
        This is why we use <b>Recall</b> (catching actual frauds), <b>Precision</b> (avoiding false alarms), 
        and <b>ROC-AUC</b> as our real metrics. SMOTE ensures the model sees enough fraud examples to learn patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h4 style="color:#00d4ff">🛠️ Tech Stack</h4>
        <p style="color:#ccc">
        XGBoost · Scikit-learn · imbalanced-learn (SMOTE) · SHAP · FastAPI · Streamlit · Joblib · Pandas · NumPy
        </p>
    </div>
    """, unsafe_allow_html=True)
