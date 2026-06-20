import streamlit as st
import pandas as pd
import json
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import google.generativeai as genai
from google import genai
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CFPB Complaint Classifier",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Dark Theme CSS ───────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: #0f172a;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background: #1e293b;
    }

    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 60%, #3b82f6 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(37,99,235,0.3);
        text-align: center;
    }
    .hero h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .hero p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.05rem;
        max-width: 650px;
        margin: 0 auto;
    }

    .result-card {
        padding: 2rem 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.06);
        min-height: 220px;
    }
    .result-card h3 {
        margin: 0 0 1rem 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.7;
    }
    .result-card .prediction {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .result-card .confidence {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-top: 0.25rem;
    }
    .result-card .explanation {
        font-size: 0.88rem;
        opacity: 0.7;
        margin-top: 1rem;
        line-height: 1.5;
        text-align: left;
        padding: 0.75rem;
        border-radius: 10px;
        background: rgba(255,255,255,0.05);
    }
    .ml-card {
        background: linear-gradient(135deg, #1e3a5f, #1e293b);
        border-left: 4px solid #3b82f6;
    }
    .llm-card {
        background: linear-gradient(135deg, #2d1b4e, #1e293b);
        border-left: 4px solid #a78bfa;
    }

    div[data-testid="stMetric"] {
        background: #1e293b;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }

    .stTextArea textarea {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.25) !important;
    }

    #MainMenu, header[data-testid="stHeader"] {display: none !important;}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {display: none !important;}
    section[data-testid="stSidebar"] span {display: none !important;}
    section[data-testid="stSidebar"] p {display: none !important;}
    div[data-testid="stExpander"] summary {background: #1e293b; border: 1px solid #334155; border-radius: 12px; color: #e2e8f0; padding: 0.75rem 1rem;}
    div[data-testid="stExpander"] summary:hover {background: #334155;}
    div[data-testid="stExpander"] details[open] summary {border-radius: 12px 12px 0 0;}
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {gap: 1rem;}
</style>
""", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'outputs')

# ── Load Functions ───────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    model = joblib.load(os.path.join(OUTPUTS_DIR, 'logistic_model.joblib'))
    tfidf = joblib.load(os.path.join(OUTPUTS_DIR, 'tfidf_vectorizer.joblib'))
    return model, tfidf

@st.cache_data
def load_metrics():
    path = os.path.join(OUTPUTS_DIR, 'metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data
def load_confusion_data():
    path = os.path.join(OUTPUTS_DIR, 'ml_predictions.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def get_gemini_client():
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    api_key = os.environ.get('GEMINI_API_KEY', '')

    if api_key:
        return genai.Client(api_key=api_key)

    return None

def classify_with_gemini(text):
    """Returns dict with prediction, confidence, reasoning."""
    model = get_gemini_client()
    st.write("Model Loaded:", model)
    if not model:
        return {"prediction": "Unavailable", "confidence": "N/A", "reasoning": "No API key configured."}

    words = text.split()
    truncated = ' '.join(words[:300])
    prompt = (
        "You are a complaint classification system for a financial company.\n"
        "Classify the following customer complaint into EXACTLY ONE of these two categories:\n"
        "'Debt collection' or 'Credit card'.\n\n"
        "Respond in this exact JSON format ONLY, nothing else:\n"
        '{"prediction": "<category>", "confidence": "<high/medium/low>", "reasoning": "<one sentence why>"}'
    )
    for attempt in range(3):
        try:
            response = model.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt + "\n\n" + truncated
            )

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as err:
                st.error(f"Gemini Error: {err}")
                print("GEMINI ERROR:", err)
                return {
                    "prediction": "Unavailable",
                    "confidence": "N/A",
                    "reasoning": "Gemini service temporarily unavailable. Please try again later."
                }

def get_ml_prediction(text, model, tfidf):
    """Returns dict with prediction and confidence."""
    X = tfidf.transform([text])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = float(max(probs))
    return {"prediction": pred, "confidence": f"{confidence:.0%}"}

# ── Load Assets ──────────────────────────────────────────────
model, tfidf = load_ml_model()
metrics = load_metrics()

# ── Hero Section ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📋 CFPB Complaint Classifier</h1>
    <p>Compare predictions from a traditional ML model (TF-IDF + Logistic Regression) and Google Gemini for classifying consumer complaints into <strong>Debt Collection</strong> or <strong>Credit Card</strong> categories.</p>
</div>
""", unsafe_allow_html=True)

# ── Input Section ────────────────────────────────────────────
narrative_input = st.text_area(
    "Paste a consumer complaint narrative below:",
    height=180,
    placeholder="e.g., I received a call from a debt collector about a debt I do not owe. They refused to provide validation..."
)

classify_btn = st.button("🚀  Analyze Complaint", type="primary", use_container_width=True)

# ── Classification Results ───────────────────────────────────
if classify_btn and narrative_input.strip():
    with st.spinner("Analyzing complaint..."):
        ml_result = get_ml_prediction(narrative_input, model, tfidf)
        llm_result = classify_with_gemini(narrative_input)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="result-card ml-card">
            <h3>🤖 ML Model</h3>
            <p class="prediction">{ml_result['prediction']}</p>
            <p class="confidence">Confidence: {ml_result['confidence']}</p>
            <div class="explanation">
                Logistic Regression trained on 8,000 CFPB complaints using TF-IDF vectorization (5,000 features).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card llm-card">
            <h3>🧠 Gemini LLM</h3>
            <p class="prediction">{llm_result['prediction']}</p>
            <p class="confidence">Confidence: {llm_result['confidence']}</p>
            <div class="explanation">{llm_result['reasoning']}</div>
        </div>
        """, unsafe_allow_html=True)

elif classify_btn:
    st.warning("Please enter a complaint narrative first.")

# ── Advanced Analytics (Collapsible) ─────────────────────────

with st.expander("📊  Advanced Analytics", expanded=False):
    if metrics and metrics.get('llm', {}).get('accuracy', 0) > 0:
        st.markdown("### Model Performance Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🤖 ML Model")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['ml']['accuracy']:.2%}")
            m2.metric("Precision", f"{metrics['ml']['precision']:.2%}")
            m3.metric("Recall", f"{metrics['ml']['recall']:.2%}")
            m4.metric("F1", f"{metrics['ml']['f1']:.2%}")
        with col2:
            st.markdown("#### 🧠 LLM")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['llm']['accuracy']:.2%}")
            m2.metric("Precision", f"{metrics['llm']['precision']:.2%}")
            m3.metric("Recall", f"{metrics['llm']['recall']:.2%}")
            m4.metric("F1", f"{metrics['llm']['f1']:.2%}")
    elif metrics:
        st.markdown("### 🤖 ML Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['ml']['accuracy']:.2%}")
        m2.metric("Precision", f"{metrics['ml']['precision']:.2%}")
        m3.metric("Recall", f"{metrics['ml']['recall']:.2%}")
        m4.metric("F1", f"{metrics['ml']['f1']:.2%}")
        st.caption("LLM metrics will appear here once the full evaluation notebook is run.")

    st.markdown("### 🔢 Confusion Matrix")
    preds_df = load_confusion_data()
    if preds_df is not None:
        cm = confusion_matrix(
            preds_df['true_label'], preds_df['predicted_label'],
            labels=['Credit card', 'Debt collection']
        )
        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor('#1e293b')
        ax.set_facecolor('#1e293b')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Credit card', 'Debt collection'],
                    yticklabels=['Credit card', 'Debt collection'], ax=ax,
                    annot_kws={"color": "#e2e8f0"})
        ax.set_title('ML Model — Confusion Matrix', color='#f1f5f9', fontweight='bold')
        ax.set_ylabel('True Label', color='#94a3b8')
        ax.set_xlabel('Predicted Label', color='#94a3b8')
        ax.tick_params(colors='#94a3b8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run Notebook 03 to generate confusion matrix data.")

    st.markdown("---")

    st.markdown("### 📈 Training Visualizations")
    viz_files = {
        "Class Balance": "class_balance.png",
        "Class Balance (Before/After)": "class_balance_before_after.png",
        "Word Count Distribution": "word_count_histogram.png",
        "Word Count Boxplot": "word_count_boxplot_before.png",
        "Word Count Before/After": "wordcount_before_after_boxplot.png",
    }
    viz_cols = st.columns(2)
    for idx, (title, filename) in enumerate(viz_files.items()):
        path = os.path.join(OUTPUTS_DIR, filename)
        if os.path.exists(path):
            with viz_cols[idx % 2]:
                st.image(path, caption=title, use_container_width=True)
