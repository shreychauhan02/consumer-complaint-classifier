import streamlit as st
import pandas as pd
import json
import os
import joblib
import requests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
from google import genai
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="CFPB Complaint Classifier",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

with st.sidebar:
    st.markdown("### API Settings")
    api_url = st.text_input("FastAPI URL", value=API_URL)
    api_status = st.empty()
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            api_status.success(f"API Online — {data['models_loaded']} models loaded")
        else:
            api_status.error("API Error")
    except Exception:
        api_status.warning("API Offline — start with: uvicorn backend.main:app --reload")

    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown(f"[API Docs ({api_url}/docs)]({api_url}/docs)")
    st.markdown(f"[ReDoc ({api_url}/redoc)]({api_url}/redoc)")
    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown("Built for the **Consumer Financial Protection Bureau** complaint analysis pipeline.")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #0a0e1a 0%, #0f172a 30%, #111827 100%);
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1e293b; }
    #MainMenu, header[data-testid="stHeader"] { display: none !important; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #2563eb 80%, #3b82f6 100%);
        padding: 3rem 2.5rem;
        border-radius: 24px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(37,99,235,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    .hero p {
        margin: 0;
        opacity: 0.9;
        font-size: 1.05rem;
        max-width: 700px;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(255,255,255,0.2);
        position: relative;
        z-index: 1;
    }

    .stTextArea textarea {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 4px 20px rgba(37,99,235,0.3) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 8px 30px rgba(37,99,235,0.4) !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a2332 0%, #0f172a 100%);
        padding: 1.25rem;
        border-radius: 14px;
        border: 1px solid #1e293b;
    }
    div[data-testid="stMetric"] label { color: #64748b !important; font-weight: 500 !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }

    div[data-testid="stExpander"] {
        border: 1px solid #1e293b !important;
        border-radius: 16px !important;
        overflow: hidden;
    }
    div[data-testid="stExpander"] summary {
        background: linear-gradient(135deg, #1a2332 0%, #0f172a 100%) !important;
        border: none !important;
        border-radius: 16px !important;
        color: #e2e8f0 !important;
        padding: 1rem 1.25rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stExpander"] summary:hover { background: #1e293b !important; }
    div[data-testid="stExpander"] details[open] summary { border-radius: 16px 16px 0 0 !important; }
    div[data-testid="stExpander"] details { border-radius: 0 0 16px 16px !important; }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin: 2rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1e293b;
        color: #e2e8f0;
    }

    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #475569;
        font-size: 0.8rem;
        border-top: 1px solid #1e293b;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, '..', 'outputs')

def find_model_files():
    candidates = {
        'Logistic Regression': ['logistic_regression_model.joblib', 'logistic_model.joblib'],
        'Naive Bayes': ['naive_bayes_model.joblib'],
        'Decision Tree': ['decision_tree_model.joblib'],
        'Random Forest': ['random_forest_model.joblib']
    }
    found = {}
    for name, filenames in candidates.items():
        for fname in filenames:
            path = os.path.join(OUTPUTS_DIR, fname)
            if os.path.exists(path):
                found[name] = path
                break
    return found

@st.cache_resource
def load_all_models():
    model_paths = find_model_files()
    models = {}
    for name, path in model_paths.items():
        models[name] = joblib.load(path)

    tfidf_path = os.path.join(OUTPUTS_DIR, 'tfidf_vectorizer.joblib')
    tfidf = joblib.load(tfidf_path) if os.path.exists(tfidf_path) else None

    return models, tfidf

@st.cache_data
def load_metrics():
    path = os.path.join(OUTPUTS_DIR, 'metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if 'ml' in data or 'llm' in data:
            return None
        return data
    return None

@st.cache_data
def load_predictions():
    path = os.path.join(OUTPUTS_DIR, 'ml_predictions.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'predicted_label' in df.columns:
            label_map = {0: 'Credit card', 1: 'Debt collection'}
            df['predicted_label'] = df['predicted_label'].map(label_map).fillna(df['predicted_label'])
        return df
    return None

def get_gemini_client():
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if api_key:
        return genai.Client(api_key=api_key)
    return None

def classify_with_gemini(text):
    client = get_gemini_client()
    if not client:
        return {"prediction": "Unavailable", "confidence": "N/A", "reasoning": "No API key configured."}

    truncated = ' '.join(text.split()[:300])
    prompt = (
        "You are a complaint classification system for a financial company.\n"
        "Classify the following customer complaint into EXACTLY ONE of these two categories:\n"
        "'Debt collection' or 'Credit card'.\n\n"
        "Respond in this exact JSON format ONLY, nothing else:\n"
        '{"prediction": "<category>", "confidence": "<high/medium/low>", "reasoning": "<one sentence why>"}'
    )
    try:
        response = client.models.generate_content(
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
        return {"prediction": "Unavailable", "confidence": "N/A", "reasoning": f"Service error: {err}"}

def get_ml_predictions(text, models, tfidf):
    if not models or tfidf is None:
        return {}
    X = tfidf.transform([text])
    results = {}
    for name, model in models.items():
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        if isinstance(pred, (int, np.integer)):
            label = 'Debt collection' if pred == 1 else 'Credit card'
        else:
            label = str(pred)
        results[name] = {
            'prediction': label,
            'confidence': float(max(prob))
        }
    return results


def predict_via_api(narrative, api_base):
    try:
        resp = requests.post(
            f"{api_base}/predict",
            json={"narrative": narrative},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception:
        return None

all_models, tfidf = load_all_models()
metrics = load_metrics()
preds_df = load_predictions()

st.markdown("""
<div class="hero">
    <div class="badge">POWERED BY SCIKIT-LEARN & GOOGLE GEMINI</div>
    <h1>CFPB Complaint Classifier</h1>
    <p>Automatically classify consumer complaints into <strong>Debt Collection</strong> or <strong>Credit Card</strong> categories using multiple ML models and large language models.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Classify a Complaint")
narrative_input = st.text_area(
    "Paste a consumer complaint narrative below",
    height=150,
    placeholder="e.g., I received a call from a debt collector about a debt I do not owe..."
)
classify_btn = st.button("Analyze Complaint", type="primary", width="stretch")

if classify_btn and narrative_input.strip():
    api_prediction = predict_via_api(narrative_input, api_url)

    if api_prediction:
        best_ml_name = api_prediction['best_model']
        ml_results = {k: {'prediction': v['prediction'], 'confidence': v['confidence']}
                      for k, v in api_prediction['all_models'].items()}
        data_source = "FastAPI Backend"
    else:
        if not all_models or tfidf is None:
            st.error("ML models not found. Start the FastAPI backend or run notebook 04 first.")
            st.stop()
        with st.spinner("Running local models (API offline)..."):
            ml_results = get_ml_predictions(narrative_input, all_models, tfidf)
            llm_result = classify_with_gemini(narrative_input)
        best_ml_name = max(ml_results.items(), key=lambda x: x[1]['confidence'])[0]
        if metrics and metrics.get('best_model') in ml_results:
            best_ml_name = metrics['best_model']
        data_source = "Local Models"

    if not ml_results:
        st.error("No ML models loaded successfully.")
    else:
        st.success(f"Source: {data_source}")

        st.markdown("### Results")

        col1, col2 = st.columns(2)

        with col1:
            res = ml_results[best_ml_name]
            conf_pct = res['confidence'] * 100
            st.markdown(f"**Best ML Model: {best_ml_name}**")
            st.metric("Prediction", res['prediction'])
            st.progress(res['confidence'])
            st.caption(f"Confidence: {conf_pct:.1f}%")

        with col2:
            if api_prediction:
                llm_result = classify_with_gemini(narrative_input)
            conf_map = {'high': 0.90, 'medium': 0.65, 'low': 0.35}
            conf_val = conf_map.get(llm_result.get('confidence', 'medium'), 0.50)
            st.markdown("**Google Gemini 2.5 Flash**")
            st.metric("Prediction", llm_result.get('prediction', 'N/A'))
            st.progress(conf_val)
            st.caption(f"Confidence: {llm_result.get('confidence', 'N/A')}")

        st.markdown("---")
        st.markdown("### All Model Predictions")
        pred_cols = st.columns(len(ml_results))
        for idx, (name, r) in enumerate(ml_results.items()):
            with pred_cols[idx]:
                st.metric(name, r['prediction'], f"{r['confidence']:.1%}")

elif classify_btn:
    st.warning("Please enter a complaint narrative to analyze.")

if metrics:
    st.markdown("---")
    st.markdown("### Model Performance Overview")

    model_names = [k for k in metrics.keys() if k != 'best_model']
    best = metrics.get('best_model', '')
    best_m = metrics.get(best, {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", best)
    m2.metric("Accuracy", f"{best_m.get('accuracy', 0):.2%}")
    m3.metric("F1 Score", f"{best_m.get('f1', 0):.2%}")
    m4.metric("AUC", f"{best_m.get('auc', 0):.2%}")

    st.markdown("#### All Models Comparison")

    table_data = []
    for name in model_names:
        m = metrics.get(name, {})
        table_data.append({
            'Model': name,
            'Accuracy': f"{m.get('accuracy', 0):.4f}",
            'Precision': f"{m.get('precision', 0):.4f}",
            'Recall': f"{m.get('recall', 0):.4f}",
            'F1 Score': f"{m.get('f1', 0):.4f}",
            'AUC': f"{m.get('auc', 0):.4f}"
        })
    st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

with st.expander("Advanced Analytics", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curves", "Feature Importance"])

    with tab1:
        if preds_df is not None and 'true_label' in preds_df.columns and 'predicted_label' in preds_df.columns:
            labels = sorted(preds_df['true_label'].unique())
            cm = confusion_matrix(preds_df['true_label'], preds_df['predicted_label'], labels=labels)
            fig, ax = plt.subplots(figsize=(7, 5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels, ax=ax,
                        annot_kws={"color": "#e2e8f0", "fontsize": 16})
            ax.set_title('Confusion Matrix', color='#f1f5f9', fontsize=13, fontweight='bold', pad=12)
            ax.set_ylabel('True Label', color='#94a3b8')
            ax.set_xlabel('Predicted Label', color='#94a3b8')
            ax.tick_params(colors='#94a3b8')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Run Notebook 04 to generate confusion matrix data.")

    with tab2:
        roc_path = os.path.join(OUTPUTS_DIR, 'roc_curves.png')
        if os.path.exists(roc_path):
            st.image(roc_path, width="stretch")
        else:
            st.info("Run Notebook 04 to generate ROC curves.")

    with tab3:
        fi_path = os.path.join(OUTPUTS_DIR, 'feature_importance.png')
        if os.path.exists(fi_path):
            st.image(fi_path, width="stretch")
        else:
            st.info("Run Notebook 04 to generate feature importance plot.")

    st.markdown("---")
    st.markdown("#### Training Visualizations")
    viz_files = {
        "Class Balance": "class_balance.png",
        "Class Balance (Before/After)": "class_balance_before_after.png",
        "Word Count Distribution": "word_count_histogram.png",
        "CV F1 Distribution": "cv_f1_distribution.png",
    }
    viz_cols = st.columns(2)
    for idx, (title, filename) in enumerate(viz_files.items()):
        path = os.path.join(OUTPUTS_DIR, filename)
        if os.path.exists(path):
            with viz_cols[idx % 2]:
                st.image(path, caption=title, width="stretch")

st.markdown("""
<div class="footer">
    CFPB Complaint Classifier — Built with Scikit-learn, Streamlit & Google Gemini
</div>
""", unsafe_allow_html=True)
