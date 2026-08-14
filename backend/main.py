import os
import json
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    ComplaintRequest, PredictionResponse, ModelPrediction,
    ModelsResponse, ModelInfo, HealthResponse
)

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

loaded_models = {}
tfidf_vectorizer = None
metrics_data = {}


def load_models():
    global loaded_models, tfidf_vectorizer, metrics_data

    model_files = {
        'Logistic Regression': ['logistic_regression_model.joblib', 'logistic_model.joblib'],
        'Naive Bayes': ['naive_bayes_model.joblib'],
        'Decision Tree': ['decision_tree_model.joblib'],
        'Random Forest': ['random_forest_model.joblib']
    }

    for name, filenames in model_files.items():
        for fname in filenames:
            path = os.path.join(OUTPUTS_DIR, fname)
            if os.path.exists(path):
                loaded_models[name] = joblib.load(path)
                break

    tfidf_path = os.path.join(OUTPUTS_DIR, 'tfidf_vectorizer.joblib')
    if os.path.exists(tfidf_path):
        tfidf_vectorizer = joblib.load(tfidf_path)

    metrics_path = os.path.join(OUTPUTS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
            if 'ml' not in data and 'llm' not in data:
                metrics_data = data


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="CFPB Complaint Classifier API",
    description=(
        "REST API for classifying consumer complaints submitted to the "
        "Consumer Financial Protection Bureau (CFPB) into **Debt Collection** "
        "or **Credit Card** categories using multiple ML models."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root():
    return {
        "name": "CFPB Complaint Classifier API",
        "version": "1.0.0",
        "description": "Classify consumer complaints using ML models",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "models": "GET /models",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health_check():
    return HealthResponse(
        status="healthy",
        models_loaded=len(loaded_models),
        tfidf_ready=tfidf_vectorizer is not None
    )


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
def get_models():
    available = list(loaded_models.keys())
    best = metrics_data.get('best_model', available[0] if available else '')

    metrics = {}
    for name in available:
        m = metrics_data.get(name, {})
        metrics[name] = ModelInfo(
            name=name,
            accuracy=m.get('accuracy', 0),
            precision=m.get('precision', 0),
            recall=m.get('recall', 0),
            f1=m.get('f1', 0),
            auc=m.get('auc', 0)
        )

    return ModelsResponse(
        available_models=available,
        best_model=best,
        metrics=metrics
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: ComplaintRequest):
    if not loaded_models or tfidf_vectorizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run notebook 04 first.")

    narrative = request.narrative.strip()
    if len(narrative.split()) < 5:
        raise HTTPException(status_code=400, detail="Narrative too short. Provide at least 5 words.")

    X = tfidf_vectorizer.transform([narrative])
    all_predictions = {}

    for name, model in loaded_models.items():
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]

        if isinstance(pred, (int, np.integer)):
            label = 'Debt collection' if pred == 1 else 'Credit card'
        else:
            label = str(pred)

        all_predictions[name] = ModelPrediction(
            model_name=name,
            prediction=label,
            confidence=round(float(max(prob)), 4)
        )

    best_model_name = metrics_data.get('best_model', '')
    if best_model_name not in all_predictions:
        best_model_name = max(all_predictions.items(), key=lambda x: x[1].confidence)[0]

    best = all_predictions[best_model_name]

    return PredictionResponse(
        narrative=narrative,
        best_model=best_model_name,
        best_prediction=best.prediction,
        best_confidence=best.confidence,
        all_models=all_predictions
    )
