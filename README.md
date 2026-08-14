# CFPB Complaint Classifier

A production-ready machine learning and generative AI system that classifies consumer complaints submitted to the Consumer Financial Protection Bureau (CFPB). The project implements multiple traditional ML classifiers (Logistic Regression, Naive Bayes, Decision Tree, Random Forest) with cross-validation and hyperparameter tuning, alongside an LLM-based classifier (Google Gemini 2.5 Flash), exposed through a FastAPI REST API and a Streamlit web interface.

---

## What Problem Does It Solve?

The CFPB receives thousands of unstructured, free-text consumer complaints daily. Manually reading, triaging, and routing these complaints to the appropriate departments or financial institutions is slow, labor-intensive, and prone to human error.

This project solves this bottleneck by:

1. **Automated Classification**: Instantly categorizing incoming free-text narratives into Debt Collection or Credit Card categories.
2. **Model Comparison**: Evaluating 4 different ML algorithms with cross-validation, hyperparameter tuning via GridSearchCV, and AUC-ROC analysis.
3. **Hybrid Intelligence**: Providing side-by-side predictions from classical ML models and a contextual LLM (Gemini).
4. **REST API**: Exposing model predictions through a FastAPI backend with OpenAPI documentation.
5. **Explainable AI**: Leveraging the LLM to output structured reasoning explaining why the complaint belongs to a category.

---

## Project Structure

```
CFPB/
├── data/
│   ├── complaints.csv              # Raw CFPB dataset
│   ├── cleaned_train.csv           # Training set (6,400 samples)
│   └── cleaned_test.csv            # Test set (1,600 samples)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      # Cleaning, deduplication, downsampling
│   ├── 03_model_training.ipynb     # Logistic Regression + Gemini evaluation
│   └── 04_model_comparison.ipynb   # 4 classifiers, CV, tuning, AUC-ROC
├── backend/
│   ├── main.py                     # FastAPI application
│   └── models.py                   # Pydantic request/response models
├── app/
│   └── app.py                      # Streamlit web interface
├── outputs/
│   ├── logistic_regression_model.joblib
│   ├── naive_bayes_model.joblib
│   ├── decision_tree_model.joblib
│   ├── random_forest_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── metrics.json                # All model metrics
│   ├── ml_predictions.csv          # Predictions on test set
│   ├── roc_curves.png              # AUC-ROC curves for all models
│   ├── confusion_matrices.png      # Confusion matrices
│   ├── feature_importance.png      # Top TF-IDF features
│   ├── cv_f1_distribution.png      # Cross-validation F1 distribution
│   └── model_comparison.png        # Side-by-side model comparison
├── requirements.txt
└── .env
```

---

## Pipeline Breakdown

### 1. Exploratory Data Analysis (01_eda.ipynb)

- Analyzed raw CFPB dataset (~115k complaints)
- Identified class imbalance (2.2:1 ratio) and resolved 20,679 duplicate narratives
- Visualized word counts, distribution histograms, and redacted text patterns

### 2. Preprocessing and Downsampling (02_preprocessing.ipynb)

- Standardized text by stripping duplicates, null entries, and outliers
- Balanced classes via stratified downsampling to create 8,000 complaints (4,000 per category)
- Split 80/20 into train (6,400) and test (1,600) sets

### 3. Single Model Training (03_model_training.ipynb)

- Built TF-IDF + Logistic Regression pipeline
- Serialized model and vectorizer with joblib
- Evaluated Gemini LLM on a test sample for comparison

### 4. Model Comparison and Evaluation (04_model_comparison.ipynb)

- Trained and evaluated 4 classifiers: Logistic Regression, Naive Bayes, Decision Tree, Random Forest
- 5-fold stratified cross-validation for robust performance estimates
- GridSearchCV hyperparameter tuning for each model
- AUC-ROC curves, confusion matrices, and feature importance analysis
- F1 distribution plots across cross-validation folds

### 5. FastAPI Backend (backend/)

- REST API exposing prediction endpoints
- GET /health - API health check with model status
- GET /models - Available models and their metrics
- POST /predict - Classify a complaint narrative using all models
- Auto-generated OpenAPI docs at /docs

### 6. Streamlit Web Interface (app/)

- Dark-themed UI with production-grade styling
- Calls FastAPI backend when available, falls back to local models
- Displays predictions from all models side-by-side
- Gemini LLM prediction with reasoning
- Advanced Analytics section with confusion matrices, ROC curves, and feature importance

---

## Model Performance

| Model                | Accuracy | Precision | Recall  | F1 Score | AUC    |
|----------------------|----------|-----------|---------|----------|--------|
| Logistic Regression  | 93.75%   | 93.75%    | 93.75%  | 93.75%   | 98.12% |
| Random Forest        | 91.87%   | 91.15%    | 92.75%  | 91.95%   | 97.52% |
| Naive Bayes          | 89.50%   | 89.80%    | 89.12%  | 89.46%   | 95.46% |
| Decision Tree        | 88.06%   | 88.50%    | 87.50%  | 87.99%   | 88.06% |

**Best Model**: Logistic Regression (AUC: 98.12%)

---

## Tech Stack

| Domain                | Technologies                                            |
|-----------------------|---------------------------------------------------------|
| Core Language         | Python 3.13+                                            |
| Data Science and ML   | pandas, numpy, scikit-learn                             |
| Generative AI         | Google GenAI SDK (gemini-2.5-flash)                     |
| Web Interface         | Streamlit with custom dark mode CSS                     |
| REST API              | FastAPI, Uvicorn, Pydantic                              |
| Visualization         | matplotlib, seaborn                                     |
| Model Serialization   | joblib                                                  |

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root folder:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Run the Full Pipeline (Optional)

```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_model_comparison.ipynb
```

### Start the FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

API docs available at http://localhost:8000/docs

### Start the Streamlit Web App

```bash
streamlit run app/app.py
```

The app connects to the FastAPI backend at http://localhost:8000. If the backend is offline, it falls back to loading models locally.
