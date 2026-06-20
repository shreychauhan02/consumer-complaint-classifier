# 📋 CFPB Complaint Classifier["https://consumer-complaint-classifier-juykdoh6dzcqgidnciuhd4.streamlit.app"]

A production-ready machine learning and generative AI system that automatically classifies consumer complaints submitted to the **Consumer Financial Protection Bureau (CFPB)**. The project implements a traditional NLP pipeline (TF-IDF + Logistic Regression) alongside an LLM-based classifier (Google Gemini 2.5 Flash), offering a comparison of performance, latency, and explainability through a sleek, custom-designed dark-themed Streamlit web interface.

---

## 💡 What Problem Does It Solve?

The CFPB receives thousands of unstructured, free-text consumer complaints daily. Manually reading, triaging, and routing these complaints to the appropriate departments or financial institutions is slow, labor-intensive, and prone to human error. 

This project solves this bottleneck by:
1. **Automated Classification**: Instantly categorizing incoming free-text narratives into specific product categories (specifically **Debt Collection** or **Credit Card**).
2. **Hybrid Intelligence**: Providing side-by-side predictions from a fast, low-cost classical machine learning model and a highly contextual, zero-shot LLM (Gemini).
3. **Explainable AI**: Leveraging the LLM to output a structured reasoning explaining *why* the complaint belongs to that category, facilitating faster human review.

---

## ⚙️ Architecture & Data Pipeline

The workflow is broken down into structured, reproducible Jupyter notebooks and a modular Streamlit web app:

```
CFPB/
├── data/
│   ├── complaints.csv          # Raw CFPB dataset
│   ├── cleaned_train.csv       # Preprocessed training dataset (6,400 samples)
│   └── cleaned_test.csv        # Preprocessed test dataset (1,600 samples)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis & visualization
│   ├── 02_preprocessing.ipynb  # Cleaning, deduplication, & downsampling
│   └── 03_model_training.ipynb # Model training, serialization & Gemini evaluation
├── app/
│   └── app.py                  # Streamlit application with custom styling
├── outputs/
│   ├── logistic_model.joblib   # Serialized Logistic Regression model
│   ├── tfidf_vectorizer.joblib # Serialized TF-IDF Vectorizer
│   ├── metrics.json            # Performance metrics
│   ├── ml_predictions.csv      # Model predictions on test set
│   └── *.png                   # Visualizations (confusion matrix, class balance, etc.)
├── requirements.txt            # Python dependencies
└── .env                        # Local environment variables
```

### 1. Exploratory Data Analysis (`01_eda.ipynb`)
- Analyzed the raw CFPB dataset (~115k complaints).
- Identified a significant class imbalance (approx. 2.2:1 ratio) and resolved 20,679 duplicate narratives.
- Visualized word counts, distribution histograms, and analyzed redacted text patterns (e.g., `XXXX`).

### 2. Preprocessing & Downsampling (`02_preprocessing.ipynb`)
- Standardized text by stripping duplicates, null entries, and extreme outliers.
- Balanced classes via stratified downsampling to create a robust dataset of **8,000 complaints** (4,000 Debt Collection, 4,000 Credit Card).
- Split the balanced data 80/20 into train (`cleaned_train.csv`) and test (`cleaned_test.csv`) sets.

### 3. Model Training & Validation (`03_model_training.ipynb`)
- Built a text classification pipeline using **TF-IDF Vectorization** (limited to 5,000 features) and **Logistic Regression**.
- Serialized the trained components (`joblib`) to the `outputs/` directory.
- Evaluated Google Gemini using the new Google GenAI SDK on a test sample to construct comparison metrics.

### 4. Interactive Web Application (`app/app.py`)
- Built a dark-themed Streamlit user interface matching production-grade AI SaaS dashboards.
- Users can paste any complaint narrative, and the app outputs:
  - **ML Model Prediction**: A fast prediction with a probability-based confidence score.
  - **Gemini LLM Prediction**: Zero-shot categorization using `gemini-2.5-flash` complete with structured reasoning.
- **Advanced Analytics Section**: A collapsible dashboard displaying:
  - Side-by-side performance metrics comparison (Accuracy, Precision, Recall, F1).
  - Confusion Matrix heatmap generated dynamically from test set predictions.
  - Explanatory training visualizations (class balance, word distributions before/after downsampling).

---

## 🛠️ Tech Stack

| Domain | Technologies |
|---|---|
| **Core Language** | Python 3.13+ |
| **Data Science & ML** | `pandas`, `numpy`, `scikit-learn` (TF-IDF + Logistic Regression) |
| **Generative AI** | Google GenAI SDK (`google-genai` using `gemini-2.5-flash`) |
| **Web Interface** | `streamlit` with customized dark mode CSS |
| **Visualization** | `matplotlib`, `seaborn` |
| **Model Serialization** | `joblib` |

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install all required python libraries:
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
Create a file named `.env` in the root folder of the project and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run the Pipeline (Optional)
To re-run the data cleaning, exploratory analysis, and model training:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb
```
*Note: Run Notebook 03 with your Gemini API key set to calculate LLM comparison metrics.*

### 4. Run the Streamlit Web Application
Launch the local web server to interact with the classifier:
```bash
streamlit run app/app.py
```

---

## 📊 Model Performance Results

Here are the evaluation results for the machine learning classifier computed against the test set:

| Metric | Traditional ML Model (LogReg) | Google Gemini LLM (Zero-shot)* |
| :--- | :--- | :--- |
| **Accuracy** | **93.67%** | *Evaluated on-demand* |
| **Precision** | **93.71%** | *Evaluated on-demand* |
| **Recall** | **94.30%** | *Evaluated on-demand* |
| **F1-Score** | **94.01%** | *Evaluated on-demand* |

> \* *LLM performance can be evaluated across the complete validation test set in Notebook 03 when a valid API key is present.*
