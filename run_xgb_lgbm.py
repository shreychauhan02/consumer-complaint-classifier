import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')

# Load data
train_df = pd.read_csv('data/cleaned_train.csv')
test_df = pd.read_csv('data/cleaned_test.csv')

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train = tfidf.fit_transform(train_df['Consumer complaint narrative'])
X_test = tfidf.transform(test_df['Consumer complaint narrative'])
y_train = train_df['Product'].values
y_test = test_df['Product'].values

label_map = {'Credit card': 0, 'Debt collection': 1}
y_train_bin = np.array([label_map[y] for y in y_train])
y_test_bin = np.array([label_map[y] for y in y_test])

# Models with fixed params (no GridSearch to save time)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1),
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_estimators=100, max_depth=6, learning_rate=0.1),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1, n_estimators=100, max_depth=6, learning_rate=0.1, num_leaves=31)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== Cross-Validation ===")
cv_results = {}
for name, model in models.items():
    acc_scores = cross_val_score(model, X_train, y_train_bin, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X_train, y_train_bin, cv=cv, scoring='f1')
    cv_results[name] = {
        'accuracy_mean': acc_scores.mean(),
        'accuracy_std': acc_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std()
    }
    print(f"{name}: Accuracy={acc_scores.mean():.4f}±{acc_scores.std():.4f} F1={f1_scores.mean():.4f}±{f1_scores.std():.4f}")

print("\n=== Training on full data and evaluating on test ===")
test_results = {}
for name, model in models.items():
    model.fit(X_train, y_train_bin)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    test_results[name] = {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy_score(y_test_bin, y_pred),
        'precision': precision_score(y_test_bin, y_pred),
        'recall': recall_score(y_test_bin, y_pred),
        'f1': f1_score(y_test_bin, y_pred)
    }
    print(f"--- {name} ---")
    print(f"  Accuracy: {test_results[name]['accuracy']:.4f}")
    print(f"  Precision: {test_results[name]['precision']:.4f}")
    print(f"  Recall: {test_results[name]['recall']:.4f}")
    print(f"  F1: {test_results[name]['f1']:.4f}")

# Save models
for name, model in models.items():
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(model, f'outputs/{safe_name}_model.joblib')
joblib.dump(tfidf, 'outputs/tfidf_vectorizer.joblib')

# Summary
summary = pd.DataFrame({
    'Model': list(test_results.keys()),
    'Accuracy': [r['accuracy'] for r in test_results.values()],
    'Precision': [r['precision'] for r in test_results.values()],
    'Recall': [r['recall'] for r in test_results.values()],
    'F1 Score': [r['f1'] for r in test_results.values()]
})
summary = summary.sort_values('F1 Score', ascending=False).reset_index(drop=True)
print("\n=== Final Model Comparison Summary ===")
print(summary.to_string(index=False))

best_model_name = summary.iloc[0]['Model']
print(f"\nBest model: {best_model_name} (F1 = {summary.iloc[0]['F1 Score']:.4f})")

# Save metrics
all_metrics = {}
for name, res in test_results.items():
    fpr, tpr, _ = roc_curve(y_test_bin, res['y_prob'])
    all_metrics[name] = {
        'accuracy': round(res['accuracy'], 4),
        'precision': round(res['precision'], 4),
        'recall': round(res['recall'], 4),
        'f1': round(res['f1'], 4),
        'auc': round(auc(fpr, tpr), 4)
    }
all_metrics['best_model'] = best_model_name

with open('outputs/metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("\nMetrics saved to outputs/metrics.json")
print(json.dumps(all_metrics, indent=2))