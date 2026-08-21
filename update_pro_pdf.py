import PyPDF2
from PyPDF2 import PdfWriter

# Read the existing PDF
pdf = PyPDF2.PdfReader('C:\\Users\\ADMIN\\OneDrive\\Desktop\\CFPB\\pro.pdf')

# Page 3 - Update models section (page index 2)
page3 = pdf.pages[2]
page3_text = page3.extract_text()

# Update the models text
old_text = """Three classifiers were trained and evaluated:
-Logistic Regression - Linear model, fast training, strong baseline for text classification
-Naive Bayes - Probabilistic model, works well with TF-IDF features, very fast
-Decision Tree - Non-linear model, interpretable, prone to overfitting without tuning
-Random Forest - Ensemble of decision trees, reduces overfitting, robust performance"""

new_text = """Six classifiers were trained and evaluated:
-Logistic Regression - Linear model, fast training, strong baseline for text classification
-Naive Bayes - Probabilistic model, works well with TF-IDF features, very fast
-Decision Tree - Non-linear model, interpretable, prone to overfitting without tuning
-Random Forest - Ensemble of decision trees, reduces overfitting, robust performance
-XGBoost (eXtreme Gradient Boosting) - Gradient boosting framework, handles sparse features well, robust performance with tuned hyperparameters
-LightGBM - Light Gradient Boosting Machine, efficient training with large feature sets, strong performance on tabular data"""

# Since modifying existing PDF text is complex, let's create a summary of what needs to be updated
print("Page 3 update needed:")
print("  Replace old models text with new 6-model text")
print()

# Page 4 - Update architecture section
page4 = pdf.pages[3]
page4_text = page4.extract_text()

old_arch = "It loads all four trained models and the TF-IDF vectorizer into memory on startup."
new_arch = "It loads all six trained models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM) and the TF-IDF vectorizer into memory on startup."

print("Page 4 update needed:")
print("  Replace architecture text from 4 models to 6 models")
print()

# Page 5 - Update conclusion section (page index 4)
page5 = pdf.pages[4] if len(pdf.pages) > 4 else None
if page5:
    page5_text = page5.extract_text()
    print("Page 5: Check for model references")
else:
    print("Page 5: Not found or doesn't exist")

# Save the PDF (even if text not modified, the structure is preserved)
writer = PdfWriter()
for page in pdf.pages:
    writer.add_page(page)
with open('C:\\Users\\ADMIN\\OneDrive\\Desktop\\CFPB\\pro_updated.pdf', 'wb') as f:
    writer.write(f)
print("\nSaved pro_updated.pdf (structure preserved)")
print("Note: Text content modification in existing PDF is complex via PyPDF2")
print("Consider using reportlab to create a new PDF with updated content")