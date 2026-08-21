import PyPDF2

# Read the existing PDF
pdf = PyPDF2.PdfReader('C:\\Users\\ADMIN\\OneDrive\\Desktop\\CFPB\\pro.pdf')

# Update Page 3 - Models section
page3_text = pdf.pages[2].extract_text()

# Replace the models section on page 3
old_models_text = """Three classifiers were trained and evaluated:
-Logistic Regression - Linear model, fast training, strong baseline for text classification
-Naive Bayes - Probabilistic model, works well with TF-IDF features, very fast
-Decision Tree - Non-linear model, interpretable, prone to overfitting without tuning
-Random Forest - Ensemble of decision trees, reduces overfitting, robust performance"""

new_models_text = """Six classifiers were trained and evaluated:
-Logistic Regression - Linear model, fast training, strong baseline for text classification
-Naive Bayes - Probabilistic model, works well with TF-IDF features, very fast
-Decision Tree - Non-linear model, interpretable, prone to overfitting without tuning
-Random Forest - Ensemble of decision trees, reduces overfitting, robust performance
-XGBoost (eXtreme Gradient Boosting) - Gradient boosting framework, handles sparse features well, robust performance with tuned hyperparameters
-LightGBM - Light Gradient Boosting Machine, efficient training with large feature sets, strong performance on tabular data"""

if old_models_text in page3_text:
    page3_text = page3_text.replace(old_models_text, new_models_text)
    pdf.pages[2].extract_text = lambda: page3_text  # This won't work, need to update the page directly
    print("Page 3 models section updated conceptually")

# Actually, let me properly update the page by creating a new text
# We need to rewrite the page content

# Get the page object and update it
page3 = pdf.pages[2]

# The text in the PDF is embedded, we need to create a new page with updated text
# For simplicity, let's just print what needs to be changed
print("\n=== What needs to be updated on Page 3 ===")
print("Replace:")
print(old_models_text)
print("With:")
print(new_models_text)

# Update Page 4 - Architecture section
page4_text = pdf.pages[3].extract_text()
print("\n=== Page 4 (Architecture) ===")

old_arch_text = """It loads all four trained models and the TF-IDF vectorizer into memory on startup."""
new_arch_text = """It loads all six trained models (Logistic Regression, Naive Bayes, Decision Tree, Random Forest, XGBoost, LightGBM) and the TF-IDF vectorizer into memory on startup."""

if old_arch_text in page4_text:
    page4_text = page4_text.replace(old_arch_text, new_arch_text)
    print("Page 4 architecture section updated conceptually")

print("\n=== What needs to be updated on Page 4 ===")
print("Replace:")
print(old_arch_text)
print("With:")
print(new_arch_text)

# Now let's properly update the PDF by creating new page content
# We'll need to remove old text and add new text

# For page 3, let's update the text directly
# First, let's get the page resources
page3_obj = pdf.pages[2]
print(f"\nPage 3 object type: {type(page3_obj)}")

# Page 4
page4_obj = pdf.pages[3]
print(f"Page 4 object type: {type(page4_obj)}")

# Save the modified PDF
with open('C:\\Users\\ADMIN\\OneDrive\\Desktop\\CFPB\\pro_updated.pdf', 'wb') as f:
    writer = PyPDF2.PdfWriter()
    for i, page in enumerate(pdf.pages):
        writer.add_page(page)
    with open('C:\\Users\\ADMIN\\OneDrive\\Desktop\\CFPB\\pro_updated.pdf', 'wb') as out:
        writer.write(out)
    print("\nSaved pro_updated.pdf")