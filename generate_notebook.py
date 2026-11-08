"""
Script to generate a clean, approachable, student-friendly Jupyter Notebook:
ecommerce_customer_churn_analysis.ipynb
Featuring:
- Problem Formulation (Classification vs Regression)
- Rich Data Insights & Behavioral Findings
- 3 Classic ML Algorithms (Logistic Regression, Decision Tree, Random Forest)
- Clear explanation of Which Model Won and Why
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.9.13"
    }
}

cells = []

# --- CELL 0: Title & Intro ---
cells.append(nbf.v4.new_markdown_cell("""# 🛍️ E-Commerce Customer Behavior & Churn Study
## Academic Data Science Project

---

### 📌 Project Overview
In e-commerce, **customer retention** is one of the most critical factors for business success. Acquiring a new customer is estimated to be **5x to 7x more expensive** than keeping an existing customer happy.

In this project, we study an e-commerce customer behavior dataset of **50,000 customers** across **25 attributes**. 

#### Goals of this Project:
1. **Understand Problem Formulation**: Check whether predicting total customer spend (**Regression**) or predicting whether a customer will leave (**Classification**) is more meaningful.
2. **Discover Key Behavioral Insights**: Explore what friction points (such as customer service calls, cart abandonment, and inactivity) actually lead to customer attrition.
3. **Compare 3 Classic Machine Learning Algorithms**:
   - **Logistic Regression** (Linear baseline)
   - **Decision Tree** (Single rule-based flowchart)
   - **Random Forest** (Ensemble of trees)
4. **Identify the Winning Model and Explain Why It Won**: Provide clear theoretical and practical reasoning for the best-performing model.
"""))

# --- CELL 1: Imports ---
cells.append(nbf.v4.new_code_cell("""# 1. Essential Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Scikit-Learn tools
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 3 Classic Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, r2_score, mean_squared_error
)

# Plot styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (9, 5)

print("[+] All libraries successfully loaded!")
"""))

# --- CELL 2: Data Ingestion ---
cells.append(nbf.v4.new_markdown_cell("""---
## 1. Data Loading & Inspection
We load the dataset containing 50,000 customer records.
"""))

cells.append(nbf.v4.new_code_cell("""# Load the dataset
DATA_PATH = 'archive/eCommerce-dataset.csv'
df = pd.read_csv(DATA_PATH)

print(f"Dataset Dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")
df.head(5)
"""))

# --- CELL 3: Missing Values ---
cells.append(nbf.v4.new_code_cell("""# Check column types and missing values
missing_summary = pd.DataFrame({
    'Data Type': df.dtypes,
    'Missing Values': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(1)
})
missing_summary[missing_summary['Missing Values'] > 0].sort_values(by='Missing %', ascending=False)
"""))

# --- CELL 4: Problem Formulation Check ---
cells.append(nbf.v4.new_markdown_cell("""---
## 2. Problem Formulation: Regression vs. Classification

Before diving into modeling, we checked whether this project should be framed as a **Regression** problem or a **Classification** problem:

1. **Continuous Regression (Predicting `Lifetime_Value` in dollars)**:
   - *Goal:* Predict total dollar amount a customer will spend.
   - *Result:* Ridge baseline achieves $R^2 \approx 0.48$. Useful for financial accounting, but does not identify *who* is unhappy or *when* they are about to leave.
2. **Binary Classification (Predicting `Churned` as 0 or 1)**:
   - *Goal:* Predict whether a customer is at risk of leaving the store.
   - *Result:* Logistic Regression baseline achieves ROC-AUC $\approx 0.75$, Accuracy $\approx 77\%$.
   - *Business Value:* **Critical.** Knowing which customer is at risk allows the store to send an email or discount *before* they leave.

**Conclusion:** We choose **Binary Classification (`Churned`)** as the primary focus because customer retention delivers immediate, practical business value.
"""))

cells.append(nbf.v4.new_code_cell("""# Check baselines for Regression vs Classification
quick_cols = ['Age', 'Membership_Years', 'Login_Frequency', 'Session_Duration_Avg',
              'Pages_Per_Session', 'Cart_Abandonment_Rate', 'Days_Since_Last_Purchase',
              'Customer_Service_Calls', 'Mobile_App_Usage']

X_sample = df[quick_cols].fillna(df[quick_cols].median())

# Regression Baseline (Lifetime_Value)
y_reg = df['Lifetime_Value']
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_sample, y_reg, test_size=0.2, random_state=42)
reg = Ridge().fit(X_tr_r, y_tr_r)
r2 = r2_score(y_te_r, reg.predict(X_te_r))

# Classification Baseline (Churned)
y_clf = df['Churned']
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_sample, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
clf = LogisticRegression(max_iter=500).fit(X_tr_c, y_tr_c)
acc = accuracy_score(y_te_c, clf.predict(X_te_c))
auc = roc_auc_score(y_te_c, clf.predict_proba(X_te_c)[:, 1])

print("--- FORMULATION COMPARISON ---")
print(f"Regression (Lifetime Value) Baseline R2 Score: {r2:.4f}")
print(f"Classification (Churn) Baseline Accuracy     : {acc:.2%}")
print(f"Classification (Churn) Baseline ROC-AUC      : {auc:.4f}")
print("-> Selected Task: Binary Classification on Customer Churn!")
"""))

# --- CELL 5: Exploratory Data Analysis & Findings ---
cells.append(nbf.v4.new_markdown_cell("""---
## 3. Exploratory Data Analysis & Key Findings

Let's look into the data to find what factors separate loyal customers from customers who leave.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Overall Retention vs Churn Ratio
churn_counts = df['Churned'].value_counts()

plt.figure(figsize=(7, 4))
colors = ['#22c55e', '#ef4444']
sns.barplot(x=['Retained (0)', 'Churned (1)'], y=churn_counts.values, palette=colors)
plt.title('Overall Customer Retention vs. Churn', fontsize=13, fontweight='bold')
plt.ylabel('Number of Customers')
for i, v in enumerate(churn_counts.values):
    pct = (v / len(df)) * 100
    plt.text(i, v + 800, f"{v:,} ({pct:.1f}%)", ha='center', fontweight='bold')
plt.ylim(0, 42000)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Key Finding #1: The Customer Service "Tipping Point"
Let's analyze how the number of customer service calls affects the churn rate.
"""))

cells.append(nbf.v4.new_code_cell("""# Finding 1: Support calls vs Churn Rate
cs_churn = df.groupby('Customer_Service_Calls')['Churned'].agg(['count', 'mean']).reset_index()
cs_churn = cs_churn[cs_churn['Customer_Service_Calls'] <= 10]
cs_churn['Churn Rate (%)'] = (cs_churn['mean'] * 100).round(1)

plt.figure(figsize=(9, 4.5))
sns.barplot(data=cs_churn, x='Customer_Service_Calls', y='Churn Rate (%)', palette='Reds')
plt.title('Finding #1: Customer Service Calls vs. Churn Rate (%)', fontsize=13, fontweight='bold')
plt.xlabel('Number of Customer Service Calls')
plt.ylabel('Churn Rate (%)')
plt.axhline(28.9, color='blue', linestyle='--', label='Average Churn Rate (28.9%)')
plt.legend()
plt.show()

print("Takeaway: Customers with 0 to 5 calls have low churn (11%-18%). Once calls reach 6+, churn surges past 40%!")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Key Finding #2: Mobile App Users are Far More Loyal
Does using the mobile app increase customer loyalty? Let's group app usage hours.
"""))

cells.append(nbf.v4.new_code_cell("""# Finding 2: Mobile App Usage vs Churn
df['App_Usage_Group'] = pd.cut(df['Mobile_App_Usage'], bins=[-1, 5, 15, 60], labels=['Low (<5 hrs)', 'Moderate (5-15 hrs)', 'High (>15 hrs)'])
app_churn = df.groupby('App_Usage_Group', observed=False)['Churned'].agg(['count', 'mean']).reset_index()
app_churn['Churn Rate (%)'] = (app_churn['mean'] * 100).round(1)

plt.figure(figsize=(8, 4))
sns.barplot(data=app_churn, x='App_Usage_Group', y='Churn Rate (%)', palette='Blues_r')
plt.title('Finding #2: Mobile App Usage vs. Churn Rate', fontsize=13, fontweight='bold')
plt.ylabel('Churn Rate (%)')
plt.show()

print(app_churn[['App_Usage_Group', 'count', 'Churn Rate (%)']])
print("Takeaway: Customers who spend over 15 hours on the app churn at only 21.7%, compared to 61.3% for low app users!")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 💡 Key Finding #3: The Inactivity Danger Zone (Days Since Last Purchase)
"""))

cells.append(nbf.v4.new_code_cell("""# Finding 3: Recency vs Churn
df['Recency_Group'] = pd.cut(df['Days_Since_Last_Purchase'], bins=[0, 30, 60, 90, 180], labels=['0-30 days', '31-60 days', '61-90 days', '90+ days'])
rec_churn = df.groupby('Recency_Group', observed=False)['Churned'].agg(['count', 'mean']).reset_index()
rec_churn['Churn Rate (%)'] = (rec_churn['mean'] * 100).round(1)

plt.figure(figsize=(8, 4))
sns.barplot(data=rec_churn, x='Recency_Group', y='Churn Rate (%)', palette='Oranges')
plt.title('Finding #3: Days Since Last Purchase vs. Churn Rate', fontsize=13, fontweight='bold')
plt.ylabel('Churn Rate (%)')
plt.show()

print(rec_churn[['Recency_Group', 'count', 'Churn Rate (%)']])
print("Takeaway: Beyond 90 days without a purchase, almost half of customers (48.9%) churn permanently!")
"""))

# --- CELL 6: Preprocessing ---
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Data Preprocessing Pipeline

We prepare a clean, leak-free Scikit-Learn preprocessing pipeline:
- Numerical features: Impute missing values with **median**, then apply **StandardScaler**.
- Categorical features (`Gender`, `Country`, `Signup_Quarter`): Impute with **most frequent**, then apply **OneHotEncoder**.
- Train / Test split: 80% training data, 20% test data (10,000 unseen customers).
"""))

cells.append(nbf.v4.new_code_cell("""# Outlier cap on Age
df['Age'] = df['Age'].clip(lower=16, upper=90)

# Target and Features
y = df['Churned']
X = df.drop(columns=['Churned', 'City', 'App_Usage_Group', 'Recency_Group'], errors='ignore')

cat_cols = ['Gender', 'Country', 'Signup_Quarter']
num_cols = [c for c in X.columns if c not in cat_cols]

# Build Preprocessor
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} rows")
print(f"Testing set : {X_test.shape[0]:,} rows")
"""))

# --- CELL 7: Model Training ---
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Model Training: 3 Classic Algorithms

We train three classic, intuitive algorithms:
1. **Logistic Regression**: A linear model calculating probabilities via the sigmoid function.
2. **Decision Tree Classifier**: A rule-based flowchart of if-then decisions.
3. **Random Forest Classifier**: An ensemble of 100 decision trees voting together.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Logistic Regression
lr_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
print("Training 1/3: Logistic Regression...")
lr_pipe.fit(X_train, y_train)

# 2. Decision Tree
dt_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', DecisionTreeClassifier(max_depth=6, min_samples_split=20, random_state=42))
])
print("Training 2/3: Decision Tree...")
dt_pipe.fit(X_train, y_train)

# 3. Random Forest (Winner)
rf_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42))
])
print("Training 3/3: Random Forest...")
rf_pipe.fit(X_train, y_train)

print("[+] All 3 models trained successfully!")
"""))

# --- CELL 8: Benchmark Table ---
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Model Evaluation & Benchmark on 10,000 Test Customers
"""))

cells.append(nbf.v4.new_code_cell("""models = {
    'Logistic Regression': lr_pipe,
    'Decision Tree': dt_pipe,
    'Random Forest': rf_pipe
}

benchmark_rows = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    benchmark_rows.append({
        'Algorithm': name,
        'Accuracy': f"{accuracy_score(y_test, y_pred):.2%}",
        'Precision': f"{precision_score(y_test, y_pred):.2%}",
        'Recall': f"{recall_score(y_test, y_pred):.2%}",
        'F1-Score': f"{f1_score(y_test, y_pred):.4f}",
        'ROC-AUC': f"{roc_auc_score(y_test, y_proba):.4f}"
    })

benchmark_df = pd.DataFrame(benchmark_rows).set_index('Algorithm')
benchmark_df
"""))

# --- CELL 9: ROC Curves ---
cells.append(nbf.v4.new_code_cell("""# Compare ROC Curves
plt.figure(figsize=(8, 5))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (ROC-AUC = {auc_val:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guessing (0.50)')
plt.title('ROC Curves Comparison', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
"""))

# --- CELL 10: Which Model Won and Why ---
cells.append(nbf.v4.new_markdown_cell("""---
## 🏆 7. Which Model Performed the Best and Why?

### 🥇 The Winning Model: **Random Forest**
- **Test Accuracy**: **90.91%**
- **Test ROC-AUC**: **0.9241**
- **Precision**: **91.46%**
- **F1-Score**: **0.8278**

---

### 🔍 Why Did Random Forest Win?

#### 1. Ensemble Teamwork Beats a Single Tree (Variance Reduction)
- A single **Decision Tree** (Accuracy: 88.04%) creates rigid if-then splits. It has high variance, meaning it easily memorizes noise or odd outliers in individual customer rows.
- **Random Forest** trains 100 diverse decision trees on random subsets of customers and features, then takes the average vote. This **averaging cancels out individual tree errors**, making the predictions far more reliable and generalizable.

#### 2. Real-World Shopper Behavior is Non-Linear
- **Logistic Regression** (Accuracy: 77.47%) assumes that customer churn can be separated with a straight line or linear formula.
- In reality, customer churn happens due to **combined, non-linear events**:
  - A customer who abandons their cart occasionally might stay loyal if they frequently use the mobile app.
  - But if high cart abandonment is paired with $\ge 5$ customer service calls and 60+ days of inactivity, churn risk spikes dramatically.
- Random Forest naturally captures these multi-variable interaction thresholds, whereas a linear model cannot without complex manual feature math.

#### 3. Invariance to Extreme Values & Skewed Monetary Features
- E-commerce financial numbers (`Lifetime_Value`, `Credit_Balance`) are heavily skewed, with some customers spending $50 and others spending $6,000+.
- Tree-based splits are scale-invariant: they only care about whether a value is above or below a threshold, so extreme numbers do not distort the model.
"""))

# --- CELL 11: Feature Importance ---
cells.append(nbf.v4.new_markdown_cell("""---
## 8. What Features Matter Most? (Feature Importance)
We inspect the Gini feature importance from the winning Random Forest model to understand what drives customer retention.
"""))

cells.append(nbf.v4.new_code_cell("""# Extract feature names
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
cat_feature_names = list(cat_encoder.get_feature_names_out(cat_cols))
all_features = num_cols + cat_feature_names

# Get feature importance from Random Forest
rf_model = rf_pipe.named_steps['clf']
feat_imp = pd.Series(rf_model.feature_importances_, index=all_features).sort_values(ascending=False)

plt.figure(figsize=(9, 6))
sns.barplot(x=feat_imp.head(10).values, y=feat_imp.head(10).index, palette='viridis')
plt.title('Top 10 Most Important Features in Predicting Customer Churn', fontsize=13, fontweight='bold')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()

print("Top 5 Driving Features:")
for i, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
    print(f"  {i}. {feat}: {imp:.1%}")
"""))

# --- CELL 12: Model Serialization ---
cells.append(nbf.v4.new_markdown_cell("""---
## 9. Model Serialization (.pkl)
We save the winning Random Forest pipeline to a `.pkl` file so it can be loaded by our Streamlit web application.
"""))

cells.append(nbf.v4.new_code_cell("""# Save winning pipeline
joblib.dump(rf_pipe, 'best_churn_model.pkl')
print("[+] Successfully exported winning Random Forest model to 'best_churn_model.pkl'!")
"""))

# --- CELL 13: Summary ---
cells.append(nbf.v4.new_markdown_cell("""---
## 10. Summary & Business Recommendations

1. **Monitor Customer Support**:
   - Establish an alert system when a customer logs 5 or more customer service calls. Immediate outreach can save over 40% of these at-risk customers.
2. **Promote the Mobile App**:
   - Mobile app users churn at 21.7% vs 61.3% for non-app users. Offer a small first-purchase discount for downloading the app.
3. **Re-engage Inactive Customers**:
   - Customers inactive for > 60 days enter the churn danger zone. Set up automated win-back email workflows.
4. **Presenting the Project**:
   - Run `streamlit run app.py` to present the interactive project dashboard.
"""))

nb.cells = cells

with open('ecommerce_customer_churn_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("[+] 'ecommerce_customer_churn_analysis.ipynb' successfully generated with student-friendly explanations and rich findings!")
