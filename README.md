# 🛍️ E-Commerce Customer Behavior & Churn Study

An approachable, clean Data Science project analyzing customer retention across 50,000 online shoppers, discovering key behavioural patterns, and comparing **3 classic machine learning algorithms** with a presentation-ready **Streamlit web application** and **Jupyter Notebook (`.ipynb`)**.

---

## 📌 Project Summary

- **Dataset**: 50,000 customer records, 25 attributes
- **Target Variable**: `Churned` (0 = Stayed, 1 = Left)
- **Baseline Retention Rate**: 71.1% retained, 28.9% churned

### ❓ Problem Formulation: Classification vs. Regression
Before modeling, we evaluated whether to predict total customer spend (**Regression**) or customer attrition (**Classification**):
- **Continuous Regression (`Lifetime_Value`)**: Useful for high-level budgeting ($R^2 \approx 0.48$), but does not identify who is unhappy or when they will leave.
- **Binary Classification (`Churned`)**: **Selected as the primary task** because customer acquisition is **5x to 7x more expensive** than retention. Catching at-risk customers early enables proactive retention campaigns.

---

## 💡 Top 5 Key Discoveries from the Data

1. **🚨 The Customer Support "Tipping Point"**:
   - Customers with 0 to 5 calls have a low churn rate (11%–18%).
   - Once calls reach **6 or more**, the churn rate **doubles to over 40%**!
2. **📱 Mobile App Usage Creates Loyalty**:
   - Customers who use the mobile app < 5 hours/month have a **61.3% churn rate**.
   - Customers using the app > 15 hours/month have only a **21.7% churn rate** (almost 3x lower).
3. **⏰ The Inactivity Danger Zone**:
   - Beyond 90 days without a purchase, almost half of customers (48.9%) leave permanently.
4. **🛒 Cart Abandonment Friction**:
   - Regular cart abandonment strongly signals pricing or shipping friction preceding departure.
5. **🌍 Behavior Matters More than Demographics**:
   - Churn rates across all 8 countries and between genders are virtually identical (~28%–29%), proving that retention is driven by customer experience, not location.

---

## 🤖 3 Classic Algorithms & Scorecard

Evaluated on **10,000 unseen test customers**:

| Algorithm | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Baseline | 77.47% | 67.67% | 42.21% | 0.5199 | 0.7910 |
| **Decision Tree** | Single Rule-Based Tree | 88.04% | 85.56% | 70.52% | 0.7731 | 0.8987 |
| **Random Forest** 🏆 | Ensemble of 100 Trees | **90.91%** | **91.46%** | **75.61%** | **0.8278** | **0.9241** |

---

## 🏆 Which Model Won and Why?

### 🥇 Winner: **Random Forest**
1. **Teamwork Beats a Single Tree (Variance Reduction)**: A single Decision Tree tries to memorize individual customer outliers, leading to high variance. Random Forest averages 100 trees trained on random subsets, canceling out individual errors.
2. **Real Shopper Behavior is Non-Linear**: Logistic Regression uses a flat linear formula. In e-commerce, churn happens when several friction points combine (e.g. cart abandonment + repeated support calls + inactivity). Trees naturally capture these multi-way thresholds.
3. **Resistant to Spending Outliers**: Customer lifetime values range from $50 to $6,000+. Decision trees are scale-invariant and aren't distorted by extreme spending spikes.

---

## 🚀 How to Run and Present

### 1. Launch the Streamlit App (4 Clear Presentation Pages)
```powershell
& "C:\Users\salin\AppData\Local\Programs\Python\Python39\python.exe" -m streamlit run app.py
```
*(Or double-click `run_app.bat`)*

**The 4 App Sections in Sidebar Navigation:**
- 📌 **Executive Overview**: High-level metrics, problem formulation check, methodology.
- 📊 **Exploratory Data Analysis (EDA)**: Full interactive charts for demographics (Country, Gender, Quarter), behavioral boxplots & distributions, correlation rankings, and raw data view.
- 💡 **Data Insights & Findings**: Visual breakdown of the 5 key discoveries (support call tipping point, app loyalty, inactivity danger zone).
- 🤖 **Algorithm Benchmark & Winner**: 3-model comparison scorecard, comparison bar chart, confusion matrix, and "Which Model Won and Why" explanation.

### 2. View the Jupyter Notebook
```powershell
& "C:\Users\salin\AppData\Local\Programs\Python\Python39\python.exe" -m jupyter notebook
```
Open [**`ecommerce_customer_churn_analysis.ipynb`**](file:///c:/Users/salin/Desktop/Zoople%20project/ecommerce_customer_churn_analysis.ipynb).
