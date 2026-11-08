"""
E-Commerce Customer Behavior & Churn Study
A Clean, Student-Friendly Data Science Project
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="E-Commerce Customer Behavior Study",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, approachable styling
st.markdown("""
<style>
    .title-text {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }
    .finding-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: #000000 !important;
    }
    .finding-card * {
        color: #000000 !important;
    }
    .finding-card h4 {
        color: #1e3a8a !important;
    }
    .winner-card {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #22c55e;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model_artifacts():
    path = os.path.join(BASE_DIR, 'model_artifacts.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_raw_dataset():
    path = os.path.join(BASE_DIR, 'archive', 'eCommerce-dataset.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

artifacts = load_model_artifacts()
raw_df = load_raw_dataset()

# Clean Sidebar Navigation
st.sidebar.title("🛍️ Customer Study")
st.sidebar.markdown("**E-Commerce Behavior Analysis**")

page = st.sidebar.radio(
    "Select Project Section:",
    [
        "📌 Executive Overview",
        "📊 Exploratory Data Analysis",
        "💡 Data Insights & Findings",
        "🤖 Algorithm Benchmark & Winner"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Academic Project Scope:**\n- 50,000 Customers\n- 3 ML Algorithms\n- Key Behavioral Findings")

# ==============================================================================
# 1. EXECUTIVE OVERVIEW
# ==============================================================================
if page == "📌 Executive Overview":
    st.markdown('<div class="title-text">E-Commerce Customer Behavior & Churn Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">A study exploring why online shoppers leave and evaluating 3 classic machine learning algorithms</div>', unsafe_allow_html=True)
    
    # 4 Clean Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Dataset Size</div>
            <div class="metric-val">50,000 Customers</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Retained Customers</div>
            <div class="metric-val" style="color: #16a34a;">71.1% (35,550)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Churned Customers</div>
            <div class="metric-val" style="color: #dc2626;">28.9% (14,450)</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-label">Best Model Accuracy</div>
            <div class="metric-val" style="color: #2563eb;">90.9% (Random Forest)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Problem Formulation: Classification vs Regression
    st.markdown("### ❓ Problem Formulation: Classification vs. Regression")
    st.markdown("""
    When beginning this project, we checked two different ways to frame the machine learning problem:
    """)
    
    col_clf, col_reg = st.columns(2)
    with col_clf:
        st.success("#### ✅ Choice 1: Binary Classification (Customer Churn)")
        st.markdown("""
        - **Question:** *Will this customer stay active, or are they about to leave (churn)?*
        - **Target:** `Churned` (0 = Stayed, 1 = Left)
        - **Why we chose this:** In business, preventing a customer from leaving is **5x to 7x cheaper** than acquiring a new customer. Knowing exactly who is about to churn allows the store to send an email or discount before they leave forever.
        """)
    with col_reg:
        st.info("#### ℹ️ Choice 2: Continuous Regression (Customer Lifetime Value)")
        st.markdown("""
        - **Question:** *How much total money will this customer spend in dollars?*
        - **Target:** `Lifetime_Value` (Continuous USD amount)
        - **Why it was secondary:** While predicting sales volume is helpful for budgeting, it does not tell the store *when* or *why* a customer is unhappy. We kept `Lifetime_Value` as a feature to help predict churn instead.
        """)

    st.markdown("---")
    st.markdown("### 🗺️ Project Methodology")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""
        **Step 1: Data Cleaning**
        - Handled missing values with median & mode
        - Checked for unrealistic age values
        - Prepared 23 feature variables
        """)
    with s2:
        st.markdown("""
        **Step 2: Data Exploration**
        - Identified key behavioral trends
        - Examined customer service friction
        - Measured app usage impact
        """)
    with s3:
        st.markdown("""
        **Step 3: Model Training**
        - Logistic Regression
        - Decision Tree
        - Random Forest
        """)
    with s4:
        st.markdown("""
        **Step 4: Model Evaluation**
        - Tested on 10,000 unseen rows
        - Compared accuracy & ROC-AUC
        - Identified the winning algorithm
        """)

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS (RESTORED AS ORIGINAL)
# ==============================================================================
elif page == "📊 Exploratory Data Analysis":
    st.markdown('<div class="title-text">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Investigate demographic breakdowns, behavioral feature distributions, and correlation patterns</div>', unsafe_allow_html=True)
    
    if raw_df is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Target & Demographics", "🕹️ Behavioral Distributions", "🔥 Correlation Analysis", "📋 Raw Data Preview"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                churn_counts = raw_df['Churned'].value_counts().reset_index()
                churn_counts.columns = ['Status', 'Count']
                churn_counts['Label'] = churn_counts['Status'].map({0: 'Retained (71.1%)', 1: 'Churned (28.9%)'})
                fig_pie = px.pie(
                    churn_counts, names='Label', values='Count',
                    title='Target Distribution: Retained vs Churned',
                    color='Label',
                    color_discrete_map={'Retained (71.1%)': '#22c55e', 'Churned (28.9%)': '#ef4444'},
                    hole=0.45
                )
                fig_pie.update_layout(margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col2:
                country_churn = raw_df.groupby('Country')['Churned'].mean().reset_index()
                country_churn['Churn Rate (%)'] = (country_churn['Churned'] * 100).round(1)
                fig_country = px.bar(
                    country_churn.sort_values(by='Churn Rate (%)', ascending=False),
                    x='Country', y='Churn Rate (%)',
                    title='Churn Rate by Country',
                    color='Churn Rate (%)',
                    color_continuous_scale='Reds'
                )
                fig_country.update_layout(margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig_country, use_container_width=True)
                
            col3, col4 = st.columns(2)
            with col3:
                gender_churn = raw_df.groupby('Gender')['Churned'].mean().reset_index()
                gender_churn['Churn Rate (%)'] = (gender_churn['Churned'] * 100).round(1)
                fig_gen = px.bar(
                    gender_churn, x='Gender', y='Churn Rate (%)',
                    title='Churn Rate by Gender',
                    color='Gender',
                    color_discrete_sequence=['#3b82f6', '#ec4899', '#8b5cf6']
                )
                st.plotly_chart(fig_gen, use_container_width=True)
                
            with col4:
                q_churn = raw_df.groupby('Signup_Quarter')['Churned'].mean().reset_index()
                q_churn['Churn Rate (%)'] = (q_churn['Churned'] * 100).round(1)
                fig_q = px.bar(
                    q_churn, x='Signup_Quarter', y='Churn Rate (%)',
                    title='Churn Rate by Signup Quarter',
                    color='Signup_Quarter',
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                st.plotly_chart(fig_q, use_container_width=True)
                
        with tab2:
            st.markdown("#### Behavioral Feature Distributions by Churn Status")
            feature_selected = st.selectbox(
                "Select Feature to Compare:",
                [
                    'Customer_Service_Calls',
                    'Cart_Abandonment_Rate',
                    'Session_Duration_Avg',
                    'Pages_Per_Session',
                    'Email_Open_Rate',
                    'Days_Since_Last_Purchase',
                    'Mobile_App_Usage',
                    'Lifetime_Value'
                ]
            )
            
            sample_plot_df = raw_df.sample(min(5000, len(raw_df)), random_state=42).copy()
            sample_plot_df['Churn Label'] = sample_plot_df['Churned'].map({0: 'Retained', 1: 'Churned'})
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_box = px.box(
                    sample_plot_df, x='Churn Label', y=feature_selected,
                    color='Churn Label',
                    color_discrete_map={'Retained': '#22c55e', 'Churned': '#ef4444'},
                    title=f'{feature_selected} Box Plot'
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
            with col_b2:
                fig_hist = px.histogram(
                    sample_plot_df, x=feature_selected, color='Churn Label',
                    barmode='overlay', nbins=30,
                    color_discrete_map={'Retained': '#22c55e', 'Churned': '#ef4444'},
                    title=f'{feature_selected} Distribution'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
        with tab3:
            st.markdown("#### Feature Correlation with Churn Status")
            numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
            corr_churn = raw_df[numeric_cols].corr()['Churned'].drop('Churned').sort_values().reset_index()
            corr_churn.columns = ['Feature', 'Correlation']
            corr_churn['Direction'] = np.where(corr_churn['Correlation'] > 0, 'Risk Factor (Increases Churn)', 'Protective Factor (Decreases Churn)')
            
            fig_corr = px.bar(
                corr_churn, x='Correlation', y='Feature', orientation='h',
                color='Direction',
                color_discrete_map={
                    'Risk Factor (Increases Churn)': '#ef4444',
                    'Protective Factor (Decreases Churn)': '#22c55e'
                },
                title='Feature Correlations with Customer Churn'
            )
            fig_corr.update_layout(height=650)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with tab4:
            st.markdown("#### Raw Dataset Explorer")
            st.dataframe(raw_df.head(100), use_container_width=True)
    else:
        st.warning("Dataset file could not be found.")

# ==============================================================================
# 3. DATA INSIGHTS & FINDINGS (DEDICATED PAGE WITH 5 FINDINGS)
# ==============================================================================
elif page == "💡 Data Insights & Findings":
    st.markdown('<div class="title-text">Data Insights & Key Behavioral Findings</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">What does the data tell us about why online shoppers leave? Here are the 5 most important discoveries:</div>', unsafe_allow_html=True)
    
    if raw_df is not None:
        # Finding 1
        st.markdown("""
        <div class="finding-card">
            <h4 style="margin: 0; color: #1e3a8a !important;">🚨 Finding #1: The Customer Service "Tipping Point"</h4>
            <p style="margin-top: 0.4rem; color: #000000 !important;">
            Calling customer service is a direct indicator of customer friction. Customers with <strong style="color: #000000 !important;">0 to 5 calls</strong> have a normal churn rate around <strong style="color: #000000 !important;">11% - 18%</strong>. 
            However, once customer service calls reach <strong style="color: #000000 !important;">6 or more</strong>, the churn rate surges to over <strong style="color: #000000 !important;">40% (more than double!)</strong>. 
            <br><em style="color: #000000 !important;">Takeaway: Customers don't mind 1 or 2 calls for minor questions, but repeated unresolved problems drive them away immediately.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        cs_churn = raw_df.groupby('Customer_Service_Calls')['Churned'].agg(['count', 'mean']).reset_index()
        cs_churn = cs_churn[cs_churn['Customer_Service_Calls'] <= 10]
        cs_churn['Churn Rate (%)'] = (cs_churn['mean'] * 100).round(1)
        
        fig_cs = px.bar(
            cs_churn, x='Customer_Service_Calls', y='Churn Rate (%)',
            title="Customer Service Calls vs. Churn Rate (%)",
            color='Churn Rate (%)',
            color_continuous_scale='Reds',
            labels={'Customer_Service_Calls': 'Number of Customer Service Calls', 'Churn Rate (%)': 'Churn Rate (%)'}
        )
        fig_cs.update_layout(height=350, margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_cs, use_container_width=True)
        
        # Finding 2
        st.markdown("""
        <div class="finding-card">
            <h4 style="margin: 0; color: #1e3a8a !important;">📱 Finding #2: Mobile App Users are Far More Loyal</h4>
            <p style="margin-top: 0.4rem; color: #000000 !important;">
            There is a massive relationship between mobile app usage and customer retention:
            </p>
            <ul style="color: #000000 !important; margin-bottom: 0.4rem;">
                <li style="color: #000000 !important;">Customers using the mobile app <strong style="color: #000000 !important;">less than 5 hours/month</strong> have an alarming <strong style="color: #000000 !important;">61.3% churn rate</strong>.</li>
                <li style="color: #000000 !important;">Customers using the app <strong style="color: #000000 !important;">more than 15 hours/month</strong> have only a <strong style="color: #000000 !important;">21.7% churn rate</strong> (almost 3x lower!).</li>
            </ul>
            <p style="color: #000000 !important; margin: 0;">
            <em style="color: #000000 !important;">Takeaway: Mobile app adoption creates a sticky shopping habit. Encouraging users to download the app is one of the best retention strategies.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Finding 3
        st.markdown("""
        <div class="finding-card">
            <h4 style="margin: 0; color: #1e3a8a !important;">⏰ Finding #3: The Inactivity Danger Zone (Days Since Last Purchase)</h4>
            <p style="margin-top: 0.4rem; color: #000000 !important;">
            How long has it been since their last purchase?
            </p>
            <ul style="color: #000000 !important; margin-bottom: 0.4rem;">
                <li style="color: #000000 !important;"><strong style="color: #000000 !important;">0 - 30 days:</strong> Churn rate is <strong style="color: #000000 !important;">25.0%</strong>.</li>
                <li style="color: #000000 !important;"><strong style="color: #000000 !important;">31 - 60 days:</strong> Churn rate increases to <strong style="color: #000000 !important;">32.0%</strong>.</li>
                <li style="color: #000000 !important;"><strong style="color: #000000 !important;">Over 90 days:</strong> Churn rate rises to <strong style="color: #000000 !important;">48.9% (nearly half of them leave)</strong>.</li>
            </ul>
            <p style="color: #000000 !important; margin: 0;">
            <em style="color: #000000 !important;">Takeaway: After 45 to 60 days without a purchase, an automated "We miss you" email with a small discount is urgently needed.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Finding 4 & 5
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("""
            <div class="finding-card">
                <h4 style="margin: 0; color: #1e3a8a !important;">🛒 Finding #4: Cart Abandonment Matters</h4>
                <p style="margin-top: 0.4rem; color: #000000 !important;">
                Customers who regularly add items to their cart but leave without buying have a much higher churn probability. High cart abandonment signals checkout obstacles, unexpected shipping fees, or lost interest.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with f_col2:
            st.markdown("""
            <div class="finding-card">
                <h4 style="margin: 0; color: #1e3a8a !important;">🌍 Finding #5: Behavior Trumps Demographics</h4>
                <p style="margin-top: 0.4rem; color: #000000 !important;">
                Across all 8 countries (USA, UK, Canada, France, Germany, Australia, India, Japan), the churn rate stays almost identical between <strong style="color: #000000 !important;">27% and 30%</strong>. Men and women also churn at almost the same rate (28%-29%). 
                <br><strong style="color: #000000 !important;">Conclusion:</strong> Churn is caused by <em style="color: #000000 !important;">customer experience and friction</em>, not geography or gender!
                </p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Dataset file not found.")

# ==============================================================================
# 4. ALGORITHM BENCHMARK & WINNER
# ==============================================================================
elif page == "🤖 Algorithm Benchmark & Winner":
    st.markdown('<div class="title-text">Machine Learning Algorithm Comparison & Winner</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">We trained 3 classic machine learning algorithms and evaluated them on 10,000 unseen test customers</div>', unsafe_allow_html=True)
    
    # Simple descriptions of the 3 algorithms
    st.markdown("### 📚 The 3 Algorithms Tested")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.info("""
        **1. Logistic Regression**
        - **What it is:** A classic linear model that calculates log-odds probabilities.
        - **Analogy:** Drawing a straight line to separate retained vs. churned customers.
        - **Strength:** Fast and easy to interpret.
        """)
    with a2:
        st.info("""
        **2. Decision Tree**
        - **What it is:** A single rule-based flowchart of if-then questions.
        - **Analogy:** "Is customer service calls > 5? If yes, check inactivity..."
        - **Strength:** Transparent and visual, but can overfit.
        """)
    with a3:
        st.info("""
        **3. Random Forest (Winner)**
        - **What it is:** An ensemble of 100 decision trees voting together.
        - **Analogy:** Asking 100 diverse experts and taking the majority vote.
        - **Strength:** Highly accurate, handles non-linear patterns.
        """)
        
    st.markdown("---")
    
    # WINNER BOX
    st.markdown("""
    <div class="winner-card">
        <h3 style="margin: 0; color: #166534;">🏆 Which Model Won and Why?</h3>
        <h4 style="margin: 0.3rem 0; color: #15803d;">Winner: Random Forest (Accuracy: 90.9% | ROC-AUC: 92.4%)</h4>
        <p style="margin-top: 0.6rem; color: #1e293b; line-height: 1.6;">
        <strong>Why did Random Forest perform better than the others?</strong>
        <br>
        <strong>1. Teamwork Beats a Single Tree (Variance Reduction):</strong> A single Decision Tree achieved 88.0% accuracy because it tries too hard to memorize specific customer outliers. Random Forest trains 100 different trees on random subsets of customers and averages their opinions. This prevents single mistakes and boosts overall accuracy.
        <br>
        <strong>2. Real-World Behavior is Not a Straight Line:</strong> Logistic Regression (77.5% accuracy) assumes a flat, straight boundary. But in e-commerce, churn happens when multiple things go wrong at once (e.g., high cart abandonment + 4 service calls + 60 days inactivity). Random Forest easily detects these multi-variable combinations.
        <br>
        <strong>3. Doesn't Get Confused by Spending Spikes:</strong> Customer spending (Lifetime Value) has huge variations from $50 to $6,000. Decision trees aren't affected by extreme numbers, whereas linear models struggle with extreme values.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison Table
    if artifacts and 'results' in artifacts:
        results = artifacts['results']
        
        st.markdown("### 📊 Performance Scorecard (Tested on 10,000 Customers)")
        
        score_rows = []
        for name, m in results.items():
            if "Default" not in name:
                clean_name = name.replace(" (Tuned)", "")
                score_rows.append({
                    'Algorithm': clean_name,
                    'Accuracy': f"{m['Accuracy']:.1%}",
                    'Precision': f"{m['Precision']:.1%}",
                    'Recall': f"{m['Recall']:.1%}",
                    'F1-Score': f"{m['F1_Score']:.3f}",
                    'ROC-AUC Score': f"{m['ROC_AUC']:.3f}"
                })
                
        score_df = pd.DataFrame(score_rows)
        st.dataframe(score_df.set_index('Algorithm'), use_container_width=True)
        
        # Simple Comparison Bar Chart
        comp_simple = []
        for row in score_rows:
            comp_simple.append({'Algorithm': row['Algorithm'], 'Metric': 'Accuracy', 'Score': float(row['Accuracy'].replace('%', ''))})
            comp_simple.append({'Algorithm': row['Algorithm'], 'Metric': 'ROC-AUC (x100)', 'Score': float(row['ROC-AUC Score']) * 100})
            
        fig_bench = px.bar(
            pd.DataFrame(comp_simple),
            x='Algorithm', y='Score', color='Metric', barmode='group',
            title='Accuracy and ROC-AUC Comparison Across the 3 Models',
            color_discrete_sequence=['#3b82f6', '#10b981']
        )
        fig_bench.update_layout(yaxis=dict(range=[0, 105]), height=400)
        st.plotly_chart(fig_bench, use_container_width=True)
        
        # Simple Confusion Matrix for Winner
        st.markdown("### 🎯 How Well Did the Winning Model Classify Customers?")
        c_left, c_right = st.columns([1, 1.2])
        with c_left:
            rf_res = results.get('Random Forest (Tuned)', results.get('Random Forest', {}))
            if 'Confusion_Matrix' in rf_res:
                cm = np.array(rf_res['Confusion_Matrix'])
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Model Predicted", y="Actual Customer State", color="Customers"),
                    x=['Stayed', 'Left'],
                    y=['Stayed', 'Left'],
                    color_continuous_scale='Greens',
                    title='Random Forest Confusion Matrix'
                )
                fig_cm.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig_cm, use_container_width=True)
        with c_right:
            st.markdown("""
            #### Interpreting the Confusion Matrix:
            - **True Negatives (Top Left):** Successfully identified customers who stayed.
            - **True Positives (Bottom Right):** Successfully caught customers who were going to churn.
            - **False Alarms (Top Right):** Only a tiny fraction of active customers were misidentified.
            - **Conclusion:** Random Forest achieved over **90% accuracy**, demonstrating that customer attrition can be accurately predicted using behavioral signals.
            """)
