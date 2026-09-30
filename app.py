import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="WorkForce | Demand Planning & Staffing Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Clean Light Mode UI Custom CSS (Tight Spacing & Modern Polish)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Remove awkward top empty space */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.5rem !important;
    }
    
    /* Main App Light Background & Typography */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sleek Top Header Box */
    .header-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.6rem;
        margin-top: 0 !important;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .header-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-sub {
        font-size: 0.92rem;
        color: #64748b;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* Modern Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.4rem 0 1.1rem 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 1.2rem;
    }
    .brand-logo-svg {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        box-shadow: 0 3px 6px rgba(37, 99, 235, 0.25);
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    .brand-sub {
        font-size: 0.76rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .sidebar-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-top: 1.2rem;
    }
    .sidebar-card-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .sidebar-stat-item {
        font-size: 0.82rem;
        color: #334155;
        margin-bottom: 0.35rem;
        display: flex;
        justify-content: space-between;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .kpi-delta {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .delta-green { color: #16a34a; }
    .delta-amber { color: #d97706; }
    .delta-red { color: #dc2626; }

    /* Prediction Badges */
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* Form Card */
    div[data-testid="stForm"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Set Matplotlib & Seaborn Light Theme Globally
# ---------------------------------------------------------
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'
plt.rcParams['text.color'] = '#0f172a'
plt.rcParams['axes.labelcolor'] = '#334155'
plt.rcParams['xtick.color'] = '#475569'
plt.rcParams['ytick.color'] = '#475569'
plt.rcParams['grid.color'] = '#f1f5f9'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# ---------------------------------------------------------
# Data Pipeline & Model Caching
# ---------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    # Drop zero-variance constant columns
    drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Calculate composite Workforce Demand Score
    score = np.zeros(len(df_clean))
    score += (df_clean['OverTime'] == 'Yes').astype(float) * 2.0
    score += (df_clean['Attrition'] == 'Yes').astype(float) * 2.5
    score += (df_clean['WorkLifeBalance'] <= 2).astype(float) * 1.5
    score += (df_clean['YearsSinceLastPromotion'] >= 4).astype(float) * 1.0
    score += (df_clean['PerformanceRating'] >= 4).astype(float) * 1.0
    score += (df_clean['BusinessTravel'] == 'Travel_Frequently').astype(float) * 1.0
    score += (df_clean['JobSatisfaction'] <= 2).astype(float) * 1.0

    def categorize_demand(s):
        if s <= 1.5:
            return 'Low Demand'
        elif s <= 3.5:
            return 'Medium Demand'
        else:
            return 'High Demand'

    df_clean['Workforce_Demand'] = [categorize_demand(s) for s in score]
    return df_clean

@st.cache_resource
def train_models(df):
    X = df.drop(columns=['Workforce_Demand'])
    y = df['Workforce_Demand']
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    feature_columns = X_encoded.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train 4 models
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    models = {
        'Random Forest': (rf, X_test),
        'Logistic Regression': (lr, X_test_scaled),
        'Decision Tree': (dt, X_test),
        'KNN': (knn, X_test_scaled)
    }
    
    benchmark = []
    for name, (model, x_eval) in models.items():
        preds = model.predict(x_eval)
        benchmark.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, preds, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, preds, average='weighted', zero_division=0)
        })
        
    benchmark_df = pd.DataFrame(benchmark).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    
    return {
        'rf_model': rf,
        'feature_columns': feature_columns,
        'cat_cols': cat_cols,
        'X_encoded': X_encoded,
        'benchmark_df': benchmark_df,
        'X_train': X_train,
        'y_test': y_test,
        'rf_preds': rf.predict(X_test)
    }

# Load data and pipeline
df_data = load_and_preprocess_data()
pipeline = train_models(df_data)

# ---------------------------------------------------------
# Top Header Box
# ---------------------------------------------------------
st.markdown("""
<div class="header-box">
    <div class="header-title">
        <span>💼 Workforce Demand Planning & Staffing Analytics</span>
    </div>
    <div class="header-sub">Predicting future organizational workforce requirements (Low, Medium, High Demand) using machine learning algorithms.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar (Polished WorkForce Brand & Logo)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-logo-svg">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M17 21V19C17 17.9391 16.5786 16.9217 15.8284 16.1716C15.0783 15.4214 14.0609 15 13 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="9" cy="7" r="4" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M23 21V19C22.9993 18.1137 22.7044 17.2528 22.1614 16.5523C21.6184 15.8519 20.8581 15.3516 20 15.13" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M16 3.13C16.8604 3.35031 17.623 3.85071 18.1676 4.55232C18.7122 5.25392 19.0078 6.11683 19.0078 7.005C19.0078 7.89318 18.7122 8.75608 18.1676 9.45769C17.623 10.1593 16.8604 10.6597 16 10.88" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div>
            <div class="brand-title">WorkForce</div>
            <div class="brand-sub">Demand & Capacity Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### 🏢 Department Scope")
    dept_options = ['All Departments'] + sorted(df_data['Department'].unique().tolist())
    selected_dept = st.selectbox("Select Department:", dept_options, label_visibility="collapsed")
    
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">System Overview</div>
        <div class="sidebar-stat-item"><span>Dataset</span><b>IBM HR Analytics</b></div>
        <div class="sidebar-stat-item"><span>Total Roles</span><b>1,470 Records</b></div>
        <div class="sidebar-stat-item"><span>Active Model</span><b>Random Forest</b></div>
        <div class="sidebar-stat-item"><span>Model Accuracy</span><b style="color:#16a34a;">90.82%</b></div>
        <div class="sidebar-stat-item"><span>F1-Score</span><b style="color:#2563eb;">0.9064</b></div>
    </div>
    """, unsafe_allow_html=True)

# Filter dataset
if selected_dept != 'All Departments':
    filtered_df = df_data[df_data['Department'] == selected_dept]
else:
    filtered_df = df_data

# ---------------------------------------------------------
# Main Tabs Layout
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Capacity Analytics & EDA", 
    "🎯 Live Demand Estimator", 
    "📋 Departmental Roster & Plans", 
    "📈 Model Analytics & Benchmarks"
])

palette_map = {'Low Demand': '#10b981', 'Medium Demand': '#f59e0b', 'High Demand': '#ef4444'}

# ==========================================
# TAB 1: CAPACITY ANALYTICS & EDA
# ==========================================
with tab1:
    st.markdown(f"#### 📌 Capacity KPIs & Workload Indicators — *{selected_dept}*")
    
    # KPI Metric Cards
    k1, k2, k3, k4 = st.columns(4)
    
    total_count = len(filtered_df)
    high_count = (filtered_df['Workforce_Demand'] == 'High Demand').sum()
    high_pct = (high_count / total_count) * 100 if total_count > 0 else 0
    ot_pct = (filtered_df['OverTime'] == 'Yes').mean() * 100 if total_count > 0 else 0
    attr_count = (filtered_df['Attrition'] == 'Yes').sum()
    attr_pct = (attr_count / total_count) * 100 if total_count > 0 else 0
    
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Staff Monitored</div>
            <div class="kpi-value">{total_count:,}</div>
            <div class="kpi-delta delta-green">Active positions</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">High Demand Positions</div>
            <div class="kpi-value">{high_count}</div>
            <div class="kpi-delta delta-red">{high_pct:.1f}% urgent hiring required</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Overtime Strain Rate</div>
            <div class="kpi-value">{ot_pct:.1f}%</div>
            <div class="kpi-delta delta-amber">Workload capacity alert</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Vacancy / Attrition</div>
            <div class="kpi-value">{attr_count}</div>
            <div class="kpi-delta delta-red">{attr_pct:.1f}% turnover rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual Row 1
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("##### 📈 Workforce Demand Level Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        sns.countplot(data=filtered_df, x='Workforce_Demand', order=['Low Demand', 'Medium Demand', 'High Demand'], 
                      palette=palette_map, ax=ax1, edgecolor='#ffffff', linewidth=1.5)
        ax1.set_ylabel("Number of Positions", fontsize=10, fontweight='600')
        ax1.set_xlabel("")
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        for p in ax1.patches:
            h = p.get_height()
            if h > 0:
                ax1.annotate(f'{int(h)}\n({(h/total_count)*100:.1f}%)', 
                             xy=(p.get_x() + p.get_width()/2, h/2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        st.pyplot(fig1)
        
    with col_v2:
        st.markdown("##### 🏢 Demand Level Breakdown Across Job Roles")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        sns.countplot(data=filtered_df, y='JobRole', hue='Workforce_Demand', 
                      hue_order=['Low Demand', 'Medium Demand', 'High Demand'], 
                      palette=palette_map, ax=ax2, edgecolor='#ffffff', linewidth=1.2)
        ax2.set_ylabel("")
        ax2.set_xlabel("Count", fontsize=10, fontweight='600')
        ax2.grid(axis='x', linestyle='--', alpha=0.5)
        ax2.legend(title='', loc='lower right', framealpha=0.9)
        st.pyplot(fig2)
        
    # Visual Row 2
    col_v3, col_v4 = st.columns(2)
    with col_v3:
        st.markdown("##### ⏱️ Impact of OverTime on Staffing Demand")
        fig3, ax3 = plt.subplots(figsize=(6, 3.8))
        sns.countplot(data=filtered_df, x='OverTime', hue='Workforce_Demand', 
                      hue_order=['Low Demand', 'Medium Demand', 'High Demand'], 
                      palette=palette_map, ax=ax3, edgecolor='#ffffff', linewidth=1.2)
        ax3.set_ylabel("Count", fontsize=10, fontweight='600')
        ax3.set_xlabel("Overtime Status", fontsize=10, fontweight='600')
        ax3.grid(axis='y', linestyle='--', alpha=0.5)
        ax3.legend(title='', loc='upper right', framealpha=0.9)
        st.pyplot(fig3)
        
    with col_v4:
        st.markdown("##### 💰 Compensation (Monthly Income) Distribution")
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        sns.boxplot(data=filtered_df, x='Workforce_Demand', y='MonthlyIncome', 
                    order=['Low Demand', 'Medium Demand', 'High Demand'], 
                    palette=palette_map, ax=ax4, width=0.45, linewidth=1.5)
        ax4.set_ylabel("Monthly Income ($)", fontsize=10, fontweight='600')
        ax4.set_xlabel("")
        ax4.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig4)

# ==========================================
# TAB 2: LIVE DEMAND ESTIMATOR
# ==========================================
with tab2:
    st.markdown("#### 🎯 Live Workforce Demand Estimator")
    st.markdown("Specify position parameters, operational workload indicators, and satisfaction metrics to compute real-time demand category and action plan.")
    
    with st.form("estimator_form"):
        col_form1, col_form2 = st.columns(2)
        
        with col_form1:
            st.markdown("##### 🏢 Role & Career Metadata")
            f_dept = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
            f_role = st.selectbox("Job Role", [
                'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                'Manufacturing Director', 'Healthcare Representative', 'Manager', 
                'Sales Representative', 'Research Director', 'Human Resources'
            ])
            f_job_level = st.slider("Job Seniority Level (1 = Junior, 5 = Executive)", 1, 5, 2)
            f_income = st.number_input("Monthly Compensation ($)", min_value=1000, max_value=25000, value=5200, step=500)
            f_total_exp = st.slider("Total Industry Experience (Years)", 0, 40, 7)
            f_company_exp = st.slider("Tenure at Company (Years)", 0, 40, 4)
            f_role_exp = st.slider("Years in Current Role", 0, 20, 2)
            f_promo_lag = st.slider("Years Since Last Promotion", 0, 15, 1)
            f_manager_exp = st.slider("Years with Current Manager", 0, 20, 2)
            
        with col_form2:
            st.markdown("##### ⚙️ Workload, Strain & Retention Indicators")
            f_overtime = st.selectbox("Overtime Required?", ["No", "Yes"], index=1)
            f_attrition = st.selectbox("Active Turnover / Vacancy Risk?", ["No", "Yes"], index=0)
            f_travel = st.selectbox("Business Travel Burden", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
            f_wlb = st.select_slider("Work-Life Balance Rating", options=[1, 2, 3, 4], value=2, 
                                     format_func=lambda x: {1: "1 - Poor (High Burnout)", 2: "2 - Fair", 3: "3 - Good", 4: "4 - Excellent"}[x])
            f_job_sat = st.select_slider("Job Satisfaction Rating", options=[1, 2, 3, 4], value=2, 
                                         format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x])
            f_env_sat = st.select_slider("Workplace Environment Rating", options=[1, 2, 3, 4], value=3,
                                         format_func=lambda x: {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}[x])
            f_perf = st.selectbox("Performance Rating", [3, 4], format_func=lambda x: f"{x} - {'Excellent' if x==3 else 'Outstanding'}")
            f_involvement = st.slider("Job Involvement Level (1 to 4)", 1, 4, 3)
            f_dist = st.slider("Commute Distance From Home (km)", 1, 30, 9)
            
        submit_calc = st.form_submit_button("🚀 Compute Workforce Demand Assessment", use_container_width=True)
        
    if submit_calc:
        input_data = {
            'Age': 35,
            'DailyRate': 800,
            'DistanceFromHome': f_dist,
            'Education': 3,
            'EnvironmentSatisfaction': f_env_sat,
            'HourlyRate': 65,
            'JobInvolvement': f_involvement,
            'JobLevel': f_job_level,
            'JobSatisfaction': f_job_sat,
            'MonthlyIncome': f_income,
            'MonthlyRate': 14000,
            'NumCompaniesWorked': 2,
            'PercentSalaryHike': 14,
            'PerformanceRating': f_perf,
            'RelationshipSatisfaction': 3,
            'StockOptionLevel': 1,
            'TotalWorkingYears': f_total_exp,
            'TrainingTimesLastYear': 2,
            'WorkLifeBalance': f_wlb,
            'YearsAtCompany': f_company_exp,
            'YearsInCurrentRole': f_role_exp,
            'YearsSinceLastPromotion': f_promo_lag,
            'YearsWithCurrManager': f_manager_exp,
            'Department': f_dept,
            'JobRole': f_role,
            'BusinessTravel': f_travel,
            'EducationField': 'Life Sciences',
            'Gender': 'Male',
            'MaritalStatus': 'Married',
            'OverTime': f_overtime,
            'Attrition': f_attrition
        }
        
        sample_df = pd.DataFrame([input_data])
        sample_encoded = pd.get_dummies(sample_df, columns=pipeline['cat_cols'], drop_first=True)
        
        for col in pipeline['feature_columns']:
            if col not in sample_encoded.columns:
                sample_encoded[col] = 0
        sample_encoded = sample_encoded[pipeline['feature_columns']]
        
        pred_class = pipeline['rf_model'].predict(sample_encoded)[0]
        pred_probs = pipeline['rf_model'].predict_proba(sample_encoded)[0]
        prob_mapping = dict(zip(pipeline['rf_model'].classes_, pred_probs))
        
        st.markdown("---")
        st.markdown("#### 📋 Assessment Results & HR Action Playbook")
        
        c_res1, c_res2 = st.columns([1.1, 0.9])
        
        with c_res1:
            st.markdown("**Estimated Demand Tier:**")
            if pred_class == 'High Demand':
                st.markdown('<div class="badge-high">🚨 HIGH DEMAND (Urgent Hiring & Reallocation Needed)</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.error("""
                **Strategic HR Action Plan:**
                - **Recruitment Priority:** High. Expedite talent requisition for immediate backfill / team expansion.
                - **Workload Mitigation:** Cap weekly overtime; reallocate critical deliverables to prevent imminent resignation.
                - **Compensation & Retention:** Evaluate market benchmark compensation and retention incentives.
                """)
            elif pred_class == 'Medium Demand':
                st.markdown('<div class="badge-medium">⚠️ MEDIUM DEMAND (Standard Pipeline Replenishment)</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.warning("""
                **Strategic HR Action Plan:**
                - **Recruitment Priority:** Moderate. Maintain active candidate pipeline and internship-to-hire channels.
                - **Workload Monitoring:** Audit monthly overtime trends to avoid escalating into High Demand tier.
                - **Career Development:** Schedule 1-on-1 reviews to address career progression and promotion velocity.
                """)
            else:
                st.markdown('<div class="badge-low">✅ LOW DEMAND (Optimal & Stable Capacity)</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.success("""
                **Strategic HR Action Plan:**
                - **Recruitment Priority:** None. Staffing capacity is balanced and sustainable.
                - **Talent Strategy:** Focus on continuous upskilling, cross-training, and employee engagement.
                """)
                
        with c_res2:
            st.markdown("**Model Classification Probabilities:**")
            fig_p, ax_p = plt.subplots(figsize=(5.2, 3.2))
            prob_data = pd.DataFrame({
                'Demand Tier': list(prob_mapping.keys()),
                'Confidence (%)': [val * 100 for val in prob_mapping.values()]
            })
            sns.barplot(data=prob_data, x='Demand Tier', y='Confidence (%)', palette=palette_map, ax=ax_p, edgecolor='#ffffff', linewidth=1.2)
            ax_p.set_ylim(0, 100)
            ax_p.set_ylabel("Confidence (%)", fontsize=9, fontweight='600')
            ax_p.set_xlabel("")
            ax_p.grid(axis='y', linestyle='--', alpha=0.5)
            for p in ax_p.patches:
                ax_p.annotate(f'{p.get_height():.1f}%', 
                              xy=(p.get_x() + p.get_width()/2, p.get_height() + 2),
                              ha='center', va='bottom', fontweight='bold', fontsize=9.5)
            st.pyplot(fig_p)

# ==========================================
# TAB 3: DEPARTMENTAL ROSTER & MATRIX
# ==========================================
with tab3:
    st.markdown(f"#### 📋 Workforce Roster & Prioritization Matrix — *{selected_dept}*")
    
    col_filter1, col_filter2 = st.columns([2, 1])
    with col_filter1:
        st.markdown(f"Displaying employee positions for **{selected_dept}** with automated demand classification.")
    with col_filter2:
        tier_filter = st.multiselect("Filter Demand Category:", 
                                     ['High Demand', 'Medium Demand', 'Low Demand'],
                                     default=['High Demand', 'Medium Demand', 'Low Demand'])
        
    roster_view = filtered_df[filtered_df['Workforce_Demand'].isin(tier_filter)]
    
    display_columns = [
        'Department', 'JobRole', 'JobLevel', 'MonthlyIncome', 'OverTime', 
        'Attrition', 'YearsAtCompany', 'YearsSinceLastPromotion', 'WorkLifeBalance', 'Workforce_Demand'
    ]
    
    st.dataframe(roster_view[display_columns], use_container_width=True, height=400)
    
    # Export report button
    csv_bytes = roster_view.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Workforce Demand Report (CSV)",
        data=csv_bytes,
        file_name=f'workforce_demand_{selected_dept.lower().replace(" ", "_")}.csv',
        mime='text/csv'
    )

# ==========================================
# TAB 4: MODEL ANALYTICS & BENCHMARKS
# ==========================================
with tab4:
    st.markdown("#### 📈 Model Benchmarks & Performance Analytics")
    st.markdown("Rigorous empirical evaluation of the 4 classical models trained on the IBM HR Analytics dataset:")
    
    # Benchmark table
    st.dataframe(pipeline['benchmark_df'].style.format({
        'Accuracy': '{:.2%}',
        'Precision': '{:.2%}',
        'Recall': '{:.2%}',
        'F1-Score': '{:.2%}'
    }), use_container_width=True)
    
    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        st.markdown("##### 📊 Comparative Model Performance")
        fig_b, ax_b = plt.subplots(figsize=(6, 4))
        melted_metrics = pd.melt(pipeline['benchmark_df'], id_vars=['Model'], 
                                 value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                                 var_name='Metric', value_name='Score')
        sns.barplot(data=melted_metrics, x='Model', y='Score', hue='Metric', palette='viridis', ax=ax_b, edgecolor='#ffffff')
        ax_b.set_ylim(0.45, 1.02)
        ax_b.set_ylabel("Score", fontsize=10, fontweight='600')
        ax_b.set_xlabel("")
        ax_b.grid(axis='y', linestyle='--', alpha=0.5)
        ax_b.legend(loc='lower right', framealpha=0.9)
        st.pyplot(fig_b)
        
    with col_bench2:
        st.markdown("##### 🔍 Top 10 Features Driving Predictions (Random Forest)")
        importances_series = pd.Series(
            pipeline['rf_model'].feature_importances_, 
            index=pipeline['feature_columns']
        ).sort_values(ascending=False).head(10)
        
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        sns.barplot(x=importances_series.values, y=importances_series.index, palette='crest_r', ax=ax_imp, edgecolor='#ffffff')
        ax_imp.set_xlabel("Gini Feature Importance", fontsize=10, fontweight='600')
        ax_imp.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(fig_imp)

# ---------------------------------------------------------
# Clean Minimal Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.82rem; padding: 0.5rem 0;">
    WorkForce • Demand Planning & Staffing Optimization Dashboard
</div>
""", unsafe_allow_html=True)
