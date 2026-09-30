# WorkForce — Workforce Demand Planning & Staffing Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://workforce-demand-planning.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-WorkForce-blue?logo=github)](https://github.com/Shweta-Tech-creator/WorkForce.git)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)

**B.Tech CSE - Semester V | Machine Learning Laboratory Project (Case Study 92)**

An end-to-end Machine Learning system and interactive Streamlit dashboard to estimate future workforce requirements (**Low Demand**, **Medium Demand**, **High Demand**) across organizational departments and job roles.

---

## 🌐 Live Application Link

🚀 **Experience the Live Dashboard:**  
**[https://workforce-demand-planning.streamlit.app](https://workforce-demand-planning.streamlit.app)**

*(Alternative direct deployment link: `https://shweta-tech-creator-workforce.streamlit.app`)*

---

## 📌 Problem Overview
Organizations frequently face operational disruptions caused by unforeseen talent shortages, excessive employee overtime, and unbalanced workload distribution. This project models workforce demand planning as a **multi-class classification problem** using empirical indicators of workload stress, turnover risk, and career progression velocity.

- **Low Demand:** Stable capacity, no immediate hiring needed.
- **Medium Demand:** Standard talent pipeline replenishment and regular backfilling.
- **High Demand:** Critical positions experiencing acute overtime strain and turnover risk requiring urgent recruitment.

---

## 📊 Dataset
- **Dataset:** IBM HR Analytics Employee Attrition & Performance
- **Source:** Kaggle (`pavansubhasht/ibm-hr-analytics-attrition-dataset`)
- **Records:** 1,470 employees $\times$ 35 attributes

---

## 🤖 Models Benchmarked (Strictly 4 Models)
All models were trained on an 80:20 stratified split of the dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier (Selected)** | **90.82%** | **0.9136** | **0.9082** | **0.9064** |
| **Logistic Regression** | **89.12%** | **0.8911** | **0.8912** | **0.8910** |
| **Decision Tree Classifier** | **88.44%** | **0.8833** | **0.8844** | **0.8819** |
| **K-Nearest Neighbors (KNN)** | **59.18%** | **0.5991** | **0.5918** | **0.5732** |

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Shweta-Tech-creator/WorkForce.git
cd WorkForce
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

### 4. Run the Jupyter Notebook
Open and run `Workforce_Demand_Planning.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.

---

## 📂 Repository Structure
```text
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Kaggle HR Dataset
├── Workforce_Demand_Planning.ipynb        # Complete Machine Learning Notebook
├── app.py                                 # Interactive Streamlit Web Dashboard
├── requirements.txt                       # Project Python Dependencies
├── .gitignore                             # Ignored files
└── README.md                              # Project Documentation
```
