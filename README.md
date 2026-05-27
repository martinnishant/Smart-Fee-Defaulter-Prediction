# Smart Fee Defaulter Prediction Model 🚀

An AI-powered machine learning system that predicts students who are likely to delay or default on fee payments using historical payment behavior, financial indicators, and engineered risk features.

Built using Python, XGBoost, Streamlit, and Data Analytics with an end-to-end ML workflow including:
- Synthetic dataset generation
- Feature engineering
- Risk prediction
- Delay forecasting
- Interactive dashboard
- Exploratory Data Analysis (EDA)

---

# 📌 Problem Statement

Educational institutions often struggle with:
- Late fee payments
- Revenue uncertainty
- Manual tracking of risky students
- Lack of predictive financial insights

This project helps schools proactively identify:
- High-risk students
- Expected payment delays
- Fee collection trends
- Class-wise financial risk patterns

---

# 🎯 Project Objectives

✅ Predict potential fee defaulters  
✅ Generate risk scores (0–100)  
✅ Forecast payment delays  
✅ Analyze collection trends  
✅ Visualize financial risk through dashboards  
✅ Support proactive financial decision-making

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| ML Models | XGBoost Classifier & Regressor |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| Notebook Analysis | Jupyter Notebook |
| Model Persistence | Joblib |

---

# 📂 Project Structure

```bash
Smart-Fee-Defaulter-Prediction/
│
├── data/
│   ├── raw/
│   │   └── fee_data.csv
│   │
│   └── processed/
│       └── features.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   └── train_model.py
│
├── dashboard/
│   └── app.py
│
├── outputs/
│   ├── risk_report.csv
│   ├── clf_model.pkl
│   ├── reg_model.pkl
│   └── feature_importance.png
│
├── screenshots/
│
├── requirements.txt
└── README.md