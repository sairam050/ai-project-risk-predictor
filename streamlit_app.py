# streamlit_app.py - Final Polished AI Project Risk & Delay Predictor

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="AI Project Risk & Delay Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 AI Project Risk & Delay Predictor")
st.caption("Enter project details, test scenarios, and download reports instantly.")

# ========== LOAD MODELS ==========
risk_model = joblib.load("rf_risk_classifier.joblib")
delay_model = joblib.load("rf_delay_regressor.joblib")

# ========== SIDEBAR INPUTS ==========
st.sidebar.header("Project Inputs")

planned_duration_days = st.sidebar.number_input("Planned Duration (days)", 30, 1000, 180)
team_size = st.sidebar.number_input("Team Size", 2, 100, 10)
budget_k = st.sidebar.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests = st.sidebar.number_input("Change Requests", 0, 20, 1)
pct_resource_util = st.sidebar.slider("Resource Utilization (%)", 0.1, 2.0, 1.0)
complexity_score = st.sidebar.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct = st.sidebar.slider("Onshore %", 0.0, 1.0, 0.5)

# Build input DataFrame
input_df = pd.DataFrame([[
    planned_duration_days, team_size, budget_k, num_change_requests,
    pct_resource_util, complexity_score, onshore_pct
]], columns=[
    "planned_duration_days", "team_size", "budget_k", "num_change_requests",
    "pct_resource_util", "complexity_score", "onshore_pct"
])

# ========== MAIN PREDICTION ==========
if st.sidebar.button("🔮 Predict"):
    proba = float(risk_model.predict_proba(input_df)[:, 1][0])
    delay_pred = float(delay_model.predict(input_df)[0])

    # Layout two columns for results
    col1, col2 = st.columns(2)

    with col1:
        if proba > 0.66:
            st.error(f"⚠️ High Risk — {proba:.1%}")
        elif proba > 0.33:
            st.warning(f"🟠 Medium Risk — {proba:.1%}")
        else:
            st.success(f"✅ Low Risk — {proba:.1%}")

        st.metric("Risk Probability", f"{proba:.2%}")

    with col2:
        st.metric("Expected Delay", f"{delay_pred:.1f} days")
        if delay_pred > planned_duration_days * 0.15:
            st.info("Projected delay > 15% of planned duration — consider mitigation actions.")
        else:
            st.success("Projected delay looks within acceptable range.")

    st.caption("Thresholds: Low < 33%, Medium 33–66%, High > 66%")

# ========== SCENARIO SIMULATION ==========
st.header("🔮 Scenario Simulation")
st.write("Compare base, optimistic, and pessimistic project assumptions.")

scenarios = {
    "Base Case": [planned_duration_days, team_size, budget_k, num_change_requests,
                  pct_resource_util, complexity_score, onshore_pct],
    "Optimistic": [planned_duration_days*0.9, team_size+2, budget_k*1.2,
                   max(0, num_change_requests-1), pct_resource_util*0.9,
                   complexity_score*0.8, min(1.0, onshore_pct+0.1)],
    "Pessimistic": [planned_duration_days*1.2, max(2, team_size-2), budget_k*0.8,
                    num_change_requests+2, pct_resource_util*1.1,
                    min(1.0, complexity_score*1.2), max(0, onshore_pct-0.1)]
}

results = {}
for label, vals in scenarios.items():
    df = pd.DataFrame([vals], columns=input_df.columns)
    risk_proba = float(risk_model.predict_proba(df)[:, 1][0])
    delay_est = float(delay_model.predict(df)[0])
    results[label] = (risk_proba, delay_est)

# Table
comparison = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T
comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda x: f"{x:.1%}")
st.subheader("📊 Scenario Comparison Table")
st.dataframe(comparison)

# Chart
st.subheader("📈 Scenario Comparison Chart")
comparison_numeric = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T

fig, ax1 = plt.subplots(figsize=(8, 5))
comparison_numeric["Risk Probability"].plot(kind="bar", ax=ax1, color="tomato", position=0, width=0.4, label="Risk (%)")
ax2 = ax1.twinx()
comparison_numeric["Expected Delay (days)"].plot(kind="bar", ax=ax2, color="skyblue", position=1, width=0.4, label="Delay (days)")

ax1.set_ylabel("Risk Probability (%)", color="tomato")
ax2.set_ylabel("Expected Delay (days)", color="skyblue")
ax1.set_xticklabels(c_
