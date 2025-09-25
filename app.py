import streamlit as st
import pandas as pd
import joblib

# Load trained models
risk_model = joblib.load("rf_risk_classifier.joblib")
delay_model = joblib.load("rf_delay_regressor.joblib")

st.title("📊 AI Project Risk & Delay Predictor")
st.write("Enter project details and predict risk & delays instantly!")

# User inputs
planned_duration_days = st.number_input("Planned Duration (days)", 30, 1000, 180)
team_size = st.number_input("Team Size", 2, 100, 10)
budget_k = st.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests = st.number_input("Number of Change Requests", 0, 20, 1)
pct_resource_util = st.slider("Resource Utilization (%)", 0.1, 2.0, 1.0)
complexity_score = st.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct = st.slider("Onshore %", 0.0, 1.0, 0.5)

# Predict
if st.button("Predict"):
    input_data = pd.DataFrame([[
        planned_duration_days, team_size, budget_k, num_change_requests,
        pct_resource_util, complexity_score, onshore_pct
    ]], columns=["planned_duration_days", "team_size", "budget_k", "num_change_requests",
                 "pct_resource_util", "complexity_score", "onshore_pct"])

    risk_pred = risk_model.predict(input_data)[0]
    delay_pred = delay_model.predict(input_data)[0]

    st.subheader("✅ Prediction Results")
    st.write(f"**Risk Flag:** {'⚠️ Risky' if risk_pred==1 else '✅ Safe'}")
    st.write(f"**Expected Delay:** {delay_pred:.1f} days")
