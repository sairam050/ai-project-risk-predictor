# streamlit_app.py - simple scaffold for the Risk Predictor demo
# To run locally:
#   pip install -r requirements.txt
#   streamlit run streamlit_app.py
#
# The app loads pre-trained models and exposes simple input controls for demo purposes.

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Project Risk Predictor - Demo", layout="centered")

st.title("AI Project Risk Predictor — Demo (scaffold)")

st.markdown("Adjust project inputs on the left and click Predict. Replace the model files with your trained models.")

# Sidebar inputs
st.sidebar.header("Project inputs")
planned_duration_days = st.sidebar.slider("Planned duration (days)", 30, 720, 120)
team_size = st.sidebar.slider("Team size", 1, 100, 8)
budget_k = st.sidebar.slider("Budget (k$)", 1, 10000, 150)
num_change_requests = st.sidebar.slider("Number of change requests", 0, 50, 2)
pct_resource_util = st.sidebar.slider("Resource utilization (0-1.5)", 0.0, 1.5, 0.85)
complexity_score = st.sidebar.slider("Complexity score (0-1)", 0.0, 1.0, 0.35)
onshore_pct = st.sidebar.slider("Onshore % (0-1)", 0.0, 1.0, 0.7)

# Build input dataframe
input_df = pd.DataFrame([{
    'planned_duration_days': planned_duration_days,
    'team_size': team_size,
    'budget_k': budget_k,
    'num_change_requests': num_change_requests,
    'pct_resource_util': pct_resource_util,
    'complexity_score': complexity_score,
    'onshore_pct': onshore_pct
}])

st.write("Input example:")
st.dataframe(input_df)

# Load models (ensure these joblib files are in same folder)
try:
    clf = joblib.load('rf_risk_classifier.joblib')
    reg = joblib.load('rf_delay_regressor.joblib')
    proba = clf.predict_proba(input_df)[:,1][0]
    delay_pred = reg.predict(input_df)[0]
    st.metric("Predicted risk (probability)", f"{proba:.2f}")
    st.metric("Predicted delay (days)", f"{delay_pred:.1f}")
except Exception as e:
    st.warning("Could not load models. Place rf_risk_classifier.joblib and rf_delay_regressor.joblib in this folder after training.")
    st.write("Error:", e)

st.markdown("""---
This is a scaffold to connect your trained models to an interactive demo. Expand with SHAP plots, scenario sliders, and a 'Run scenario' panel to show mitigation impact.
""")

# ================== Scenario Simulation Panel ==================
st.header("🔮 Scenario Simulation")

st.write("Test how changing certain project factors impacts risk & delay.")

# Inputs for simulation
sim_team_size = st.slider("Simulated Team Size", 2, 100, team_size)
sim_budget = st.slider("Simulated Budget (in $1000s)", 100, 10000, budget_k)
sim_complexity = st.slider("Simulated Complexity Score", 0.0, 1.0, complexity_score)

if st.button("Run Simulation"):
    sim_data = pd.DataFrame([[
        planned_duration_days, sim_team_size, sim_budget, num_change_requests,
        pct_resource_util, sim_complexity, onshore_pct
    ]], columns=[
        "planned_duration_days", "team_size", "budget_k", "num_change_requests",
        "pct_resource_util", "complexity_score", "onshore_pct"
    ])

sim_proba = float(clf.predict_proba(sim_data)[:, 1][0])
sim_delay = float(reg.predict(sim_data)[0])


    st.subheader("📊 Simulation Results")
    st.write(f"**Risk Probability:** {sim_proba:.1%}")
    st.write(f"**Expected Delay:** {sim_delay:.1f} days")

    if sim_delay > planned_duration_days * 0.15:
        st.warning("⚠️ Projected delay exceeds 15% of planned duration.")
    else:
        st.success("✅ Projected delay looks manageable.")

# ========== Step 2: Scenario Simulation Panel ==========
st.sidebar.header("🔮 Scenario Simulation")

if st.sidebar.button("Run Scenario Simulation"):
    st.subheader("📊 Scenario Simulation Results")
    st.write("Here you can compare different 'what-if' cases side by side.")

    # Example scenarios (you can expand later)
    scenarios = {
        "Base Case": [planned_duration_days, team_size, budget_k, num_change_requests,
                      pct_resource_util, complexity_score, onshore_pct],
        "Optimistic": [planned_duration_days*0.9, team_size+2, budget_k*1.2, 
                       max(0, num_change_requests-1), pct_resource_util*0.9,
                       complexity_score*0.8, onshore_pct+0.1],
        "Pessimistic": [planned_duration_days*1.2, max(2, team_size-2), budget_k*0.8,
                        num_change_requests+2, pct_resource_util*1.1,
                        complexity_score*1.2, max(0, onshore_pct-0.1)]
    }

    results = {}
    for label, vals in scenarios.items():
        df = pd.DataFrame([vals], columns=[
            "planned_duration_days", "team_size", "budget_k", "num_change_requests",
            "pct_resource_util", "complexity_score", "onshore_pct"
        ])
risk_proba = float(clf.predict_proba(df)[:, 1][0])
delay_est = float(reg.predict(df)[0])

        results[label] = (risk_proba, delay_est)

    st.write("### Comparison Table")
    comparison = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T
    comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda x: f"{x:.1%}")
    st.dataframe(comparison)

