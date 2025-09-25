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
