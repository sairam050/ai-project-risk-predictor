# streamlit_app.py - AI Project Risk & Delay Predictor
# Run locally with:
#   pip install -r requirements.txt
#   streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================== Page Config ==================
st.set_page_config(page_title="AI Project Risk & Delay Predictor", layout="wide")

st.title("📊 AI Project Risk & Delay Predictor- UPDATED")
st.write("Enter project details and predict risk & delays instantly!")

# ================== Load Models ==================
try:
    risk_model = joblib.load("rf_risk_classifier.joblib")
    delay_model = joblib.load("rf_delay_regressor.joblib")
except Exception as e:
    st.error("❌ Could not load models. Ensure .joblib files are in the repo.")
    st.stop()

# ================== Presets ==================
presets = {
    "Select a preset": None,
    "Conservative (Low Risk)": {
        "planned_duration_days": 90, "team_size": 12, "budget_k": 800,
        "num_change_requests": 0, "pct_resource_util": 0.95,
        "complexity_score": 0.2, "onshore_pct": 0.8
    },
    "Typical (Medium Risk)": {
        "planned_duration_days": 180, "team_size": 8, "budget_k": 300,
        "num_change_requests": 2, "pct_resource_util": 1.0,
        "complexity_score": 0.5, "onshore_pct": 0.5
    },
    "Risky (High Risk)": {
        "planned_duration_days": 360, "team_size": 4, "budget_k": 80,
        "num_change_requests": 8, "pct_resource_util": 1.3,
        "complexity_score": 0.9, "onshore_pct": 0.2
    }
}

st.sidebar.header("Presets & Samples")
selected_preset = st.sidebar.selectbox("Choose a preset", list(presets.keys()))

if st.sidebar.button("Load preset"):
    if presets[selected_preset]:
        for k, v in presets[selected_preset].items():
            st.session_state[k] = v
        st.experimental_rerun()

# ================== User Inputs ==================
planned_duration_days = st.number_input("Planned Duration (days)", 30, 1000,
                                        value=st.session_state.get("planned_duration_days", 180))
team_size = st.number_input("Team Size", 2, 100,
                            value=st.session_state.get("team_size", 10))
budget_k = st.number_input("Budget (in $1000s)", 100, 10000,
                           value=st.session_state.get("budget_k", 500))
num_change_requests = st.number_input("Number of Change Requests", 0, 20,
                                      value=st.session_state.get("num_change_requests", 1))
pct_resource_util = st.slider("Resource Utilization (%)", 0.1, 2.0,
                              value=st.session_state.get("pct_resource_util", 1.0))
complexity_score = st.slider("Complexity Score", 0.0, 1.0,
                             value=st.session_state.get("complexity_score", 0.5))
onshore_pct = st.slider("Onshore %", 0.0, 1.0,
                        value=st.session_state.get("onshore_pct", 0.5))

# ================== Prediction ==================
if st.button("Predict"):
    input_data = pd.DataFrame([[
        planned_duration_days, team_size, budget_k, num_change_requests,
        pct_resource_util, complexity_score, onshore_pct
    ]], columns=[
        "planned_duration_days", "team_size", "budget_k", "num_change_requests",
        "pct_resource_util", "complexity_score", "onshore_pct"
    ])

    # Risk prediction
    proba = float(risk_model.predict_proba(input_data)[:, 1][0])
    if proba > 0.66:
        risk_state = ("High risk", "error")
    elif proba > 0.33:
        risk_state = ("Medium risk", "warning")
    else:
        risk_state = ("Low risk", "success")

    # Delay prediction
    delay_pred = float(delay_model.predict(input_data)[0])
    delay_range = None
    if hasattr(delay_model, "estimators_"):  # RandomForest CI
        all_preds = np.array([est.predict(input_data)[0] for est in delay_model.estimators_])
        delay_range = (np.percentile(all_preds, 10), np.percentile(all_preds, 90))

    # Layout: two columns
    col_left, col_right = st.columns([1, 1])

    with col_left:
        label, mode = risk_state
        if mode == "error":
            st.error(f"⚠️ {label} — {proba:.1%}")
        elif mode == "warning":
            st.warning(f"🟠 {label} — {proba:.1%}")
        else:
            st.success(f"✅ {label} — {proba:.1%}")
        st.metric("Risk probability", f"{proba:.2%}")

        # Show top feature contributors
        if hasattr(risk_model, "feature_importances_"):
            importances = risk_model.feature_importances_
            feature_names = input_data.columns
            top_idx = np.argsort(importances)[::-1][:3]
            st.markdown("**Top contributors (model):**")
            for idx in top_idx:
                st.write(f"- **{feature_names[idx]}** ({importances[idx]:.2%})")

    with col_right:
        st.metric("Expected delay (days)", f"{delay_pred:.1f}")
        if delay_range:
            st.write(f"**Delay 80% CI:** {delay_range[0]:.1f} – {delay_range[1]:.1f} days")
        else:
            st.write("**Delay 80% CI:** Not available")

        if delay_pred > planned_duration_days * 0.15:
            st.info("Projected delay > 15% of planned duration — consider mitigation actions.")
        else:
            st.info("Projected delay looks within acceptable range.")

    with st.expander("Why did the model predict this? (quick guide)"):
        st.write("""
        - **Top contributors** above are the most important features for risk.
        - Try scenario testing below to see how changes affect risk/delay.
        - Thresholds: Low < 33%, Medium 33–66%, High > 66%.
        """)

# ================== Scenario Simulation Panel ==================
st.header("🔮 Scenario Simulation")
st.write("Test how changing certain project factors impacts risk & delay.")

# User-controlled sliders for testing scenarios
sim_team_size = st.slider("Simulated Team Size", 2, 100, team_size)
sim_budget = st.slider("Simulated Budget (in $1000s)", 100, 10000, budget_k)
sim_complexity = st.slider("Simulated Complexity Score", 0.0, 1.0, complexity_score)

# Main simulation button
if st.button("Run Simulation"):
    sim_data = pd.DataFrame([[
        planned_duration_days, sim_team_size, sim_budget, num_change_requests,
        pct_resource_util, sim_complexity, onshore_pct
    ]], columns=input_df.columns)   # ✅ FIXED (was input_data)

    sim_proba = float(risk_model.predict_proba(sim_data)[:, 1][0])
    sim_delay = float(delay_model.predict(sim_data)[0])

    st.subheader("📊 Simulation Results")
    st.write(f"**Risk Probability:** {sim_proba:.1%}")
    st.write(f"**Expected Delay:** {sim_delay:.1f} days")

    if sim_delay > planned_duration_days * 0.15:
        st.warning("⚠️ Projected delay exceeds 15% of planned duration.")
    else:
        st.success("✅ Projected delay looks manageable.")

# Sidebar shortcut: prebuilt scenarios
st.sidebar.header("🔮 Quick Scenario Simulation")
if st.sidebar.button("Run Scenario Simulation"):
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
        df = pd.DataFrame([vals], columns=input_df.columns)   # ✅ Correct reference
        risk_proba = float(risk_model.predict_proba(df)[:, 1][0])
        delay_est = float(delay_model.predict(df)[0])
        results[label] = (risk_proba, delay_est)

    st.subheader("📊 Scenario Comparison Table")
    comparison = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T
    comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda x: f"{x:.1%}")
    st.dataframe(comparison)
