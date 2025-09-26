import streamlit as st
import pandas as pd
import joblib

import numpy as np

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
    # build input dataframe
    input_data = pd.DataFrame([[
        planned_duration_days, team_size, budget_k, num_change_requests,
        pct_resource_util, complexity_score, onshore_pct
    ]], columns=[
        "planned_duration_days", "team_size", "budget_k", "num_change_requests",
        "pct_resource_util", "complexity_score", "onshore_pct"
    ])

    # 1) Risk probability & label
    proba = float(risk_model.predict_proba(input_data)[:, 1][0])
    if proba > 0.66:
        risk_state = ("High risk", "error")     # show red
    elif proba > 0.33:
        risk_state = ("Medium risk", "warning") # show orange
    else:
        risk_state = ("Low risk", "success")    # show green

    # 2) Delay prediction + ensemble CI (if available)
    delay_pred = float(delay_model.predict(input_data)[0])
    delay_range = None
    try:
        # If the regressor is an ensemble (RandomForest), get per-estimator preds
        if hasattr(delay_model, "estimators_"):
            all_preds = np.array([est.predict(input_data)[0] for est in delay_model.estimators_])
            lower = np.percentile(all_preds, 10)
            upper = np.percentile(all_preds, 90)
            delay_range = (lower, upper)
    except Exception:
        delay_range = None

    # 3) Nice layout: two columns for quick comparison
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # colored summary box
        label, mode = risk_state
        if mode == "error":
            st.error(f"⚠️  {label} — {proba:.1%}")
        elif mode == "warning":
            st.warning(f"🟠  {label} — {proba:.1%}")
        else:
            st.success(f"✅  {label} — {proba:.1%}")

        # a clean metric
        st.metric("Risk probability", f"{proba:.2%}")

        # top 3 feature contributors (quick feature_importances fallback)
        try:
            feature_names = [
                "planned_duration_days", "team_size", "budget_k",
                "num_change_requests", "pct_resource_util", "complexity_score", "onshore_pct"
            ]
            if hasattr(risk_model, "feature_importances_"):
                importances = np.array(risk_model.feature_importances_)
                top_idx = np.argsort(importances)[::-1][:3]
                st.markdown("**Top contributors (model):**")
                for idx in top_idx:
                    st.write(f"- **{feature_names[idx]}** ({importances[idx]:.2%})")
        except Exception:
            # if something goes wrong, fail silently
            pass

    with col_right:
        # Delay result & CI
        st.metric("Expected delay (days)", f"{delay_pred:.1f}")
        if delay_range is not None:
            st.write(f"**Delay 80% CI:** {delay_range[0]:.1f} – {delay_range[1]:.1f} days")
        else:
            st.write("**Delay 80% CI:** Not available")

        # quick interpretation text
        if delay_pred > planned_duration_days * 0.15:
            st.info("Projected delay > 15% of planned duration — consider mitigation actions.")
        else:
            st.info("Projected delay looks within acceptable range.")

    # 4) Optional: collapsible explainability / tips
    with st.expander("Why did the model predict this? (quick guide)"):
        st.write("""
        - **Top contributors** above are the model's most important features for risk.
        - For deeper explanations (SHAP plots), we can add interactive charts or precomputed images.
        - Use the Scenario panel (coming next) to test mitigation actions and see ROI.
        """)

    # 5) Small footnote with thresholds
    st.caption("Thresholds: Low < 33%, Medium 33–66%, High > 66% (adjustable).")

    
