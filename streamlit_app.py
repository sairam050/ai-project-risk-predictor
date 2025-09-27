# streamlit_app.py – Final Polished AI Project Risk & Delay Predictor

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Project Risk & Delay Predictor",
    layout="wide",
    page_icon="📊"
)

# ================== LOAD MODELS ==================
@st.cache_resource
def load_models():
    risk_model = joblib.load("rf_risk_classifier.joblib")
    delay_model = joblib.load("rf_delay_regressor.joblib")
    return risk_model, delay_model

try:
    risk_model, delay_model = load_models()
except Exception as e:
    st.error("❌ Could not load models. Make sure `.joblib` files are in the repo.")
    st.stop()

# ================== HEADER ==================
st.title("📊 AI Project Risk & Delay Predictor")
st.markdown("Enter project details, test scenarios, and download polished reports instantly.")

# ================== INPUTS ==================
st.sidebar.header("📂 Project Inputs")

planned_duration_days = st.sidebar.number_input("Planned Duration (days)", 30, 1000, 180)
team_size = st.sidebar.number_input("Team Size", 2, 100, 10)
budget_k = st.sidebar.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests = st.sidebar.number_input("Change Requests", 0, 20, 1)
pct_resource_util = st.sidebar.slider("Resource Utilization (%)", 0.1, 2.0, 1.0)
complexity_score = st.sidebar.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct = st.sidebar.slider("Onshore %", 0.0, 1.0, 0.5)

# Build DataFrame
input_df = pd.DataFrame([[
    planned_duration_days, team_size, budget_k, num_change_requests,
    pct_resource_util, complexity_score, onshore_pct
]], columns=[
    "planned_duration_days", "team_size", "budget_k", "num_change_requests",
    "pct_resource_util", "complexity_score", "onshore_pct"
])

# ================== PREDICTION ==================
if st.sidebar.button("🚀 Predict"):
    proba = float(risk_model.predict_proba(input_df)[:, 1][0])
    delay_pred = float(delay_model.predict(input_df)[0])

    # Risk status
    if proba > 0.66:
        st.error(f"⚠️ High Risk — {proba:.1%}")
    elif proba > 0.33:
        st.warning(f"🟠 Medium Risk — {proba:.1%}")
    else:
        st.success(f"✅ Low Risk — {proba:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Probability", f"{proba:.1%}")
    with col2:
        st.metric("Expected Delay (days)", f"{delay_pred:.1f}")

    if delay_pred > planned_duration_days * 0.15:
        st.info("📌 Projected delay > 15% of planned duration — consider mitigation actions.")

    st.caption("Thresholds: Low < 33%, Medium 33–66%, High > 66%")

    # ================== FEATURE IMPORTANCE ==================
    st.subheader("🔍 Why did the model predict this?")
    try:
        feature_names = input_df.columns
        importances = risk_model.feature_importances_
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(fi["Feature"], fi["Importance"], color="orange")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance (Risk Model)")
        st.pyplot(fig)
    except Exception:
        st.info("ℹ️ Feature importance not available for this model.")

# ================== SCENARIO SIMULATION ==================
st.subheader("🔮 Scenario Simulation")
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

comparison = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T
comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda x: f"{x:.1%}")
st.table(comparison)

# ================== VISUALIZATION ==================
st.subheader("📊 Scenario Comparison Chart")
fig, ax1 = plt.subplots(figsize=(7, 4))

labels = list(results.keys())
risk_vals = [results[l][0] for l in labels]
delay_vals = [results[l][1] for l in labels]

ax1.bar(labels, risk_vals, color="tomato", alpha=0.7, label="Risk (%)")
ax1.set_ylabel("Risk Probability (%)", color="tomato")
ax1.set_ylim(0, 1)

ax2 = ax1.twinx()
ax2.bar(labels, delay_vals, color="skyblue", alpha=0.7, label="Delay (days)")
ax2.set_ylabel("Expected Delay (days)", color="skyblue")

st.pyplot(fig)

# ================== PDF REPORT ==================
st.subheader("📄 Download Report")
def create_pdf_report(proba, delay_pred, comparison):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("📊 AI Project Risk & Delay Predictor — Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Risk Probability: {proba:.1%}", styles['Normal']))
    story.append(Paragraph(f"Expected Delay: {delay_pred:.1f} days", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Scenario Comparison", styles['Heading2']))
    table_data = [["Scenario", "Risk Probability", "Expected Delay (days)"]] + comparison.reset_index().values.tolist()
    story.append(Table(table_data))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Thresholds: Low < 33%, Medium 33–66%, High > 66%", styles['Italic']))

    doc.build(story)
    buffer.seek(0)
    return buffer

if st.sidebar.button("⬇️ Download PDF Report"):
    proba = float(risk_model.predict_proba(input_df)[:, 1][0])
    delay_pred = float(delay_model.predict(input_df)[0])
    pdf_buffer = create_pdf_report(proba, delay_pred, comparison)
    st.download_button("📥 Download PDF", data=pdf_buffer, file_name="risk_delay_report.pdf", mime="application/pdf")
