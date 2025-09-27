# 📊 AI Project Risk & Delay Predictor with Explainability
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Page config
st.set_page_config(page_title="AI Project Risk & Delay Predictor", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 1100px;
            margin: auto;
        }
        .stMetric {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load trained models
risk_model = joblib.load("rf_risk_classifier.joblib")
delay_model = joblib.load("rf_delay_regressor.joblib")

# Title
st.title("📊 AI Project Risk & Delay Predictor")
st.write("Enter project details, test scenarios, and download reports instantly.")

# Sidebar Inputs
st.sidebar.header("Project Inputs")
planned_duration_days = st.sidebar.number_input("Planned Duration (days)", 30, 1000, 180)
team_size = st.sidebar.number_input("Team Size", 2, 100, 10)
budget_k = st.sidebar.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests = st.sidebar.number_input("Change Requests", 0, 20, 1)
pct_resource_util = st.sidebar.slider("Resource Utilization (%)", 0.1, 2.0, 1.0)
complexity_score = st.sidebar.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct = st.sidebar.slider("Onshore %", 0.0, 1.0, 0.5)

# Build input dataframe
input_df = pd.DataFrame([[
    planned_duration_days, team_size, budget_k, num_change_requests,
    pct_resource_util, complexity_score, onshore_pct
]], columns=[
    "planned_duration_days", "team_size", "budget_k", "num_change_requests",
    "pct_resource_util", "complexity_score", "onshore_pct"
])

# Predictions
if st.sidebar.button("Predict"):
    proba = float(risk_model.predict_proba(input_df)[:, 1][0])
    delay_pred = float(delay_model.predict(input_df)[0])

    # Risk alert
    if proba > 0.66:
        st.error(f"⚠️ High Risk — {proba:.1%}")
    elif proba > 0.33:
        st.warning(f"🟠 Medium Risk — {proba:.1%}")
    else:
        st.success(f"✅ Low Risk — {proba:.1%}")

    # Delay result
    st.metric("Expected Delay", f"{delay_pred:.1f} days")
    if delay_pred > planned_duration_days * 0.15:
        st.info("⚠️ Projected delay > 15% of planned duration — consider mitigation actions.")
    else:
        st.info("✅ Delay looks manageable.")

    st.caption("Thresholds: Low < 33%, Medium 33–66%, High > 66%")

    # ===== Scenario Simulation =====
    st.header("✨ Scenario Simulation")
    st.write("Compare base, optimistic, and pessimistic project assumptions.")

    scenarios = {
        "Base Case": [planned_duration_days, team_size, budget_k, num_change_requests,
                      pct_resource_util, complexity_score, onshore_pct],
        "Optimistic": [planned_duration_days*0.9, team_size+2, budget_k*1.2,
                       max(0, num_change_requests-1), pct_resource_util*0.9,
                       complexity_score*0.8, min(1.0, onshore_pct+0.1)],
        "Pessimistic": [planned_duration_days*1.2, max(2, team_size-2), budget_k*0.8,
                        num_change_requests+2, min(2.0, pct_resource_util*1.1),
                        min(1.0, complexity_score*1.2), max(0, onshore_pct-0.1)]
    }

    results = {}
    for label, vals in scenarios.items():
        df = pd.DataFrame([vals], columns=input_df.columns)
        risk_proba = float(risk_model.predict_proba(df)[:, 1][0])
        delay_est = float(delay_model.predict(df)[0])
        results[label] = {"Risk Probability": f"{risk_proba:.1%}", "Expected Delay (days)": delay_est}

    comparison = pd.DataFrame(results).T
    st.subheader("📋 Scenario Comparison Table")
    st.dataframe(comparison)

    # ===== Chart =====
    st.subheader("📈 Scenario Comparison Chart")
    comparison_numeric = pd.DataFrame(
        {k: [float(v["Risk Probability"].strip('%'))/100, v["Expected Delay (days)"]] for k, v in results.items()},
        index=["Risk Probability", "Expected Delay (days)"]
    ).T

    fig, ax1 = plt.subplots(figsize=(8, 5))
    comparison_numeric["Risk Probability"].plot(kind="bar", ax=ax1, color="tomato", width=0.4, position=0, label="Risk (%)")
    ax2 = ax1.twinx()
    comparison_numeric["Expected Delay (days)"].plot(kind="bar", ax=ax2, color="skyblue", width=0.4, position=1, label="Delay (days)")
    ax1.set_ylabel("Risk Probability (%)", color="tomato")
    ax2.set_ylabel("Expected Delay (days)", color="skyblue")
    ax1.set_xticklabels(comparison_numeric.index, rotation=0)
    fig.suptitle("Scenario Comparison: Risk vs Delay", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    st.pyplot(fig)

# ================== SHAP Explainability ==================
import shap
import matplotlib.pyplot as plt

st.subheader("🔍 Why did the model predict this?")

try:
    # Only works if model supports predict_proba
    explainer = shap.Explainer(risk_model, input_df)
    shap_values = explainer(input_df)

    # Plot top feature importances
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, max_display=7, show=False)
    st.pyplot(fig)

except Exception as e:
    st.info("ℹ️ SHAP explanation not available for this model.")
    st.caption(f"(Debug: {e})")

# ================== PDF Export ==================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

if st.button("📥 Download PDF Report"):
    # Create PDF
    doc = SimpleDocTemplate("report.pdf")
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Project Risk & Delay Predictor — Report", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Risk Probability: {proba:.1%}", styles['Normal']))
    story.append(Paragraph(f"Expected Delay: {delay_pred:.1f} days", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Scenario Comparison", styles['Heading2']))

    data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
    for label, (r, d) in results.items():
        data.append([label, f"{r:.1%}", f"{d:.2f}"])
    table = Table(data)
    story.append(table)

    story.append(Spacer(1, 12))
    story.append(Paragraph("Thresholds: Low < 33%, Medium 33–66%, High > 66%", styles['Italic']))

    doc.build(story)

    with open("report.pdf", "rb") as f:
        st.download_button("⬇️ Save Report", f, file_name="risk_report.pdf")
