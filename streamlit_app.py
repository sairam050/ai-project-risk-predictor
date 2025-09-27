import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from io import BytesIO
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# ================= App Config =================
st.set_page_config(page_title="AI Project Risk & Delay Predictor", layout="wide")

# Load models
risk_model = joblib.load("rf_risk_classifier.joblib")
delay_model = joblib.load("rf_delay_regressor.joblib")

st.sidebar.header("📂 Project Inputs")
planned_duration_days = st.sidebar.number_input("Planned Duration (days)", 30, 1000, 180)
team_size = st.sidebar.number_input("Team Size", 2, 100, 10)
budget_k = st.sidebar.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests = st.sidebar.number_input("Number of Change Requests", 0, 20, 1)
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

# ================= Prediction =================
if st.sidebar.button("🚀 Predict"):
    st.title("📊 AI Project Risk & Delay Predictor")
    st.write("Enter project details, test scenarios, and download polished reports instantly.")

    # ---- Risk Prediction ----
    risk_proba = float(risk_model.predict_proba(input_df)[:, 1][0])
    delay_pred = float(delay_model.predict(input_df)[0])

    if risk_proba > 0.66:
        st.error(f"⚠️ High Risk — {risk_proba:.1%}")
    elif risk_proba > 0.33:
        st.warning(f"🟠 Medium Risk — {risk_proba:.1%}")
    else:
        st.success(f"✅ Low Risk — {risk_proba:.1%}")

    st.metric("Risk Probability", f"{risk_proba:.2%}")
    st.metric("Expected Delay", f"{delay_pred:.1f} days")

    if delay_pred > planned_duration_days * 0.15:
        st.info("Projected delay > 15% of planned duration — consider mitigation actions.")
    else:
        st.info("Projected delay looks within acceptable range.")

    # ---- Scenario Simulation ----
    st.subheader("🔮 Scenario Simulation")
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
        df = pd.DataFrame([vals], columns=input_df.columns)
        p = float(risk_model.predict_proba(df)[:, 1][0])
        d = float(delay_model.predict(df)[0])
        results[label] = (p, d)

    comparison = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T
    comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda x: f"{x:.1%}")
    st.table(comparison)

    # ---- Scenario Chart ----
    st.subheader("📊 Scenario Comparison Chart")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    labels = list(results.keys())
    risk_vals = [results[l][0] for l in labels]
    delay_vals = [results[l][1] for l in labels]

    ax1.bar(labels, risk_vals, color="salmon", alpha=0.7, label="Risk (%)")
    ax1.set_ylabel("Risk Probability (%)", color="red")
    ax2 = ax1.twinx()
    ax2.bar(labels, delay_vals, color="skyblue", alpha=0.6, label="Delay (days)")
    ax2.set_ylabel("Expected Delay (days)", color="blue")
    st.pyplot(fig)

    # Save chart for PDF
    chart_buf = BytesIO()
    fig.savefig(chart_buf, format="png")
    chart_buf.seek(0)

    # ---- SHAP Explainability ----
    st.subheader("🔍 Why did the model predict this?")
    try:
        explainer = shap.TreeExplainer(risk_model)
        shap_values = explainer.shap_values(input_df)
        shap_fig = plt.figure()
        shap.summary_plot(shap_values, input_df, show=False)
        st.pyplot(shap_fig)

        shap_buf = BytesIO()
        shap_fig.savefig(shap_buf, format="png")
        shap_buf.seek(0)
    except Exception:
        shap_buf = None
        st.info("ℹ️ SHAP explanation not available for this model.")

    # ---- PDF Report ----
    st.subheader("📑 Download Report")

    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("📊 AI Project Risk & Delay Predictor — Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Risk Probability: {risk_proba:.1%}", styles["Normal"]))
        story.append(Paragraph(f"Expected Delay: {delay_pred:.1f} days", styles["Normal"]))
        story.append(Spacer(1, 12))

        data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
        for k, v in results.items():
            data.append([k, f"{v[0]:.1%}", f"{v[1]:.1f}"])
        story.append(Table(data))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Thresholds: Low < 33%, Medium 33–66%, High > 66%", styles["Italic"]))

        # Add chart
        story.append(Image(chart_buf, width=400, height=250))

        # Add SHAP if available
        if shap_buf:
            story.append(Paragraph("Feature Importance / SHAP Explanation", styles["Heading2"]))
            story.append(Image(shap_buf, width=400, height=250))

        doc.build(story)
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf()
    st.download_button("📥 Download PDF Report", data=pdf_buffer,
                       file_name="risk_delay_report.pdf", mime="application/pdf")
