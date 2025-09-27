# ================== Imports ==================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import datetime

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors


# ================== Load Models ==================
@st.cache_resource
def load_models():
    risk_model = joblib.load("rf_risk_classifier.joblib")
    delay_model = joblib.load("rf_delay_regressor.joblib")
    return risk_model, delay_model

risk_model, delay_model = load_models()


# ================== Streamlit UI ==================
st.set_page_config(page_title="AI Project Risk Predictor", layout="wide")
st.title("📊 AI Project Risk & Delay Predictor")
st.write("Enter project details and predict risk & delays instantly!")


# ================== Inputs ==================
planned_duration_days = st.number_input("Planned Duration (days)", 30, 1000, 180)
team_size = st.number_input("Team Size", 2, 100, 10)
budget_k = st.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests = st.number_input("Number of Change Requests", 0, 20, 1)
pct_resource_util = st.slider("Resource Utilization (%)", 0.1, 2.0, 1.0)
complexity_score = st.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct = st.slider("Onshore %", 0.0, 1.0, 0.5)

input_df = pd.DataFrame([[
    planned_duration_days, team_size, budget_k, num_change_requests,
    pct_resource_util, complexity_score, onshore_pct
]], columns=[
    "planned_duration_days", "team_size", "budget_k", "num_change_requests",
    "pct_resource_util", "complexity_score", "onshore_pct"
])


# ================== Prediction ==================
if st.button("🚀 Predict"):
    risk_proba = float(risk_model.predict_proba(input_df)[:, 1][0])
    delay_pred = float(delay_model.predict(input_df)[0])

    # Save results in session state
    st.session_state.last_results = {
        "risk_proba": risk_proba,
        "delay_pred": delay_pred,
        "input_df": input_df
    }

    # Display prediction
    if risk_proba > 0.66:
        st.error(f"⚠️ High risk — {risk_proba:.1%}")
    elif risk_proba > 0.33:
        st.warning(f"🟠 Medium risk — {risk_proba:.1%}")
    else:
        st.success(f"✅ Low risk — {risk_proba:.1%}")

    st.metric("Expected Delay", f"{delay_pred:.1f} days")


# ================== Scenario Simulation ==================
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
    rp = float(risk_model.predict_proba(df)[:, 1][0])
    dl = float(delay_model.predict(df)[0])
    results[label] = (rp, dl)

comparison = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T
comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda x: f"{x:.1%}")
st.dataframe(comparison)

# Save to session state for PDF
if "last_results" in st.session_state:
    st.session_state.last_results["comparison"] = comparison


# ================== Charts ==================
st.subheader("📈 Scenario Comparison Chart")
fig, ax = plt.subplots()
x = comparison.index
y1 = [float(val.strip('%')) for val in comparison["Risk Probability"]]
y2 = comparison["Expected Delay (days)"]

ax.bar(x, y1, color="salmon", label="Risk Probability (%)")
ax.set_ylabel("Risk Probability (%)", color="salmon")
ax2 = ax.twinx()
ax2.plot(x, y2, marker="o", color="blue", label="Expected Delay (days)")
ax2.set_ylabel("Expected Delay (days)", color="blue")

st.pyplot(fig)

# Save chart for PDF
buf = BytesIO()
fig.savefig(buf, format="png")
buf.seek(0)
chart_buf = buf


# ================== SHAP Explainability ==================
st.subheader("🔎 Why did the model predict this?")
try:
    explainer = shap.TreeExplainer(risk_model)
    shap_values = explainer(input_df)
    fig2, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False, max_display=7)
    st.pyplot(fig2)

    # Save SHAP for PDF
    shap_buf = BytesIO()
    fig2.savefig(shap_buf, format="png")
    shap_buf.seek(0)
except Exception:
    st.warning("⚠️ SHAP explanation unavailable.")
    shap_buf = None


# ================== PDF Report ==================
def generate_pdf(results, chart_img, shap_img=None, candidate="Sairam Thonuunuri", logo_path=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Header
    story.append(Paragraph("📊 AI Project Risk & Delay Predictor — Report", styles['Title']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Prepared for:</b> {candidate}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated on:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 16))

    # Summary
    story.append(Paragraph("<b>🔹 Summary</b>", styles['Heading2']))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Risk Probability:</b> {results['risk_proba']:.1%}", styles['Normal']))
    story.append(Paragraph(f"<b>Expected Delay:</b> {results['delay_pred']:.1f} days", styles['Normal']))
    story.append(Spacer(1, 16))

    # Scenario Table
    story.append(Paragraph("<b>📊 Scenario Comparison</b>", styles['Heading2']))
    data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
    table_styles = []
    for i, (label, row) in enumerate(results["comparison"].iterrows(), start=1):
        risk_val = float(row["Risk Probability"].strip('%')) / 100
        delay_val = f"{row['Expected Delay (days)']:.1f}"
        if risk_val < 0.33:
            bg_color = colors.lightgreen
        elif risk_val < 0.66:
            bg_color = colors.lightyellow
        else:
            bg_color = colors.salmon
        data.append([label, row["Risk Probability"], delay_val])
        table_styles.append(('BACKGROUND', (0, i), (-1, i), bg_color))

    table = Table(data, colWidths=[150, 120, 150])
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.6, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ] + table_styles))
    story.append(table)
    story.append(Spacer(1, 20))

    # Scenario Chart
    story.append(Paragraph("<b>📈 Scenario Comparison Chart</b>", styles['Heading2']))
    story.append(Image(chart_img, width=400, height=250))
    story.append(Spacer(1, 20))

    # SHAP Chart
    if shap_img:
        story.append(Paragraph("<b>🔎 Feature Importance (SHAP)</b>", styles['Heading2']))
        story.append(Image(shap_img, width=400, height=250))
        story.append(Spacer(1, 20))

    # Footer
    story.append(Paragraph("<i>Thresholds: Low < 33%, Medium 33–66%, High > 66%</i>", styles['Italic']))
    story.append(Spacer(1, 8))
    story.append(Paragraph("© 2025 Project Risk AI — Demo Report", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ================== Sidebar PDF Download ==================
st.sidebar.subheader("📑 Download Report")
if "last_results" in st.session_state and "comparison" in st.session_state.last_results:
    pdf_buffer = generate_pdf(st.session_state.last_results, chart_buf, shap_buf)
    st.sidebar.download_button(
        "⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name="risk_delay_report.pdf",
        mime="application/pdf"
    )
else:
    st.sidebar.info("Run a prediction first to enable PDF download.")
