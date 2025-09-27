# ================== Imports ==================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import datetime
import shap
import os
import gdown

# PDF (ReportLab)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# ================== Download Models ==================
RISK_URL = "https://drive.google.com/uc?id=1EKFSP7P1Tx57NiKGm8G3CA5Viecm-g-f"
DELAY_URL = "https://drive.google.com/uc?id=1QHdEDIldTQV-ZfoikVXGIF8URT1eE71-"

if not os.path.exists("rf_risk_classifier.joblib"):
    gdown.download(RISK_URL, "rf_risk_classifier.joblib", quiet=False)

if not os.path.exists("rf_delay_regressor.joblib"):
    gdown.download(DELAY_URL, "rf_delay_regressor.joblib", quiet=False)

# ================== Page Config ==================
st.set_page_config(page_title="AI Project Risk & Delay Predictor", layout="wide")

st.title("📊 AI Project Risk & Delay Predictor")
st.caption(
    "Enter project details in the left sidebar and click **Predict**. "
    "You’ll get risk & delay estimates, scenario comparisons, an explanation (if available), "
    "and a polished PDF you can download."
)

# ================== Helpers ==================
@st.cache_resource
def load_models():
    """Load models once and cache them for faster reruns."""
    risk_model = joblib.load("rf_risk_classifier.joblib")
    delay_model = joblib.load("rf_delay_regressor.joblib")
    return risk_model, delay_model


def make_shap_figure(model, X):
    """Return a SHAP bar plot or None if not supported."""
    try:
        explainer = shap.TreeExplainer(model)
        fig = plt.figure(figsize=(6, 4))
        explanation = explainer(X)
        shap.plots.bar(explanation, show=False, max_display=7)
        return fig
    except Exception:
        return None


def fig_to_png_bytes(fig):
    """Save Matplotlib figure to PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_pdf(results, candidate_name="Your Name / Org"):
    """Generate polished PDF with summary and scenario comparison."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("📊 AI Project Risk & Delay Predictor — Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Prepared for:</b> {candidate_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated on:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>🔹 Summary</b>", styles["Heading2"]))
    story.append(Paragraph(f"<b>Risk Probability:</b> {results['risk_proba']:.1%}", styles["Normal"]))
    story.append(Paragraph(f"<b>Expected Delay:</b> {results['delay_pred']:.1f} days", styles["Normal"]))
    story.append(Spacer(1, 16))

    data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
    for k, v in results["results_map"].items():
        data.append([k, f"{v[0]:.1%}", f"{v[1]:.1f}"])
    tab = Table(data, colWidths=[150, 150, 150])
    tab.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ]))
    story.append(tab)
    story.append(Spacer(1, 18))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ================== Load Models ==================
try:
    risk_model, delay_model = load_models()
except Exception:
    st.error("❌ Could not load models. Ensure download links are correct.")
    st.stop()

# ================== Sidebar Inputs ==================
st.sidebar.header("📂 Project Inputs")
candidate_name = st.sidebar.text_input("Name/Company for PDF", value="Sairam Thonuunuri")

planned_duration_days = st.sidebar.number_input("Planned Duration (days)", 30, 1000, 180)
team_size            = st.sidebar.number_input("Team Size", 2, 100, 10)
budget_k             = st.sidebar.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests  = st.sidebar.number_input("Change Requests", 0, 20, 1)
pct_resource_util    = st.sidebar.slider("Resource Utilization (%)", 0.1, 2.0, 1.0)
complexity_score     = st.sidebar.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct          = st.sidebar.slider("Onshore %", 0.0, 1.0, 0.5)

input_df = pd.DataFrame([[
    planned_duration_days, team_size, budget_k, num_change_requests,
    pct_resource_util, complexity_score, onshore_pct
]], columns=[
    "planned_duration_days", "team_size", "budget_k", "num_change_requests",
    "pct_resource_util", "complexity_score", "onshore_pct"
])

# ================== Predict & Render ==================
if st.sidebar.button("🚀 Predict"):
    risk_proba = float(risk_model.predict_proba(input_df)[:, 1][0])
    delay_pred = float(delay_model.predict(input_df)[0])

    scenarios = {
        "Base Case": [planned_duration_days, team_size, budget_k, num_change_requests,
                      pct_resource_util, complexity_score, onshore_pct],
        "Optimistic": [planned_duration_days*0.9, team_size+2, budget_k*1.2,
                       max(0, num_change_requests-1), pct_resource_util*0.9,
                       complexity_score*0.8, min(1.0, onshore_pct+0.1)],
        "Pessimistic": [planned_duration_days*1.2, max(2, team_size-2), budget_k*0.8,
                        num_change_requests+2, pct_resource_util*1.1,
                        min(1.0, complexity_score*1.2), max(0.0, onshore_pct-0.1)]
    }

    results_map = {}
    for label, vals in scenarios.items():
        df = pd.DataFrame([vals], columns=input_df.columns)
        p = float(risk_model.predict_proba(df)[:, 1][0])
        d = float(delay_model.predict(df)[0])
        results_map[label] = (p, d)

    comparison = pd.DataFrame(results_map, index=["Risk Probability", "Expected Delay (days)"]).T
    comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda v: f"{v:.1%}")

    if risk_proba > 0.70:
        st.error(f"⚠️ High risk — {risk_proba:.1%}")
    elif risk_proba > 0.40:
        st.warning(f"🟠 Medium risk — {risk_proba:.1%}")
    else:
        st.success(f"✅ Low risk — {risk_proba:.1%}")

    st.metric("Risk Probability", f"{risk_proba:.2%}")
    st.metric("Expected Delay", f"{delay_pred:.1f} days")

    st.subheader("🔮 Scenario Simulation")
    st.dataframe(comparison)

    st.subheader("📑 Download Report")
    pdf_buf = generate_pdf({
        "risk_proba": risk_proba,
        "delay_pred": delay_pred,
        "results_map": results_map,
    }, candidate_name=candidate_name)
    st.download_button("⬇️ Download PDF Report", data=pdf_buf, file_name="risk_delay_report.pdf", mime="application/pdf")
