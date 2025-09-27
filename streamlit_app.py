# ================== Imports ==================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import datetime
import gdown  # for Google Drive downloads

# PDF (ReportLab)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors


# ================== Page Config ==================
st.set_page_config(page_title="AI Project Risk & Delay Predictor", layout="wide")

st.title("📊 AI Project Risk & Delay Predictor")
st.caption(
    "Enter project details in the left sidebar and click **Predict**. "
    "You’ll get risk & delay estimates, scenario comparisons, and a polished PDF you can download."
)


# ================== Download & Load Models ==================
@st.cache_resource
def load_models():
    """Fetch models from Google Drive and load them."""
    risk_url = "https://drive.google.com/uc?id=1EKFSP7P1Tx57NiKGm8G3CA5Viecm-g-f"
    delay_url = "https://drive.google.com/uc?id=1QHdEDIldTQV-ZfoikVXGIF8URT1eE71-"

    risk_file = "rf_risk_classifier.joblib"
    delay_file = "rf_delay_regressor.joblib"

    gdown.download(risk_url, risk_file, quiet=False)
    gdown.download(delay_url, delay_file, quiet=False)

    risk_model = joblib.load(risk_file)
    delay_model = joblib.load(delay_file)

    return risk_model, delay_model


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_pdf(results, candidate_name="Your Name / Org", logo_path=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Header
    story.append(Paragraph("📊 AI Project Risk & Delay Predictor — Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Prepared for:</b> {candidate_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated on:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 16))

    # Summary
    story.append(Paragraph("<b>🔹 Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Risk Probability:</b> {results['risk_proba']:.1%}", styles["Normal"]))
    story.append(Paragraph(f"<b>Expected Delay:</b> {results['delay_pred']:.1f} days", styles["Normal"]))
    story.append(Spacer(1, 16))

    # Scenario Table
    story.append(Paragraph("<b>📊 Scenario Comparison</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
    row_styles = []
    for i, (label, (prob, delay)) in enumerate(results["results_map"].items(), start=1):
        if prob < 0.33: bg = colors.lightgreen
        elif prob < 0.66: bg = colors.lightyellow
        else: bg = colors.salmon
        data.append([label, f"{prob:.1%}", f"{delay:.1f}"])
        row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))
    tab = Table(data, colWidths=[150, 120, 150])
    tab.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
    ] + row_styles))
    story.append(tab)
    story.append(Spacer(1, 18))

    # Chart
    story.append(Paragraph("<b>📈 Scenario Comparison Chart</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    try:
        story.append(Image(BytesIO(results["chart_png"]), width=400, height=250))
    except Exception:
        story.append(Paragraph("Chart unavailable.", styles["Italic"]))
    story.append(Spacer(1, 18))

    story.append(Paragraph("<i>Thresholds: Low < 33%, Medium 33–66%, High > 66%</i>", styles["Italic"]))
    doc.build(story)
    buffer.seek(0)
    return buffer


# ================== Load Models ==================
try:
    risk_model, delay_model = load_models()
except Exception as e:
    st.error(f"❌ Could not load models. Error: {e}")
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

input_df = pd.DataFrame([[planned_duration_days, team_size, budget_k, num_change_requests,
                          pct_resource_util, complexity_score, onshore_pct]],
                        columns=["planned_duration_days", "team_size", "budget_k", "num_change_requests",
                                 "pct_resource_util", "complexity_score", "onshore_pct"])


# ================== Predict & Render ==================
clicked = st.sidebar.button("🚀 Predict")

if clicked or ("__last__" in st.session_state):

    if clicked:
        risk_proba = float(risk_model.predict_proba(input_df)[:, 1][0])
        delay_pred = float(delay_model.predict(input_df)[0])

        # Scenarios
        scenarios = {
            "Base Case": [planned_duration_days, team_size, budget_k, num_change_requests,
                          pct_resource_util, complexity_score, onshore_pct],
            "Optimistic": [planned_duration_days * 0.90, team_size + 2, budget_k * 1.2,
                           max(0, num_change_requests - 1), pct_resource_util * 0.90,
                           complexity_score * 0.80, min(1.0, onshore_pct + 0.10)],
            "Pessimistic": [planned_duration_days * 1.20, max(2, team_size - 2), budget_k * 0.80,
                            num_change_requests + 2, pct_resource_util * 1.10,
                            min(1.0, complexity_score * 1.20), max(0.0, onshore_pct - 0.10)]
        }

        results_map = {}
        for label, vals in scenarios.items():
            df = pd.DataFrame([vals], columns=input_df.columns)
            p = float(risk_model.predict_proba(df)[:, 1][0])
            d = float(delay_model.predict(df)[0])
            results_map[label] = (p, d)

        # Chart
        fig_chart, ax1 = plt.subplots(figsize=(7, 4))
        labels = list(results_map.keys())
        risk_vals = [results_map[k][0] for k in labels]
        delay_vals = [results_map[k][1] for k in labels]
        ax1.bar(labels, risk_vals, color="salmon", alpha=0.75)
        ax1.set_ylabel("Risk Probability", color="red")
        ax2 = ax1.twinx()
        ax2.plot(labels, delay_vals, marker="o", color="blue")
        ax2.set_ylabel("Expected Delay (days)", color="blue")
        ax1.set_title("Scenario Comparison: Risk vs Delay")
        chart_png = fig_to_png_bytes(fig_chart)

        st.session_state["__last__"] = {
            "risk_proba": risk_proba,
            "delay_pred": delay_pred,
            "results_map": results_map,
            "chart_png": chart_png
        }

    R = st.session_state["__last__"]

    # Risk categories (fixed thresholds)
    if R["risk_proba"] > 0.66:
        st.error(f"⚠️ High risk — {R['risk_proba']:.1%}")
    elif R["risk_proba"] > 0.33:
        st.warning(f"🟠 Medium risk — {R['risk_proba']:.1%}")
    else:
        st.success(f"✅ Low risk — {R['risk_proba']:.1%}")

    # Metrics
    c1, c2 = st.columns(2)
    with c1: st.metric("Risk Probability", f"{R['risk_proba']:.2%}")
    with c2: st.metric("Expected Delay", f"{R['delay_pred']:.1f} days")

    # Chart
    st.subheader("📈 Scenario Comparison Chart")
    st.image(BytesIO(R["chart_png"]), use_container_width=True)

    # PDF download
    st.subheader("📑 Download Report")
    pdf_buf = generate_pdf(R, candidate_name=candidate_name, logo_path=None)
    st.download_button("⬇️ Download PDF Report", data=pdf_buf, file_name="risk_delay_report.pdf", mime="application/pdf")

else:
    st.info("Adjust inputs on the left and click **Predict** to generate results.")
