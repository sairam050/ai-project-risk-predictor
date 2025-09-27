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


# ================== Page Config ==================
st.set_page_config(page_title="AI Project Risk & Delay Predictor", layout="wide")
st.title("📊 AI Project Risk & Delay Predictor")
st.caption(
    "Enter project details in the left sidebar and click **Predict**. "
    "You’ll get risk & delay estimates, scenario comparisons, explanations (if available), "
    "and a polished PDF you can download."
)


# ================== Download Models from Google Drive ==================
def download_models():
    files = {
        "rf_risk_classifier.joblib": "1cVk3B8PEVFZMBWBfnwgUNtotWtWe-7eP",
        "rf_delay_regressor.joblib": "1rMO-fgZX2SRuhz1R3RH5w90alAgUQ6Gq",
    }
    for fname, fid in files.items():
        if not os.path.exists(fname):
            url = f"https://drive.google.com/uc?id={fid}"
            st.info(f"⬇️ Downloading {fname} ...")
            gdown.download(url, fname, quiet=False)

download_models()


# ================== Helpers ==================
@st.cache_resource
def load_models():
    """Load models once and cache them."""
    risk_model = joblib.load("rf_risk_classifier.joblib")
    delay_model = joblib.load("rf_delay_regressor.joblib")
    return risk_model, delay_model


def make_shap_figure(model, X):
    """Return SHAP bar plot or None if not supported."""
    try:
        explainer = shap.TreeExplainer(model)
        fig = plt.figure(figsize=(6, 4))
        explanation = explainer(X)
        shap.plots.bar(explanation, show=False, max_display=7)
        return fig
    except Exception:
        return None


def make_importance_figure(model, feature_names):
    """Return feature importance figure if available."""
    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(6, 4))
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values().tail(10)
        ax.barh(imp.index, imp.values)
        ax.set_title("Feature Importance (model)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        return fig
    return None


def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_pdf(results, candidate_name="Your Name / Org", logo_path=None):
    """Generate PDF with summary, scenarios, and charts."""
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
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Risk Probability:</b> {results['risk_proba']:.1%}", styles["Normal"]))
    story.append(Paragraph(f"<b>Expected Delay:</b> {results['delay_pred']:.1f} days", styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>📊 Scenario Comparison</b>", styles["Heading2"]))
    data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
    for label, (prob, delay) in results["results_map"].items():
        data.append([label, f"{prob:.1%}", f"{delay:.1f}"])
    tab = Table(data, colWidths=[150, 120, 150])
    tab.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    story.append(tab)
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>📈 Scenario Chart</b>", styles["Heading2"]))
    try:
        story.append(Image(BytesIO(results["chart_png"]), width=400, height=250))
    except:
        story.append(Paragraph("Chart unavailable.", styles["Italic"]))

    if results.get("shap_png"):
        story.append(Spacer(1, 16))
        story.append(Paragraph("<b>🔎 Feature Importance / SHAP</b>", styles["Heading2"]))
        try:
            story.append(Image(BytesIO(results["shap_png"]), width=400, height=250))
        except:
            story.append(Paragraph("Explanation unavailable.", styles["Italic"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ================== Load Models ==================
try:
    risk_model, delay_model = load_models()
except Exception:
    st.error("❌ Could not load models. Check if Google Drive links are correct.")
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

        comparison = pd.DataFrame(results_map, index=["Risk Probability", "Expected Delay (days)"]).T
        comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda v: f"{v:.1%}")

        # Chart
        fig_chart, ax1 = plt.subplots(figsize=(7, 4))
        labels = list(results_map.keys())
        ax1.bar(labels, [results_map[k][0] for k in labels], color="salmon")
        ax2 = ax1.twinx()
        ax2.plot(labels, [results_map[k][1] for k in labels], marker="o", color="blue")
        chart_png = fig_to_png_bytes(fig_chart)

        shap_png = None
        fig_shap = make_shap_figure(risk_model, input_df)
        if fig_shap:
            shap_png = fig_to_png_bytes(fig_shap)
        else:
            fig_imp = make_importance_figure(risk_model, input_df.columns)
            if fig_imp:
                shap_png = fig_to_png_bytes(fig_imp)

        st.session_state["__last__"] = {
            "risk_proba": risk_proba,
            "delay_pred": delay_pred,
            "comparison": comparison,
            "results_map": results_map,
            "chart_png": chart_png,
            "shap_png": shap_png
        }

    R = st.session_state["__last__"]

   # 🔧 Tweaked thresholds for clearer demo screenshots
if R["risk_proba"] > 0.85:
    st.error(f"⚠️ High risk — {R['risk_proba']:.1%}")
elif R["risk_proba"] > 0.65:
    st.warning(f"🟠 Medium risk — {R['risk_proba']:.1%}")
else:
    st.success(f"✅ Low risk — {R['risk_proba']:.1%}")


    st.metric("Risk Probability", f"{R['risk_proba']:.2%}")
    st.metric("Expected Delay", f"{R['delay_pred']:.1f} days")

    st.subheader("🔮 Scenario Simulation")
    st.dataframe(R["comparison"])

    st.subheader("📈 Scenario Comparison Chart")
    st.image(BytesIO(R["chart_png"]), use_container_width=True)

    st.subheader("🔎 Why did the model predict this?")
    if R["shap_png"]:
        st.image(BytesIO(R["shap_png"]), caption="Top drivers of risk")
    else:
        st.info("No SHAP/feature importance explanation available.")

    st.subheader("📑 Download Report")
    pdf_buf = generate_pdf(R, candidate_name=candidate_name)
    st.download_button("⬇️ Download PDF", data=pdf_buf, file_name="risk_delay_report.pdf", mime="application/pdf")

else:
    st.info("Adjust inputs on the left and click **Predict** to generate results.")
