# ================== Imports ==================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import datetime
import shap
import gdown
import os

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
    "You’ll get risk & delay estimates, scenario comparisons, an explanation (if available), "
    "and a polished PDF you can download."
)


# ================== Helpers ==================
def download_if_missing(url, output):
    """Download from Google Drive if file is missing."""
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)


@st.cache_resource
def load_models():
    """Download models if missing and load them once (cached)."""
    # Google Drive direct links (update with your own IDs)
    risk_url  = "https://drive.google.com/uc?id=1EKFSP7P1Tx57NiKGm8G3CA5Viecm-g-f"
    delay_url = "https://drive.google.com/uc?id=1QHdEDIldTQV-ZfoikVXGIF8URT1eE71-"

    download_if_missing(risk_url, "rf_risk_classifier.joblib")
    download_if_missing(delay_url, "rf_delay_regressor.joblib")

    risk_model = joblib.load("rf_risk_classifier.joblib")
    delay_model = joblib.load("rf_delay_regressor.joblib")

    return risk_model, delay_model


def make_shap_figure(model, X):
    """Return a SHAP bar plot or None if not supported."""
    try:
        explainer = shap.TreeExplainer(model)
        fig = plt.figure(figsize=(6, 4))
        try:
            explanation = explainer(X)
            shap.plots.bar(explanation, show=False, max_display=7)
        except Exception:
            shap_values = explainer.shap_values(X)
            vals = np.abs(shap_values).mean(axis=0)
            order = np.argsort(vals)[::-1][:7]
            ax = plt.gca()
            ax.barh(np.array(X.columns)[order][::-1], vals[order][::-1])
            ax.set_title("Feature Importance (approx.)")
            ax.set_xlabel("Mean |SHAP value|")
            plt.tight_layout()
        return fig
    except Exception:
        return None


def make_importance_figure(model, feature_names):
    """Return feature importance figure if available, else None."""
    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(6, 4))
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(10)
        ax.barh(imp.index, imp.values)
        ax.set_title("Feature Importance (model)")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        return fig
    return None


def fig_to_png_bytes(fig):
    """Save Matplotlib figure to PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_pdf(results, candidate_name="Your Name / Org", logo_path=None):
    """Generate polished PDF with summary, scenario table, charts, and optional SHAP."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    if logo_path:
        try:
            story.append(Image(logo_path, width=80, height=80))
            story.append(Spacer(1, 10))
        except Exception:
            pass

    story.append(Paragraph("📊 AI Project Risk & Delay Predictor — Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Prepared for:</b> {candidate_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Generated on:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>🔹 Summary</b>", styles["Heading2"]))
    story.append(Paragraph(f"<b>Risk Probability:</b> {results['risk_proba']:.1%}", styles["Normal"]))
    story.append(Paragraph(f"<b>Expected Delay:</b> {results['delay_pred']:.1f} days", styles["Normal"]))
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>📊 Scenario Comparison</b>", styles["Heading2"]))
    data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
    row_styles = []
    for i, (label, (prob, delay)) in enumerate(results["results_map"].items(), start=1):
        if prob < 0.40: bg = colors.lightgreen
        elif prob < 0.75: bg = colors.lightyellow
        else: bg = colors.salmon
        data.append([label, f"{prob:.1%}", f"{delay:.1f}"])
        row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))
    tab = Table(data, colWidths=[150, 120, 150])
    tab.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ] + row_styles))
    story.append(tab)
    story.append(Spacer(1, 16))

    story.append(Paragraph("<b>📈 Scenario Comparison Chart</b>", styles["Heading2"]))
    try:
        story.append(Image(BytesIO(results["chart_png"]), width=400, height=250))
    except Exception:
        story.append(Paragraph("Chart unavailable.", styles["Italic"]))
    story.append(Spacer(1, 16))

    if results.get("shap_png"):
        story.append(Paragraph("<b>🔎 Feature Importance / SHAP</b>", styles["Heading2"]))
        try:
            story.append(Image(BytesIO(results["shap_png"]), width=400, height=250))
        except Exception:
            story.append(Paragraph("Explanation unavailable.", styles["Italic"]))
        story.append(Spacer(1, 16))

    story.append(Paragraph("<i>Thresholds: Low < 40%, Medium 40–75%, High > 75%</i>", styles["Italic"]))
    story.append(Paragraph("© 2025 Project Risk AI — Demo Report", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ================== Load Models ==================
try:
    risk_model, delay_model = load_models()
    print("✅ Models loaded:", type(risk_model), type(delay_model))
except Exception as e:
    st.error(f"❌ Could not load models: {e}")
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
        "Optimistic": [planned_duration_days * 0.9, team_size + 2, budget_k * 1.2,
                       max(0, num_change_requests - 1), pct_resource_util * 0.9,
                       complexity_score * 0.8, min(1.0, onshore_pct + 0.1)],
        "Pessimistic": [planned_duration_days * 1.2, max(2, team_size - 2), budget_k * 0.8,
                        num_change_requests + 2, pct_resource_util * 1.1,
                        min(1.0, complexity_score * 1.2), max(0.0, onshore_pct - 0.1)]
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
    risk_vals = [results_map[k][0] for k in labels]
    delay_vals = [results_map[k][1] for k in labels]
    ax1.bar(labels, risk_vals, color="salmon", alpha=0.75)
    ax1.set_ylabel("Risk Probability", color="red")
    ax2 = ax1.twinx()
    ax2.plot(labels, delay_vals, marker="o", color="blue")
    ax2.set_ylabel("Expected Delay (days)", color="blue")
    ax1.set_title("Scenario Comparison: Risk vs Delay")
    chart_png = fig_to_png_bytes(fig_chart)

    # SHAP / importance
    shap_png = None
    fig_shap = make_shap_figure(risk_model, input_df)
    if fig_shap is not None:
        shap_png = fig_to_png_bytes(fig_shap)
    else:
        fig_imp = make_importance_figure(risk_model, input_df.columns)
        if fig_imp is not None:
            shap_png = fig_to_png_bytes(fig_imp)

# Banner (Adjusted thresholds for demo clarity)
        # Banner (Adjusted thresholds for demo clarity)
    if risk_proba > 0.85:
        st.error(f"⚠️ High risk — {risk_proba:.1%}")
    elif risk_proba > 0.50:
        st.warning(f"🟠 Medium risk — {risk_proba:.1%}")
    else:
        st.success(f"✅ Low risk — {risk_proba:.1%}")


    # Metrics
    c1, c2 = st.columns(2)
    with c1: st.metric("Risk Probability", f"{risk_proba:.2%}")
    with c2: st.metric("Expected Delay", f"{delay_pred:.1f} days")

    # Results
    st.subheader("🔮 Scenario Simulation")
    st.dataframe(comparison)

    st.subheader("📈 Scenario Comparison Chart")
    st.image(BytesIO(chart_png), use_container_width=True)

    st.subheader("🔎 Why did the model predict this?")
    if shap_png:
        st.image(BytesIO(shap_png), caption="Top drivers of risk")
    else:
        st.info("ℹ️ Explainability not available for this model.")

    st.subheader("📑 Download Report")
    pdf_buf = generate_pdf({
        "risk_proba": risk_proba,
        "delay_pred": delay_pred,
        "results_map": results_map,
        "chart_png": chart_png,
        "shap_png": shap_png,
    }, candidate_name=candidate_name)
    st.download_button("⬇️ Download PDF Report", data=pdf_buf, file_name="risk_delay_report.pdf", mime="application/pdf")
