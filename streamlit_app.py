# ================== Imports ==================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import datetime
import shap

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
    "You’ll get risk & delay estimates, scenario comparisons, explainability (if available), "
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
    """Return a SHAP plot as matplotlib figure or None."""
    try:
        explainer = shap.TreeExplainer(model)
        fig = plt.figure(figsize=(6, 4))
        try:
            explanation = explainer(X)  # newer SHAP
            shap.plots.bar(explanation, show=False, max_display=7)
        except Exception:
            shap_values = explainer.shap_values(X)
            shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=7)
        return fig
    except Exception:
        return None


def make_importance_figure(model, feature_names):
    """Return a feature importance figure if available, else None."""
    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots(figsize=(6, 4))
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values().tail(10)
        ax.barh(imp.index, imp.values)
        ax.set_title("Feature Importance (model)")
        plt.tight_layout()
        return fig
    return None


def fig_to_png_bytes(fig):
    """Convert matplotlib fig to PNG bytes."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_pdf(results, candidate_name="Your Name / Org", logo_path=None):
    """Generate a polished PDF report with scenario table, chart, and optional SHAP/importance."""
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
    story.append(Paragraph(f"<b>Risk Probability:</b> {results['risk_proba']:.1%}", styles["Normal"]))
    story.append(Paragraph(f"<b>Expected Delay:</b> {results['delay_pred']:.1f} days", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Scenario Table
    story.append(Paragraph("<b>📊 Scenario Comparison</b>", styles["Heading2"]))
    data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
    row_styles = []
    for i, (label, (prob, delay)) in enumerate(results["results_map"].items(), start=1):
        if prob < 0.33:
            bg = colors.lightgreen
        elif prob < 0.66:
            bg = colors.lightyellow
        else:
            bg = colors.salmon
        data.append([label, f"{prob:.1%}", f"{delay:.1f}"])
        row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))
    table = Table(data, colWidths=[150, 120, 150])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ] + row_styles))
    story.append(table)
    story.append(Spacer(1, 12))

    # Scenario Chart
    story.append(Paragraph("<b>📈 Scenario Comparison Chart</b>", styles["Heading2"]))
    if results.get("chart_png"):
        story.append(Image(BytesIO(results["chart_png"]), width=400, height=250))
    else:
        story.append(Paragraph("ℹ️ Chart unavailable", styles["Italic"]))
    story.append(Spacer(1, 12))

    # Explainability
    story.append(Paragraph("<b>🔎 Feature Importance / SHAP</b>", styles["Heading2"]))
    if results.get("shap_png"):
        try:
            story.append(Image(BytesIO(results["shap_png"]), width=400, height=250))
        except Exception:
            story.append(Paragraph("⚠️ Explanation image could not be rendered.", styles["Italic"]))
    else:
        story.append(Paragraph("ℹ️ Explainability not available (SHAP/feature importances unavailable).", styles["Italic"]))
    story.append(Spacer(1, 12))

    # Footer
    story.append(Paragraph("<i>Thresholds: Low < 33%, Medium 33–66%, High > 66%</i>", styles["Italic"]))
    story.append(Paragraph("© 2025 Project Risk AI — Demo Report", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ================== Load Models ==================
try:
    risk_model, delay_model = load_models()
except Exception:
    st.error("❌ Models not found. Ensure joblib files are in app folder.")
    st.stop()


# ================== Sidebar ==================
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
                        columns=["planned_duration_days", "team_size", "budget_k",
                                 "num_change_requests", "pct_resource_util",
                                 "complexity_score", "onshore_pct"])


# ================== Predict & Render ==================
if st.sidebar.button("🚀 Predict") or ("__last__" in st.session_state):

    if "__last__" not in st.session_state:
        # Predictions
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

        # Comparison DF
        comparison = pd.DataFrame(results_map, index=["Risk Probability", "Expected Delay (days)"]).T
        comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda v: f"{v:.1%}")

        # Chart
        fig_chart, ax1 = plt.subplots(figsize=(7, 4))
        labels = list(results_map.keys())
        ax1.bar(labels, [results_map[k][0] for k in labels], color="salmon")
        ax1.set_ylabel("Risk Probability", color="red")
        ax2 = ax1.twinx()
        ax2.plot(labels, [results_map[k][1] for k in labels], marker="o", color="blue")
        ax2.set_ylabel("Expected Delay (days)", color="blue")
        ax1.set_title("Scenario Comparison: Risk vs Delay")
        chart_png = fig_to_png_bytes(fig_chart)

        # Explainability
        shap_png = None
        fig_shap = make_shap_figure(risk_model, input_df)
        if fig_shap is not None:
            shap_png = fig_to_png_bytes(fig_shap)
        else:
            fig_imp = make_importance_figure(risk_model, input_df.columns)
            if fig_imp is not None:
                shap_png = fig_to_png_bytes(fig_imp)

        st.session_state["__last__"] = dict(
            risk_proba=risk_proba,
            delay_pred=delay_pred,
            results_map=results_map,
            comparison=comparison,
            chart_png=chart_png,
            shap_png=shap_png
        )

    R = st.session_state["__last__"]

    # Risk banner
    if R["risk_proba"] > 0.66:
        st.error(f"⚠️ High risk — {R['risk_proba']:.1%}")
    elif R["risk_proba"] > 0.33:
        st.warning(f"🟠 Medium risk — {R['risk_proba']:.1%}")
    else:
        st.success(f"✅ Low risk — {R['risk_proba']:.1%}")

    # Metrics
    st.metric("Risk Probability", f"{R['risk_proba']:.2%}")
    st.metric("Expected Delay", f"{R['delay_pred']:.1f} days")

    # Scenario table & chart
    st.subheader("🔮 Scenario Simulation")
    st.dataframe(R["comparison"])
    st.image(BytesIO(R["chart_png"]), use_container_width=True)

    # Explainability
    st.subheader("🔎 Why did the model predict this?")
    if R["shap_png"]:
        st.image(BytesIO(R["shap_png"]), caption="Top drivers of risk")
    else:
        st.info("ℹ️ Explainability not available (SHAP/feature importances unavailable).")

    # PDF download
    pdf_buf = generate_pdf(R, candidate_name=candidate_name)
    st.download_button("⬇️ Download PDF Report", data=pdf_buf,
                       file_name="risk_delay_report.pdf", mime="application/pdf")

else:
    st.info("Adjust inputs on the left and click **Predict** to generate results.")
