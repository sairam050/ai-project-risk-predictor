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

# PDF (ReportLab)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# Download helper
try:
    import gdown
except ImportError:
    st.error("gdown module missing. Please add it in requirements.txt: \n\n gdown==5.1.0")
    st.stop()

# ================== Google Drive Model IDs ==================
RISK_MODEL_ID = "1EKFSP7P1Tx57NiKGm8G3CA5Viecm-g-f"
DELAY_MODEL_ID = "1QHdEDIldTQV-ZfoikVXGIF8URT1eE71-"

RISK_MODEL_PATH = "rf_risk_classifier.joblib"
DELAY_MODEL_PATH = "rf_delay_regressor.joblib"


# ================== Download Models ==================
def download_models():
    if not os.path.exists(RISK_MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={RISK_MODEL_ID}", RISK_MODEL_PATH, quiet=False)
    if not os.path.exists(DELAY_MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={DELAY_MODEL_ID}", DELAY_MODEL_PATH, quiet=False)

download_models()


# ================== Page Config ==================
st.set_page_config(page_title="AI Project Risk & Delay Predictor", layout="wide")
st.title("📊 AI Project Risk & Delay Predictor")
st.caption("Enter project details in the left sidebar and click **Predict**. "
           "You’ll get risk & delay estimates, scenario comparisons, explanations, "
           "and a polished PDF report.")


# ================== Helpers ==================
@st.cache_resource
def load_models():
    """Load models once and cache them for faster reruns."""
    risk_model = joblib.load(RISK_MODEL_PATH)
    delay_model = joblib.load(DELAY_MODEL_PATH)
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
    for label, (prob, delay) in results["results_map"].items():
        data.append([label, f"{prob:.1%}", f"{delay:.1f}"])
    tab = Table(data)
    tab.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black)]))
    story.append(tab)

    story.append(Spacer(1, 16))
    story.append(Paragraph("📈 Scenario Comparison Chart", styles["Heading2"]))
    try:
        story.append(Image(BytesIO(results["chart_png"]), width=400, height=250))
    except Exception:
        story.append(Paragraph("Chart unavailable.", styles["Italic"]))

    if results.get("shap_png"):
        story.append(Spacer(1, 16))
        story.append(Paragraph("🔎 Feature Importance / SHAP", styles["Heading2"]))
        try:
            story.append(Image(BytesIO(results["shap_png"]), width=400, height=250))
        except Exception:
            story.append(Paragraph("Explanation unavailable.", styles["Italic"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ================== Load Models ==================
try:
    risk_model, delay_model = load_models()
except Exception as e:
    st.error(f"❌ Could not load models: {e}")
    st.stop()


# ================== Sidebar Inputs ==================
st.sidebar.header("📂 Project Inputs")
candidate_name = st.sidebar.text_input("Name/Company for PDF", value="Sairam Thonuunuri")

planned_duration_days = st.sidebar.number_input("Planned Duration (days)", 30, 1000, 180)
team_size = st.sidebar.number_input("Team Size", 2, 100, 10)
budget_k = st.sidebar.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests = st.sidebar.number_input("Change Requests", 0, 20, 1)
pct_resource_util = st.sidebar.slider("Resource Utilization (%)", 0.1, 2.0, 1.0)
complexity_score = st.sidebar.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct = st.sidebar.slider("Onshore %", 0.0, 1.0, 0.5)

input_df = pd.DataFrame([[planned_duration_days, team_size, budget_k, num_change_requests,
                          pct_resource_util, complexity_score, onshore_pct]],
                        columns=["planned_duration_days", "team_size", "budget_k", "num_change_requests",
                                 "pct_resource_util", "complexity_score", "onshore_pct"])


# ================== Predict & Render ==================
if st.sidebar.button("🚀 Predict") or "__last__" in st.session_state:
    if "__last__" not in st.session_state:
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

        # Chart
        fig, ax1 = plt.subplots(figsize=(6, 4))
        labels = list(results_map.keys())
        risks = [results_map[k][0] for k in labels]
        delays = [results_map[k][1] for k in labels]
        ax1.bar(labels, risks, color="salmon", alpha=0.7)
        ax1.set_ylabel("Risk Probability", color="red")
        ax2 = ax1.twinx()
        ax2.plot(labels, delays, marker="o", color="blue")
        ax2.set_ylabel("Expected Delay (days)", color="blue")
        chart_png = fig_to_png_bytes(fig)

        # SHAP / feature importance
        shap_png = None
        fig_shap = make_shap_figure(risk_model, input_df)
        if fig_shap is not None:
            shap_png = fig_to_png_bytes(fig_shap)
        else:
            fig_imp = make_importance_figure(risk_model, input_df.columns)
            if fig_imp is not None:
                shap_png = fig_to_png_bytes(fig_imp)

        st.session_state["__last__"] = {
            "risk_proba": risk_proba,
            "delay_pred": delay_pred,
            "results_map": results_map,
            "chart_png": chart_png,
            "shap_png": shap_png
        }

    # Display results
    R = st.session_state["__last__"]
    if R["risk_proba"] > 0.7:
        st.error(f"⚠️ High risk — {R['risk_proba']:.1%}")
    elif R["risk_proba"] > 0.45:
        st.warning(f"🟠 Medium risk — {R['risk_proba']:.1%}")
    else:
        st.success(f"✅ Low risk — {R['risk_proba']:.1%}")

    st.metric("Risk Probability", f"{R['risk_proba']:.1%}")
    st.metric("Expected Delay", f"{R['delay_pred']:.1f} days")

    st.subheader("📊 Scenario Simulation")
    st.write(pd.DataFrame(R["results_map"]).T)

    st.subheader("📈 Scenario Comparison Chart")
    st.image(BytesIO(R["chart_png"]))

    st.subheader("🔎 Why did the model predict this?")
    if R["shap_png"]:
        st.image(BytesIO(R["shap_png"]), caption="Top drivers of risk")
    else:
        st.info("Explainability not available.")

    st.subheader("📑 Download Report")
    pdf_buf = generate_pdf(R, candidate_name=candidate_name)
    st.download_button("⬇️ Download PDF Report", data=pdf_buf, file_name="risk_delay_report.pdf", mime="application/pdf")
else:
    st.info("Adjust inputs on the left and click **Predict** to generate results.")
