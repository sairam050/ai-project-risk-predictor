# streamlit_app.py — Final polished AI Project Risk & Delay Predictor

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# PDF (install 'reportlab' in requirements.txt)
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    PDF_AVAILABLE = True
except Exception:
    PDF_AVAILABLE = False

# Optional SHAP (safe import)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ===================== Page setup & light styling =====================
st.set_page_config(
    page_title="AI Project Risk & Delay Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* tighten the page a bit and center content */
.block-container {max-width: 1100px; padding-top: 1rem; padding-bottom: 2.0rem;}
/* nice look for metrics */
[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Project Risk & Delay Predictor")
st.caption("Enter project details, run predictions, compare scenarios, and export a report.")


# ===================== Load models =====================
try:
    risk_model = joblib.load("rf_risk_classifier.joblib")
    delay_model = joblib.load("rf_delay_regressor.joblib")
except Exception as e:
    st.error("❌ Couldn't load model files (`rf_risk_classifier.joblib`, `rf_delay_regressor.joblib`). "
             "Place them in the repo root. Error:\n\n" + str(e))
    st.stop()


# ===================== Sidebar inputs =====================
st.sidebar.header("Project Inputs")

planned_duration_days = st.sidebar.number_input("Planned Duration (days)", 30, 1000, 180)
team_size             = st.sidebar.number_input("Team Size", 2, 100, 10)
budget_k              = st.sidebar.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests   = st.sidebar.number_input("Change Requests", 0, 20, 1)
pct_resource_util     = st.sidebar.slider("Resource Utilization (× capacity)", 0.1, 2.0, 1.0)
complexity_score      = st.sidebar.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct           = st.sidebar.slider("Onshore %", 0.0, 1.0, 0.5)

# build input dataframe used everywhere below
input_df = pd.DataFrame([[
    planned_duration_days, team_size, budget_k, num_change_requests,
    pct_resource_util, complexity_score, onshore_pct
]], columns=[
    "planned_duration_days", "team_size", "budget_k", "num_change_requests",
    "pct_resource_util", "complexity_score", "onshore_pct"
])


# ===================== Helpers =====================
def store_prediction_to_session(prob: float, delay: float):
    st.session_state["pred_proba"] = float(prob)
    st.session_state["pred_delay"] = float(delay)


def get_pred_from_session():
    return (
        st.session_state.get("pred_proba", None),
        st.session_state.get("pred_delay", None),
    )


def create_pdf_report(prob: float, delay: float, table_df: pd.DataFrame) -> BytesIO:
    """Return a PDF buffer with a simple one-page report."""
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("AI Project Risk & Delay Predictor — Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Risk Probability: <b>{prob:.1%}</b>", styles["Normal"]))
    story.append(Paragraph(f"Expected Delay: <b>{delay:.1f} days</b>", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Scenario Comparison", styles["Heading2"]))
    # table data (header + rows)
    table_data = [["Scenario"] + list(table_df.columns)]
    for idx, row in table_df.iterrows():
        table_data.append([idx] + list(row.values))
    story.append(Table(table_data))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Thresholds: Low < 33%, Medium 33–66%, High > 66%.", styles["Italic"]))

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    doc.build(story)
    buf.seek(0)
    return buf


# ===================== Prediction (top cards) =====================
left, right = st.columns(2)

with left:
    if st.sidebar.button("🔮 Predict"):
        proba = float(risk_model.predict_proba(input_df)[:, 1][0])
        store_prediction_to_session(proba, float(delay_model.predict(input_df)[0]))

# read back latest prediction (if any)
proba_session, delay_session = get_pred_from_session()

if proba_session is not None and delay_session is not None:
    col1, col2 = st.columns(2)
    with col1:
        if proba_session > 0.66:
            st.error(f"⚠️ High Risk — {proba_session:.1%}")
        elif proba_session > 0.33:
            st.warning(f"🟠 Medium Risk — {proba_session:.1%}")
        else:
            st.success(f"✅ Low Risk — {proba_session:.1%}")
        st.metric("Risk Probability", f"{proba_session:.2%}")
        st.caption("Thresholds: Low < 33%, Medium 33–66%, High > 66%.")

    with col2:
        st.metric("Expected Delay", f"{delay_session:.1f} days")
        if delay_session > planned_duration_days * 0.15:
            st.info("Projected delay > 15% of planned duration — consider mitigation actions (buffer, scope trim, add resources).")
        else:
            st.success("Projected delay looks within an acceptable range.")

else:
    st.info("Click **Predict** in the left panel to generate top-line results.")


# ===================== Scenario Simulation =====================
st.header("🪄 Scenario Simulation")
st.write("Compare **Base**, **Optimistic**, and **Pessimistic** assumptions to see how risk and delay move.")

scenarios = {
    "Base Case": [
        planned_duration_days, team_size, budget_k, num_change_requests,
        pct_resource_util, complexity_score, onshore_pct
    ],
    "Optimistic": [
        planned_duration_days * 0.90, team_size + 2, budget_k * 1.20,
        max(0, num_change_requests - 1), pct_resource_util * 0.90,
        complexity_score * 0.80, min(1.0, onshore_pct + 0.10)
    ],
    "Pessimistic": [
        planned_duration_days * 1.20, max(2, team_size - 2), budget_k * 0.80,
        num_change_requests + 2, pct_resource_util * 1.10,
        min(1.0, complexity_score * 1.20), max(0.0, onshore_pct - 0.10)
    ],
}

results = {}
for label, vals in scenarios.items():
    df = pd.DataFrame([vals], columns=input_df.columns)
    p = float(risk_model.predict_proba(df)[:, 1][0])
    d = float(delay_model.predict(df)[0])
    results[label] = (p, d)

# Table (pretty)
comparison = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T
comparison_display = comparison.copy()
comparison_display["Risk Probability"] = comparison_display["Risk Probability"].apply(lambda x: f"{x:.1%}")

st.subheader("📊 Scenario Comparison Table")
st.dataframe(comparison_display, use_container_width=True)

# Chart (grouped bars: risk on left axis, delay on right axis)
st.subheader("📈 Scenario Comparison: Risk vs Delay")
comparison_numeric = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T

fig, ax1 = plt.subplots(figsize=(8, 5))
comparison_numeric["Risk Probability"].plot(
    kind="bar", ax=ax1, color="tomato", position=0, width=0.4, label="Risk (%)"
)
ax2 = ax1.twinx()
comparison_numeric["Expected Delay (days)"].plot(
    kind="bar", ax=ax2, color="skyblue", position=1, width=0.4, label="Delay (days)"
)
ax1.set_ylabel("Risk Probability", color="tomato")
ax2.set_ylabel("Expected Delay (days)", color="skyblue")
ax1.set_xticklabels(comparison_numeric.index, rotation=0)
fig.suptitle("Scenario Comparison: Risk vs Delay", fontsize=14, fontweight="bold")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
st.pyplot(fig)


# ===================== Explainability (optional SHAP) =====================
with st.expander("🔍 Why did the model predict this? (Explainability)"):
    if SHAP_AVAILABLE:
        try:
            # Use model-agnostic Explainer for broad compatibility
            explainer = shap.Explainer(risk_model)
            sv = explainer(input_df)

            st.write("Most influential features for the current inputs:")
            fig_shap, ax_shap = plt.subplots()
            # For a single row, waterfall/force plots can be noisy in Streamlit; bar plot is clean
            shap.plots.bar(sv, max_display=7, show=False)
            st.pyplot(fig_shap)
        except Exception as e:
            st.warning("Couldn't generate SHAP plot. Falling back to model feature importances.")
            if hasattr(risk_model, "feature_importances_"):
                importances = risk_model.feature_importances_
                feat = pd.Series(importances, index=input_df.columns).sort_values(ascending=False)
                st.bar_chart(feat)
            else:
                st.info("Feature importances not available for this model.")
    else:
        st.info("Install SHAP to enable explainability (already included in requirements).")


# ===================== Downloads =====================
st.header("📥 Download")
colA, colB = st.columns(2)

with colA:
    st.download_button(
        label="Download Scenario Table (CSV)",
        data=comparison_display.to_csv(index=True).encode("utf-8"),
        file_name="scenario_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

with colB:
    if PDF_AVAILABLE:
        # use latest prediction if available, otherwise compute from base case now
        base_prob = proba_session if proba_session is not None else float(risk_model.predict_proba(input_df)[:, 1][0])
        base_delay = delay_session if delay_session is not None else float(delay_model.predict(input_df)[0])
        pdf_buf = create_pdf_report(base_prob, base_delay, comparison_display)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buf,
            file_name="project_risk_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("PDF export requires `reportlab` (already listed in requirements.txt).")

