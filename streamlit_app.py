# ---------------- Imports ----------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import tempfile
import shap

# ---------------- Page Config ----------------
st.set_page_config(page_title="AI Project Risk & Delay Predictor", layout="wide")

# ---------------- Load Models ----------------
try:
    risk_model = joblib.load("rf_risk_classifier.joblib")
    delay_model = joblib.load("rf_delay_regressor.joblib")
except:
    st.error("⚠️ Models not found! Please upload rf_risk_classifier.joblib and rf_delay_regressor.joblib")
    st.stop()

# ---------------- Sidebar: Project Inputs ----------------
st.sidebar.header("📂 Project Inputs")

planned_duration_days = st.sidebar.number_input("Planned Duration (days)", 30, 1000, 180)
team_size = st.sidebar.number_input("Team Size", 2, 100, 10)
budget_k = st.sidebar.number_input("Budget (in $1000s)", 100, 10000, 500)
num_change_requests = st.sidebar.number_input("Change Requests", 0, 20, 1)
pct_resource_util = st.sidebar.slider("Resource Utilization (%)", 0.1, 2.0, 1.0)
complexity_score = st.sidebar.slider("Complexity Score", 0.0, 1.0, 0.5)
onshore_pct = st.sidebar.slider("Onshore %", 0.0, 1.0, 0.5)

input_df = pd.DataFrame([[
    planned_duration_days, team_size, budget_k, num_change_requests,
    pct_resource_util, complexity_score, onshore_pct
]], columns=[
    "planned_duration_days", "team_size", "budget_k", "num_change_requests",
    "pct_resource_util", "complexity_score", "onshore_pct"
])

# ---------------- Main Title ----------------
st.title("📊 AI Project Risk & Delay Predictor")
st.write("Enter project details, test scenarios, and download polished reports instantly.")

# ---------------- Prediction ----------------
if st.sidebar.button("🚀 Predict"):
    risk_proba = float(risk_model.predict_proba(input_df)[:, 1][0])
    delay_pred = float(delay_model.predict(input_df)[0])

    if risk_proba > 0.66:
        st.error(f"⚠️ High Risk — {risk_proba:.1%}")
    elif risk_proba > 0.33:
        st.warning(f"🟠 Medium Risk — {risk_proba:.1%}")
    else:
        st.success(f"✅ Low Risk — {risk_proba:.1%}")

    st.metric("Expected Delay", f"{delay_pred:.1f} days")
    st.write("Thresholds: Low < 33%, Medium 33–66%, High > 66%")

    # ---------------- Scenario Simulation ----------------
    st.header("🔮 Scenario Simulation")
    st.write("Compare base, optimistic, and pessimistic project assumptions.")

    scenarios = {
        "Base Case": [planned_duration_days, team_size, budget_k, num_change_requests,
                      pct_resource_util, complexity_score, onshore_pct],
        "Optimistic": [planned_duration_days*0.9, team_size+2, budget_k*1.2,
                       max(0, num_change_requests-1), pct_resource_util*0.9,
                       complexity_score*0.8, min(1.0, onshore_pct+0.1)],
        "Pessimistic": [planned_duration_days*1.2, max(2, team_size-2), budget_k*0.8,
                        num_change_requests+2, pct_resource_util*1.1,
                        min(1.0, complexity_score*1.2), max(0, onshore_pct-0.1)]
    }

    results = {}
    for label, vals in scenarios.items():
        df = pd.DataFrame([vals], columns=input_df.columns)
        prob = float(risk_model.predict_proba(df)[:, 1][0])
        delay = float(delay_model.predict(df)[0])
        results[label] = (prob, delay)

    comparison = pd.DataFrame(results, index=["Risk Probability", "Expected Delay (days)"]).T
    comparison["Risk Probability"] = comparison["Risk Probability"].apply(lambda x: f"{x:.1%}")
    st.table(comparison)

    # ---------------- Chart ----------------
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

    # ---------------- Feature Importance / Explainability ----------------
    st.subheader("🔎 Why did the model predict this?")
    shap_path = None
    try:
        explainer = shap.TreeExplainer(risk_model)
        shap_values = explainer(input_df)
        st.set_option("deprecation.showPyplotGlobalUse", False)
        shap.plots.bar(shap_values, show=False, max_display=7)
        st.pyplot(bbox_inches="tight")
    except Exception:
        if hasattr(risk_model, "feature_importances_"):
            feat_imp = pd.Series(risk_model.feature_importances_, index=input_df.columns)
            st.bar_chart(feat_imp.sort_values(ascending=False).head(7))
        else:
            st.info("ℹ️ SHAP/feature importance not available for this model.")

    # ---------------- PDF Report Functions ----------------
    def generate_pdf_report(proba, delay_pred, comparison, chart_path, shap_path=None):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            doc = SimpleDocTemplate(tmpfile.name, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            story.append(Paragraph("📊 AI Project Risk & Delay Predictor — Report", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Risk Probability: {proba:.1%}", styles['Normal']))
            story.append(Paragraph(f"Expected Delay: {delay_pred:.1f} days", styles['Normal']))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Scenario Comparison", styles['Heading2']))
            table_data = [["Scenario", "Risk Probability", "Expected Delay (days)"]]
            for label, row in comparison.iterrows():
                table_data.append([label, row["Risk Probability"], f"{row['Expected Delay (days)']:.1f}"])
            story.append(Table(table_data))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Scenario Comparison Chart", styles['Heading2']))
            story.append(Image(chart_path, width=400, height=250))
            story.append(Spacer(1, 12))

            if shap_path:
                story.append(Paragraph("Feature Importance / SHAP Explanation", styles['Heading2']))
                story.append(Image(shap_path, width=400, height=250))
                story.append(Spacer(1, 12))

            story.append(Paragraph("Thresholds: Low < 33%, Medium 33–66%, High > 66%", styles['Italic']))

            doc.build(story)
            return tmpfile.name

    def generate_explanation_chart(clf, input_df):
        shap_path = None
        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(input_df)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as shap_file:
                shap.plots.bar(shap_values, show=False, max_display=7)
                plt.savefig(shap_file.name, bbox_inches="tight")
                plt.close()
                shap_path = shap_file.name
        except Exception:
            if hasattr(clf, "feature_importances_"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as shap_file:
                    feat_imp = pd.Series(clf.feature_importances_, index=input_df.columns)
                    feat_imp.nlargest(7).plot(kind="barh", color="skyblue")
                    plt.title("Feature Importance (Fallback)")
                    plt.xlabel("Importance")
                    plt.tight_layout()
                    plt.savefig(shap_file.name)
                    plt.close()
                    shap_path = shap_file.name
        return shap_path

    # ---------------- PDF Download Button ----------------
    st.sidebar.subheader("📑 Download Report")
    if st.sidebar.button("Download PDF Report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as chart_file:
            fig, ax1 = plt.subplots(figsize=(6,4))
            ax2 = ax1.twinx()

            comparison["Risk Float"] = comparison["Risk Probability"].str.rstrip('%').astype(float) / 100.0
            ax1.bar(comparison.index, comparison["Risk Float"], color="salmon", alpha=0.7, label="Risk (%)")
            ax2.bar(comparison.index, comparison["Expected Delay (days)"], color="skyblue", alpha=0.7, label="Delay (days)")

            ax1.set_ylabel("Risk Probability (%)", color="red")
            ax2.set_ylabel("Expected Delay (days)", color="blue")
            plt.title("Scenario Comparison: Risk vs Delay")
            plt.tight_layout()
            plt.savefig(chart_file.name)
            plt.close()
            chart_path = chart_file.name

        shap_path = generate_explanation_chart(risk_model, input_df)
        pdf_path = generate_pdf_report(risk_proba, delay_pred, comparison, chart_path, shap_path)

        with open(pdf_path, "rb") as f:
            st.sidebar.download_button("⬇️ Download PDF Report", f, file_name="risk_delay_report.pdf")
