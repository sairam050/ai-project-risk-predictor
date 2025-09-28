# 🚀 AI Project Risk & Delay Predictor

### Predict project risks and delays instantly with AI.

---

## 🔹 Quick Overview (For Recruiters)

* ✅ Predicts **probability of project failure** (Low / Medium / High)
* ✅ Estimates **expected schedule delays** (in days)
* ✅ Runs **what-if scenarios** (Base / Optimistic / Pessimistic)
* ✅ Explains predictions with **AI-powered feature importance**
* ✅ Generates a polished **PDF report** for stakeholders

👉 **[Open the App on Streamlit](https://ai-project-risk-predictor-p2yc7gde6kf8khzpeayn7h.streamlit.app/)**
*(No setup needed, runs directly in the browser!)*

---

## 📊 Example Predictions

| Scenario         | Risk Probability | Expected Delay |
| ---------------- | ---------------- | -------------- |
| Low Risk Project | ~15%             | ~12 days       |
| Medium Risk      | ~45–60%          | ~30 days       |
| High Risk        | ~80–100%         | ~50+ days      |

---

## 🛠 Tech Stack

* **Python** – scikit-learn, pandas, numpy
* **Streamlit** – interactive web app
* **SHAP** – model explainability
* **Matplotlib + ReportLab** – charts & PDF reporting
* **Google Drive (gdown)** – model hosting & auto-download

---

# 🧑‍💻 Detailed Documentation (For Engineers)

## 📂 Project Structure

```
├── streamlit_app.py        # Main app file
├── rf_risk_classifier.joblib  # Risk classification model
├── rf_delay_regressor.joblib  # Delay prediction model
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## ⚙️ How It Works

1. **Input Features**

   * Planned Duration (days)
   * Team Size
   * Budget ($k)
   * Change Requests
   * Resource Utilization (%)
   * Complexity Score (0–1)
   * Onshore %

2. **Model Outputs**

   * **Risk Probability:** Likelihood of project being at risk
   * **Expected Delay:** Predicted schedule slip in days
   * **Risk Level:**

     * ✅ Low (< 40%)
     * 🟠 Medium (40–70%)
     * ⚠️ High (> 70%)

3. **Explainability**

   * SHAP plots show the top drivers of risk.
   * Scenario simulations allow “what-if” testing.

4. **Export**

   * Full PDF Report with summary, scenario tables, charts, and explanations.

---

## 📑 Setup (For Developers)

### 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/ai-project-risk-predictor.git
cd ai-project-risk-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
streamlit run streamlit_app.py
```

### 4. Deploy on Streamlit Cloud

* Push repo to GitHub.
* Connect GitHub repo in **Streamlit Cloud**.
* App auto-deploys at `<your-app>.streamlit.app`.

---

## 👤 Author

**Sairam Thonuunuri**
📩 [LinkedIn](https://linkedin.com/in/your-link) | [GitHub](https://github.com/your-username)

---

⚡ **Tip:**
The **top section** is short & punchy for recruiters.
The **bottom section** is detailed for engineers/reviewers.
