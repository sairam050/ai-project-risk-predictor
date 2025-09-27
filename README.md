# 📊 AI Project Risk & Delay Predictor  

An interactive web app that predicts **project risk** and **expected delays** using machine learning.  
Built with **Streamlit**, **scikit-learn**, and **SHAP**, the tool allows users to simulate scenarios, visualize outcomes, and download polished PDF reports.

---

## 🚀 Features  

- **Risk & Delay Prediction**: Estimates probability of project failure/delay using trained ML models and outputs expected delay in days.  
- **Scenario Simulation**: Run *Base Case*, *Optimistic*, and *Pessimistic* scenarios, then compare results side-by-side with tables and charts.  
- **Explainability (AI Transparency)**: Highlights which features most influenced predictions (via SHAP / feature importances), making the model interpretable for non-technical stakeholders.  
- **Polished PDF Reports**: Auto-generates downloadable reports with summary, scenario comparison, and visualizations — recruiter-friendly formatting with candidate name included.  

---

## 🛠️ Tech Stack  

- Python 3.10+  
- Streamlit (frontend & deployment)  
- scikit-learn (ML models)  
- SHAP (model explainability)  
- Matplotlib (charts & plots)  
- ReportLab (PDF generation)  

---

## 📂 Project Structure  

- **streamlit_app.py** — Main Streamlit app  
- **rf_risk_classifier.joblib** — Pretrained risk classifier  
- **rf_delay_regressor.joblib** — Pretrained delay regressor  
- **requirements.txt** — Dependencies  
- **README.md** — Project documentation  

---

## ⚡ How to Run  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/sairam050/ai-project-risk-predictor.git
   cd ai-project-risk-predictor
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app locally**  
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open in browser**  
   👉 http://localhost:8501  

---

## 🌐 Live Demo  

The app is deployed on **Streamlit Cloud**:  
👉 [AI Project Risk & Delay Predictor](https://ai-project-risk-predictor.streamlit.app)  

---

## 📑 Example Report  

A generated PDF report includes:  
- Risk probability  
- Expected delay  
- Scenario comparison (table + chart)  
- Feature importance / SHAP explanation  

---

## 🎯 Why This Project Matters  

- Demonstrates **AI + project management expertise**.  
- Showcases skills in **ML, data visualization, and app deployment**.  
- Recruiters and hiring managers can interact with the tool directly.  

---

## 👤 Author  

**Sairam Thonupunuri**  
- 📧 [Your Email]  
- 🌐 [LinkedIn Profile]  
- 💻 [Portfolio or GitHub]  
