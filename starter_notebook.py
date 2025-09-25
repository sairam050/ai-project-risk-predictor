# AI Project Risk Predictor - Starter Colab Script
# Run this in Google Colab. Execute cells in order.
#
# Overview:
# 1) Generate or load a project dataset (synthetic example included)
# 2) EDA and feature engineering
# 3) Train baseline models for (a) risk classification and (b) delay regression
# 4) Evaluate, save models, and produce simple explanations (SHAP)
#
# Note: install packages in Colab before running heavy cells:
#   !pip install -q scikit-learn xgboost shap pandas numpy matplotlib joblib

# ---------- Imports ----------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, classification_report, mean_absolute_error
import joblib

# ---------- 1) Synthetic dataset generation (quick start) ----------
rng = np.random.RandomState(42)
n = 2000
df = pd.DataFrame({
    'planned_duration_days': rng.randint(30, 540, size=n),
    'team_size': rng.randint(2, 40, size=n),
    'budget_k': rng.randint(5, 5000, size=n),
    'num_change_requests': rng.poisson(2, size=n),
    'pct_resource_util': rng.uniform(0.4, 1.3, size=n),
    'complexity_score': rng.uniform(0,1,size=n),
    'onshore_pct': rng.uniform(0,1,size=n)
})
# Simulated delay (days) with noise
df['delay_days'] = (df['num_change_requests']*6 + (df['planned_duration_days']/120)*4 + (df['complexity_score']*40)
                    + (1 - df['onshore_pct'])*5 + rng.normal(0,12,size=n)).round().astype(int)
df['delay_days'] = df['delay_days'].clip(lower=0)
# Binary risk flag: delay > 10% of planned duration
df['risk_flag'] = (df['delay_days'] > (df['planned_duration_days'] * 0.10)).astype(int)

df.to_csv('synthetic_projects.csv', index=False)
print("Synthetic dataset saved to synthetic_projects.csv; sample:")
print(df.head())

# ---------- 2) Train/test split ----------
features = ['planned_duration_days','team_size','budget_k','num_change_requests','pct_resource_util','complexity_score','onshore_pct']
X = df[features]
y_class = df['risk_flag']
y_reg = df['delay_days']

X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42)

# ---------- 3) Baseline classification model (risk) ----------
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, yc_train)
probs = clf.predict_proba(X_test)[:,1]
auc = roc_auc_score(yc_test, probs)
print(f"Baseline Risk Classifier AUC: {auc:.3f}")
print(classification_report(yc_test, clf.predict(X_test)))

# ---------- 4) Baseline regression model (delay days) ----------
reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
reg.fit(X_train, yr_train)
preds = reg.predict(X_test)
mae = mean_absolute_error(yr_test, preds)
print(f"Baseline Delay Regression MAE: {mae:.2f} days")

# ---------- 5) Save models ----------
joblib.dump(clf, 'rf_risk_classifier.joblib')
joblib.dump(reg, 'rf_delay_regressor.joblib')
print("Models saved: rf_risk_classifier.joblib, rf_delay_regressor.joblib")

# ---------- 6) Next steps (in Colab) ----------
print("\nNEXT STEPS (recommended):")
print(" - Perform detailed EDA (correlations, missing values, distributions)")
print(" - Feature engineering: add derived features (change request rate, buffer ratio, resource volatility)")
print(" - Try XGBoost/LightGBM and cross-validation; tune hyperparameters")
print(" - Use SHAP for explainability and create a Streamlit demo app")
print(" - Create a one-page executive summary and a 3-minute demo video")
