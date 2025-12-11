# app/app.py

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ------------ Load Model ------------
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.joblib"

st.set_page_config(page_title="Telecom Churn Prediction", layout="centered")
st.title("📞 Telecom Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

# Check if model exists
if not MODEL_PATH.exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`.\n\n"
        "👉 Run `python -m src.models.train` to generate the model file."
    )
    st.stop()

# Load model
model = joblib.load(MODEL_PATH)

# ------------ Input Form ------------
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

with col2:
    total_charges = st.number_input("Total Charges", min_value=0.0, value=50.0)

contract_type = st.selectbox(
    "Contract Type",
    ["month-to-month", "one_year", "two_year"]
)

tenure_band = st.selectbox(
    "Tenure Band",
    ["0-2", "3-6", "7-12", "13-24", "25+"]
)

# Derived field (same logic as in feature engineering)
high_monthly = 1 if monthly_charges > 50 else 0

# ------------ Prepare Input Data ------------
input_df = pd.DataFrame([{
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "tenure": tenure,
    "contract_type": contract_type,
    "tenure_band": tenure_band,
    "high_monthly": high_monthly
}])

# ------------ Prediction ------------
if st.button("Predict Churn Probability"):
    probability = model.predict_proba(input_df)[0][1]
    st.success(f"Churn Probability: **{probability:.2%}**")

    if probability > 0.5:
        st.warning("⚠️ High risk of churn. Consider retention strategies.")
    else:
        st.info("✅ Low churn risk.")

