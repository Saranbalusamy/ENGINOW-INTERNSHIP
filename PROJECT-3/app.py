import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and preprocessor
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📉 Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn and identify risk level.")

# ------------------------
# User Input Form
# ------------------------
st.header("🧾 Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=2000.0)

# ------------------------
# Prediction
# ------------------------
if st.button("🔍 Predict Churn"):
    
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    # Feature engineering (same as training)
    input_data["AvgRevenue"] = input_data["TotalCharges"] / (input_data["tenure"] + 1)
    input_data["ServiceCount"] = (
        (input_data[["OnlineSecurity","OnlineBackup","DeviceProtection",
                     "TechSupport","StreamingTV","StreamingMovies"]] == "Yes").sum(axis=1)
    )
    input_data["ContractRisk"] = input_data["Contract"].map({
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1
    })

    input_data["TenureGroup"] = pd.cut(
        input_data["tenure"],
        bins=[0, 12, 36, 72],
        labels=["New", "Mid", "Loyal"]
    )

    # Preprocess & predict
    processed = preprocessor.transform(input_data)
    churn_prob = model.predict_proba(processed)[0][1]

    # Risk category
    if churn_prob > 0.7:
        risk = "🔴 High Risk"
    elif churn_prob > 0.4:
        risk = "🟠 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.subheader("📊 Prediction Result")
    st.write(f"**Churn Probability:** {churn_prob:.2f}")
    st.write(f"**Risk Level:** {risk}")
