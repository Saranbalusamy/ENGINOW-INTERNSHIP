import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("credit_risk_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")


feature_names = list(preprocessor.feature_names_in_)

st.title("Credit Risk Prediction System")

st.write("Enter basic applicant details:")


income = st.number_input("Applicant Income", min_value=0.0)
credit = st.number_input("Loan Amount", min_value=0.0)
age = st.number_input("Age (years)", min_value=18, max_value=100)


input_data = pd.DataFrame(
    data=np.nan,
    columns=feature_names,
    index=[0]
)


input_data.loc[0, "AMT_INCOME_TOTAL"] = income
input_data.loc[0, "AMT_CREDIT"] = credit
input_data.loc[0, "AGE_YEARS"] = age
input_data.loc[0, "DAYS_BIRTH"] = -age * 365

input_data.loc[0, "DTI_RATIO"] = credit / income if income > 0 else 0


if age < 30:
    input_data.loc[0, "AGE_GROUP"] = "20-30"
elif age < 40:
    input_data.loc[0, "AGE_GROUP"] = "30-40"
elif age < 50:
    input_data.loc[0, "AGE_GROUP"] = "40-50"
elif age < 60:
    input_data.loc[0, "AGE_GROUP"] = "50-60"
else:
    input_data.loc[0, "AGE_GROUP"] = "60+"


if st.button("Predict Risk"):
    processed = preprocessor.transform(input_data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    if prediction == 1:
        st.error(f"High Risk Applicant (Probability: {probability:.2f})")
    else:
        st.success(f"Low Risk Applicant (Probability: {probability:.2f})")
