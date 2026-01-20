# ======================================================
# Customer Purchase Prediction
# Decision Tree, Random Forest, KNN
# FINAL & ERROR-FREE VERSION
# ======================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load dataset

file_path = r"S:\ENGINOW\PROJECT-1\Mall_Customers.xlsx"
data = pd.read_excel(file_path)

print("Dataset Loaded Successfully")
print(data.head())

# 2. Drop CustomerID
data.drop("CustomerID", axis=1, inplace=True)


# 3. Encode Gender
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])  # Male=1, Female=0


# 4. Define FEATURES and TARGET
FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = data[FEATURE_COLUMNS]
y = data["Purchased"]


# 5. Feature Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 6. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 8. RANDOM FOREST

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))




# REAL-TIME PREDICTION 

def realtime_prediction(model, model_name):
    print(f"\n--- Real-Time Prediction using {model_name} ---")

    gender = input("Gender (Male/Female): ").strip().lower()
    gender = 1 if gender == "male" else 0

    age = int(input("Age: "))
    income = float(input("Annual Income (k$): "))
    spending = float(input("Spending Score (1-100): "))

    user_df = pd.DataFrame(
        [[gender, age, income, spending]],
        columns=FEATURE_COLUMNS
    )

    user_scaled = scaler.transform(user_df)
    prediction = model.predict(user_scaled)

    if prediction[0] == 1:
        print("Customer WILL purchase")
    else:
        print("Customer will NOT purchase")
realtime_prediction(rf_model, "Random Forest")

