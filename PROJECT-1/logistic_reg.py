import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset

file_path = r"S:\ENGINOW\PROJECT-1\Mall_Customers.xlsx"
data = pd.read_excel(file_path)

print("First 5 rows:\n", data.head())
print("\nDataset Shape:", data.shape)
print("\nDataset Info:")
print(data.info())


# 2. Drop CustomerID (not a feature)

data.drop("CustomerID", axis=1, inplace=True)


# 3. Encode Gender
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])  # Male=1, Female=0


# 4. Define FEATURES and TARGET (CRITICAL STEP)

FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = data[FEATURE_COLUMNS]
y = data["Purchased"]


# 5. Feature Scaling (ONLY X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 6. Train–Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# 7. Train Logistic Regression Model

model = LogisticRegression()
model.fit(X_train, y_train)


# 8. Model Evaluation

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# REAL-TIME PREDICTION 
def predict_purchase():
    print("\n--- Enter Customer Details ---")

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
predict_purchase()
