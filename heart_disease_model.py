# Heart Disease Prediction using Logistic Regression 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_auc_score)

# ---------- 1. Load the data ----------
df = pd.read_csv(r"C:\Users\Rishabh\OneDrive\Desktop\heart-disease-logistic-regression\heart-disease-logistic-regression\heart.csv")          
print(df.shape)
print(df.head())
print(df.info())

# ---------- 2. Data cleaning ----------
# 2a. Missing values
print("Missing values:\n", df.isnull().sum())

# Cholesterol and RestingBP have 0 values, which is impossible -> treat as missing
df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
df["RestingBP"] = df["RestingBP"].replace(0, np.nan)

# Fill the missing values with the median
df["Cholesterol"] = df["Cholesterol"].fillna(df["Cholesterol"].median())
df["RestingBP"] = df["RestingBP"].fillna(df["RestingBP"].median())

# 2b. Duplicates
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# ---------- 3. Encode categorical data ----------
df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
df["ExerciseAngina"] = df["ExerciseAngina"].map({"N": 0, "Y": 1})
df = pd.get_dummies(df, columns=["ChestPainType", "RestingECG", "ST_Slope"],
                    drop_first=True, dtype=int)

# ---------- 4. Correlation ----------
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

print(df.corr()["HeartDisease"].sort_values(ascending=False))

# ---------- 5. Train-test split ----------
X = df.drop("HeartDisease", axis=1)    # features
y = df["HeartDisease"]                 # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features (fit on train only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- 6. Train the model ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------- 7. Evaluate the model ----------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy :", accuracy_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
