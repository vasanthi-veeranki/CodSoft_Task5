# CodSoft Data Science Internship - Task 5
# Credit Card Fraud Detection
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
# Load Dataset

df = pd.read_csv("creditcard.csv")

print("=" * 50)
print("First 5 Records")
print("=" * 50)
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistical Summary")
print(df.describe())

# Fraud Distribution
print("\nTransaction Distribution")
print(df["Class"].value_counts())

plt.figure(figsize=(6,4))
df["Class"].value_counts().plot(kind="bar")
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0,1], ["Genuine (0)", "Fraud (1)"], rotation=0)
plt.tight_layout()
plt.show()

# Split Features and Target

X = df.drop("Class", axis=1)
y = df["Class"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Build Model

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
# Prediction

y_pred = model.predict(X_test)

# Evaluation

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("Model Evaluation")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nTask 5 Completed Successfully!")