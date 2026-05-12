import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load Dataset
df = pd.read_csv("Social_Network_Ads.csv")

print("First 5 Rows:")
print(df.head())

# Input and Output Variables
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Extract Values
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTrue Positive:", TP)
print("True Negative:", TN)
print("False Positive:", FP)
print("False Negative:", FN)

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print("\nAccuracy:", accuracy)

# Error Rate
error_rate = (FP + FN) / (TP + TN + FP + FN)
print("Error Rate:", error_rate)

# Precision
precision = TP / (TP + FP)
print("Precision:", precision)

# Recall
recall = TP / (TP + FN)
print("Recall:", recall)

# optional

import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()