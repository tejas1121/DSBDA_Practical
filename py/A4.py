import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
boston = fetch_openml(name='boston', version=1, as_frame=True)

df = boston.frame

print("First 5 Rows:")
print(df.head())

# Shape
print("\nShape of Dataset:")
print(df.shape)

#make all the datatypes in float
df = df.astype(float)

# check all the datatypes are in float
df.dtypes

# Input and Output Variables
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Compare Results
result = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

print("\nActual vs Predicted Prices:")
print(result.head())

# Error Calculation
mse = mean_squared_error(y_test, y_pred)

print("\nMean Squared Error:", mse)

#optional

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2) # Diagonal line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Boston House Prices')
plt.show()