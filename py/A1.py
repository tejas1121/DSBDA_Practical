import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display first 5 rows
print(df.head())

# Shape of dataset
print("Shape of dataset:")
print(df.shape)

# Dataset information   
print("Information about dataset:")
print(df.info())

# Missing values
print("Missing values:")
print(df.isnull().sum())

# Statistical summary
print("Statistical summary:")
print(df.describe())

# Datatypes
print("Datatypes of variables:")
print(df.dtypes)

# Fill missing values in Age column
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Convert Age datatype
df['Age'] = df['Age'].astype(int)

# Convert categorical variable into numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Display updated dataset
print("Updated dataset:")
print(df.head())