import pandas as pd
import numpy as np
import seaborn as sns

# -----------------------------
# PART 1
# -----------------------------

data = {
    'Name': ['Amit', 'Sonal', 'Riya', 'Karan', 'Neha', 'Raj'],
    
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
    
    'Age': [20, 21, 19, 22, 20, 23],
    
    'Income': [25000, 30000, 28000, 35000, 32000, 40000]
}

df = pd.DataFrame(data)

print("Student Dataset:")
print(df)

# Grouped Statistics
grouped = df.groupby('Gender')['Income']

print("\nMean:")
print(grouped.mean())

print("\nMedian:")
print(grouped.median())

print("\nMinimum:")
print(grouped.min())

print("\nMaximum:")
print(grouped.max())

print("\nStandard Deviation:")
print(grouped.std())

# List for each category
male_income = df[df['Gender'] == 'Male']['Income'].tolist()

female_income = df[df['Gender'] == 'Female']['Income'].tolist()

print("\nMale Income List:", male_income)

print("Female Income List:", female_income)

# -----------------------------
# PART 2
# -----------------------------

iris = sns.load_dataset("Iris.csv")

print("\nFirst 5 rows of Iris Dataset:")
print(iris.head())

print("\nSpecies Names:")
print(iris['species'].unique())

# Separate species
setosa = iris[iris['species'] == 'setosa']

versicolor = iris[iris['species'] == 'versicolor']

virginica = iris[iris['species'] == 'virginica']

# Statistics
print("\nStatistics for Iris-setosa")
print(setosa.describe())

print("\nStatistics for Iris-versicolor")
print(versicolor.describe())

print("\nStatistics for Iris-virginica")
print(virginica.describe())