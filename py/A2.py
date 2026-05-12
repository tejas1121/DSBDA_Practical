import pandas as pd
import numpy as np

# Create Dataset
data = {
    'Name': ['Amit', 'Sonal', 'Riya', 'Karan', 'Neha', 'Raj', 'Pooja', 'Vikas'],
    
    'Math': [85, 90, np.nan, 105, 76, 45, 89, 150],
    
    'Science': [92, 85, 88, 95, np.nan, 40, 91, 35],
    
    'Attendance': [80, 95, 78, 110, 85, 60, np.nan, 72],
    
    'CGPA': [8.5, 9.1, 7.8, 10.5, 8.2, 5.5, 9.0, 20]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Missing Values
df['Math'].fillna(df['Math'].mean(), inplace=True)

df['Science'].fillna(df['Science'].mean(), inplace=True)

df['Attendance'].fillna(df['Attendance'].mean(), inplace=True)

# Handle Inconsistencies
df.loc[df['Math'] > 100, 'Math'] = 100

df.loc[df['Attendance'] > 100, 'Attendance'] = 100

df.loc[df['CGPA'] > 10, 'CGPA'] = 10

# Detect Outliers using IQR
Q1 = df['Math'].quantile(0.25)

Q3 = df['Math'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

print("\nLower Limit:", lower)

print("Upper Limit:", upper)

# Remove Outliers
df = df[(df['Math'] >= lower) & (df['Math'] <= upper)]

print("\nDataset after removing outliers:")
print(df)

# Normalization
df['Math_Normalized'] = (df['Math'] - df['Math'].min()) / (df['Math'].max() - df['Math'].min())

print("\nNormalized Data:")
print(df[['Math', 'Math_Normalized']])