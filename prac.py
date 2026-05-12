#cell1
import pandas as pd
import numpy as np

#cell2
df=pd.read_csv(r"C:\Users\Tejas Burkul\Downloads\archive (2)\Titanic-Dataset.csv")

#cell3
print(df)

#cell4
df

#cell5
df.isnull()

#cell6
df.isnull().sum()

#cell7
df.notnull()

#cell8
df.notnull().sum()

#cell9
df.describe()

#cell10
df.size

#cell11
df.shape

#cell12
df.info

#cell13
df=df.dropna()

#cell14
df

#cell15
df["Fare"]=df["Fare"].astype(int)

#cell16
df

#cell17
df["Sex"] = df["Sex"].replace({"female": 0, "male": 1})

#cell18
df


