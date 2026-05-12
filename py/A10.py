import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
iris = sns.load_dataset('iris')

# Display first 5 rows
print(iris.head())

# Dataset Information
print(iris.info())

# Data Types
print(iris.dtypes)

# Statistical Summary
print(iris.describe())

# -----------------------------
# Histograms
# -----------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes.flatten()

sns.histplot(iris['sepal_length'], ax=ax[0])
ax[0].set_title("Sepal Length")

sns.histplot(iris['sepal_width'], ax=ax[1])
ax[1].set_title("Sepal Width")

sns.histplot(iris['petal_length'], ax=ax[2])
ax[2].set_title("Petal Length")

sns.histplot(iris['petal_width'], ax=ax[3])
ax[3].set_title("Petal Width")

plt.tight_layout()

plt.show()

# -----------------------------
# Boxplots
# -----------------------------

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes.flatten()

sns.boxplot(y=iris['sepal_length'], ax=ax[0])
ax[0].set_title("Sepal Length")

sns.boxplot(y=iris['sepal_width'], ax=ax[1])
ax[1].set_title("Sepal Width")

sns.boxplot(y=iris['petal_length'], ax=ax[2])
ax[2].set_title("Petal Length")

sns.boxplot(y=iris['petal_width'], ax=ax[3])
ax[3].set_title("Petal Width")

plt.tight_layout()

plt.show()