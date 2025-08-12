import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load dataset
df = pd.read_csv("iris.csv")

# Fix column names
df.columns = df.columns.str.replace(".", "_", regex=False)

# --- Basic Info ---
print("📌 First 5 rows:\n", df.head(), "\n")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("Data Types:\n", df.dtypes, "\n")
print("Missing Values:\n", df.isnull().sum(), "\n")

# --- Convert features to NumPy array ---
features = df.iloc[:, :-1].to_numpy()

# --- NumPy Calculations ---
print("📊 Statistics per Feature:")
print("Mean:", np.round(np.mean(features, axis=0), 2))
print("Median:", np.round(np.median(features, axis=0), 2))
print("Std Dev:", np.round(np.std(features, axis=0), 2), "\n")

print(f"Min Petal Length: {df['petal_length'].min()} | Max Petal Length: {df['petal_length'].max()}\n")

# Correlation Matrix
print("🔗 Correlation Matrix:")
print(np.round(np.corrcoef(features, rowvar=False), 2), "\n")

# --- Grouping ---
species_avg = df.groupby("variety").mean(numeric_only=True)
print("📌 Average per Species:\n", species_avg, "\n")

largest_species = species_avg["petal_length"].idxmax()
print(f"🌸 Species with largest average petal length: {largest_species}\n")

# --- New Feature ---
df["Petal_Area"] = df["petal_length"] * df["petal_width"]

# --- Plots ---
plt.figure(figsize=(6,4))
plt.hist(df["sepal_length"], bins=10, color="skyblue", edgecolor="black")
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(df["petal_length"], df["petal_width"], c="red", alpha=0.7)
plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

plt.figure(figsize=(6,4))
species_avg["sepal_length"].plot(kind="bar", color=["blue", "green", "orange"])
plt.title("Average Sepal Length by Species")
plt.ylabel("Average Sepal Length")
plt.xlabel("Species")
plt.show()