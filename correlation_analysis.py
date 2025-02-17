import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
fetal_df = pd.read_csv("f_health.csv")  # Fetal Health Data
maternal_df = pd.read_csv("maternal_health.csv")  # Maternal Health Data

# Display first few rows (Optional)
print("Fetal Health Dataset:\n", fetal_df.head())
print("\nMaternal Health Dataset:\n", maternal_df.head())

# Merge datasets (If they have a common identifier like 'Patient_ID')
if 'Patient_ID' in fetal_df.columns and 'Patient_ID' in maternal_df.columns:
    merged_df = pd.merge(fetal_df, maternal_df, on="Patient_ID")
else:
    print("⚠️ No common column found! Merging by index (Not Recommended).")
    merged_df = pd.concat([fetal_df, maternal_df], axis=1)

# Drop non-numeric columns
merged_df_numeric = merged_df.select_dtypes(include=['number'])

# Compute Correlation Matrix
correlation_matrix = merged_df_numeric.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Between Maternal and Fetal Health Parameters")
plt.show()
