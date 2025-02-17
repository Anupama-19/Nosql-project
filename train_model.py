import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# -------------------- Fetal Health Model --------------------

# Load the fetal health dataset
df_fetal = pd.read_csv("f_health.csv")

# Separate features and target variable for fetal health
X_fetal = df_fetal.drop(columns=["fetal_health"])  # Drop the target column
y_fetal = df_fetal["fetal_health"]  # Target column for fetal health

# Split the dataset into training and testing sets for fetal health
X_train_fetal, X_test_fetal, y_train_fetal, y_test_fetal = train_test_split(X_fetal, y_fetal, test_size=0.2, random_state=42)

# Train the fetal health model
model_fetal = RandomForestClassifier(random_state=42)
model_fetal.fit(X_train_fetal, y_train_fetal)

# Test the fetal health model
y_pred_fetal = model_fetal.predict(X_test_fetal)
accuracy_fetal = accuracy_score(y_test_fetal, y_pred_fetal)
print(f"Fetal Health Model Accuracy: {accuracy_fetal:.2f}")

# Save the fetal health model
joblib.dump(model_fetal, "fetal_health_model.pkl")
print("Fetal Health Model saved as 'fetal_health_model.pkl'")


# -------------------- Maternal Health Model --------------------

# Load the maternal health dataset
df_maternal = pd.read_csv("maternal_health.csv")

# Separate features and target variable for maternal health
X_maternal = df_maternal.drop(columns=["RiskLevel"])  # Drop the 'RiskLevel' column (target column)
y_maternal = df_maternal["RiskLevel"]  # Target column for maternal health

# Split the dataset into training and testing sets for maternal health
X_train_maternal, X_test_maternal, y_train_maternal, y_test_maternal = train_test_split(X_maternal, y_maternal, test_size=0.2, random_state=42)

# Train the maternal health model
model_maternal = RandomForestClassifier(random_state=42)
model_maternal.fit(X_train_maternal, y_train_maternal)

# Test the maternal health model
y_pred_maternal = model_maternal.predict(X_test_maternal)
accuracy_maternal = accuracy_score(y_test_maternal, y_pred_maternal)
print(f"Maternal Health Model Accuracy: {accuracy_maternal:.2f}")

# Save the maternal health model
joblib.dump(model_maternal, "maternal_health_model.pkl")
print("Maternal Health Model saved as 'maternal_health_model.pkl'")
