import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error
import joblib

# Load the dataset we generated
data = pd.read_csv("training_data.csv")

# These are the INPUTS the model sees (what a real sensor would give us)
feature_cols = ["pm25_start", "pm25_at_10min", "pm25_end", "decline_rate_10min", "room_volume", "cadr_rated"]
X = data[feature_cols]

# Split into train/test so we can check accuracy on unseen scenarios
X_train, X_test, y_train_fault, y_test_fault = train_test_split(
    X, data["fault_label"], test_size=0.2, random_state=42
)

# --- Model 1: classify which fault is happening ---
fault_model = RandomForestClassifier(n_estimators=200, random_state=42)
fault_model.fit(X_train, y_train_fault)
fault_preds = fault_model.predict(X_test)

print("=== Fault Classification Report ===")
print(classification_report(y_test_fault, fault_preds))

# --- Model 2: predict the actual ACH value ---
y_train_ach = data.loc[X_train.index, "true_ach"]
y_test_ach = data.loc[X_test.index, "true_ach"]

ach_model = RandomForestRegressor(n_estimators=200, random_state=42)
ach_model.fit(X_train, y_train_ach)
ach_preds = ach_model.predict(X_test)

mae = mean_absolute_error(y_test_ach, ach_preds)
print(f"\n=== ACH Prediction ===")
print(f"Mean Absolute Error: {mae:.2f} ACH units")
print(f"(For reference, target ACH is ~5, so this tells you how far off we typically are)")

# Save both models so we can use them later without retraining
joblib.dump(fault_model, "fault_model.pkl")
joblib.dump(ach_model, "ach_model.pkl")
print("\nModels saved: fault_model.pkl, ach_model.pkl")