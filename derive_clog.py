import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("training_data.csv")
feature_cols = ["pm25_start", "pm25_at_10min", "pm25_end", "decline_rate_10min", "room_volume", "cadr_rated"]

ach_model = joblib.load("ach_model.pkl")
door_model = joblib.load("door_pct_model.pkl")

X = data[feature_cols]
ach_pred = ach_model.predict(X)
door_pred = door_model.predict(X)

# Algebraic derivation instead of a third ML model
clog_pred = 100 * (1 - (ach_pred - door_pred/100 * 9) * data["room_volume"] / data["cadr_rated"])
clog_pred = clog_pred.clip(0, 100)  # physically can't be outside this range

mae = mean_absolute_error(data["true_clog_pct"], clog_pred)
print(f"Derived clog_pct MAE: {mae:.2f} (vs ML model's 22.90)")