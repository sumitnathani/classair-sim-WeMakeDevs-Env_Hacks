import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

data = pd.read_csv("training_data.csv")

feature_cols = ["pm25_start", "pm25_at_10min", "pm25_end", "decline_rate_10min", "room_volume", "cadr_rated"]
X = data[feature_cols]

targets = {
    "ach": "true_ach",
    "clog_pct": "true_clog_pct",
    "door_pct": "true_door_pct"
}

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

models = {}
for name, col in targets.items():
    y = data[col]
    y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    print(f"{name}: MAE = {mae:.2f} (target range: {y.min():.1f} to {y.max():.1f})")

    models[name] = model
    joblib.dump(model, f"{name}_model.pkl")

print("\nModels saved: ach_model.pkl, clog_pct_model.pkl, door_pct_model.pkl")