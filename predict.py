import pandas as pd
import joblib

ach_model = joblib.load("ach_model.pkl")
door_model = joblib.load("door_pct_model.pkl")

def predict_classroom_status(pm25_start, pm25_at_10min, pm25_end, decline_rate_10min, room_volume, cadr_rated):
    features = pd.DataFrame([{
        "pm25_start": pm25_start,
        "pm25_at_10min": pm25_at_10min,
        "pm25_end": pm25_end,
        "decline_rate_10min": decline_rate_10min,
        "room_volume": room_volume,
        "cadr_rated": cadr_rated
    }])

    ach_pred = ach_model.predict(features)[0]
    door_pred = door_model.predict(features)[0]
    clog_pred = 100 * (1 - (ach_pred - door_pred/100 * 9) * room_volume / cadr_rated)
    clog_pred = max(0, min(100, clog_pred))

    # Simple readiness rule: CEEW's target is ~5 ACH
    is_compliant = ach_pred >= 5

    return {
        "predicted_ach": round(float(ach_pred), 2),
        "predicted_clog_pct": round(float(clog_pred), 1),
        "predicted_door_open_pct": round(float(door_pred), 1),
        "ceew_compliant": bool(is_compliant),
        "target_ach": 5
    }

if __name__ == "__main__":
    # Test with a made-up "bad" scenario
    result = predict_classroom_status(
        pm25_start=150, pm25_at_10min=95, pm25_end=60,
        decline_rate_10min=5.5, room_volume=180, cadr_rated=280
    )
    print(result)