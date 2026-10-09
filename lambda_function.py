import json
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
    is_compliant = ach_pred >= 5

    return {
        "predicted_ach": round(float(ach_pred), 2),
        "predicted_clog_pct": round(float(clog_pred), 1),
        "predicted_door_open_pct": round(float(door_pred), 1),
        "ceew_compliant": bool(is_compliant),
        "target_ach": 5
    }

def lambda_handler(event, context):
    # Lambda receives input as a JSON string in event["body"] when called via API Gateway
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event

    result = predict_classroom_status(
        pm25_start=body["pm25_start"],
        pm25_at_10min=body["pm25_at_10min"],
        pm25_end=body["pm25_end"],
        decline_rate_10min=body["decline_rate_10min"],
        room_volume=body["room_volume"],
        cadr_rated=body["cadr_rated"]
    )

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result)
    }