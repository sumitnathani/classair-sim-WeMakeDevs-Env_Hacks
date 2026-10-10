from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
ach_model = joblib.load("ach_model.pkl")
door_model = joblib.load("door_pct_model.pkl")

def predict_classroom_status(pm25_start, pm25_at_10min, pm25_end, decline_rate_10min, room_volume, cadr_rated):
    features = [[pm25_start, pm25_at_10min, pm25_end, decline_rate_10min, room_volume, cadr_rated]]
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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_classroom_status(
        pm25_start=data["pm25_start"],
        pm25_at_10min=data["pm25_at_10min"],
        pm25_end=data["pm25_end"],
        decline_rate_10min=data["decline_rate_10min"],
        room_volume=data["room_volume"],
        cadr_rated=data["cadr_rated"]
    )
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return "ClassAir predictor is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
