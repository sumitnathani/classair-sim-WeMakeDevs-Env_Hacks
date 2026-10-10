import os
from flask import Flask, request, jsonify, render_template
import joblib
import google.generativeai as genai

app = Flask(__name__)
ach_model = joblib.load("ach_model.pkl")
door_model = joblib.load("door_pct_model.pkl")

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

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

def generate_fallback_report(result):
    if result["ceew_compliant"]:
        return f"This classroom meets the CEEW ventilation target with an estimated {result['predicted_ach']} ACH. No immediate action needed."
    else:
        cause = "a clogged filter" if result["predicted_clog_pct"] > result["predicted_door_open_pct"] else "an open door reducing purifier effectiveness"
        return f"This classroom falls short of the CEEW target ({result['predicted_ach']} ACH vs 5 ACH target), likely due to {cause}. Estimated filter clog: {result['predicted_clog_pct']}%, door-open time: {result['predicted_door_open_pct']}%. Recommended action: inspect and address the likely cause above."

def generate_report(result):
    try:
        prompt = f"""You are an expert indoor air quality consultant writing a report for a school principal who has no technical background. You specialize in classroom ventilation and CEEW's air purifier guidelines for Indian schools.

Classroom sensor data:
- Predicted Air Changes per Hour (ACH): {result['predicted_ach']} (CEEW recommended target for classrooms: 5 ACH)
- Estimated filter clog level: {result['predicted_clog_pct']}%
- Estimated door-open percentage during class: {result['predicted_door_open_pct']}%
- Compliance status: {result['ceew_compliant']}

Write a confident, authoritative report with these sections:

1. **Status Summary** — one sentence stating whether the room meets the CEEW ventilation standard, in plain terms a principal would understand.
2. **What This Means** — explain, using a simple analogy, what the ACH number actually represents for the children in that room (e.g. how many times the air is fully replaced per hour, and why that matters for health).
3. **Likely Cause** — if non-compliant, diagnose the probable root cause from the clog% and door% data, and briefly explain the mechanism (e.g. why a clogged filter reduces airflow).
4. **Recommended Actions** — 2-3 specific, concrete steps, ordered by priority and ease of implementation.
5. **Technical Note** — one sentence for the facilities team with the precise numbers, for their maintenance log.

Write with expertise and confidence — this should read like it came from a specialist who has inspected thousands of classrooms, not a generic chatbot. Be specific and instructive rather than vague. Keep the total length to under 200 words.."""

        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini call failed, using fallback: {e}")
        return generate_fallback_report(result)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_classroom_status(**data)
    return jsonify(result)

@app.route("/report", methods=["POST"])
def report():
    data = request.get_json()
    result = predict_classroom_status(**data)
    result["report"] = generate_report(result)
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
