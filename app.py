import os
from flask import Flask, request, jsonify, render_template
import joblib
import google.generativeai as genai
import json as json_lib

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
        overview = f"Good news — this classroom is performing well. The purifier is maintaining an air change rate above the recommended level, which means the air is being refreshed often enough to keep pollution levels in check for the students spending their day here."
        actions = "No urgent action needed right now. Keep up routine maintenance: check the filter roughly every 2-3 months depending on usage, and continue normal door habits during class hours."
    else:
        if result["predicted_clog_pct"] >= result["predicted_door_open_pct"]:
            overview = f"This classroom is falling short of the recommended air change rate, and the main reason appears to be the purifier's filter. A clogged filter can quietly undercut a purifier's performance well before anyone notices, since the unit keeps running and sounding normal even as it does less and less actual cleaning."
            actions = (f"Priority fix: replace or thoroughly clean the purifier's filter. A filter clogged to roughly {result['predicted_clog_pct']}% is forcing the unit's fan to push air through a much narrower effective opening, which sharply cuts down the real volume of air it can clean per hour even though it appears to be running fine. Over time, trapped dust and particulate matter can also start recirculating back into the room instead of being captured. Inspect the filter today, replace it if it's visibly grey or dust-caked, and set a recurring reminder to check it every 4-6 weeks given Delhi's pollution load. Also take a look at the door — it's estimated to be open around "
                       f"{result['predicted_door_open_pct']}% of class time, which independently lets outdoor pollution back in and works against the purifier regardless of filter condition.")
        else:
            overview = f"This classroom is falling short of the recommended air change rate, and the main reason appears to be the door being left open during class. Even a well-maintained purifier can't keep up if outdoor air is continuously flowing back into the room."
            actions = (f"Priority fix: address the open door. It's estimated to be open roughly {result['predicted_door_open_pct']}% of class time, and every time it opens, outdoor pollution pours back into the room faster than the purifier can filter it out — effectively resetting the air quality gains the unit just made. Start with a simple, low-cost fix: brief teachers and students on keeping the door closed during class, and consider a door closer/spring hinge so it shuts automatically instead of relying on habit. Also check the filter — it's estimated at "
                      f"{result['predicted_clog_pct']}% clogged, which is a secondary but still meaningful contributor, so don't ignore it once the door habit is fixed.")
    return {"overview": overview, "actions": actions}
def generate_report(result):
    try:
        prompt = f"""You are a veteran indoor air quality consultant who has personally inspected thousands of Indian school classrooms. You're writing a report for a school principal with no technical background, but you talk like someone who really knows this field deeply and cares about the kids breathing that air every day.

Classroom sensor data:
- Predicted Air Changes per Hour (ACH): {result['predicted_ach']} (CEEW recommended target: 5 ACH)
- Estimated filter clog level: {result['predicted_clog_pct']}%
- Estimated door-open percentage during class: {result['predicted_door_open_pct']}%
- Compliance status: {result['ceew_compliant']}

Respond with ONLY a valid JSON object, no markdown formatting, no code fences, in exactly this shape:

{{
  "overview": "A warm, confident, conversational 5-7 sentence paragraph. Do NOT lead with numbers. Explain in plain language whether this room is safe and performing well, what the single biggest issue is (if any), and why it matters for the children in that room day after day. Write like a specialist explaining this to a worried parent, not a report generator.",
  "actions": "A detailed, confident, highly instructive action plan, 150-250 words. If the filter is a major contributor to the problem (compare clog% vs door%), go deep on exactly why a clogged filter fails, what's physically happening inside it, step by step what the facilities team should do to replace or clean it, how often this should be checked going forward, and what happens if it's ignored. If the open door is a major contributor, go equally deep on exactly why an open door defeats the purifier's work, the airflow/pressure mechanism behind it, concrete behavioral or physical fixes (door closers, signage, teacher habit changes, physical door seals), and what to monitor afterward. Address whichever is the bigger contributor first and in more depth, then briefly mention the other if it's also elevated. Write with real expertise and specificity — this should sound like it came from someone who has fixed this exact problem hundreds of times, not generic advice."
}}

Do not include any text outside the JSON object."""

        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json_lib.loads(raw.strip())
        return {"overview": parsed["overview"], "actions": parsed["actions"]}
    except Exception as e:
        print(f"Gemini call failed, using fallback: {e}")
        return generate_fallback_report(result)

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    result = predict_classroom_status(**data)
    return jsonify(result)

@app.route("/report", methods=["POST"])
def report():
    data = request.get_json()
    result = predict_classroom_status(**data)
    report_data = generate_report(result)
    result["overview"] = report_data["overview"]
    result["actions"] = report_data["actions"]
    return jsonify(result)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
