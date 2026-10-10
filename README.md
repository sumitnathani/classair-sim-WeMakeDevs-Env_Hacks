# ClassAir — Classroom Air Purifier Verification

Built for WeMakeDevs Environmental Hacks (AWS × WeMakeDevs), Heat and Water / Indoor Air track.

## The Problem

Delhi's government is installing air purifiers in 10,000+ school classrooms to combat
pollution-driven closures under GRAP. But buying a purifier doesn't guarantee it's
working — clogged filters, open doors, and undersized units can all silently defeat
a purifier's effectiveness, and there's currently no system to verify real-world
performance in a classroom.

ClassAir estimates a classroom's real ventilation performance (Air Changes per Hour,
or ACH) from PM2.5 sensor readings, flags the likely cause if it's underperforming
(clogged filter vs. open door), and generates a plain-language compliance report.

## How It Works

1. **Simulator** (`simulate.py`, `generate_dataset.py`) — generates physics-based
   synthetic PM2.5 decay curves for classrooms under varying conditions (filter clog %,
   door-open %, room size, purifier CADR rating, sensor noise). This is simulated data,
   not live hardware — see "Honest Limitations" below.
2. **Model** (`train_model.py`) — trains Random Forest regressors on the simulated data
   to predict ACH and door-open % from sensor-derived features. Filter clog % is
   derived algebraically from the predicted ACH and door% (see `app.py`), since this
   relationship is physically known rather than needing to be learned.
3. **API + Web App** (`app.py`, `templates/index.html`) — a Flask server, deployed on
   AWS EC2, that takes sensor readings and returns predicted ACH, clog %, door %, and
   a CEEW-compliance flag (target: 5 ACH).
4. **AI Report** — predictions are passed to an LLM (Gemini, after hitting AWS Bedrock's
   default new-account token quota — see below) to generate a plain-English report
   for school staff, with a rule-based fallback if the AI call fails.

## AWS Usage

- **EC2**: hosts the full application (model inference + Flask API + web interface),
  running live and publicly accessible.
- **Bedrock**: originally used for AI report generation via the Converse API
  (Amazon Nova / Anthropic Claude). We hit Bedrock's default daily token quota for new
  AWS accounts during testing, documented in `flask.log` history. We kept the Bedrock
  integration path in the codebase and switched the live report-generation call to
  Google's Gemini API as a practical workaround, with a rule-based fallback for
  reliability regardless of which AI provider is used.
- **IAM**: a dedicated role scoped to the EC2 instance for Bedrock access.

## Honest Limitations

- **Simulated, not live sensor data.** We built a physics-based simulator rather than
  physical hardware, due to sourcing/calibration constraints within the hackathon
  timeframe. A real deployment would need a calibration pass against actual PM2.5/CO2
  sensors (uncalibrated low-cost sensors are known to drift at high PM and humidity).
- **Filter clog % is the least certain prediction** of the three outputs (mean absolute
  error ~12.6 percentage points on simulated test data), since it's a secondary
  variable derived algebraically rather than directly measured.
- **AI-generated report quality depends on the provider's availability** — the
  rule-based fallback ensures the app always returns a usable report even if the
  AI call fails or is rate-limited.

## Running It
