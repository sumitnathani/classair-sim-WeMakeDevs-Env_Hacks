import numpy as np
import pandas as pd
from simulate import simulate_decay_curve

np.random.seed(42)  # reproducible results

rows = []
n_samples = 500  # number of simulated classroom scenarios

for i in range(n_samples):
    clog_pct = np.random.uniform(0, 100)
    door_pct = np.random.uniform(0, 100)
    room_volume = np.random.uniform(140, 220)   # vary classroom size a bit
    cadr = np.random.uniform(200, 400)          # vary purifier strength
    noise = np.random.uniform(2, 8)             # vary sensor noise

    curve = simulate_decay_curve(
        clog_pct=clog_pct,
        door_pct=door_pct,
        room_volume=room_volume,
        cadr=cadr,
        noise_level=noise
    )

    # Extract simple features from the curve instead of keeping every point:
    # these are the inputs our model will actually see
    pm25_start = curve["pm25"].iloc[0]
    pm25_end = curve["pm25"].iloc[-1]
    pm25_at_10min = curve["pm25"].iloc[10]
    decline_rate = (pm25_start - pm25_at_10min) / 10  # per-minute drop, early window

    # Simple fault label logic: what's the DOMINANT problem here?
    if clog_pct > 60 and door_pct > 60:
        fault = "clogged_filter_and_door_open"
    elif clog_pct > 60:
        fault = "clogged_filter"
    elif door_pct > 60:
        fault = "door_open"
    else:
        fault = "normal"

    rows.append({
        "scenario_id": i,
        "pm25_start": pm25_start,
        "pm25_at_10min": pm25_at_10min,
        "pm25_end": pm25_end,
        "decline_rate_10min": decline_rate,
        "room_volume": room_volume,
        "cadr_rated": cadr,
        "true_ach": curve["ach"].iloc[0],
        "true_clog_pct": clog_pct,
        "true_door_pct": door_pct,
        "fault_label": fault
    })

#final call
dataset = pd.DataFrame(rows)
dataset.to_csv("training_data.csv", index=False)
print(f"Generated {len(dataset)} scenarios")
print(dataset.head())
print("\nFault label distribution:")
print(dataset["fault_label"].value_counts())