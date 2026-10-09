import numpy as np
import pandas as pd

def simulate_decay_curve(
    clog_pct,      # 0-100, how clogged the filter is
    door_pct,      # 0-100, how open the door is
    room_volume=180,   # cubic meters, typical classroom
    cadr=300,          # purifier's rated clean air delivery rate (m3/hr)
    initial_pm25=150,  # starting PM2.5 level (ug/m3)
    outdoor_pm25=120,  # outdoor reference level (infiltration source)
    duration_min=30,   # how long we simulate, in minutes
    noise_level=5      # random sensor noise
):
    # Clogging reduces effective CADR
    effective_cadr = cadr * (1 - clog_pct / 100)

    # Open door increases infiltration (air leaking back in from outside)
    infiltration_rate = (door_pct / 100) * 0.15 * room_volume  # m3/min equivalent

    # Natural air changes per hour from purifier + infiltration
    ach = (effective_cadr / room_volume) + (infiltration_rate * 60 / room_volume)

    # Decay constant (per minute)
    k = ach / 60

    timestamps = np.arange(0, duration_min, 1)
    pm25_values = []

    for t in timestamps:
        # Exponential decay toward a floor set by outdoor infiltration
        floor = outdoor_pm25 * (door_pct / 100) * 0.3
        value = floor + (initial_pm25 - floor) * np.exp(-k * t)
        # Add sensor noise
        value += np.random.normal(0, noise_level)
        pm25_values.append(max(value, 0))

    return pd.DataFrame({
        "minute": timestamps,
        "pm25": pm25_values,
        "clog_pct": clog_pct,
        "door_pct": door_pct,
        "ach": ach
    })

# Quick test: generate one sample curve and save it
if __name__ == "__main__":
    df = simulate_decay_curve(clog_pct=20, door_pct=10)
    df.to_csv("test_sample.csv", index=False)
    print(df.head(10))
    print(f"\nComputed ACH: {df['ach'].iloc[0]:.2f}")