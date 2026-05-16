"""Generate a realistic mock physiological stream for the live dashboard.

Phases: calm baseline -> rising stress -> elevated anxiety -> recovery -> calm.
Each row = one snapshot at ~1.5s cadence. 200 rows = ~5 minutes of "wear time".
"""
import csv
import math
import random
from pathlib import Path

random.seed(7)
N = 200
HERE = Path(__file__).parent
OUT = HERE / "sample_stream.csv"


def jitter(amp):
    return random.uniform(-amp, amp)


def phase(progress, calm, peak):
    """Smooth ramp from calm -> peak using a half-cosine."""
    p = (1 - math.cos(progress * math.pi)) / 2
    return calm + (peak - calm) * p


rows = []
for i in range(N):
    t = i / (N - 1)  # 0..1
    if t < 0.20:                # calm baseline
        hr  = 68 + jitter(2.5)
        eda = 0.22 + jitter(0.05)
        st  = 32.2 + jitter(0.15)
    elif t < 0.45:              # rising stress (0.20 -> 0.45)
        p = (t - 0.20) / 0.25
        hr  = phase(p, 70, 96)  + jitter(2.5)
        eda = phase(p, 0.25, 0.85) + jitter(0.07)
        st  = phase(p, 32.3, 33.6) + jitter(0.12)
    elif t < 0.65:              # elevated plateau
        hr  = 105 + jitter(4)
        eda = 1.10 + jitter(0.15)
        st  = 34.0 + jitter(0.18)
    elif t < 0.85:              # recovery (0.65 -> 0.85)
        p = (t - 0.65) / 0.20
        hr  = phase(p, 108, 75) + jitter(2.5)
        eda = phase(p, 1.05, 0.30) + jitter(0.08)
        st  = phase(p, 33.9, 32.5) + jitter(0.14)
    else:                       # back to calm
        hr  = 72 + jitter(2)
        eda = 0.25 + jitter(0.05)
        st  = 32.4 + jitter(0.12)

    rows.append({
        "t":   i,
        "hr":  round(hr, 1),
        "eda": round(max(0.05, eda), 3),
        "st":  round(st, 2),
    })

with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["t", "hr", "eda", "st"])
    w.writeheader()
    w.writerows(rows)

print(f"Wrote {len(rows)} rows -> {OUT}")
