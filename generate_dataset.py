"""
Creaing a synthetic Geotab telematics shape dataset for a taxi fleet operating in Manitoba, Canada (Winnipeg-centred, with Brandon / Portage la
Prairie / Steinbach spread).

Output: geotab_taxi_manitoba_demo.csv

One row per event, plus one row per event-free trip (event columns blank).
Keeping trip-level (km, hours) and event-level behaviour in a single file.

Columns:
    VehicleID, VIN, DeviceID, TripID, StartTimestamp, EndTimestamp, DistanceKM,
    DurationSeconds, LocationID, Latitude, Longitude, EventTimestamp, EventID,
    EventType, Severity, Timestamp, ScoreID, Period, Score
"""

from __future__ import annotations

import argparse
import math
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 20260813
random.seed(SEED)
np.random.seed(SEED)

N_VEHICLES = 300
DAYS = 365 
END_DATE = datetime(2026, 8, 1)
START_DATE = END_DATE - timedelta(days=DAYS)

# Manitoba geography (lat, lon, weight, spread_deg) 
CITIES = [
    ("WPG", 49.8951, -97.1384, 0.86, 0.055),
    ("BDN", 49.8483, -99.9501, 0.06, 0.020),
    ("PLP", 49.9730, -98.2926, 0.04, 0.015),
    ("STB", 49.5258, -96.6843, 0.04, 0.015),
]

HOTSPOTS = [
    ("Portage & Main", 49.8955, -97.1385),
    ("Portage & Memorial", 49.8886, -97.1500),
    ("Bishop Grandin & St Mary's", 49.8231, -97.1069),
    ("McPhillips & Inkster", 49.9349, -97.1560),
    ("Pembina & Bishop Grandin", 49.8213, -97.1524),
    ("Regent & Lagimodiere", 49.8896, -97.0421),
    ("Kenaston & Sterling Lyon", 49.8493, -97.2079),
    ("Main & Higgins", 49.9036, -97.1329),
]

EVENT_TYPES = {
    # type                base_rate_per_100km, severity distribution (1..5)
    "HarshBraking":       (2.60, [0.34, 0.30, 0.20, 0.11, 0.05]),
    "HarshAcceleration":  (1.90, [0.38, 0.30, 0.19, 0.09, 0.04]),
    "HarshCornering":     (1.10, [0.40, 0.31, 0.18, 0.08, 0.03]),
    "Speeding":           (2.20, [0.30, 0.28, 0.22, 0.14, 0.06]),
    "Idling":             (3.10, [0.55, 0.28, 0.12, 0.04, 0.01]),
    "SeatbeltUnbuckled":  (0.70, [0.25, 0.30, 0.25, 0.15, 0.05]),
    "DistractedDriving":  (0.55, [0.20, 0.26, 0.26, 0.19, 0.09]),
    "Collision":          (0.012, [0.05, 0.10, 0.25, 0.35, 0.25]),  # rare, severe
}

VIN_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # no I, O, Q real VIN rule


def make_vin(i: int) -> str:
    body = "".join(random.choice(VIN_CHARS) for _ in range(8))
    return f"2C4{body}{i:06d}"[:17].ljust(17, "0")


def pick_city():
    r = random.random()
    acc = 0.0
    for code, lat, lon, w, spread in CITIES:
        acc += w
        if r <= acc:
            return code, lat, lon, spread
    return CITIES[0][0], CITIES[0][1], CITIES[0][2], CITIES[0][4]


def winter_factor(dt: datetime) -> float:
    """Manitoba winters (ice, snow, dark at 16:30) measurably raise harsh-event
    rates. Sinusoid peaking in January, trough in July."""
    doy = dt.timetuple().tm_yday
    return 1.0 + 0.35 * math.cos(2 * math.pi * (doy - 15) / 365.0)


def hour_factor(hour: int) -> float:
    # Rush hours and 02:00-03:00 are the risky windows for taxis
    table = {0: 1.15, 1: 1.30, 2: 1.55, 3: 1.40, 4: 1.05, 5: 0.85, 6: 0.90,
             7: 1.25, 8: 1.35, 9: 1.05, 10: 0.90, 11: 0.95, 12: 1.05, 13: 1.00,
             14: 1.00, 15: 1.20, 16: 1.40, 17: 1.45, 18: 1.20, 19: 1.00,
             20: 0.95, 21: 1.00, 22: 1.10, 23: 1.15}
    return table[hour]


def build() -> pd.DataFrame:
    vehicles = []
    for i in range(1, N_VEHICLES + 1):
        code, clat, clon, spread = pick_city()
        vehicles.append({
            "VehicleID": f"V{i:04d}",
            "VIN": make_vin(i),
            "DeviceID": f"GO9-{random.randint(100000, 999999)}",
            "home": (code, clat, clon, spread),
            # Latent per-driver risk multiplier: log-normal, so a minority of drivers are markedly riskier - the classic telematics long tail.
            "risk": float(np.clip(np.random.lognormal(0.0, 0.45), 0.25, 4.0)),
            # Shift bias: 0 = day driver, 1 = night driver.
            "night": random.random() < 0.38,
            "util": float(np.clip(np.random.normal(1.0, 0.25), 0.35, 1.9)),
        })

    rows = []
    trip_n = 0
    event_n = 0

    for v in vehicles:
        code, clat, clon, spread = v["home"]
        for d in range(DAYS):
            day = START_DATE + timedelta(days=d)
            # For day sampling, each vehicle contributes approximately ~28% of the calendar year - across all 12 months ()
            if random.random() < 0.72:
                continue
            n_trips = max(1, int(np.random.poisson(14 * v["util"])))
            for _ in range(n_trips):
                trip_n += 1
                trip_id = f"T{trip_n:07d}"

                if v["night"]:
                    hour = int(np.random.choice(
                        range(24),
                        p=np.array([.09,.09,.09,.07,.04,.02,.02,.02,.02,.02,.02,.02,.02,.02,.02,.03,.03,.03,.04,.05,.05,.06,.06,.07])
                        / np.array([.09,.09,.09,.07,.04,.02,.02,.02,.02,.02,.02,.02,.02,.02,.02,.03,.03,.03,.04,.05,.05,.06,.06,.07]).sum()))
                else:
                    hour = int(np.random.choice(
                        range(24),
                        p=np.array([.01,.01,.01,.01,.01,.02,.05,.08,.08,.06,.05,.05,.06,.05,.05,.06,.08,.08,.06,.04,.03,.02,.02,.01])
                        / np.array([.01,.01,.01,.01,.01,.02,.05,.08,.08,.06,.05,.05,.06,.05,.05,.06,.08,.08,.06,.04,.03,.02,.02,.01]).sum()))

                start = day.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
                # Urban taxi trips: mostly short, occasional airport/inter-city run.
                dist = float(np.clip(np.random.lognormal(1.35, 0.75), 0.4, 120.0))
                # Average speed depends on hour (congestion) and trip length.
                avg_kph = np.clip(np.random.normal(30 + 12 * math.log1p(dist), 6), 8, 95)
                dur = int(dist / avg_kph * 3600)
                end = start + timedelta(seconds=dur)

                lat = clat + np.random.normal(0, spread)
                lon = clon + np.random.normal(0, spread * 1.5)
                loc_id = f"L{code}-{abs(hash((round(lat,3), round(lon,3)))) % 99999:05d}"

                exposure = dist / 100.0
                mult = v["risk"] * winter_factor(start) * hour_factor(hour)

                trip_events = []
                for etype, (rate, sev_p) in EVENT_TYPES.items():
                    lam = rate * exposure * mult
                    for _ in range(np.random.poisson(lam)):
                        event_n += 1
                        # 45% of events snap to a known hotspot corridor.
                        if code == "WPG" and random.random() < 0.45:
                            _, hlat, hlon = random.choice(HOTSPOTS)
                            elat = hlat + np.random.normal(0, 0.0035)
                            elon = hlon + np.random.normal(0, 0.0050)
                        else:
                            elat = lat + np.random.normal(0, spread * 0.4)
                            elon = lon + np.random.normal(0, spread * 0.6)
                        ets = start + timedelta(seconds=random.randint(0, max(1, dur)))
                        trip_events.append({
                            "EventID": f"E{event_n:08d}",
                            "EventType": etype,
                            "Severity": int(np.random.choice([1, 2, 3, 4, 5], p=sev_p)),
                            "EventTimestamp": ets,
                            "Latitude": round(float(elat), 6),
                            "Longitude": round(float(elon), 6),
                        })

                base = {
                    "VehicleID": v["VehicleID"], "VIN": v["VIN"],
                    "DeviceID": v["DeviceID"], "TripID": trip_id,
                    "StartTimestamp": start, "EndTimestamp": end,
                    "DistanceKM": round(dist, 3), "DurationSeconds": dur,
                    "LocationID": loc_id,
                }
                if trip_events:
                    for e in trip_events:
                        rows.append({**base, **e})
                else:
                    # Event-free trip: keeps the exposure denominator honest.
                    rows.append({**base, "EventID": None, "EventType": None,
                                 "Severity": None, "EventTimestamp": None,
                                 "Latitude": round(float(lat), 6),
                                 "Longitude": round(float(lon), 6)})

    df = pd.DataFrame(rows)

    # Geotab-style Safety Score, per vehicle per monthly Period: a 0-100 score where higher = safer.
    # We derive it from the vehicle's own severity-weighted event rate so the score is consistent with the raw events 
    # essential, otherwise the analysis would be fitting noise instead of behaviour.
    df["Period"] = pd.to_datetime(df["StartTimestamp"]).dt.to_period("M").astype(str)
    grp = df.groupby(["VehicleID", "Period"])
    agg = grp.agg(km=("DistanceKM", lambda s: s.groupby(df.loc[s.index, "TripID"]).first().sum()),
                  wsev=("Severity", "sum")).reset_index()
    agg["rate"] = agg["wsev"].fillna(0) / (agg["km"] / 100.0).replace(0, np.nan)
    rmax = agg["rate"].quantile(0.98)
    agg["Score"] = (100 - 100 * (agg["rate"] / rmax).clip(0, 1)).round(1)
    agg["Score"] = (agg["Score"] + np.random.normal(0, 1.5, len(agg))).clip(0, 100).round(1)
    agg["ScoreID"] = ["S" + f"{i:07d}" for i in range(1, len(agg) + 1)]

    df = df.merge(agg[["VehicleID", "Period", "ScoreID", "Score"]],
                  on=["VehicleID", "Period"], how="left")

    # "Timestamp" = the score record's stamp (period end) -- part of the spec.
    df["Timestamp"] = pd.PeriodIndex(df["Period"], freq="M").to_timestamp(how="end").floor("s")

    cols = ["VehicleID", "VIN", "DeviceID", "TripID", "StartTimestamp", "EndTimestamp",
            "DistanceKM", "DurationSeconds", "LocationID", "Latitude", "Longitude",
            "EventTimestamp", "EventID", "EventType", "Severity", "Timestamp",
            "ScoreID", "Period", "Score"]
    return df[cols].sort_values(["VehicleID", "StartTimestamp", "EventTimestamp"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", default="geotab_taxi_manitoba_demo.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out = build()
    out.to_csv(a.out, index=False)
    print(f"rows={len(out):,}  vehicles={out.VehicleID.nunique()}  "
          f"trips={out.TripID.nunique():,}  events={out.EventID.notna().sum():,}")
    print(f"written -> {a.out}")
