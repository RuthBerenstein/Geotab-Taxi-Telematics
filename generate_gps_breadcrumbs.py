"""
Adding the missing piece needed to draw REAL road paths: a timestamped GPS
breadcrumb log ("LogRecord" in MyGeotab terms).

TIMESTAMP ALONE IS NOT ENOUGH
The question was: "if we add a timestamp to each coordinate reading, can you show the real road path?".  
The honest answer is: a timestamp only helps if there are MANY coordinate readings per trip.  
The delivered flat CSV holds a single coordinate per trip (its start / pickup point), so ordering those points
in time can only ever produce straight hops between fares -- which is exactly the light-blue connector the dashboard used 
to draw.

A real Geotab GO device does not work that way: it streams a LogRecord every few seconds 
(curve-logging: denser through turns, sparser on a straight highway).  
That stream, ordered by timestamp, IS the driven path.  So this script synthesises that stream at Geotab-like fidelity:

    1. For each sampled trip, take the real pickup point from the main CSV and derive a drop-off point at the trip's 
      recorded DistanceKM (road distance, so the straight-line offset is scaled down by a ~1.35 detour factor).
    2. Ask OSRM (OpenStreetMap routing) for the actual driving route between those two points -> a polyline that follows 
      Winnipeg's real streets.
    3. Resample that polyline every ~10 s of travel time, spread the timestamps between the trip's StartTimestamp and 
      EndTimestamp, derive an instant speed per breadcrumb, and add ~4 m of GPS jitter so it looks like real hardware rather 
      than a perfect geometry.

With breadcrumbs the dashboard can finally answer route-level questions:
which CORRIDORS a risky driver actually uses, whether harsh events cluster on a particular approach, 
how much of a shift is spent on high-speed arterials vs. residential streets.  
Straight-line hops can answer none of that.

Output: geotab_taxi_manitoba_gps_logs.csv
    LogID, VehicleID, DeviceID, TripID, Timestamp, Latitude, Longitude, SpeedKPH, SequenceNo
"""

from __future__ import annotations

import argparse
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

SEED = 20260818
random.seed(SEED)
np.random.seed(SEED)

OSRM = "https://router.project-osrm.org/route/v1/driving"
# Road distance is always longer than the crow-flies distance. 1.35 is the
# classic urban circuity factor -- using it keeps the generated drop-off point
# roughly consistent with the trip's recorded DistanceKM.
DETOUR = 1.35
SAMPLE_SECONDS = 10.0  # breadcrumb cadence, close to Geotab curve-logging average
GPS_JITTER_DEG = 0.00004  # ~4 m of positional noise, like a real GO device


def destination(lat: float, lon: float, bearing_deg: float, km: float):
    """Great-circle offset -- gives a plausible drop-off point for a trip."""
    R = 6371.0
    br = math.radians(bearing_deg)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(km / R)
                     + math.cos(lat1) * math.sin(km / R) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(km / R) * math.cos(lat1),
                             math.cos(km / R) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def haversine_km(a, b) -> float:
    R = 6371.0
    dlat = math.radians(b[0] - a[0])
    dlon = math.radians(b[1] - a[1])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def osrm_route(p1, p2, session: requests.Session):
    """Real road geometry between two points. Returns [(lat, lon), ...] or None."""
    url = (f"{OSRM}/{p1[1]:.6f},{p1[0]:.6f};{p2[1]:.6f},{p2[0]:.6f}"
           "?overview=full&geometries=geojson")
    for attempt in range(3):
        try:
            r = session.get(url, timeout=25)
            if r.status_code != 200:
                continue
            js = r.json()
            if js.get("code") != "Ok" or not js.get("routes"):
                return None
            coords = js["routes"][0]["geometry"]["coordinates"]
            return [(c[1], c[0]) for c in coords]
        except Exception:
            continue
    return None


def resample(path, start_ts, duration_s: int):
    """
    Turning geometry into a *log*.
    OSRM returns shape points, not time-stamped positions. We walk the polyline, accumulate distance, and place a breadcrumb 
    every SAMPLE_SECONDS of the trip's own duration. That preserves two things the dashboard needs:
      - the timestamps line up with the trip window (and therefore with events),
      - instantaneous speed varies with the local geometry (dense shape points = turns/congestion = lower speed), 
      which is what makes a replayed route look and behave like real telematics.
    """
    if len(path) < 2 or duration_s <= 0:
        return []
    seg = [haversine_km(path[i], path[i + 1]) for i in range(len(path) - 1)]
    total = sum(seg)
    if total <= 0:
        return []
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    n = max(2, min(240, int(duration_s / SAMPLE_SECONDS) + 1))
    out = []
    prev = None
    for k in range(n):
        frac = k / (n - 1)
        target = frac * total
        j = int(np.searchsorted(cum, target, side="right") - 1)
        j = max(0, min(j, len(seg) - 1))
        local = (target - cum[j]) / seg[j] if seg[j] > 0 else 0.0
        lat = path[j][0] + (path[j + 1][0] - path[j][0]) * local
        lon = path[j][1] + (path[j + 1][1] - path[j][1]) * local
        lat += np.random.normal(0, GPS_JITTER_DEG)
        lon += np.random.normal(0, GPS_JITTER_DEG * 1.5)
        ts = start_ts + pd.Timedelta(seconds=frac * duration_s)
        if prev is None:
            spd = 0.0
        else:
            dt = (ts - prev[0]).total_seconds()
            spd = haversine_km((prev[1], prev[2]), (lat, lon)) / dt * 3600 if dt > 0 else 0.0
        out.append((ts, round(lat, 6), round(lon, 6), round(float(min(spd, 120.0)), 1)))
        prev = (ts, lat, lon)
    return out


def build(csv_path: str, trips_per_vehicle: int, workers: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["VehicleID", "DeviceID", "TripID", "StartTimestamp",
                                        "EndTimestamp", "DistanceKM", "DurationSeconds",
                                        "Latitude", "Longitude"],
                     parse_dates=["StartTimestamp", "EndTimestamp"])
    trips = (df.sort_values("TripID").groupby("TripID", as_index=False).first())
    # Keep the breadcrumb file to a sane size: a recent, representative slice of
    # each vehicle's shifts. Long inter-city runs are dropped -- they route fine
    # but dominate the map and add little coaching value.
    trips = trips[(trips["DistanceKM"] >= 0.6) & (trips["DistanceKM"] <= 35)]
    trips = (trips.sort_values("StartTimestamp")
                  .groupby("VehicleID", group_keys=False)
                  .tail(trips_per_vehicle)
                  .reset_index(drop=True))
    print(f"routing {len(trips):,} trips for {trips.VehicleID.nunique()} vehicles ...")

    session = requests.Session()

    def one(rec):
        lat, lon = float(rec.Latitude), float(rec.Longitude)
        bearing = random.uniform(0, 360)
        km = max(0.4, float(rec.DistanceKM) / DETOUR)
        dlat, dlon = destination(lat, lon, bearing, km)
        path = osrm_route((lat, lon), (dlat, dlon), session)
        if not path:
            return []
        pts = resample(path, rec.StartTimestamp, int(rec.DurationSeconds))
        return [{"VehicleID": rec.VehicleID, "DeviceID": rec.DeviceID, "TripID": rec.TripID,
                 "Timestamp": ts, "Latitude": la, "Longitude": lo, "SpeedKPH": sp,
                 "SequenceNo": i}
                for i, (ts, la, lo, sp) in enumerate(pts)]

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(one, trips.itertuples(index=False)):
            rows.extend(res)
            done += 1
            if done % 250 == 0:
                print(f"  {done:,}/{len(trips):,} trips routed, {len(rows):,} breadcrumbs")

    out = pd.DataFrame(rows).sort_values(["VehicleID", "Timestamp"]).reset_index(drop=True)
    out.insert(0, "LogID", ["G" + f"{i:08d}" for i in range(1, len(out) + 1)])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="geotab_taxi_manitoba_demo.csv")
    ap.add_argument("--out", default="geotab_taxi_manitoba_gps_logs.csv")
    ap.add_argument("--trips-per-vehicle", type=int, default=15)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    g = build(a.csv, a.trips_per_vehicle, a.workers)
    g.to_csv(a.out, index=False)
    print(f"breadcrumbs={len(g):,}  trips={g.TripID.nunique():,}  "
          f"vehicles={g.VehicleID.nunique()}  ->  {a.out}")
