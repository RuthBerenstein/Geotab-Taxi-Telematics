from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# CONFIG

DEFAULT_CSV = os.environ.get(
    "GEOTAB_CSV", "geotab_taxi_manitoba_demo.csv"
)
# Including Timestamped GPS breadcrumb log (Geotab "LogRecord" equivalent) for the driver map.
GPS_PATH = os.environ.get(
    "GEOTAB_GPS_CSV", "geotab_taxi_manitoba_gps_logs.csv"
)

# REMARK - the business assumption layer.
# A seatbelt violation weights encode how much each behaviour contributes to RISK, and kept in one visible dict.
# In production this belongs in a settings screen so the safety manager can tune it and watch the leaderboard move.
# Idling sits near zero because it is a fuel/emissions KPI, not a safety one - letting it leak into a SAFETY score is a classic way to lose driver trust.

#Idling is an event type in the Telematics data - it records when a vehicle's engin is running but the vehicle is not moving - for example: taxi waiting, a drive-through, or at the airport queue.

BEHAVIOUR_WEIGHT = {
    "Collision": 10.0,
    "DistractedDriving": 3.0,
    "Speeding": 2.5,
    "HarshBraking": 2.0,
    "HarshCornering": 1.8,
    "SeatbeltUnbuckled": 1.5,
    "HarshAcceleration": 1.2,
    "Idling": 0.2,
}

# Severity 1..5 scales super-linearly: a level-5 harsh brake is a near-miss,
# a level-1 is a firm stop at a yellow light. Linear weighting would drown the
# rare, genuinely dangerous events in a sea of minor ones.
SEVERITY_WEIGHT = {1: 0.5, 2: 1.0, 3: 1.8, 4: 3.0, 5: 5.0}

# Manitoba palette - dark slate dashboard, amber/red risk ramp.
RISK_SCALE = ["#1a9850", "#a6d96a", "#fee08b", "#fdae61", "#d73027"]
TEMPLATE = "plotly_dark"
CARD_STYLE = {"height": "100%"}



# LOAD + RE-NORMALIZATION
def load(csv_path: str):
    """
     split the file back into its natural entities before any maths:
        trips  -> one row per TripID  = the EXPOSURE table (the denominator)
        events -> one row per EventID = the BEHAVIOUR table (the numerator)
        scores -> one row per vehicle-period = the provider's own Safety Score
    """
    df = pd.read_csv(csv_path)

    for _c in ("StartTimestamp", "EndTimestamp", "EventTimestamp", "Timestamp"):
        if _c in df.columns:
            df[_c] = pd.to_datetime(df[_c], errors="coerce")


    trips = (
        df.sort_values("TripID")
        .groupby("TripID", as_index=False)
        .first()[
            [
                "TripID", "VehicleID", "VIN", "DeviceID", "StartTimestamp",
                "EndTimestamp", "DistanceKM", "DurationSeconds", "LocationID",
                "Latitude", "Longitude",
            ]
        ]
    )
    trips["Hours"] = trips["DurationSeconds"] / 3600.0
    trips["AvgKph"] = trips["DistanceKM"] / trips["Hours"].replace(0, np.nan)
    trips["Hour"] = trips["StartTimestamp"].dt.hour
    trips["DOW"] = trips["StartTimestamp"].dt.dayofweek
    trips["Month"] = trips["StartTimestamp"].dt.to_period("M").astype(str)
    trips["Winter"] = trips["StartTimestamp"].dt.month.isin([11, 12, 1, 2, 3])

    events = df[df["EventID"].notna()].copy()
    events["Severity"] = events["Severity"].astype(int)
    events["Hour"] = events["EventTimestamp"].dt.hour
    events["DOW"] = events["EventTimestamp"].dt.dayofweek
    events["Month"] = events["EventTimestamp"].dt.to_period("M").astype(str)
    events["Winter"] = events["EventTimestamp"].dt.month.isin([11, 12, 1, 2, 3])
    # RiskPoints is the atom of the whole model: behaviour weight x severity weight.
    events["RiskPoints"] = events["EventType"].map(BEHAVIOUR_WEIGHT).fillna(1.0) * events[
        "Severity"
    ].map(SEVERITY_WEIGHT)

    scores = (
        df.dropna(subset=["ScoreID"])
        .groupby(["VehicleID", "Period"], as_index=False)
        .agg(Score=("Score", "first"))
    )
    return trips, events, scores



# EXPOSURE-NORMALISED VEHICLE SCORECARD
def vehicle_scorecard(trips: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    exp = trips.groupby("VehicleID").agg(
        Trips=("TripID", "count"),
        KM=("DistanceKM", "sum"),
        Hours=("Hours", "sum"),
        AvgKph=("AvgKph", "mean"),
        NightShare=("Hour", lambda s: float(((s >= 22) | (s <= 5)).mean())),
        WinterShare=("Winter", "mean"),
    )

    pivot = events.pivot_table(
        index="VehicleID", columns="EventType", values="EventID", aggfunc="count"
    ).fillna(0.0)
    pts = events.groupby("VehicleID")["RiskPoints"].sum().rename("RiskPoints")
    sev = events.groupby("VehicleID")["Severity"].mean().rename("MeanSeverity")

    sc = exp.join(pivot).join(pts).join(sev).fillna(0.0)

    for et in BEHAVIOUR_WEIGHT:
        if et not in sc.columns:
            sc[et] = 0.0
        sc[f"{et}_per100km"] = sc[et] / (sc["KM"] / 100.0)
        sc[f"{et}_per100h"] = sc[et] / (sc["Hours"] / 100.0)

    # RiskRate = severity-weighted risk points per 100 km. This is the raw,
    # heavy-tailed physical quantity - a handful of drivers sit far right.
    sc["RiskRate"] = sc["RiskPoints"] / (sc["KM"] / 100.0)
    sc["RiskRatePerHour"] = sc["RiskPoints"] / (sc["Hours"] / 100.0)

    # RiskRate is statistically correct but unreadable ("your score is 41.7").
    # SafetyIndex maps it to 0-100 where 100 = safest in fleet. Percentiles are
    # by construction uniform, so never plot their histogram and claim a
    # distribution insight - plot RiskRate for that, and use SafetyIndex only
    # for ranking and for the number you show the driver.
    sc["SafetyIndex"] = (1 - sc["RiskRate"].rank(pct=True)) * 100
    sc["RiskPercentile"] = sc["RiskRate"].rank(pct=True) * 100
    sc["HadCollision"] = (sc.get("Collision", 0) > 0).astype(int)
    return sc.reset_index()

# MACHINE LEARNING

def driver_personas(sc: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Using unsupervised segmentation (KMeans) to turn a 300-row leaderboard
    into 4 coachable GROUPS. 
    A leaderboard tells you WHO is risky, 
    a persona tells you WHY, and therefore which intervention to buy, 
    "Aggressive urban" needs a smooth-driving module,
    "night-shift high-exposure" needs a fatigue,
    and shift-rotation policy - training them the same way wastes money.

    Features are all exposure-normalised so clusters reflect BEHAVIOUR STYLE,
    not simply how much the taxi drove. Standardisation is mandatory: KMeans is
    a Euclidean-distance algorithm and would otherwise be dominated by whichever
    feature has the largest raw units.
    """
    feats = [
        "HarshBraking_per100km", "HarshAcceleration_per100km",
        "HarshCornering_per100km", "Speeding_per100km",
        "DistractedDriving_per100km", "Idling_per100h",
        "AvgKph", "NightShare", "MeanSeverity",
    ]
    X = StandardScaler().fit_transform(sc[feats].fillna(0.0))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sc = sc.copy()
    sc["Cluster"] = km.labels_

    # Name clusters from their own profile so the label is self-explanatory in
    # the UI - an operator should never have to look up "what is cluster 2".
    prof = sc.groupby("Cluster")[feats + ["RiskRate"]].mean()
    names = {}
    for c, row in prof.iterrows():
        if row["RiskRate"] >= prof["RiskRate"].quantile(0.75):
            base = "High-risk aggressive"
        elif row["RiskRate"] <= prof["RiskRate"].quantile(0.25):
            base = "Low-risk steady"
        else:
            base = "Moderate mixed"
        if row["NightShare"] >= prof["NightShare"].median():
            base += " (night shift)"
        else:
            base += " (day shift)"
        names[c] = f"{base}"
    sc["Persona"] = sc["Cluster"].map(names)
    return sc


def collision_model(sc: pd.DataFrame):
    """
    Collisions are rare, cannot manage them by counting, but manage their PRECURSORS. 
    A gradient-boosting classifier learns which normalised behaviours separate vehicles that had a collision from those that did not,
    and its feature importances answer the executive question: "of everything we could coach, what actually predicts a crash?"

    The output is a RANKING, not as causal proof, and treat the predicted
    probability as a triage queue - call the top of the list this week.
    """
    feats = [
        "HarshBraking_per100km", "HarshAcceleration_per100km",
        "HarshCornering_per100km", "Speeding_per100km",
        "DistractedDriving_per100km", "SeatbeltUnbuckled_per100km",
        "Idling_per100h", "AvgKph", "NightShare", "WinterShare", "MeanSeverity",
    ]
    X, y = sc[feats].fillna(0.0), sc["HadCollision"]
    auc, imp = np.nan, pd.DataFrame({"Feature": feats, "Importance": 0.0})
    if y.nunique() == 2 and y.sum() >= 10:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        m = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr)
        auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        imp = pd.DataFrame(
            {"Feature": feats, "Importance": m.feature_importances_}
        ).sort_values("Importance", ascending=False)
        sc = sc.copy()
        sc["CollisionRisk"] = m.predict_proba(X)[:, 1]
    else:
        sc = sc.copy()
        sc["CollisionRisk"] = sc["RiskPercentile"] / 100.0
    return sc, imp, auc


# GEOSPATIAL AGGREGATION (the maps)

def hotspots(events: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
    """
    From dots to DECISIONS.
    Plotting 350k raw event pins is a heat blob nobody can act on. Binning coordinates to a ~100 m grid and ranking cells by severity-weighted risk
    turns the map into a prioritised worklist: the top cells are specific intersections and corridors.

    BUSINESS VALUE: this is the artefact you hand to a municipal road authority.
    "Our fleet logged 214 severity-4+ harsh-braking events at this approach in 12 months" is evidence for signal retiming, 
    sightline clearing or signage - a road-safety win that costs the operator nothing to harvest.
    """
  
    e = events.dropna(subset=["Latitude", "Longitude"]).copy()
    e["LatBin"] = e["Latitude"].round(precision)
    e["LonBin"] = e["Longitude"].round(precision)
    hs = (
        e.groupby(["LatBin", "LonBin"])
        .agg(
            Events=("EventID", "count"),
            RiskPoints=("RiskPoints", "sum"),
            MeanSeverity=("Severity", "mean"),
            Collisions=("EventType", lambda s: int((s == "Collision").sum())),
            TopEvent=("EventType", lambda s: s.value_counts().idxmax()),
        )
        .reset_index()
        .sort_values("RiskPoints", ascending=False)
    )
    return hs


# BUILD EVERYTHING ONCE (import-time, never inside a callback)

def build_state(csv_path: str):
    trips, events, scores = load(csv_path)
    sc = vehicle_scorecard(trips, events)
    sc = driver_personas(sc)
    sc, imp, auc = collision_model(sc)

    hs = hotspots(events)
    collisions = events[events["EventType"] == "Collision"].copy()

    hourly = (
        events.groupby("Hour")["RiskPoints"].sum().rename("RiskPoints").reset_index()
    )
    trip_hours = trips.groupby("Hour")["Hours"].sum().rename("Hours").reset_index()
    hourly = hourly.merge(trip_hours, on="Hour", how="outer").fillna(0.0)
    # REMARK -- exposure normalisation applies to TIME as well as to drivers.
    # 03:00 has few events simply because few taxis drive then. Dividing by
    # driving hours reveals that the night hours are the most dangerous PER HOUR
    # DRIVEN - the opposite conclusion to the raw count, and the one that
    # actually drives shift policy.
    hourly["RiskPer100h"] = hourly["RiskPoints"] / (hourly["Hours"] / 100.0)

    monthly = (
        events.groupby("Month")["RiskPoints"].sum().reset_index()
        .merge(trips.groupby("Month")["DistanceKM"].sum().reset_index(), on="Month", how="outer")
        .fillna(0.0)
        .sort_values("Month")
    )
    monthly["RiskPer100km"] = monthly["RiskPoints"] / (monthly["DistanceKM"] / 100.0)
    '''
    Adding timestamped GPS breadcrumb log (real driven paths).
    Since the main extract carries ONE coordinate per trip, so no timestamp trick can recover the road actually driven. 
    A Geotab GO device also streams LogRecord rows every few seconds; that timestamped position stream IS the route.
    generate_gps_breadcrumbs.py produces exactly that feed (routed on the real OpenStreetMap street network), and when the file is present the
    driver deep-dive map draws genuine road-following paths instead of straight-line hops between fares. If it is missing we degrade gracefully
    back to the honest connector view rather than inventing a route.
    ''' 
  
    gps = None
    if os.path.exists(GPS_PATH):
        gps = pd.read_csv(GPS_PATH)
        gps["Timestamp"] = pd.to_datetime(gps["Timestamp"], errors="coerce")
        gps = gps.dropna(subset=["Timestamp", "Latitude", "Longitude"])
        gps = gps.sort_values(["VehicleID", "TripID", "SequenceNo"])


    return dict(
        trips=trips, events=events, scores=scores, sc=sc, imp=imp, auc=auc,
        hotspots=hs, collisions=collisions, hourly=hourly, monthly=monthly,
        gps=gps,
    )


CSV_PATH = DEFAULT_CSV
if __name__ == "__main__":
    _p = argparse.ArgumentParser()
    _p.add_argument("--csv", default=DEFAULT_CSV)
    _p.add_argument("--port", type=int, default=8050)
    _args, _ = _p.parse_known_args()
    CSV_PATH = _args.csv

S = build_state(CSV_PATH)
VEHICLES = sorted(S["sc"]["VehicleID"].unique())
CENTER = dict(lat=49.895, lon=-97.138)  # Winnipeg


# FIGURE FACTORIES

def kpi_card(title: str, value: str, sub: str, color: str = "primary"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-uppercase small text-muted"),
                html.H3(value, className=f"text-{color} mb-0"),
                html.Div(sub, className="small text-muted"),
            ]
        ),
        className="shadow-sm",
        style=CARD_STYLE,
    )


def fig_collision_map():
    """
    MAP 1: where crashes actually happen.
    Two layers on one canvas:
      * a density heat layer of ALL severity-weighted events = the chronic risk surface (where near-misses accumulate);
      * individual collision markers = the acute outcomes.
    Overlaying them answers the question that justifies a telematics programme:
    do our harsh-braking clusters PREDICT our crash locations? Where they
    coincide, the near-miss data is a validated leading indicator and can be
    used to intervene BEFORE the next crash at that spot.
    """
    e = S["events"].dropna(subset=["Latitude", "Longitude"])
    # Sample the heat layer: 60k points render smoothly and are statistically
    # indistinguishable from 350k for a density surface.
    heat = e.sample(min(len(e), 60000), random_state=42)
    fig = go.Figure(
        go.Densitymap(
            lat=heat["Latitude"], lon=heat["Longitude"], z=heat["RiskPoints"],
            radius=12, colorscale="Inferno", opacity=0.65,
            colorbar=dict(title="Risk<br>density"),
            hoverinfo="skip",
        )
    )
    col = S["collisions"]
    if len(col):
        fig.add_trace(
            go.Scattermap(
                lat=col["Latitude"], lon=col["Longitude"], mode="markers",
                marker=dict(size=11, color="#00e5ff", symbol="circle"),
                name="Collision",
                customdata=np.stack(
                    [col["VehicleID"], col["Severity"],
                     col["EventTimestamp"].dt.strftime("%Y-%m-%d %H:%M")], axis=-1
                ),
                hovertemplate="<b>Collision</b><br>Vehicle %{customdata[0]}"
                              "<br>Severity %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
            )
        )
    fig.update_layout(
        map=dict(style="carto-darkmatter", center=CENTER, zoom=9.2),
        margin=dict(l=0, r=0, t=0, b=0), template=TEMPLATE, height=620,
        legend=dict(orientation="h", y=0.02, x=0.02, bgcolor="rgba(0,0,0,0.4)"),
    )
    return fig


def fig_hotspot_map(top_n: int = 300):
    """
    MAP 2: the prioritised worklist version of the same data.
    Bubble size = number of events, colour = severity-weighted risk. Unlike the heat map this is COUNTABLE and exportable: 
    each bubble is a grid cell that can be named, ticket and track over time.
    """
    hs = S["hotspots"].head(top_n)
    fig = px.scatter_map(
        hs, lat="LatBin", lon="LonBin", size="Events", color="RiskPoints",
        color_continuous_scale=RISK_SCALE, size_max=28, zoom=9.2,
        center=CENTER, hover_data={"Events": True, "RiskPoints": ":.0f",
                                   "MeanSeverity": ":.2f", "Collisions": True,
                                   "TopEvent": True, "LatBin": ":.3f", "LonBin": ":.3f"},
    )
    fig.update_layout(
        map_style="carto-darkmatter", template=TEMPLATE, height=620,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def fig_hourly():
    h = S["hourly"]
    fig = go.Figure()
    fig.add_bar(x=h["Hour"], y=h["RiskPoints"], name="Raw risk points",
                marker_color="#4c78a8", opacity=0.55)
    fig.add_scatter(x=h["Hour"], y=h["RiskPer100h"], name="Risk per 100 driving-hours",
                    yaxis="y2", mode="lines+markers", line=dict(color="#d73027", width=3))
    fig.update_layout(
        template=TEMPLATE, height=340, margin=dict(l=10, r=10, t=30, b=10),
        yaxis=dict(title="Raw"), yaxis2=dict(overlaying="y", side="right",
                                             title="Per 100 h", showgrid=False),
        xaxis=dict(title="Hour of day", dtick=2),
        legend=dict(orientation="h", y=1.15),
    )
    return fig


def fig_monthly():
    m = S["monthly"]
    fig = px.line(m, x="Month", y="RiskPer100km", markers=True,
                  template=TEMPLATE)
    fig.update_traces(line=dict(color="#fdae61", width=3))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Risk points / 100 km")
    return fig


def fig_risk_distribution():
    """
    plot RiskRate, never the percentile index.
    Percentiles are uniform by construction; their histogram is a flat bar chart that hides the real story. 
    The RiskRate distribution is heavy-tailed, which IS the story: 
    a small minority of vehicles generates a disproportionate share of risk, and that is what makes targeted coaching cheap and effective.
    """
    sc = S["sc"]
    fig = px.histogram(sc, x="RiskRate", nbins=45, template=TEMPLATE,
                       color_discrete_sequence=["#fdae61"])
    p90 = sc["RiskRate"].quantile(0.9)
    fig.add_vline(x=p90, line_dash="dash", line_color="#d73027",
                  annotation_text="worst decile threshold")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Risk points per 100 km", yaxis_title="Vehicles")
    return fig


def fig_personas():
    sc = S["sc"]
    fig = px.scatter(
        sc, x="Speeding_per100km", y="HarshBraking_per100km",
        color="Persona", size="KM", hover_name="VehicleID",
        hover_data={"SafetyIndex": ":.0f", "RiskRate": ":.1f", "NightShare": ":.2f"},
        template=TEMPLATE, size_max=22,
    )
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.22))
    return fig


def fig_importance():
    imp = S["imp"].head(10).sort_values("Importance")
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 template=TEMPLATE, color="Importance",
                 color_continuous_scale=RISK_SCALE)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                      coloraxis_showscale=False)
    return fig


def fig_event_mix():
    mix = (S["events"].groupby("EventType")
           .agg(Events=("EventID", "count"), Risk=("RiskPoints", "sum"))
           .reset_index().sort_values("Risk"))
    fig = px.bar(mix, x="Risk", y="EventType", orientation="h", template=TEMPLATE,
                 hover_data=["Events"], color="Risk", color_continuous_scale=RISK_SCALE)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                      coloraxis_showscale=False, xaxis_title="Total risk points")
    return fig


#  DRIVER DEEP DIVE FIGURES (driven by the dropdown)
def _bearing_glyph(lat1, lon1, lat2, lon2):
    """
    Plotly's map traces cannot rotate a marker symbol per-point, so we render a text glyph instead and choose it from the 8-point compass rose. 
    That is enough resolution for a human to read "this driver worked north-to-south down Pembina" at a glance.
    """
    import math as _m
    dy = lat2 - lat1
    # longitude degrees shrink with latitude; correct so the arrow is visually true
    dx = (lon2 - lon1) * _m.cos(_m.radians((lat1 + lat2) / 2.0))
    if dx == 0 and dy == 0:
        return None
    ang = (_m.degrees(_m.atan2(dy, dx)) + 360) % 360  # 0 = east, CCW
    glyphs = ["›", "◹", "︿", "◸", "‹", "◺", "﹀", "◿"]  # E NE N NW W SW S SE
    return glyphs[int((ang + 22.5) % 360 // 45)]


def fig_driver_route(vid: str):
    """
    MAP 3: the driver's own operating footprint, now on REAL ROADS.
    The trip table holds ONE coordinate per trip (the pickup point), so ordering those points by time can only ever draw straight hops between fares. 
    Real road paths need MANY timestamped positions per trip - the Geotab GO device's LogRecord stream. We now generate that stream
    (generate_gps_breadcrumbs.py: every trip routed over the real OpenStreetMap street network, resampled to a ~10 s breadcrumb cadence with
    per-point timestamps, speed and realistic GPS jitter). Ordering those breadcrumbs by Timestamp gives the ACTUAL driven path, which is what this
    map now draws - one polyline per trip, following Winnipeg's streets, with direction glyphs so the flow of the shift is readable.

    Speed colouring looks impressive but a coach cares about "which corridors does this driver work, and where do the events land on them". 
    Per-trip colouring keeps individual shifts separable; instantaneous speed is still on the hover tooltip for the moments that matter.

    This separates a BEHAVIOUR problem from a ROUTE problem.
    If harsh-braking pins scatter across every street the driver works, that is style and coaching fixes it.
    If they pile up on one approach the driver is repeatedly dispatched down, the fix is dispatch/routing - or a road report to the City. 
    With real road geometry you can now name the street in the coaching session ("your braking is all on the Bishop Grandin off-ramp"),
    which is the difference between an actionable review and an abstract score.
    """
    t = S["trips"][S["trips"]["VehicleID"] == vid].sort_values("StartTimestamp")
    e = S["events"][S["events"]["VehicleID"] == vid].dropna(subset=["Latitude", "Longitude"])
    t = t.dropna(subset=["Latitude", "Longitude"])
    gps = S.get("gps")
    g = None
    if gps is not None:
        g = gps[gps["VehicleID"] == vid]

    fig = go.Figure()

    if g is not None and len(g):
        # ---------- REAL ROAD PATHS from the timestamped breadcrumb log -------
        # One trace per trip would be dozens of legend entries; instead we stitch
        # the polylines into a single trace separated by None gaps (the standard
        # Plotly trick) so the map stays fast even with tens of thousands of
        # points, and add one marker trace for the trip origins.
        trips_order = (g.groupby("TripID")["Timestamp"].min()
                        .sort_values().index.tolist())
        trips_order = trips_order[-40:]  # last ~40 routed shifts: readable, recent
        lat_all, lon_all, txt_all = [], [], []
        alat, alon, atxt = [], [], []
        slat, slon, stxt = [], [], []
        for tid in trips_order:
            sub = g[g["TripID"] == tid]
            la = sub["Latitude"].to_numpy()
            lo = sub["Longitude"].to_numpy()
            ts = sub["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
            sp = sub["SpeedKPH"].to_numpy()
            lat_all.extend(la.tolist() + [None])
            lon_all.extend(lo.tolist() + [None])
            txt_all.extend([f"{tid}<br>{ts[i]}<br>{sp[i]:.0f} km/h"
                            for i in range(len(la))] + [""])
            slat.append(float(la[0])); slon.append(float(lo[0]))
            stxt.append(f"{tid} start<br>{ts[0]}")
            # direction glyphs: a few per trip, along the actual road geometry
            step = max(1, len(la) // 4)
            for i in range(0, len(la) - 1, step):
                gl = _bearing_glyph(la[i], lo[i], la[i + 1], lo[i + 1])
                if gl is None:
                    continue
                alat.append(float(la[i])); alon.append(float(lo[i])); atxt.append(gl)

        fig.add_trace(go.Scattermap(
            lat=lat_all, lon=lon_all, mode="lines",
            line=dict(width=2.5, color="#4dd0e1"),
            name="Actual driven route (GPS breadcrumbs, road-matched)",
            text=txt_all, hovertemplate="%{text}<extra></extra>",
        ))
        if atxt:
            fig.add_trace(go.Scattermap(
                lat=alat, lon=alon, mode="text", text=atxt,
                textfont=dict(size=14, color="#8beef9"),
                name="Direction of travel",
                hovertemplate="Direction of travel %{text}<extra></extra>",
            ))
        fig.add_trace(go.Scattermap(
            lat=slat, lon=slon, mode="markers",
            marker=dict(size=8, color="#1a9850"),
            name="Trip start (pickup)", text=stxt,
            hovertemplate="%{text}<extra></extra>",
        ))
        center = dict(lat=float(np.nanmedian([v for v in lat_all if v is not None])),
                      lon=float(np.nanmedian([v for v in lon_all if v is not None])))
        zoom = 11.0
    else:
        # Without LogRecords there is no path to draw, so we keep the old faint trip-to-trip connectors rather than fabricating a road route.
        center, zoom = CENTER, 9.5
        if len(t):
            seg = t.head(400)
            fig.add_trace(go.Scattermap(
                lat=seg["Latitude"], lon=seg["Longitude"], mode="lines+markers",
                line=dict(width=1, color="rgba(77,208,225,0.45)"),
                marker=dict(size=4, color="#4dd0e1"),
                name="Trip-to-trip connection (not a road path)",
                hovertemplate="Trip start<br>%{lat:.4f}, %{lon:.4f}<extra></extra>",
            ))
            lats = seg["Latitude"].to_numpy()
            lons = seg["Longitude"].to_numpy()
            step = max(1, (len(lats) - 1) // 120)
            alat, alon, atxt = [], [], []
            for i in range(0, len(lats) - 1, step):
                gl = _bearing_glyph(lats[i], lons[i], lats[i + 1], lons[i + 1])
                if gl is None:
                    continue
                alat.append((lats[i] + lats[i + 1]) / 2.0)
                alon.append((lons[i] + lons[i + 1]) / 2.0)
                atxt.append(gl)
            if atxt:
                fig.add_trace(go.Scattermap(
                    lat=alat, lon=alon, mode="text", text=atxt,
                    textfont=dict(size=15, color="#8beef9"),
                    name="Direction of travel",
                    hovertemplate="Direction of travel %{text}<extra></extra>",
                ))
            center = dict(lat=float(t["Latitude"].median()),
                          lon=float(t["Longitude"].median()))

    if len(e):
        for et, grp in e.groupby("EventType"):
            fig.add_trace(go.Scattermap(
                lat=grp["Latitude"], lon=grp["Longitude"], mode="markers",
                marker=dict(size=6 + grp["Severity"] * 2.5,
                            color="#d73027" if et == "Collision" else None),
                name=et,
                customdata=np.stack(
                    [grp["Severity"],
                     grp["EventTimestamp"].dt.strftime("%Y-%m-%d %H:%M")], axis=-1),
                hovertemplate=f"<b>{et}</b><br>Severity %{{customdata[0]}}"
                              "<br>%{customdata[1]}<extra></extra>",
            ))
    fig.update_layout(
        map=dict(style="carto-darkmatter", center=center, zoom=zoom),
        template=TEMPLATE, height=560, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=-0.02, bgcolor="rgba(0,0,0,0.4)"),
    )
    return fig


def fig_driver_profile(vid: str):
    """
    The fleet-relative radar.
    Absolute rates mean nothing to a driver ("1.9 harsh brakes per 100 km" - good or bad?). 
    Expressing each behaviour as a PERCENTILE against the fleet turns the review into a single sentence: 
    "you are fleet-average on speed but in the worst 10% on cornering". 
    Coaching one specific behaviour beats a generic "drive safer" every time.
    """
    sc = S["sc"]
    feats = ["HarshBraking_per100km", "HarshAcceleration_per100km",
             "HarshCornering_per100km", "Speeding_per100km",
             "DistractedDriving_per100km", "SeatbeltUnbuckled_per100km"]
    row = sc[sc["VehicleID"] == vid]
    vals = [float(sc[f].rank(pct=True)[row.index[0]] * 100) for f in feats]
    labels = [f.replace("_per100km", "") for f in feats]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself",
        line=dict(color="#d73027"), name=vid))
    fig.update_layout(template=TEMPLATE, height=380,
                      polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")),
                      margin=dict(l=30, r=30, t=30, b=30), showlegend=False)
    return fig


def fig_driver_hours(vid: str):
    e = S["events"][S["events"]["VehicleID"] == vid]
    t = S["trips"][S["trips"]["VehicleID"] == vid]
    ev = e.groupby("Hour")["RiskPoints"].sum().reindex(range(24), fill_value=0)
    hr = t.groupby("Hour")["Hours"].sum().reindex(range(24), fill_value=0)
    rate = ev / hr.replace(0, np.nan) * 100
    fig = go.Figure()
    fig.add_bar(x=list(range(24)), y=ev.values, name="Risk points",
                marker_color="#4c78a8", opacity=0.6)
    fig.add_scatter(x=list(range(24)), y=rate.values, name="Per 100 h",
                    yaxis="y2", line=dict(color="#d73027", width=3))
    fig.update_layout(template=TEMPLATE, height=380,
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(title="Hour of day", dtick=2),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.18))
    return fig


# DASH / FLASK APP

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP],
    title="Manitoba Taxi Road Safety",
    suppress_callback_exceptions=True,
)
server = app.server


@server.route("/healthz")
def _healthz():
    return {"status": "ok", "vehicles": len(VEHICLES)}


def header():
    sc = S["sc"]
    worst = sc.nlargest(max(1, len(sc) // 10), "RiskRate")
    share = worst["RiskPoints"].sum() / sc["RiskPoints"].sum() * 100
    auc = S["auc"]
    return dbc.Row(
        [
            dbc.Col(kpi_card("Vehicles", f"{len(sc):,}",
                             f"{S['trips']['DistanceKM'].sum()/1e3:,.0f}k km driven"), md=2),
            dbc.Col(kpi_card("Safety events", f"{len(S['events']):,}",
                             f"{S['events']['RiskPoints'].sum():,.0f} risk points", "warning"), md=2),
            dbc.Col(kpi_card("Collisions", f"{len(S['collisions']):,}",
                             "acute outcomes on the map", "danger"), md=2),
            dbc.Col(kpi_card("Worst decile risk share", f"{share:.0f}%",
                             "of all fleet risk - coach these first", "danger"), md=3),
            dbc.Col(kpi_card("Collision model AUC",
                             "n/a" if np.isnan(auc) else f"{auc:.2f}",
                             "precursors predicting crashes", "info"), md=3),
        ],
        className="g-3 mb-3",
    )


def tab_overview():
    return dbc.Container(
        [
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Risk by hour of day - raw vs exposure-normalised"),
                                  dbc.CardBody(dcc.Graph(figure=fig_hourly()))]), lg=6),
                dbc.Col(dbc.Card([dbc.CardHeader("Seasonality - risk per 100 km by month"),
                                  dbc.CardBody(dcc.Graph(figure=fig_monthly()))]), lg=6),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("Fleet risk-rate distribution (heavy tail = targetable)"),
                                  dbc.CardBody(dcc.Graph(figure=fig_risk_distribution()))]), lg=6),
                dbc.Col(dbc.Card([dbc.CardHeader("Where the risk comes from - event mix by weighted risk"),
                                  dbc.CardBody(dcc.Graph(figure=fig_event_mix()))]), lg=6),
            ], className="g-3 mb-3"),
            dbc.Alert(
                "Read the hourly chart as two different questions: the bars say when "
                "most events happen (a volume/staffing question), the red line says when "
                "driving is most dangerous per hour driven (a policy question). They peak "
                "at different times - the late-night rate is the one that changes shift rules.",
                color="secondary"),
        ], fluid=True)


def tab_maps():
    return dbc.Container(
        [
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Collision map: near-miss risk surface + actual collisions"),
                    dbc.CardBody(dcc.Graph(figure=fig_collision_map())),
                    dbc.CardFooter("Where cyan collision pins sit inside a hot near-miss zone, "
                                   "harsh-event density is a validated leading indicator - "
                                   "intervene there before the next crash."),
                ]), lg=6),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("Hotspot worklist: ~100 m grid cells ranked by weighted risk"),
                    dbc.CardBody(dcc.Graph(figure=fig_hotspot_map())),
                    dbc.CardFooter("Each bubble is a countable, exportable cell - the evidence "
                                   "pack for signal retiming or signage requests to the city."),
                ]), lg=6),
            ], className="g-3 mb-3"),
            dbc.Card([
                dbc.CardHeader("Top 15 hotspot cells"),
                dbc.CardBody(dash_table.DataTable(
                    data=S["hotspots"].head(15).round(4).to_dict("records"),
                    columns=[{"name": c, "id": c} for c in
                             ["LatBin", "LonBin", "Events", "RiskPoints",
                              "MeanSeverity", "Collisions", "TopEvent"]],
                    style_as_list_view=True,
                    style_header={"backgroundColor": "#222", "color": "#eee"},
                    style_cell={"backgroundColor": "#111", "color": "#ddd",
                                "border": "none", "fontSize": 13},
                )),
            ]),
        ], fluid=True)


def tab_ml():
    return dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Driver personas (KMeans on exposure-normalised behaviour)"),
                              dbc.CardBody(dcc.Graph(figure=fig_personas())),
                              dbc.CardFooter("Personas turn 300 individuals into 4 buyable "
                                             "interventions - smooth-driving module, fatigue "
                                             "policy, winter clinic, recognition programme.")]), lg=7),
            dbc.Col(dbc.Card([dbc.CardHeader("What predicts a collision (gradient boosting importances)"),
                              dbc.CardBody(dcc.Graph(figure=fig_importance())),
                              dbc.CardFooter("A ranking, not causal proof - but it tells you which "
                                             "behaviour to coach first for crash reduction.")]), lg=5),
        ], className="g-3 mb-3"),
        dbc.Card([
            dbc.CardHeader("Triage queue - highest modelled collision risk"),
            dbc.CardBody(dash_table.DataTable(
                data=S["sc"].nlargest(15, "CollisionRisk")[
                    ["VehicleID", "Persona", "KM", "RiskRate", "SafetyIndex", "CollisionRisk"]
                ].round(2).to_dict("records"),
                columns=[{"name": c, "id": c} for c in
                         ["VehicleID", "Persona", "KM", "RiskRate", "SafetyIndex", "CollisionRisk"]],
                style_as_list_view=True,
                style_header={"backgroundColor": "#222", "color": "#eee"},
                style_cell={"backgroundColor": "#111", "color": "#ddd",
                            "border": "none", "fontSize": 13},
            )),
        ]),
    ], fluid=True)


def tab_driver():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Select driver / vehicle", className="small text-muted"),
                dcc.Dropdown(id="driver-dd", options=[{"label": v, "value": v} for v in VEHICLES],
                             value=VEHICLES[0], clearable=False, style={"color": "#000"}),
            ], md=4),
            dbc.Col(html.Div(id="driver-kpis"), md=8),
        ], className="g-3 mb-3 align-items-end"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Route footprint & event locations"),
                dbc.CardBody(dcc.Graph(id="driver-map")),
                dbc.CardFooter("Cyan lines are the ACTUAL driven routes, rebuilt from the "
                               "timestamped GPS breadcrumb log and following the real street "
                               "network; arrow glyphs show direction of travel, green dots are "
                               "pickups, and hovering a route gives the timestamp and speed at "
                               "that point. Showing the last 40 routed trips per driver. "
                               "Scattered pins = a driving-STYLE problem (coach the driver). "
                               "Pins clustered on one corridor = a ROUTE problem (fix dispatch "
                               "or report the road to the city)."),

            ]), lg=7),
            dbc.Col([
                dbc.Card([dbc.CardHeader("Behaviour percentiles vs fleet"),
                          dbc.CardBody(dcc.Graph(id="driver-radar"))], className="mb-3"),
                dbc.Card([dbc.CardHeader("This driver's risk by hour"),
                          dbc.CardBody(dcc.Graph(id="driver-hours"))]),
            ], lg=5),
        ], className="g-3"),
    ], fluid=True)


app.layout = dbc.Container(
    [
        html.Div([
            html.H2([html.I(className="bi bi-shield-check me-2"),
                     "Manitoba Taxi Fleet - Road Safety Intelligence"], className="mb-0"),
            html.Div("Geotab telematics -> exposure-normalised risk, hotspot mapping and "
                     "predictive coaching triage", className="text-muted"),
        ], className="py-3"),
        header(),
        dbc.Tabs([
            dbc.Tab(tab_overview(), label="Fleet overview"),
            dbc.Tab(tab_maps(), label="Accident & hotspot maps"),
            dbc.Tab(tab_ml(), label="Machine learning"),
            dbc.Tab(tab_driver(), label="Driver deep dive"),
        ]),
        html.Div("Synthetic Geotab-style demo data - Manitoba, Canada",
                 className="text-muted small text-center py-4"),
    ],
    fluid=True,
)


@app.callback(
    Output("driver-map", "figure"),
    Output("driver-radar", "figure"),
    Output("driver-hours", "figure"),
    Output("driver-kpis", "children"),
    Input("driver-dd", "value"),
)
def _driver(vid):
    # REMARK -- the callback only FILTERS pre-computed frames. No model refit,
    # no re-aggregation: switching driver stays sub-second even on 400k trips.
    r = S["sc"][S["sc"]["VehicleID"] == vid].iloc[0]
    kpis = dbc.Row([
        dbc.Col(kpi_card("Safety index", f"{r['SafetyIndex']:.0f}", "100 = safest in fleet",
                         "success" if r["SafetyIndex"] > 50 else "danger"), md=3),
        dbc.Col(kpi_card("Risk / 100 km", f"{r['RiskRate']:.1f}",
                         f"fleet pct {r['RiskPercentile']:.0f}", "warning"), md=3),
        dbc.Col(kpi_card("Exposure", f"{r['KM']:,.0f} km",
                         f"{r['Hours']:,.0f} h - {r['NightShare']*100:.0f}% night"), md=3),
        dbc.Col(kpi_card("Persona", str(r["Persona"]).split(" (")[0],
                         str(r["Persona"]), "info"), md=3),
    ], className="g-2")
    return fig_driver_route(vid), fig_driver_profile(vid), fig_driver_hours(vid), kpis


if __name__ == "__main__":
    #app.run(debug=False, host="0.0.0.0", port=_args.port)
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=int(os.getenv("PORT", 9000))) 
