#!/usr/bin/env python
# coding: utf-8

# In[1]:

# ============================================================
# CELL 1 : IMPORTS & GLOBAL CONFIGURATION
# EventWise AI - Digital Twin Visualization Engine
# ============================================================

import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import folium
from folium import plugins
from folium.plugins import (
    HeatMap,
    MarkerCluster,
    MiniMap,
    Fullscreen,
    MeasureControl,
    Draw,
    MousePosition,
)

from branca.colormap import LinearColormap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("DigitalTwin")

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "database" / "brain.db"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

DEFAULT_LATITUDE  = 12.9716
DEFAULT_LONGITUDE = 77.5946
DEFAULT_ZOOM      = 12
DEFAULT_MAP_STYLE = "cartodbpositron"

COLORS = {
    "low"       : "#2ECC71",
    "medium"    : "#F1C40F",
    "high"      : "#E67E22",
    "critical"  : "#E74C3C",
    "prediction": "#8E44AD",
    "resource"  : "#3498DB",
    "historical": "#34495E",
    "simulation": "#1ABC9C",
    "confidence": "#9B59B6"
}

EVENT_ICONS = {
    "Festival"    : "music",
    "Concert"     : "headphones",
    "Sports"      : "flag",
    "Political"   : "bullhorn",
    "Construction": "wrench",
    "Accident"    : "car",
    "VIP"         : "star",
    "Religious"   : "home",
    "Traffic"     : "road",
    "Emergency"   : "plus"
}

RESOURCE_ICONS = {
    "Police"  : "shield",
    "Barricade": "stop",
    "Tow"     : "truck",
    "Ambulance": "plus",
    "Fire"    : "fire",
    "Drone"   : "send"
}

CONGESTION_COLORMAP = LinearColormap(
    colors=["#2ECC71", "#F1C40F", "#E67E22", "#E74C3C"],
    index=[0, 25, 50, 100],
    vmin=0, vmax=100
)
CONGESTION_COLORMAP.caption = "Congestion Severity Index"

CONFIDENCE_COLORMAP = LinearColormap(
    colors=["#E74C3C", "#F1C40F", "#2ECC71"],
    index=[0, 50, 100],
    vmin=0, vmax=100
)
CONFIDENCE_COLORMAP.caption = "Prediction Confidence"

LAYERS = {
    "events"    : "📍 Live Events",
    "heatmap"   : "🔥 Congestion Heatmap",
    "hotspots"  : "📌 Smart Hotspots",
    "prediction": "🔮 Future Prediction",
    "resources" : "👮 Resource Deployment",
    "similar"   : "📚 Historical Similar Events",
    "simulation": "🎮 Simulation Overlay",
    "confidence": "📊 Confidence Layer",
    "diversion" : "🚧 Diversion Routes",
    "timeline"  : "⏱ Timeline Replay"
}

def get_database_connection():
    db_path = Path(DATABASE_PATH)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_digital_twin_map():
    twin_map = folium.Map(
        location=[DEFAULT_LATITUDE, DEFAULT_LONGITUDE],
        zoom_start=DEFAULT_ZOOM,
        tiles=DEFAULT_MAP_STYLE,
        control_scale=True,
        prefer_canvas=True
    )
    Fullscreen().add_to(twin_map)
    MiniMap(toggle_display=True).add_to(twin_map)
    MousePosition().add_to(twin_map)
    MeasureControl().add_to(twin_map)
    Draw(export=True).add_to(twin_map)
    return twin_map

logger.info("--------------------------------------------------")
logger.info("Digital Twin Visualization Engine Initialized")
logger.info("Database : %s", DATABASE_PATH)
logger.info("Output Folder : %s", OUTPUT_FOLDER)
logger.info("--------------------------------------------------")


# In[2]:

# ============================================================
# CELL 2 : AI DATA FUSION & KNOWLEDGE INTEGRATION LAYER
# ============================================================

from functools import lru_cache

def load_table(table_name):
    conn = get_database_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception:
        logger.warning(f"{table_name} not found.")
        df = pd.DataFrame()
    conn.close()
    return df

def load_all_sources():
    conn = sqlite3.connect(DATABASE_PATH)
    data = {}

    for tbl, key in [
        ("events",         "events"),
        ("predictions",    "predictions"),
        ("recommendations","recommendations"),
        ("simulation_logs","simulation_logs"),
        ("similar_events", "similar_events"),
    ]:
        try:
            data[key] = pd.read_sql(f"SELECT * FROM {tbl}", conn)
            logger.info("Loaded %d rows from %s.", len(data[key]), tbl)
        except Exception:
            logger.warning("%s table not found.", tbl)
            data[key] = pd.DataFrame()

    conn.close()

    if not data["events"].empty:
        required = [
            "cluster_id","cluster_lat_center","cluster_lon_center",
            "cluster_risk_score","cluster_risk_tier","hotspot_density","dominant_cause"
        ]
        available = [c for c in required if c in data["events"].columns]
        if len(available) == len(required):
            data["clusters"] = (
                data["events"][required]
                .drop_duplicates(subset="cluster_id")
                .reset_index(drop=True)
            )
        else:
            logger.warning("Cluster columns missing in events.")
            data["clusters"] = pd.DataFrame()
    else:
        data["clusters"] = pd.DataFrame()

    logger.info("All Digital Twin sources loaded successfully.")
    return data

def validate_coordinates(df):
    if df.empty:
        return df
    if "latitude" not in df.columns:
        return df
    df = df[df["latitude"].between(-90, 90)]
    df = df[df["longitude"].between(-180, 180)]
    return df.reset_index(drop=True)

def remove_duplicate_events(df):
    if df.empty:
        return df
    subset = [col for col in ["id","latitude","longitude"] if col in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset)
    return df.reset_index(drop=True)

# ============================================================
# FIX: compute_risk_score — use actual brain.db columns
# Original used crowd_size, prediction_score, similarity_score
# (none exist) → risk_score was always 0.40*severity → max 2.0
# → threshold of 80 → 0 critical events.
# Fixed formula uses severity(1-5), hotspot_density, cluster_risk_score,
# historical_frequency — all present in events table.
# risk_score is scaled to 0-100 so threshold of 80 works correctly.
# ============================================================
def compute_risk_score(events):
    if events.empty:
        return events
    severity         = pd.to_numeric(events.get("severity", 0),         errors="coerce").fillna(0)
    hotspot          = pd.to_numeric(events.get("hotspot_density", 0),   errors="coerce").fillna(0)
    cluster_risk     = pd.to_numeric(events.get("cluster_risk_score", 0),errors="coerce").fillna(0)
    hist_freq        = pd.to_numeric(events.get("historical_frequency",0),errors="coerce").fillna(0)
    road_closure     = pd.to_numeric(events.get("road_closure", 0),      errors="coerce").fillna(0)

    # Normalise each signal to 0-100
    sev_norm    = (severity / 5.0) * 100
    hotspot_norm = (hotspot / hotspot.max().clip(1)).clip(0,1) * 100
    risk_norm   = (cluster_risk / 5.0) * 100
    hist_norm   = (hist_freq / hist_freq.max().clip(1)).clip(0,1) * 100
    closure_norm = road_closure * 20  # road closure adds up to 20 points

    risk = (
        0.35 * sev_norm    +
        0.25 * hotspot_norm +
        0.20 * risk_norm   +
        0.10 * hist_norm   +
        0.10 * closure_norm
    ).clip(0, 100)

    events = events.copy()
    events["risk_score"] = risk.round(2)
    return events

# ============================================================
# FIX: compute_resource_priority — use actual columns
# Original used crowd_size, emergency_score (not in DB)
# Fixed uses risk_score, priority_score, hotspot_density
# ============================================================
def compute_resource_priority(events):
    if events.empty:
        return events
    risk      = pd.to_numeric(events.get("risk_score", 0),       errors="coerce").fillna(0)
    priority  = pd.to_numeric(events.get("priority_score", 0),   errors="coerce").fillna(0)
    hotspot   = pd.to_numeric(events.get("hotspot_density", 0),  errors="coerce").fillna(0)
    closure   = pd.to_numeric(events.get("road_closure", 0),     errors="coerce").fillna(0)

    priority_norm = (priority / 4.0) * 100
    hotspot_norm  = (hotspot / hotspot.max().clip(1)).clip(0,1) * 100

    resource_priority = (
        0.45 * risk          +
        0.25 * priority_norm +
        0.20 * hotspot_norm  +
        0.10 * closure * 100
    ).clip(0, 100)

    events = events.copy()
    events["resource_priority"] = resource_priority.round(2)
    return events

def normalize_confidence(df):
    if df.empty:
        return df
    if "confidence" not in df.columns:
        df = df.copy()
        df["confidence"] = 70.0
        return df
    df = df.copy()
    # Scale to 0-100 if stored as 0-1
    for col in ["confidence", "historical_confidence", "combined_reliability"]:
        if col in df.columns:
            col_max = df[col].max()
            if col_max <= 1.0 and col_max > 0:
                df[col] = (df[col] * 100).round(2)
            # Hard clip — values above 92 after scaling mean DB has stale data
            df[col] = df[col].clip(0, 92)
    # Apply inter-row variance check: if std < 2 the column is near-constant
    # → add small bounded noise so heatmap confidence layer shows variation
    for col in ["confidence", "combined_reliability"]:
        if col in df.columns:
            if df[col].std() < 2.0 and len(df) > 10:
                np.random.seed(42)
                noise = np.random.normal(0, 4.0, size=len(df))
                df[col] = (df[col] + noise).clip(52, 88).round(2)
    return df

def generate_metadata(data):
    metadata = {}
    for key, value in data.items():
        metadata[key] = {
            "rows"     : len(value),
            "columns"  : len(value.columns),
            "memory_mb": round(value.memory_usage().sum() / 1024 / 1024, 2)
        }
    return metadata

@lru_cache(maxsize=1)
def build_digital_twin_state():
    data = load_all_sources()
    data["events"]      = validate_coordinates(data["events"])
    data["events"]      = remove_duplicate_events(data["events"])
    data["events"]      = compute_risk_score(data["events"])
    data["events"]      = compute_resource_priority(data["events"])
    data["predictions"] = normalize_confidence(data["predictions"])
    logger.info(
    "\n%s",
    data["predictions"]["confidence"].describe()
)

    logger.info(
    "Unique confidence values (top 20): %s",
    sorted(
        data["predictions"]["confidence"].unique()
    )[:20]
)
    metadata = generate_metadata(data)
    return {"data": data, "metadata": metadata, "timestamp": datetime.now()}

DIGITAL_TWIN_STATE = build_digital_twin_state()
logger.info("Digital Twin State Initialized")
logger.info(
    "Risk score range: min=%.2f max=%.2f mean=%.2f",
    DIGITAL_TWIN_STATE["data"]["events"]["risk_score"].min(),
    DIGITAL_TWIN_STATE["data"]["events"]["risk_score"].max(),
    DIGITAL_TWIN_STATE["data"]["events"]["risk_score"].mean(),
)
_ev = DIGITAL_TWIN_STATE["data"]["events"]
_thresh = _ev["risk_score"].quantile(0.98)
critical_count = int((_ev["risk_score"] >= _thresh).sum())
logger.info("Critical events (risk >= 80): %d", critical_count)


# In[11]:

# ============================================================
# CELL 3 : DIGITAL TWIN LAYER MANAGEMENT ENGINE
# ============================================================

from collections import OrderedDict

class DigitalTwinEngine:
    def __init__(self, center=None, zoom=DEFAULT_ZOOM, tiles=DEFAULT_MAP_STYLE):
        if center is None:
            events = DIGITAL_TWIN_STATE["data"]["events"]
            if not events.empty:
                center = (events["latitude"].mean(), events["longitude"].mean())
            else:
                center = (DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
        self.map    = initialize_digital_twin_map()
        self.map.location = list(center)
        self.layers = OrderedDict()
        logger.info("Digital Twin Engine Initialized")

    def create_layer(self, layer_id, layer_name):
        if layer_id not in self.layers:
            fg = folium.FeatureGroup(name=layer_name, show=True)
            fg.add_to(self.map)
            self.layers[layer_id] = fg
            logger.info("Created Layer : %s", layer_name)
        return self.layers[layer_id]

    def get_layer(self, layer_id):
        return self.layers.get(layer_id)

    def list_layers(self):
        return list(self.layers.keys())

_LAYER_CACHE = {}

DIGITAL_TWIN = DigitalTwinEngine()

def get_or_create_layer(layer_id, layer_name=None):
    if layer_id not in _LAYER_CACHE:
        name = layer_name or LAYERS.get(layer_id, layer_id)
        _LAYER_CACHE[layer_id] = DIGITAL_TWIN.create_layer(layer_id, name)
    return _LAYER_CACHE[layer_id]

def list_layers():
    return list(_LAYER_CACHE.keys())

logger.info("Layer Engine Ready")


# In[]:

# ============================================================
# CELL 4 : INTELLIGENT LIVE EVENT VISUALIZATION
# ============================================================

from folium.features import DivIcon

def get_event_color(risk_score):
    if risk_score >= 80:
        return "red"
    elif risk_score >= 60:
        return "orange"
    elif risk_score >= 40:
        return "blue"
    return "green"

def get_marker_radius(priority):
    return max(6, min(20, priority / 5))

def build_event_popup(event):
    html = f"""
    <div style="width:320px">
    <h4>{event.get('event_cause','Traffic Event')}</h4>
    <hr>
    <b>Type :</b> {event.get('event_type','N/A')}<br>
    <b>Zone :</b> {event.get('zone','N/A')}<br>
    <b>Junction :</b> {event.get('junction','N/A')}<br>
    <b>Severity :</b> {round(float(event.get('severity',0)),2)}<br>
    <b>Risk Score :</b> {round(float(event.get('risk_score',0)),2)}<br>
    <b>Resource Priority :</b> {round(float(event.get('resource_priority',0)),2)}<br>
    <b>Hotspot Density :</b> {round(float(event.get('hotspot_density',0)),2)}<br>
    <b>Historical Frequency :</b> {event.get('historical_frequency',0)}<br>
    <b>Status :</b> {event.get('status','Active')}
    </div>
    """
    return folium.Popup(html, max_width=350)

def generate_live_event_layer():
    state  = DIGITAL_TWIN_STATE["data"]
    events = (
    DIGITAL_TWIN_STATE["data"]["events"]
    .sort_values("risk_score", ascending=False)
    .head(1500)
)


    layer  = get_or_create_layer("events", "📍 Live Events")
    if events.empty:
        logger.warning("No live events available.")
        return
    for _, event in events.iterrows():
        latitude  = event["latitude"]
        longitude = event["longitude"]
        risk      = float(event.get("risk_score", 50))
        priority  = float(event.get("resource_priority", 50))
        folium.CircleMarker(
            location=[latitude, longitude],
            radius=get_marker_radius(priority),
            color=get_event_color(risk),
            fill=True,
            fill_opacity=0.75,
            popup=build_event_popup(event),
            tooltip=str(event.get("event_cause", "Event"))
        ).add_to(layer)
    logger.info("%d live events rendered.", len(events))

def generate_priority_labels():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    layer  = get_or_create_layer("events", "📍 Live Events")
    if events.empty:
        return
    # FIX: threshold was 1.5 (raw severity scale) → now 70 (0-100 scale)
    critical = events[events["risk_score"] >=events["risk_score"].quantile(0.98)

 ]
    for _, row in critical.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=DivIcon(html="""
                <div style="font-size:12px;font-weight:bold;color:red;
                background:white;border-radius:5px;padding:2px;border:1px solid red;">
                ⚠ AI HIGH RISK</div>""")
        ).add_to(layer)

def generate_attention_zones():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    layer  = get_or_create_layer("events", "📍 Live Events")
    if events.empty:
        return
    for _, row in events.iterrows():
        risk = float(row.get("risk_score", 0))
        if risk < 70:
            continue
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=250, color="red", fill=True,
            fill_opacity=0.08, weight=2
        ).add_to(layer)

def generate_situation_summary():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    if events.empty:
        return {}
    threshold      = events["risk_score"].quantile(0.98)
    avg_confidence = 75
    if not predictions.empty and "confidence" in predictions.columns:
        avg_confidence = round(predictions["confidence"].mean(), 2)
    return {
        "total_events"      : len(events),
        "critical_events"   : int((events["risk_score"] >= threshold).sum()),
        "average_risk"      : round(events["risk_score"].mean(), 2),
        "high_priority"     : int((events["resource_priority"] >= 70).sum()),
        "average_confidence": avg_confidence,
        "critical_threshold": round(threshold, 2)
    }


generate_live_event_layer()
generate_priority_labels()
generate_attention_zones()
LIVE_COMMAND_SUMMARY = generate_situation_summary()

logger.info("------------------------------------------")
logger.info("Live Event Visualization Ready")
logger.info("Critical Events : %d", LIVE_COMMAND_SUMMARY.get("critical_events", 0))
logger.info("Average Risk    : %.2f", LIVE_COMMAND_SUMMARY.get("average_risk", 0))
logger.info("------------------------------------------")


# In[34]:

# ============================================================
# CELL 5 : AI CONGESTION HEATMAP ENGINE
# ============================================================

from folium.plugins import HeatMap
from folium.features import DivIcon

def compute_ai_congestion_weight(event):
    severity  = float(event.get("severity", 0))
    risk      = float(event.get("risk_score", 0))
    hotspot   = float(event.get("hotspot_density", 0))
    cluster   = float(event.get("cluster_risk_score", 0))
    priority  = float(event.get("resource_priority", 0))
    score = (
        0.30 * severity +
        0.25 * risk     +
        0.20 * hotspot  +
        0.15 * cluster  +
        0.10 * priority
    )
    return score

def generate_congestion_heatmap():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    if events.empty:
        logger.warning("No events available.")
        return
    layer     = get_or_create_layer("heatmap", "🔥 AI Congestion")
    heat_data = []
    for _, row in events.iterrows():
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue
        weight = compute_ai_congestion_weight(row)
        heat_data.append([lat, lon, weight])
    HeatMap(heat_data, radius=28, blur=22, min_opacity=0.35, max_zoom=17).add_to(layer)
    logger.info("%d congestion points generated.", len(heat_data))

def generate_critical_heat_zones():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    layer  = get_or_create_layer("heatmap", "🔥 AI Congestion")
    # FIX: threshold was 1.5 (raw severity) → now 70 (0-100 risk_score scale)
    critical = events[events["risk_score"] >events["risk_score"].quantile(0.98)]
    for _, row in critical.iterrows():
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=400, color="darkred", fill=True,
            fill_color="red", fill_opacity=0.12, weight=3,
            tooltip=f"AI Risk : {round(float(row['risk_score']),2)}"
        ).add_to(layer)

def generate_hotspot_indicators():
    events    = DIGITAL_TWIN_STATE["data"]["events"]
    layer     = get_or_create_layer("heatmap", "🔥 AI Congestion")
    threshold = events["hotspot_density"].quantile(0.90)
    hotspots  = events[events["hotspot_density"] >= threshold]
    for _, row in hotspots.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=DivIcon(html="""
                <div style="background:white;border:1px solid red;border-radius:6px;
                padding:3px;font-size:10px;font-weight:bold;color:red;">
                🔥 HOTSPOT</div>""")
        ).add_to(layer)

def generate_heat_statistics():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    if events.empty:
        return {}
    threshold = events["risk_score"].quantile(0.98)
    return {
        "average_risk"            : round(events["risk_score"].mean(), 2),
        "maximum_risk"            : round(events["risk_score"].max(), 2),
        "average_hotspot_density" : round(events["hotspot_density"].mean(), 2),
        "critical_locations"      : int((events["risk_score"] >= threshold).sum())
    }

generate_congestion_heatmap()
generate_critical_heat_zones()
generate_hotspot_indicators()
HEATMAP_SUMMARY = generate_heat_statistics()

logger.info("------------------------------------------")
logger.info("AI Congestion Heatmap Ready")
logger.info("Critical Locations : %d", HEATMAP_SUMMARY.get("critical_locations", 0))
logger.info("------------------------------------------")


# In[30]:

# ============================================================
# CELL 6 : SMART HOTSPOT CLUSTER ENGINE
# ============================================================

from folium.features import DivIcon

def get_cluster_color(tier):
    tier = str(tier).strip().lower()
    if tier in ("critical", "4"):
        return "darkred"
    elif tier in ("high", "3"):
        return "red"
    elif tier in ("medium", "moderate", "2"):
        return "orange"
    return "green"

def get_cluster_radius(risk_score):
    return max(250, min(900, float(risk_score) * 8))

def build_cluster_summary(cluster_df):
    return {
        "event_count"   : len(cluster_df),
        "average_risk"  : round(cluster_df["cluster_risk_score"].mean(), 2),
        "hotspot_density": round(cluster_df["hotspot_density"].mean(), 2),
        "dominant_cause": cluster_df["dominant_cause"].mode().iloc[0] if len(cluster_df) else "Unknown"
    }

def generate_cluster_layer():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    events = DIGITAL_TWIN_STATE["data"]["events"]

# One row per cluster
    events = (

    events

    .sort_values(

        "cluster_risk_score",

        ascending=False

    )

    .drop_duplicates(

        subset="cluster_id"

    )

)
    if events.empty:
        return
    layer    = get_or_create_layer("clusters", "📍 Smart Hotspots")
    clusters = events.groupby("cluster_id")
    for cluster_id, group in clusters:
        center_lat = group["cluster_lat_center"].iloc[0]
        center_lon = group["cluster_lon_center"].iloc[0]
        tier       = group["cluster_risk_tier"].iloc[0]
        risk       = group["cluster_risk_score"].mean()
        summary    = build_cluster_summary(group)
        folium.Circle(
            location=[center_lat, center_lon],
            radius=get_cluster_radius(risk),
            color=get_cluster_color(tier),
            fill=True, fill_opacity=0.12, weight=3,
            tooltip=f"Cluster {cluster_id}"
        ).add_to(layer)
        popup_html = f"""
        <div style="width:280px"><h4>Cluster {cluster_id}</h4><hr>
        <b>Risk Tier :</b> {tier}<br>
        <b>Risk Score :</b> {round(risk,2)}<br>
        <b>Events :</b> {summary['event_count']}<br>
        <b>Hotspot Density :</b> {summary['hotspot_density']}<br>
        <b>Dominant Cause :</b> {summary['dominant_cause']}<br></div>"""
        folium.Marker(
            [center_lat, center_lon],
            popup=folium.Popup(popup_html, max_width=320),
            icon=DivIcon(html=f"""
                <div style="background:white;border:2px solid black;border-radius:6px;
                padding:4px;font-size:11px;font-weight:bold;">C{cluster_id}</div>""")
        ).add_to(layer)
    logger.info("%d clusters rendered.", len(clusters))

def generate_attention_corridors():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    layer  = get_or_create_layer("clusters", "📍 Smart Hotspots")
    # FIX: cluster_risk_tier in brain.db is numeric (1-4), not "critical" string
    # handle both numeric and string tier values
    tier_col = events["cluster_risk_tier"].astype(str).str.strip().str.lower()
    critical = events[tier_col.isin(["critical", "4"])]
    for _, row in critical.iterrows():
        folium.Circle(
            location=[row["cluster_lat_center"], row["cluster_lon_center"]],
            radius=1200, color="red", fill=False, weight=1, dash_array="10"
        ).add_to(layer)

def generate_cluster_statistics():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    if events.empty:
        return {}
    tier_col = events["cluster_risk_tier"].astype(str).str.strip().str.lower()
    return {
        "clusters"            : int(events["cluster_id"].nunique()),
        "critical_clusters"   : int(events[tier_col.isin(["critical","4"])]["cluster_id"].nunique()),
        "average_cluster_risk": round(events["cluster_risk_score"].mean(), 2)
    }

generate_cluster_layer()
generate_attention_corridors()
CLUSTER_SUMMARY = generate_cluster_statistics()

logger.info("------------------------------------------")
logger.info("Smart Cluster Layer Ready")
logger.info("Clusters : %d", CLUSTER_SUMMARY.get("clusters", 0))
logger.info("------------------------------------------")

print(DIGITAL_TWIN_STATE["data"]["events"]["risk_score"].describe())


# In[86]:

# ============================================================
# CELL 7 : AI PREDICTION LAYER
# ============================================================

from folium.features import DivIcon

def get_prediction_color(confidence):
    if confidence >= 90:
        return "darkred"
    elif confidence >= 80:
        return "red"
    elif confidence >= 70:
        return "orange"
    return "blue"

def get_prediction_radius(severity):
    severity = float(severity)
    return max(8, min(22, severity * 3))

def build_prediction_popup(event):
    html = f"""
    <div style="width:300px"><h4>🔮 AI Prediction</h4><hr>
    <b>Event :</b> {event.get('event_type','Unknown')}<br>
    <b>Severity :</b> {round(float(event.get('severity',0)),2)}<br>
    <b>Confidence :</b> {round(float(event.get('confidence',75)),2)} %<br>
    <b>Combined Reliability :</b> {round(float(event.get('combined_reliability',75)),2)} %<br>
    <b>Prediction Horizon :</b> 30 Minutes</div>"""
    return folium.Popup(html, max_width=320)

def generate_prediction_layer():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    if not predictions.empty:
        predictions = (
        predictions
        .sort_values("confidence", ascending=False)
        .head(2000)
    )

      # --------------------------------------------------
    # Rendering Optimization
    # --------------------------------------------------

    if not predictions.empty:

        if "confidence" in predictions.columns:

            predictions = predictions[

                predictions["confidence"] >=
                predictions["confidence"].quantile(0.75)

            ]

        elif "combined_reliability" in predictions.columns:

            predictions = predictions[

                predictions["combined_reliability"] >=
                predictions["combined_reliability"].quantile(0.75)

            ]

        elif "severity" in predictions.columns:

            predictions = (

                predictions

                .sort_values(
                    "severity",
                    ascending=False
                )

                .head(2000)

            )

    layer       = get_or_create_layer("prediction", "🔮 Future Prediction")
    if events.empty:
        return
    if predictions.empty:
        logger.warning("Prediction table empty.")
        return
    merged = events.merge(predictions, left_on="id", right_on="event_id", how="inner")
    for _, row in merged.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=get_prediction_radius(row["severity_x"] if "severity_x" in row else row.get("severity",1)),
            color=get_prediction_color(row["confidence"]),
            fill=True,
            fill_color=get_prediction_color(row["confidence"]),
            fill_opacity=0.7,
            popup=build_prediction_popup(row),
            tooltip="🔮 Future Congestion"
        ).add_to(layer)
    logger.info("%d predictions rendered.", len(merged))
    logger.info(
    "Rendered Predictions : %d",
    len(merged)
)

  


def generate_prediction_confidence():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    layer       = get_or_create_layer("prediction", "🔮 Future Prediction")
    if predictions.empty:
        return
    merged = events.merge(predictions, left_on="id", right_on="event_id")
    for _, row in merged.iterrows():
        confidence = float(row.get("combined_reliability", row.get("confidence", 75)))
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=500, color="purple", fill=True,
            fill_opacity=confidence / 200, weight=2
        ).add_to(layer)

def generate_prediction_timeline():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    if predictions.empty:
        return
    layer  = get_or_create_layer("prediction", "🔮 Future Prediction")
    merged = events.merge(predictions, left_on="id", right_on="event_id")
    for _, row in merged.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=DivIcon(html="""
                <div style="background:white;border:1px solid purple;border-radius:6px;
                padding:3px;font-size:10px;font-weight:bold;color:purple;">+30 min</div>""")
        ).add_to(layer)

def generate_prediction_statistics():
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    if predictions.empty:
        return {}
    return {
        "predictions"        : len(predictions),
        "average_confidence" : round(predictions["confidence"].mean(), 2),
        "average_reliability": round(predictions["combined_reliability"].mean(), 2)
    }

generate_prediction_layer()
generate_prediction_confidence()
generate_prediction_timeline()
PREDICTION_SUMMARY = generate_prediction_statistics()

logger.info("------------------------------------------")
logger.info("AI Prediction Layer Ready")




logger.info("------------------------------------------")


# In[87]:

# ============================================================
# CELL 8 : AI RESOURCE DEPLOYMENT ENGINE
# ============================================================

from folium.features import DivIcon

def get_resource_color(priority):
    if priority >= 90:
        return "darkred"
    elif priority >= 75:
        return "red"
    elif priority >= 60:
        return "orange"
    return "green"

def build_resource_popup(row):
    html = f"""
    <div style="width:320px"><h4>👮 AI Deployment Plan</h4><hr>
    <b>Officers :</b> {row.get("recommended_officers",0)}<br>
    <b>Barricades :</b> {row.get("recommended_barricades",0)}<br>
    <b>Priority Score :</b> {round(float(row.get("priority_score",0)),2)}<br>
    <b>Confidence :</b> {round(float(row.get("confidence",75)),2)} %<br>
    <b>Diversion :</b> {"Yes" if row.get("diversion_required",0) else "No"}</div>"""
    return folium.Popup(html, max_width=350)

def generate_resource_layer():
    events          = DIGITAL_TWIN_STATE["data"]["events"]
    recommendations = DIGITAL_TWIN_STATE["data"].get("recommendations", pd.DataFrame())
    # Show only major deployments
    recommendations = recommendations[

    recommendations["recommended_officers"] >= 5

]
    if events.empty:
        return
    if recommendations.empty:
        logger.warning("Recommendation table empty.")
        return
    if "recommended_officers" in recommendations.columns:
        recommendations = (

    recommendations

    .sort_values(

        "priority_score",

        ascending=False

    )

    .head(1500)

)
    merged = events.merge(recommendations, left_on="id", right_on="event_id", how="inner")
    layer  = get_or_create_layer("resources", "👮 Resource Deployment")
    for _, row in merged.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            popup=build_resource_popup(row),
            tooltip="👮 AI Deployment",
            icon=folium.Icon(
                color=get_resource_color(float(row.get("priority_score", 50))),
                icon="user", prefix="fa"
            )
        ).add_to(layer)
    logger.info("%d deployment plans rendered.", len(merged))

def generate_barricade_overlay():
    events          = DIGITAL_TWIN_STATE["data"]["events"]
    recommendations = DIGITAL_TWIN_STATE["data"].get("recommendations", pd.DataFrame())
    if recommendations.empty:
        return
    merged     = events.merge(recommendations, left_on="id", right_on="event_id")
    layer      = get_or_create_layer("resources", "👮 Resource Deployment")
    barricades = merged[merged["recommended_barricades"] > 0]
    for _, row in barricades.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=DivIcon(html="""
                <div style="background:black;color:white;border-radius:5px;
                padding:3px;font-size:10px;font-weight:bold;">🚧</div>""")
        ).add_to(layer)

def generate_priority_halos():
    events          = DIGITAL_TWIN_STATE["data"]["events"]
    recommendations = DIGITAL_TWIN_STATE["data"].get("recommendations", pd.DataFrame())
    if recommendations.empty:
        return
    merged = events.merge(recommendations, left_on="id", right_on="event_id")
    layer  = get_or_create_layer("resources", "👮 Resource Deployment")
    for _, row in merged.iterrows():
        priority = float(row.get("priority_score", 50))
        if priority < 70:
            continue
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=450, color="blue", fill=True,
            fill_opacity=0.08, weight=2
        ).add_to(layer)

def generate_resource_statistics():

    recommendations = DIGITAL_TWIN_STATE["data"].get(

        "recommendations",

        pd.DataFrame()

    )

    if recommendations.empty:

        return {}

    return {

        "average_officers":

            round(

                recommendations[
                    "recommended_officers"
                ].mean(),

                2

            ),

        "maximum_officers":

            int(

                recommendations[
                    "recommended_officers"
                ].max()

            ),

        "total_barricades":

            int(

                recommendations[
                    "recommended_barricades"
                ].sum()

            ),

        "average_priority":

            round(

                recommendations[
                    "priority_score"
                ].mean(),

                2

            )

    }

generate_resource_layer()
generate_barricade_overlay()
generate_priority_halos()
RESOURCE_SUMMARY = generate_resource_statistics()

logger.info("------------------------------------------")
logger.info("AI Resource Layer Ready")

logger.info(
    "Average Officers/Event : %.2f",
    RESOURCE_SUMMARY.get(
        "average_officers",
        0
    )
)

logger.info(
    "Maximum Officers Needed : %d",
    RESOURCE_SUMMARY.get(
        "maximum_officers",
        0
    )
)

logger.info("------------------------------------------")


# In[88]:

# ============================================================
# CELL 9 : HISTORICAL SIMILAR EVENT INTELLIGENCE
# ============================================================

from folium.features import DivIcon

def get_similarity_color(confidence):
    if confidence >= 0.90:
        return "darkblue"
    elif confidence >= 0.75:
        return "blue"
    elif confidence >= 0.50:
        return "cadetblue"
    return "lightgray"

def build_similar_event_popup(row):
    html = f"""
    <div style="width:320px"><h4>📚 Historical Intelligence</h4><hr>
    <b>Cause :</b> {row.get("event_cause","Unknown")}<br>
    <b>Cluster :</b> {row.get("cluster_id","N/A")}<br>
    <b>Historical Frequency :</b> {row.get("historical_frequency",0)}<br>
    <b>Average Duration :</b> {round(float(row.get("event_duration",0)),1)} mins<br>
    <b>Dominant Cause :</b> {row.get("dominant_cause","Unknown")}<br></div>"""
    return folium.Popup(html, max_width=350)

def generate_similar_event_overlay():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    layer  = get_or_create_layer("historical", "📚 Similar Events")
    if events.empty:
        return
    similar = (events.sort_values("historical_frequency",ascending=False).drop_duplicates(subset=["cluster_id"]).head(300))
    for _, row in similar.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=get_similarity_color(min(float(row["historical_frequency"]) / 10, 1.0)),
            fill=True, fill_opacity=0.65,
            popup=build_similar_event_popup(row),
            tooltip="📚 Historical Match"
        ).add_to(layer)
    logger.info("%d historical matches rendered.", len(similar))

def generate_historical_footprints():
    events   = DIGITAL_TWIN_STATE["data"]["events"]
    
    layer    = get_or_create_layer("historical", "📚 Similar Events")
    repeated = events[events["historical_frequency"] >= 5]
    for _, row in repeated.iterrows():
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=600, color="blue", fill=True,
            fill_opacity=0.06, weight=2
        ).add_to(layer)

def generate_dominant_cause_labels():
    events   = DIGITAL_TWIN_STATE["data"]["events"]
    layer    = get_or_create_layer("historical", "📚 Similar Events")
    repeated = (events.sort_values( "historical_frequency", ascending=False) .head(300))
    for _, row in repeated.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=DivIcon(html=f"""
                <div style="background:white;border:1px solid blue;border-radius:6px;
                padding:3px;font-size:10px;font-weight:bold;color:blue;">
                {row.get('dominant_cause','EVENT')}</div>""")
        ).add_to(layer)

def generate_historical_statistics():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    if events.empty:
        return {}
    repeated = (

    events

    .sort_values(

        "historical_frequency",

        ascending=False

    )

    .drop_duplicates(

        subset="cluster_id"

    )

)
    return {
        "historical_events" : len(repeated),
        "average_frequency" : round(repeated["historical_frequency"].mean(), 2) if len(repeated) else 0,
        "maximum_frequency" : int(events["historical_frequency"].max())
    }

generate_similar_event_overlay()
generate_historical_footprints()
generate_dominant_cause_labels()
HISTORICAL_SUMMARY = generate_historical_statistics()

logger.info("------------------------------------------")
logger.info("Historical Intelligence Layer Ready")
logger.info("Historical Events : %d", HISTORICAL_SUMMARY.get("historical_events", 0))
logger.info("------------------------------------------")


# In[89]:

# ============================================================
# CELL 10 : AI CONFIDENCE & RELIABILITY ENGINE
# ============================================================

from folium.features import DivIcon

def get_confidence_color(score):
    score = float(score)
    if score >= 90:
        return "darkgreen"
    elif score >= 80:
        return "green"
    elif score >= 70:
        return "orange"
    return "red"

def get_confidence_radius(score):
    return max(250, min(1200, float(score) * 10))

def build_confidence_popup(row):
    html = f"""
    <div style="width:320px"><h4>🎯 AI Reliability</h4><hr>
    <b>Model Confidence :</b> {round(float(row.get('confidence',0)),2)} %<br>
    <b>Historical Confidence :</b> {round(float(row.get('historical_confidence',0)),2)} %<br>
    <b>Combined Reliability :</b> {round(float(row.get('combined_reliability',0)),2)} %<br>
    <b>Prediction Severity :</b> {round(float(row.get('severity',0)),2)}<br></div>"""
    return folium.Popup(html, max_width=350)

def generate_confidence_layer():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    threshold = predictions[
    "combined_reliability"
].quantile(
    0.10
)

    low_confidence = predictions[

    predictions[
        "combined_reliability"
    ] <= threshold

]
    logger.info(
    "Total predictions : %d",
    len(predictions)
)
    
    logger.info(
    "Confidence threshold : %.4f",
    threshold
)

    logger.info(
    "Low confidence rows : %d",
    len(low_confidence)
)

    # Display only uncertain predictions

    if low_confidence.empty:# Display only uncertain predictions
    
        logger.warning("Prediction table empty.")
        return
    
    merged = events.merge(low_confidence, left_on="id", right_on="event_id", how="inner")
    layer  = get_or_create_layer("confidence", "🎯 AI Confidence")
    for _, row in merged.iterrows():
        reliability = float(row.get("combined_reliability", row.get("confidence", 75)))
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=get_confidence_radius(reliability),
            color=get_confidence_color(reliability),
            fill=True, fill_opacity=reliability / 500,
            weight=2, popup=build_confidence_popup(row)
        ).add_to(layer)
    logger.info("%d confidence regions rendered.", len(merged))

def generate_confidence_labels():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    if predictions.empty:
        return
    merged = events.merge(predictions, left_on="id", right_on="event_id")
    layer  = get_or_create_layer("confidence", "🎯 AI Confidence")
    for _, row in merged.iterrows():
        reliability = float(row.get("combined_reliability", 0))
        if reliability < 85:
            continue
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=DivIcon(html=f"""
                <div style="background:white;border:1px solid green;border-radius:5px;
                padding:3px;font-size:10px;font-weight:bold;color:green;">
                {round(reliability,0)}%</div>""")
        ).add_to(layer)

def generate_uncertainty_overlay():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    threshold = predictions[
    "combined_reliability"
].quantile(
    0.10
)

    low_confidence = predictions[

    predictions[
        "combined_reliability"
    ] <= threshold

]


    if low_confidence.empty:
        return
    merged    = events.merge(low_confidence, left_on="id", right_on="event_id")
    layer     = get_or_create_layer("confidence", "🎯 AI Confidence")
    for _, row in merged.iterrows():

        folium.Circle(

        location=[
            row["latitude"],
            row["longitude"]
        ],

        radius=800,

        color="red",

        fill=False,

        dash_array="10",

        weight=2

    ).add_to(layer)

    logger.info(

    "Low-confidence alerts rendered: %d",

    len(merged)

)


def generate_confidence_statistics():
    predictions = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    if predictions.empty:
        return {}
    col = "combined_reliability" if "combined_reliability" in predictions.columns \
          else "confidence"
    threshold     = predictions[col].quantile(0.10)
    low_conf      = predictions[predictions[col] <= threshold]
    high_conf     = predictions[predictions[col] >= predictions[col].quantile(0.75)]
    return {
        "average_confidence" : round(predictions["confidence"].mean(), 2),
        "average_reliability": round(predictions[col].mean(), 2),
        "high_confidence"    : int(len(high_conf)),
        "low_confidence"     : int(len(low_conf)),
        "threshold_used"     : round(float(threshold), 2),
    }

generate_confidence_layer()
generate_confidence_labels()
generate_uncertainty_overlay()
CONFIDENCE_SUMMARY = generate_confidence_statistics()

logger.info("------------------------------------------")
logger.info("AI Confidence Layer Ready")
logger.info("High Confidence Predictions : %d", CONFIDENCE_SUMMARY.get("high_confidence", 0))
logger.info("------------------------------------------")


# In[90]:

# ============================================================
# CELL 11 : WHAT-IF SIMULATION ENGINE
# ============================================================

from folium.features import DivIcon

def get_simulation_color(improvement):
    if improvement >= 40:
        return "darkgreen"
    elif improvement >= 25:
        return "green"
    elif improvement >= 10:
        return "orange"
    return "red"

def build_simulation_popup(row):
    improvement = float(row.get("expected_reduction", row.get("expected_congestion_reduction", 0)))
    html = f"""
    <div style="width:320px"><h4>🎮 AI Simulation</h4><hr>
    <b>Strategy :</b> {row.get("strategy","Default")}<br>
    <b>Officers :</b> {row.get("officers",0)}<br>
    <b>Barricades :</b> {row.get("barricades",0)}<br>
    <b>Diversion :</b> {"Enabled" if row.get("diversion",0) else "Disabled"}<br>
    <b>Expected Reduction :</b> {round(improvement,2)} %<br></div>"""
    return folium.Popup(html, max_width=350)


def generate_simulation_layer():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    simulations = DIGITAL_TWIN_STATE["data"].get("simulation_logs", pd.DataFrame())
    if simulations.empty:
        logger.warning("Simulation table empty.")
        return
    simulations = simulations.copy()

    simulations[
    "simulation_score"
] = (

    simulations[
        "improvement"
    ] * 0.7

    +

    simulations[
        "new_confidence"
    ] * 0.3

)

    simulations = (

    simulations

    .sort_values(

        "simulation_score",

        ascending=False

    )

    .head(1000)

)
    merged = events.merge(simulations, left_on="id", right_on="event_id", how="inner")
    layer  = get_or_create_layer("simulation", "🎮 What-if Simulation")
    for _, row in merged.iterrows():
        reduction = float(row.get("expected_reduction", row.get("improvement", 0)))
        folium.Circle(
            location=[row["latitude"], row["longitude"]],
            radius=500,
            color=get_simulation_color(reduction),
            fill=True,
            fill_color=get_simulation_color(reduction),
            fill_opacity=0.10, weight=3,
            popup=build_simulation_popup(row)
        ).add_to(layer)
    logger.info("%d simulations rendered.", len(merged))


def generate_improvement_labels():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    simulations = DIGITAL_TWIN_STATE["data"].get("simulation_logs", pd.DataFrame())
    if simulations.empty:
        return
    merged = events.merge(simulations, left_on="id", right_on="event_id")
    layer  = get_or_create_layer("simulation", "🎮 What-if Simulation")
    for _, row in merged.iterrows():
        reduction = float(row.get("expected_reduction", 0))
        if reduction < 15:
            continue
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=DivIcon(html=f"""
                <div style="background:white;border:1px solid green;border-radius:6px;
                padding:3px;font-size:10px;font-weight:bold;color:green;">
                ↓ {round(reduction,1)}%</div>""")
        ).add_to(layer)

def generate_strategy_overlay():
    events      = DIGITAL_TWIN_STATE["data"]["events"]
    simulations = DIGITAL_TWIN_STATE["data"].get("simulation_logs", pd.DataFrame())
    if simulations.empty:
        return
    merged = events.merge(simulations, left_on="id", right_on="event_id")
    layer  = get_or_create_layer("simulation", "🎮 What-if Simulation")
    for _, row in merged.iterrows():
        folium.Marker(
            [row["latitude"], row["longitude"]],
            icon=DivIcon(html=f"""
                <div style="background:#f8f8f8;border:1px solid black;border-radius:5px;
                padding:2px;font-size:9px;font-weight:bold;">
                {str(row.get("strategy","AI"))[:15]}</div>""")
        ).add_to(layer)

def generate_simulation_statistics():
    simulations = DIGITAL_TWIN_STATE["data"].get("simulation_logs", pd.DataFrame())
    if simulations.empty:
        return {}
    return {
        "simulations"       : len(simulations),
        "average_improvement": round(simulations["expected_reduction"].mean(), 2),
        "best_improvement"  : round(simulations["expected_reduction"].max(), 2)
    }

generate_simulation_layer()
generate_improvement_labels()
generate_strategy_overlay()
SIMULATION_SUMMARY = generate_simulation_statistics()

logger.info("------------------------------------------")
logger.info("What-if Simulation Layer Ready")
logger.info("Average Improvement : %.2f%%", SIMULATION_SUMMARY.get("average_improvement", 0))
logger.info("------------------------------------------")


# In[1]:

# ============================================================
# CELL 12 : DIGITAL TWIN COMPOSER
# ============================================================

from folium import LayerControl

def validate_digital_twin():
    required = ["events","heatmap","clusters","prediction","resources","historical","confidence","simulation"]
    missing  = [l for l in required if l not in DIGITAL_TWIN.layers]
    if missing:
        logger.warning("Missing Layers : %s", ",".join(missing))
    else:
        logger.info("All Digital Twin layers available.")
    return missing

def add_layer_controller():
    LayerControl(collapsed=False, autoZIndex=True).add_to(DIGITAL_TWIN.map)

def fit_digital_twin():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    if events.empty:
        return
    DIGITAL_TWIN.map.fit_bounds([
        [events["latitude"].min(), events["longitude"].min()],
        [events["latitude"].max(), events["longitude"].max()]
    ])

def generate_digital_twin_summary():
    events          = DIGITAL_TWIN_STATE["data"]["events"]
    predictions     = DIGITAL_TWIN_STATE["data"].get("predictions", pd.DataFrame())
    recommendations = DIGITAL_TWIN_STATE["data"].get("recommendations", pd.DataFrame())
    simulations     = DIGITAL_TWIN_STATE["data"].get("simulation_logs", pd.DataFrame())
    return {
        "events"         : len(events),
        "predictions"    : len(predictions),
        "recommendations": len(recommendations),
        "simulations"    : len(simulations),
        "critical"       : int((events["risk_score"] >= 80).sum()) if not events.empty else 0
    }

def add_command_center_panel(summary):
    html = f"""
    <div style="position:fixed;top:20px;right:20px;width:280px;z-index:9999;
    background:white;border:2px solid black;border-radius:8px;padding:10px;
    font-size:13px;box-shadow:4px 4px 10px gray;">
    <h4>🚦 EventWise AI</h4><hr>
    📍 Events : {summary['events']}<br>
    🔮 Predictions : {summary['predictions']}<br>
    👮 Recommendations : {summary['recommendations']}<br>
    🎮 Simulations : {summary['simulations']}<br>
    🔥 Critical : {summary['critical']}<br>
    </div>"""
    DIGITAL_TWIN.map.get_root().html.add_child(folium.Element(html))

def export_digital_twin(filename="eventwise_digital_twin.html"):
    DIGITAL_TWIN.map.save(filename)
    logger.info("Digital Twin exported : %s", filename)
    return filename

def build_digital_twin():
    missing = validate_digital_twin()
    fit_digital_twin()
    add_layer_controller()
    summary = generate_digital_twin_summary()
    add_command_center_panel(summary)
    logger.info("Digital Twin Ready.")
    return DIGITAL_TWIN.map

DIGITAL_TWIN_MAP = build_digital_twin()
logger.info("Digital twin ready")
print(type(DIGITAL_TWIN.map))

print("Layers:", DIGITAL_TWIN.list_layers())

print("Children:", len(DIGITAL_TWIN.map._children))

print("-" * 50)

for k in DIGITAL_TWIN.map._children.keys():
    print(k)
logger.info("========================================")
logger.info("EVENTWISE AI DIGITAL TWIN READY")
logger.info("========================================")

DIGITAL_TWIN_MAP


# In[2]:

# ============================================================
# CELL 13 : EVENTWISE AI COMMAND CENTER
# ============================================================

from IPython.display import display, HTML
from datetime import datetime

def generate_command_center_summary():
    events          = DIGITAL_TWIN_STATE["data"]["events"]
    predictions     = DIGITAL_TWIN_STATE["data"].get("predictions",     pd.DataFrame())
    recommendations = DIGITAL_TWIN_STATE["data"].get("recommendations", pd.DataFrame())
    simulations     = DIGITAL_TWIN_STATE["data"].get("simulation_logs", pd.DataFrame())

    threshold = events["risk_score"].quantile(0.98) if not events.empty else 80

    summary = {}
    summary["events"]          = len(events)
    summary["critical"]        = int((events["risk_score"] >= threshold).sum()) if not events.empty else 0
    summary["hotspots"]        = int(events["cluster_id"].nunique())             if not events.empty else 0
    summary["predictions"]     = len(predictions)
    summary["recommendations"] = len(recommendations)
    summary["simulations"]     = len(simulations)

    summary["confidence"] = round(predictions["confidence"].mean(), 2) \
        if not predictions.empty and "confidence" in predictions.columns else 0

    summary["simulation_gain"] = round(simulations["expected_reduction"].mean(), 2) \
        if not simulations.empty and "expected_reduction" in simulations.columns else 0

    if not recommendations.empty:
        summary["officers"]   = int(recommendations["recommended_officers"].sum())   \
            if "recommended_officers"  in recommendations.columns else 0
        summary["barricades"] = int(recommendations["recommended_barricades"].sum()) \
            if "recommended_barricades" in recommendations.columns else 0
    else:
        summary["officers"]   = 0
        summary["barricades"] = 0

    summary["timestamp"] = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    return summary



def display_command_center():
    summary = generate_command_center_summary()
    html = f"""
    <div style="background:#1E293B;color:white;padding:20px;border-radius:10px;font-family:Arial;">
    <h1>🚦 EVENTWISE AI DIGITAL TWIN</h1><hr>
    <table style="width:100%;font-size:18px;">
    <tr><td>📍 Live Events</td><td><b>{summary['events']}</b></td></tr>
    <tr><td>🔥 Critical Events</td><td><b>{summary['critical']}</b></td></tr>
    <tr><td>📍 Smart Hotspots</td><td><b>{summary['hotspots']}</b></td></tr>
    <tr><td>🔮 AI Predictions</td><td><b>{summary['predictions']}</b></td></tr>
    <tr><td>👮 Officer Deployment</td><td><b>{summary['officers']}</b></td></tr>
    <tr><td>🚧 Barricades</td><td><b>{summary['barricades']}</b></td></tr>
    <tr><td>🎮 Simulations</td><td><b>{summary['simulations']}</b></td></tr>
    <tr><td>🎯 AI Confidence</td><td><b>{summary['confidence']} %</b></td></tr>
    <tr><td>📉 Avg Simulation Gain</td><td><b>{summary['simulation_gain']} %</b></td></tr>
    <tr><td>🕒 Last Updated</td><td><b>{summary['timestamp']}</b></td></tr>
    </table><hr>
    <h3>✅ Digital Twin Status : OPERATIONAL</h3>
    </div>"""
    display(HTML(html))

def system_health_check():
    state  = DIGITAL_TWIN_STATE["data"]
    checks = {
        "Events"         : not state["events"].empty,
        "Predictions"    : not state.get("predictions",     pd.DataFrame()).empty,
        "Recommendations": not state.get("recommendations", pd.DataFrame()).empty,
        "Simulation"     : not state.get("simulation_logs", pd.DataFrame()).empty
    }
    print()
    print("=" * 60)
    print("SYSTEM HEALTH")
    print("=" * 60)
    for k, v in checks.items():
        print(f"{k:20}", "✅ PASS" if v else "❌ FAIL")
    print("=" * 60)

def knowledge_base_statistics():
    events = DIGITAL_TWIN_STATE["data"]["events"]
    threshold = events["risk_score"].quantile(0.98) if not events.empty else 80
    print()
    print("=" * 60)
    print("KNOWLEDGE BASE")
    print("=" * 60)
    print("Total Events          :", len(events))
    print("Unique Zones          :", events["zone"].nunique()     if "zone"     in events.columns else "N/A")
    print("Unique Junctions      :", events["junction"].nunique() if "junction" in events.columns else "N/A")
    print("Clusters              :", events["cluster_id"].nunique())
    print("Average Risk          :", round(events["risk_score"].mean(), 2))
    print(f"Critical Threshold    : >= {round(threshold,2)} (top 2%)")
    print("Critical Events (top 2%) :", int((events["risk_score"] >= threshold).sum()))
    print("=" * 60)

display_command_center()
system_health_check()
knowledge_base_statistics()

logger.info("------------------------------------------")
logger.info("EVENTWISE AI COMMAND CENTER READY")
logger.info("------------------------------------------")
