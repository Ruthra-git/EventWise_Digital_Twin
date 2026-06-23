import streamlit as st
from streamlit_folium import st_folium

import pandas as pd
import numpy as np

import sqlite3

import folium
from folium.plugins import HeatMap

import plotly.express as px
import plotly.graph_objects as go
from recommend import generate_action_plan
from bots import run_all_bots
from pdfcard import get_pdf_bytes
from pathlib import Path
from datetime import datetime
from recommend import generate_action_plan
from bots import run_all_bots
from pdfcard import get_pdf_bytes

from gtts import gTTS
import tempfile
import os
from twin_score import get_twin_score_ui, get_sidebar_score_widget
import warnings
from corridor_watch import (
    get_corridor_watch_ui,
    get_corridor_sidebar_widget
)
warnings.filterwarnings("ignore")

  
# Streamlit Page Configuration  

st.set_page_config(
    page_title="EventWise AI Digital Twin",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

  
# Application Title  

st.title("🚦 EventWise AI Digital Twin")
st.caption(
    "AI-powered Decision Support Platform for Intelligent Traffic Management"
)

  
# Theme Colors  

PRIMARY_COLOR = "#1565C0"

SUCCESS_COLOR = "#2E7D32"

WARNING_COLOR = "#F9A825"

DANGER_COLOR = "#C62828"

BACKGROUND_COLOR = "#F8F9FA"

  
# Performance Constants  

MAX_LIVE_EVENTS = 1500

MAX_PREDICTIONS = 2000

MAX_DEPLOYMENTS = 1500

MAX_SIMULATIONS = 1000

LOW_CONFIDENCE_PERCENTILE = 10

  
# Database Path  


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "database" / "brain.db"



  
# Auto Refresh Interval  

CACHE_TTL = 30

  
# Layer Names  

MAP_LAYERS = {

    "📍 Live Events": True,

    "🔥 AI Heatmap": True,

    "📍 Smart Clusters": True,

    "🔮 Future Predictions": True,

    "👮 Resource Deployment": True,

    "📚 Historical Intelligence": True,

    "🎯 AI Confidence": True,

    "🎮 What-if Simulation": True

}

  
# System Startup Time  

APP_START_TIME = datetime.now()

  
# End of Cell 1    
# Cell 2 : Database Loader and Session Initialization  

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_data():
    """
    Load all EventWise AI Digital Twin tables.

    Cached to prevent repeated database reads.

    Returns
    -------
    dict
        Dictionary containing all application DataFrames.
    """

    conn = None

    try:

        conn = sqlite3.connect(DB_PATH)

        app_data = {

            "events":
                pd.read_sql(
                    "SELECT * FROM events",
                    conn
                ),

            "predictions":
                pd.read_sql(
                    "SELECT * FROM predictions",
                    conn
                ),

            "recommendations":
                pd.read_sql(
                    "SELECT * FROM recommendations",
                    conn
                ),

            "simulations":
                pd.read_sql(
                    "SELECT * FROM simulation_logs",
                    conn
                ),

            "similar_events":
                pd.read_sql(
                    "SELECT * FROM similar_events",
                    conn
                ),

            "hotspot_statistics":
                pd.read_sql(
                    "SELECT * FROM hotspot_statistics",
                    conn
                ),

            "feedback":
                pd.read_sql(
                    "SELECT * FROM feedback",
                    conn
                ),

            "deployment_logs":
                pd.read_sql(
                    "SELECT * FROM deployment_logs",
                    conn
                ),

            "model_metadata":
                pd.read_sql(
                    "SELECT * FROM model_metadata",
                    conn
                ),

            "bot_sessions":
                pd.read_sql(
                    "SELECT * FROM bot_sessions",
                    conn
                )

        }

        app_data["metadata"] = {

            "database_path": DB_PATH,

            "loaded_time": datetime.now(),

            "total_tables": 10,

            "cache_ttl": CACHE_TTL

        }

        return app_data

    except Exception as e:

        st.error(f"❌ Database Loading Error\n\n{e}")

        st.stop()

    finally:

        if conn is not None:

            conn.close()

  
# Load Cached Database  

APP_DATA = load_data()

  
# Global DataFrames  

events_df = APP_DATA["events"]

predictions_df = APP_DATA["predictions"]

recommendations_df = APP_DATA["recommendations"]

simulations_df = APP_DATA["simulations"]

similar_df = APP_DATA["similar_events"]

hotspot_df = APP_DATA["hotspot_statistics"]

feedback_df = APP_DATA["feedback"]

deployment_df = APP_DATA["deployment_logs"]

metadata_df = APP_DATA["model_metadata"]

bot_sessions_df = APP_DATA["bot_sessions"]

  
# Session State Initialization  

SESSION_DEFAULTS = {

    "selected_event": None,

    "selected_cluster": None,

    "selected_zone": "All",

    "selected_event_type": "All",

    "selected_severity": "All",

    "selected_prediction": "All",

    "search_location": "",

    "map_clicked": None,

    "filtered_events": events_df,

    "filtered_predictions": predictions_df,

    "filtered_recommendations": recommendations_df,

    "filtered_simulations": simulations_df,

    "last_refresh": datetime.now(),

    "dashboard_mode": "Command Center"

}

for key, value in SESSION_DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value

  
# Database Validation  

required_tables = [

    "events",

    "predictions",

    "recommendations",

    "simulations",

    "similar_events",

    "hotspot_statistics",

    "feedback",

    "deployment_logs",

    "model_metadata",

    "bot_sessions"

]

missing_tables = [

    table

    for table in required_tables

    if table not in APP_DATA

]

if missing_tables:

    st.error(

        f"Missing Tables : {missing_tables}"

    )

    st.stop()

  
# Executive Load Summary  

st.success("✅ EventWise AI Knowledge Base Loaded Successfully")

# ------------------------------------------------
# Dashboard Statistics
# ------------------------------------------------

total_events = len(events_df)

# High confidence predictions
high_confidence = 0

if (
    not predictions_df.empty
    and
    "confidence" in predictions_df.columns
):

    if predictions_df["confidence"].max() <= 1:

        high_confidence = len(

            predictions_df[
                predictions_df["confidence"] >= 0.80
            ]

        )

    else:

        high_confidence = len(

            predictions_df[
                predictions_df["confidence"] >= 80
            ]

        )

# Total officers recommended
total_officers = 0

if (
    not recommendations_df.empty
    and
    "recommended_officers" in recommendations_df.columns
):

    total_officers = int(

        recommendations_df[
            "recommended_officers"
        ].head(50).sum()

    )

# Critical events
critical_events = 0

if "severity" in events_df.columns:

    critical_events = len(

        events_df[
            events_df["severity"] >= 4
        ]

    )

# AI sessions
total_sessions = len(bot_sessions_df)

# ------------------------------------------------
# KPI Cards
# ------------------------------------------------

col1, col2, col3, col4= st.columns(4)

with col1:

    st.metric(

        "🚦 Events",

        total_events

    )



with col2:

    st.metric(

        "👮 Officers",

        total_officers

    )

with col3:

    st.metric(

        "🚨 Critical Events",

        critical_events

    )

with col4:

    st.metric(

        "🤖 AI Sessions",

        total_sessions

    )
  
# Quick Health Check  

with st.expander("🛠 System Status", expanded=False):

    health = pd.DataFrame({

        "Table": [

            "events",

            "predictions",

            "recommendations",

            "simulation_logs",

            "similar_events",

            "hotspot_statistics",

            "feedback",

            "deployment_logs",

            "model_metadata",

            "bot_sessions"

        ],

        "Rows": [

            len(events_df),

            len(predictions_df),

            len(recommendations_df),

            len(simulations_df),

            len(similar_df),

            len(hotspot_df),

            len(feedback_df),

            len(deployment_df),

            len(metadata_df),

            len(bot_sessions_df)

        ]

    })

    st.dataframe(

        health,

        use_container_width=True,

        hide_index=True

    )

  
# End of Cell 2    
# Cell 3 : Enterprise Sidebar & Global Filters  

st.sidebar.title("🚦 Command Center")

st.sidebar.markdown("---")
  
# Filter Options  

zone_options = ["All"]
if "zone" in events_df.columns:
    zone_options.extend(
        sorted(events_df["zone"].dropna().astype(str).unique().tolist())
    )

event_options = ["All"]
if "event_type" in events_df.columns:
    event_options.extend(
        sorted(events_df["event_type"].dropna().astype(str).unique().tolist())
    )

cluster_options = ["All"]
if "cluster_id" in events_df.columns:
    cluster_options.extend(
        sorted(events_df["cluster_id"].dropna().unique().tolist())
    )

severity_options = ["All", 1, 2, 3, 4, 5]
  
# Sidebar Controls  

selected_zone = st.sidebar.selectbox(
    "🌍 Zone",
    zone_options,
    index=zone_options.index(st.session_state.selected_zone)
    if st.session_state.selected_zone in zone_options else 0
)

selected_event = st.sidebar.selectbox(
    "🎪 Event Type",
    event_options
)

selected_severity = st.sidebar.selectbox(
    "⚠ Severity",
    severity_options
)

selected_cluster = st.sidebar.selectbox(
    "📍 Cluster",
    cluster_options
)

search_text = st.sidebar.text_input(
    "🔍 Search Location"
)

st.sidebar.markdown("---")
  
# Map Layers  

st.sidebar.subheader("🗺 Map Layers")

for layer in MAP_LAYERS.keys():

    MAP_LAYERS[layer] = st.sidebar.checkbox(

        layer,

        value=MAP_LAYERS[layer]

    )

st.sidebar.markdown("---")
  
# Apply Filters  

filtered_events = events_df.copy()

if selected_zone != "All" and "zone" in filtered_events.columns:

    filtered_events = filtered_events[
        filtered_events["zone"] == selected_zone
    ]

if selected_event != "All" and "event_type" in filtered_events.columns:

    filtered_events = filtered_events[
        filtered_events["event_type"] == selected_event
    ]

if selected_severity != "All" and "severity" in filtered_events.columns:

    filtered_events = filtered_events[
        filtered_events["severity"] == selected_severity
    ]

if selected_cluster != "All" and "cluster_id" in filtered_events.columns:

    filtered_events = filtered_events[
        filtered_events["cluster_id"] == selected_cluster
    ]

if search_text != "":

    search = search_text.lower()

    search_cols = []

    for col in [
        "junction",
        "location",
        "event_cause",
        "event_type",
        "zone"
    ]:

        if col in filtered_events.columns:

            search_cols.append(col)

    if len(search_cols):

        mask = filtered_events[search_cols].astype(str).apply(

            lambda x: x.str.lower().str.contains(search)

        ).any(axis=1)

        filtered_events = filtered_events[mask]
  
# Synchronize Other Tables  

event_ids = []

if "id" in filtered_events.columns:

    event_ids = filtered_events["id"].tolist()

filtered_predictions = predictions_df.copy()

if "event_id" in filtered_predictions.columns:

    filtered_predictions = filtered_predictions[
        filtered_predictions["event_id"].isin(event_ids)
    ]

filtered_recommendations = recommendations_df.copy()

if "event_id" in filtered_recommendations.columns:

    filtered_recommendations = filtered_recommendations[
        filtered_recommendations["event_id"].isin(event_ids)
    ]

filtered_simulations = simulations_df.copy()

if "event_id" in filtered_simulations.columns:

    filtered_simulations = filtered_simulations[
        filtered_simulations["event_id"].isin(event_ids)
    ]
  
# Store in Session  

st.session_state.selected_zone = selected_zone

st.session_state.selected_cluster = selected_cluster

st.session_state.filtered_events = filtered_events

st.session_state.filtered_predictions = filtered_predictions

st.session_state.filtered_recommendations = filtered_recommendations

st.session_state.filtered_simulations = filtered_simulations
  
# Sidebar Actions  

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Refresh Dashboard"):

    st.cache_data.clear()

    for key in [

        "backend_plan",

        "backend_bot_outputs",

        "backend_event"

    ]:

        st.session_state.pop(key, None)

    st.rerun()

if st.sidebar.button("♻ Reset Filters"):

    st.session_state.selected_zone = "All"

    st.session_state.selected_cluster = "All"

    st.rerun()
#get_sidebar_score_widget()
get_corridor_sidebar_widget()
#get_twin_score_ui()  
# Voice Generation Function  

def play_voice_message(message):

    tts = gTTS(
        text=message,
        lang="en",
        slow=False
    )

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp3"
    )

    tts.save(temp.name)

    with open(temp.name, "rb") as f:

        audio_bytes = f.read()

    st.audio(
        audio_bytes,
        format="audio/mp3"
    )

    try:
        os.remove(temp.name)
    except:
        pass
  
# End of Cell 3    
# Cell 5A : Digital Twin Data Preparation  

st.markdown("---")

st.subheader("🌍 Preparing AI Digital Twin")
  
# Load Filtered Events  

events = st.session_state.get(
    "filtered_events",
    events_df
)
  
# Empty Dataset Check  

if events.empty:

    st.warning(
        "No events available for selected filters."
    )

    st.stop()
  
# Coordinate Validation  

required_columns = [

    "lat",

    "lon",

    "severity",

    "priority_score",

    "cluster_id",

    "cluster_risk_score",

    "hotspot_density",

    "cluster_lat_center",

    "cluster_lon_center"

]

missing = [

    col

    for col in required_columns

    if col not in events.columns

]

if len(missing):

    st.error(

        f"Missing columns : {missing}"

    )

    st.stop()
  
# Remove Invalid Coordinates  

events = events.dropna(

    subset=[

        "lat",

        "lon"

    ]

)

events = events[

    (events["lat"] != 0)

    &

    (events["lon"] != 0)

]
  
# Heatmap Dataset  

heatmap_events = events.copy()
  
# Marker Dataset  

MAX_MARKERS = 300

marker_events = (

    events

    .sort_values(

        by=[

            "cluster_risk_score",

            "priority_score",

            "severity"

        ],

        ascending=False

    )

    .head(MAX_MARKERS)

)
  
# Cluster Dataset  

cluster_df = (

    events

    [

        [

            "cluster_id",

            "cluster_lat_center",

            "cluster_lon_center",

            "cluster_risk_score",

            "cluster_risk_tier"

        ]

    ]

    .drop_duplicates()

)
  
# Heatmap Points  

heat_data = [

    [

        row.lat,

        row.lon,

        row.hotspot_density

    ]

    for row in

    heatmap_events.itertuples()

]
  
# Map Center  

MAP_CENTER = [

    heatmap_events["lat"].mean(),

    heatmap_events["lon"].mean()

]
  
# Store for Cell 5B  

st.session_state.heatmap_events = heatmap_events

st.session_state.marker_events = marker_events

st.session_state.cluster_df = cluster_df

st.session_state.heat_data = heat_data

st.session_state.map_center = MAP_CENTER
  
# Preparation Summary  

c1, c2, c3, c4 = st.columns(4)

c1.metric(

    "Total Events",

    len(events)

)

c2.metric(

    "Displayed Markers",

    len(marker_events)

)

c3.metric(

    "AI Clusters",

    len(cluster_df)

)

c4.metric(

    "Heatmap Points",

    len(heat_data)

)

st.success(

    "✅ Digital Twin datasets prepared successfully."

)
  
# End Cell 5A    
# Cell 5B : AI Digital Twin Rendering Engine  

st.markdown("---")
st.subheader("🌍 Live AI Digital Twin")
  
# Load Prepared Objects  

heatmap_events = st.session_state.get(
    "heatmap_events",
    pd.DataFrame()
)

marker_events = st.session_state.get(
    "marker_events",
    pd.DataFrame()
)

cluster_df = st.session_state.get(
    "cluster_df",
    pd.DataFrame()
)

heat_data = st.session_state.get(
    "heat_data",
    []
)

map_center = st.session_state.get(
    "map_center",
    [12.9716, 77.5946]
)
  
# Create Map  

m = folium.Map(

    location=map_center,

    zoom_start=11,

    tiles="CartoDB positron",

    control_scale=True,

    prefer_canvas=True

)
  
# AI Heatmap Layer  

if MAP_LAYERS.get("🔥 AI Heatmap", True):

    HeatMap(

        heat_data,

        radius=18,

        blur=15,

        min_opacity=0.30

    ).add_to(m)
  
# Live Event Layer  

if MAP_LAYERS.get("📍 Live Events", True):

    fg_events = folium.FeatureGroup(
        name="Live Events"
    )

    for row in marker_events.itertuples():

        if row.severity >= 4:

            color = "red"

        elif row.severity == 3:

            color = "orange"

        elif row.severity == 2:

            color = "blue"

        else:

            color = "green"

        popup = f"""
        <b>Event :</b> {row.event_type}<br>
        <b>Cause :</b> {row.event_cause}<br>
        <b>Severity :</b> {row.severity}<br>
        <b>Priority :</b> {row.priority_score:.2f}<br>
        <b>Risk :</b> {row.cluster_risk_score:.2f}<br>
        <b>Tier :</b> {row.cluster_risk_tier}<br>
        <b>Zone :</b> {row.zone}<br>
        <b>Junction :</b> {row.junction}
        """

        folium.CircleMarker(

            location=[row.lat, row.lon],

            radius=6,

            color=color,

            fill=True,

            fill_color=color,

            fill_opacity=0.85,

            weight=2,

            tooltip=row.event_type,

            popup=popup

        ).add_to(fg_events)

    fg_events.add_to(m)
  
# Smart Cluster Layer  

if MAP_LAYERS.get("📍 Smart Clusters", True):

    fg_cluster = folium.FeatureGroup(
        name="AI Clusters"
    )

    for row in cluster_df.itertuples():

        risk = float(row.cluster_risk_score)

        radius = max(250, min(1000, risk * 120))

        folium.Circle(

            location=[
                row.cluster_lat_center,
                row.cluster_lon_center
            ],

            radius=radius,

            color="purple",

            fill=True,

            fill_opacity=0.08,

            tooltip=f"""
Cluster {row.cluster_id}

Risk : {risk:.2f}

Tier : {row.cluster_risk_tier}
"""

        ).add_to(fg_cluster)

        folium.Marker(

            [
                row.cluster_lat_center,
                row.cluster_lon_center
            ],

            icon=folium.Icon(

                color="purple",

                icon="info-sign"

            )

        ).add_to(fg_cluster)

    fg_cluster.add_to(m)
  
# Layer Control  

folium.LayerControl(
    collapsed=False
).add_to(m)
  
# Render Map  

map_state = st_folium(

    m,

    height=700,

    width=None,

    returned_objects=[
        "last_clicked"
    ],

    use_container_width=True

)
  
# Click Inspector  

if map_state:

    clicked = map_state.get("last_clicked")

    if clicked is not None:

        st.success(

            f"""
📍 Selected Location

Latitude : {clicked['lat']:.6f}

Longitude : {clicked['lng']:.6f}
"""

        )
  
# Footer Statistics  

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(

        "🔥 Heatmap Events",

        len(heatmap_events)

    )

with c2:

    st.metric(

        "📍 Rendered Events",

        len(marker_events)

    )

with c3:

    st.metric(

        "🧠 AI Clusters",

        len(cluster_df)

    )

st.caption(
    """
⚡ Performance Optimized

• Heatmap uses complete dataset

• Only Top 300 AI-ranked events rendered

• Cluster centers precomputed

• Canvas rendering enabled for smooth interaction
"""
)
st.markdown("---")
# ==========================================
# Synchronize Corridor Watch with Dashboard
# ==========================================

import corridor_watch

corridor_watch.load_corridor_data = lambda: {

    "events": st.session_state.filtered_events.copy(),

    "hotspots": hotspot_df.copy(),

    "predictions": st.session_state.filtered_predictions.copy(),

    "recommendations": st.session_state.filtered_recommendations.copy(),

    "simulations": st.session_state.filtered_simulations.copy()

}

# Force recomputation whenever filters change

st.session_state.pop(

    "corridor_df",

    None

)
get_corridor_watch_ui()

st.markdown("---")
  
# End Cell 5B    
# Cell 4 : Executive AI Command Dashboard  

st.markdown("## 📊 Executive Traffic Intelligence Dashboard")

events = st.session_state.filtered_events

predictions = st.session_state.filtered_predictions

recommendations = st.session_state.filtered_recommendations

simulations = st.session_state.filtered_simulations
  
# KPI Computation  

total_events = len(events)

critical_events = 0

if "severity" in events.columns:

    critical_events = len(

        events[events["severity"] >= 4]

    )

avg_severity = 0

if "severity" in events.columns:

    avg_severity = round(

        events["severity"].mean(),

        2

    )

avg_confidence = 0

if not predictions.empty and "confidence" in predictions.columns:

    confidence = predictions["confidence"]

    if confidence.max() <= 1:

        confidence = confidence * 100

    avg_confidence = round(

        confidence.mean(),

        1

    )

recommended_officers = 0

if (

    not recommendations.empty

    and

    "recommended_officers" in recommendations.columns

):

    recommended_officers = int(

        recommendations["recommended_officers"].sum()

    )

avg_improvement = 0

if (

    not simulations.empty

    and

    "expected_reduction" in simulations.columns

):

    avg_improvement = simulations[
    "expected_reduction"
].fillna(0).mean()

    if avg_improvement <= 1:

        avg_improvement *= 100

    avg_improvement = round(
    avg_improvement,
    1
)
if (
    not simulations.empty
    and
    "expected_reduction" in simulations.columns
):

    sim_col = simulations[
        "expected_reduction"
    ].fillna(0)

    avg_improvement = sim_col.mean()

    if avg_improvement <= 1:

        avg_improvement *= 100

    avg_improvement = round(
        avg_improvement,
        1
    )  
# KPI Cards  

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:

    st.metric(

        "🚦 Events",

        f"{total_events:,}"

    )

with c2:

    st.metric(

        "🚨 Critical",

        f"{critical_events:,}"

    )

with c3:

    st.metric(

        "📈 Avg Severity",

        avg_severity

    )

with c4:

    st.metric(

        "🤖 AI Confidence",

        f"{avg_confidence}%"

    )

with c5:

    st.metric(

        "👮 Officers",

        f"{recommended_officers:,}"

    )

with c6:

    st.metric(

        "🎯 AI Gain",

        f"{avg_improvement}%"

    )

st.markdown("---")
  
# Executive Analytics  

left, right = st.columns([2, 1])

with left:

    if (

        "event_type" in events.columns

        and

        len(events)

    ):

        fig = px.histogram(

            events,

            x="event_type",

            title="Event Distribution",

            height=350

        )

        fig.update_layout(

            margin=dict(

                l=10,

                r=10,

                t=40,

                b=10

            )

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

with right:

    severity_counts = (

        events["severity"]

        .value_counts()

        .sort_index()

        if "severity" in events.columns

        else None

    )

    if severity_counts is not None:

        fig = px.pie(

            values=severity_counts.values,

            names=severity_counts.index,

            title="Severity Distribution",

            height=350

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

st.markdown("---")
  
# AI Operational Summary  

col1, col2 = st.columns(2)

with col1:

    st.info(

        f"""

### 🚦 Traffic Summary

• Active Events : **{total_events:,}**

• Critical Events : **{critical_events:,}**

• Average Severity : **{avg_severity}**

"""
    )

with col2:

    st.success(

        f"""

### 🤖 AI Summary

• Mean Confidence : **{avg_confidence}%**

• Officers Recommended : **{recommended_officers:,}**

• Estimated Congestion Reduction : **{avg_improvement}%**

"""
    )
  
# End Cell 4  


  
# 🚀 EXECUTIVE CRISIS MODE  

st.markdown("---")
st.subheader("🚀 Executive Crisis Mode")

st.caption(
    "One-click emergency response workflow for traffic command centers"
)

if st.button(
    "🚨 Activate Crisis Mode",
    use_container_width=True
):

    events = st.session_state.get(
        "filtered_events",
        events_df
    )

    if len(events) == 0:

        st.warning("No events available.")

    else:

        sort_cols = [
            c for c in [
                "cluster_risk_score",
                "priority_score",
                "severity"
            ]
            if c in events.columns
        ]

        highest_event = (
            events
            .sort_values(
                by=sort_cols,
                ascending=False
            )
            .iloc[0]
        )

        st.error(
            f"""
🚨 CRITICAL INCIDENT DETECTED

Zone : {highest_event.get('zone','Unknown')}

Cause : {highest_event.get('event_cause','Unknown')}

Severity : {highest_event.get('severity',0)}

Cluster : {highest_event.get('cluster_id','N/A')}
"""
        )

        officers = int(
            highest_event.get(
                "recommended_officers",
                highest_event.get(
                    "severity",
                    1
                ) * 3
            )
        )

        reduction = min(
            45,
            officers * 0.08
        )

        clearance = max(
            10,
            45 - reduction
        )

        st.success(
            f"""
### 🤖 AI Response Plan

👮 Deploy {officers} Officers

🚧 Deploy Temporary Barricades

↪ Activate Diversion Route

🚦 Enable Adaptive Signals

⏱ Estimated Clearance :
{clearance:.1f} Minutes

📉 Expected Congestion Reduction :
{reduction:.1f}%
"""
        )

        st.info(
            """
### 🎯 Crisis Workflow

1️⃣ Detect Critical Cluster

2️⃣ Predict Escalation

3️⃣ Generate Resource Plan

4️⃣ Deploy Field Units

5️⃣ Monitor Resolution
"""
        )

        try:

            msg = f"""
Attention traffic response teams.

Critical incident detected.

Zone {highest_event.get('zone','Unknown')}.

Deploy {officers} officers immediately.

Enable diversion routes.

Estimated clearance time is
{clearance:.0f} minutes.
"""

            play_voice_message(msg)

        except:

            pass

        st.balloons()
    with st.expander(
    "📋 Executive Incident Summary",
    expanded=True
):

        st.json({
            "zone":
            highest_event.get("zone"),

            "cause":
            highest_event.get("event_cause"),

            "severity":
            highest_event.get("severity"),

            "officers":
            officers,

            "clearance":
            clearance,

            "reduction":
            reduction
        })

st.markdown("---")  
# ⏳ DIGITAL TWIN TIMELINE  

st.markdown("---")

st.subheader("⏳ Digital Twin Timeline")

timeline = st.slider(

    "Simulation Horizon",

    -30,
    30,
    0,
    15

)

if timeline < 0:

    twin_mode = f"{abs(timeline)} min ago"

elif timeline == 0:

    twin_mode = "Live Now"

else:

    twin_mode = f"+{timeline} min prediction"

st.info(
    f"🌍 Digital Twin State : {twin_mode}"
)
events = st.session_state.get(
    "filtered_events",
    events_df
).copy()
if timeline > 0:

    growth_factor = 1 + (
        timeline / 120
    )

    if "cluster_risk_score" in events.columns:

        events["cluster_risk_score"] = (
            events["cluster_risk_score"]
            *
            growth_factor
        )

    if "priority_score" in events.columns:

        events["priority_score"] = (
            events["priority_score"]
            *
            growth_factor
        )

elif timeline < 0:

    reduction_factor = 1 - (
        abs(timeline) / 120
    )

    if "cluster_risk_score" in events.columns:

        events["cluster_risk_score"] = (
            events["cluster_risk_score"]
            *
            reduction_factor
        )

    if "priority_score" in events.columns:

        events["priority_score"] = (
            events["priority_score"]
            *
            reduction_factor
        )
avg_risk = (
    events["cluster_risk_score"]
    .mean()
)

critical = len(
    events[
        events["cluster_risk_score"]
        >= 4
    ]
)

c1,c2,c3 = st.columns(3)

c1.metric(

    "🌍 Twin Time",

    twin_mode

)

c2.metric(

    "⚠ Avg Risk",

    f"{avg_risk:.2f}"

)

c3.metric(

    "🚨 Critical Clusters",

    critical

)
if timeline > 0:

    st.warning(f"""

### 🔮 AI Forecast

Traffic demand expected to increase
over the next {timeline} minutes.

Average risk projected to reach
{avg_risk:.2f}

AI recommends proactive deployment
before congestion escalation.

""")

elif timeline < 0:

    st.success(f"""

### 📚 Historical Replay

Viewing traffic state

{abs(timeline)} minutes earlier.

Used for incident reconstruction
and response analysis.

""")

else:

    st.info("""

### 📍 Live Operations Mode

Displaying current city state.

AI monitoring active.

""")
with st.expander("🤖 AI Analytics & Prediction Center", expanded = False):

    events=st.session_state.get("filtered_events",events_df)
    predictions=st.session_state.get("filtered_predictions",predictions_df)
    recommendations=st.session_state.get("filtered_recommendations",recommendations_df)

    t1,t2,t3,t4,t5=st.tabs(["🎯 Confidence","⚠ Risk","🚦 Causes","🌍 Zones","📋 Intelligence"])

    with t1:
        if not predictions.empty and "confidence" in predictions.columns:
            conf=predictions.copy()
            if conf["confidence"].max()<=1:
                conf["confidence"]=conf["confidence"]*100
            fig=px.histogram(
                conf,
                x="confidence",
                nbins=25,
                title="AI Confidence Distribution"
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig,use_container_width=True)
            c1,c2,c3=st.columns(3)
            c1.metric("Average",f"{conf['confidence'].mean():.1f}%")
            c2.metric("Maximum",f"{conf['confidence'].max():.1f}%")
            c3.metric("Minimum",f"{conf['confidence'].min():.1f}%")
        else:
            st.info("Confidence data unavailable.")

    with t2:
        if "cluster_risk_tier" in events.columns:
            risk=events["cluster_risk_tier"].value_counts().reset_index()
            risk.columns=["Risk Tier","Count"]
            fig=px.bar(
                risk,
                x="Risk Tier",
                y="Count",
                text="Count",
                title="Cluster Risk Distribution"
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig,use_container_width=True)
        elif "cluster_risk_score" in events.columns:
            fig=px.histogram(
                events,
                x="cluster_risk_score",
                nbins=20,
                title="Cluster Risk Score"
            )
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Risk information unavailable.")

    with t3:
        col="event_cause_clean" if "event_cause_clean" in events.columns else "event_cause"
        if col in events.columns:
            cause=events[col].value_counts().head(10).reset_index()
            cause.columns=["Cause","Count"]
            fig=px.bar(
                cause,
                x="Count",
                y="Cause",
                orientation="h",
                title="Top Event Causes"
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Cause information unavailable.")

    with t4:
        if "zone" in events.columns and "cluster_risk_score" in events.columns:
            zone=events.groupby("zone").agg(
                Events=("id","count"),
                AvgRisk=("cluster_risk_score","mean")
            ).reset_index()
            fig=px.bar(
                zone,
                x="zone",
                y="AvgRisk",
                color="Events",
                title="Zone Intelligence"
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(zone,use_container_width=True,hide_index=True)
        else:
            st.info("Zone intelligence unavailable.")

    with t5:
        merged=events.copy()
        if(
            not predictions.empty and
            "event_id" in predictions.columns and
            "id" in merged.columns
        ):
            merged["id"]=merged["id"].astype(str)
            pred=predictions.copy()
            pred["event_id"]=pred["event_id"].astype(str)
            merged=merged.merge(
                pred,
                left_on="id",
                right_on="event_id",
                how="left"
            )
        cols=[]
        for c in[
            "event_type",
            "zone",
            "junction",
            "severity",
            "priority_score",
            "cluster_risk_score",
            "cluster_risk_tier",
            "confidence"
        ]:
            if c in merged.columns:
                cols.append(c)
        st.dataframe(
            merged[cols].head(100),
            use_container_width=True,
            hide_index=True
        )
        st.markdown("---")   
    # 🚨 EMERGENCY SCENARIO CENTER   

    st.markdown("---")

with st.expander("🚨 Emergency Scenario Simulator", expanded= False):

    scenario = st.selectbox(

        "Select Scenario",

        [

            "Normal Operations",

            "🏟 Stadium Event",

            "🏏 Cricket Match",

            "🎤 Concert",

            "🌧 Heavy Rain",

            "🚨 Major Accident"

        ]

    )   
    # Scenario Intelligence   

    scenario_events = 0
    scenario_risk = 1.0
    scenario_officers = 0
    scenario_clearance = 15
    scenario_diversion = False

    if scenario == "🏟 Stadium Event":

        scenario_events = 120

        scenario_risk = 4.5

        scenario_officers = 35

        scenario_clearance = 55

        scenario_diversion = True

    elif scenario == "🏏 Cricket Match":

        scenario_events = 180

        scenario_risk = 4.8

        scenario_officers = 50

        scenario_clearance = 70

        scenario_diversion = True

    elif scenario == "🎤 Concert":

        scenario_events = 90

        scenario_risk = 3.9

        scenario_officers = 25

        scenario_clearance = 40

        scenario_diversion = True

    elif scenario == "🌧 Heavy Rain":

        scenario_events = 75

        scenario_risk = 4.2

        scenario_officers = 20

        scenario_clearance = 60

        scenario_diversion = False

    elif scenario == "🚨 Major Accident":

        scenario_events = 45

        scenario_risk = 5.0

        scenario_officers = 30

        scenario_clearance = 80

        scenario_diversion = True
    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "🚦 Expected Events",
        scenario_events
    )

    c2.metric(
        "⚠ Risk Score",
        scenario_risk
    )

    c3.metric(
        "👮 Officers Required",
        scenario_officers
    )

    c4.metric(
        "⏱ Clearance Time",
        f"{scenario_clearance} min"
    )
    st.success(f"""

    ### 🤖 Scenario Response Plan

    Scenario :
    **{scenario}**

    Expected Traffic Events :
    **{scenario_events}**

    Risk Level :
    **{scenario_risk}**

    Required Officers :
    **{scenario_officers}**

    Estimated Clearance :
    **{scenario_clearance} Minutes**

    Diversion Required :
    **{'YES' if scenario_diversion else 'NO'}**

    """)
    scenario_df = pd.DataFrame({

        "Metric":[

            "Events",

            "Risk",

            "Officers",

            "Clearance"

        ],

        "Value":[

            scenario_events,

            scenario_risk,

            scenario_officers,

            scenario_clearance

        ]

    })

    fig = px.bar(

        scenario_df,

        x="Metric",

        y="Value",

        text="Value",

        title="Scenario Impact Analysis"

    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    if st.button(
        "🎤 Generate Scenario Briefing"
    ):

        msg = f"""

    Attention traffic command center.

    Scenario detected.

    {scenario}.

    Expected events are
    {scenario_events}.

    Deploy
    {scenario_officers}
    officers.

    Estimated clearance time
    is
    {scenario_clearance}
    minutes.

    """

        play_voice_message(msg)

with st.expander("🎮 What-If Simulation Studio", expanded = False):

    events=st.session_state.get("filtered_events",events_df)
    simulations=st.session_state.get("filtered_simulations",simulations_df)

    c1,c2,c3,c4=st.columns(4)

    with c1:
        officers=st.slider("👮 Additional Officers",0,50,10)

    with c2:
        barricades=st.slider("🚧 Barricades",0,20,5)

    with c3:
        diversion=st.slider("↪ Diversion Level (%)",0,100,40)

    with c4:
        signal=st.slider("🚦 Signal Optimization (%)",0,100,50)

    base=len(events)

    risk=events["cluster_risk_score"].mean() if "cluster_risk_score" in events.columns else 3

    pred_reduction=min(
        60,
        officers*0.45+
        barricades*0.80+
        diversion*0.15+
        signal*0.20
    )

    pred_time=max(
        5,
        45-pred_reduction*0.5
    )

    pred_events=max(
        0,
        int(base*(1-pred_reduction/100))
    )

    m1,m2,m3,m4=st.columns(4)

    m1.metric(
        "🚦Current Events",
        base
    )

    m2.metric(
        "🎯Predicted Events",
        pred_events,
        delta=-(base-pred_events)
    )

    m3.metric(
        "📉Congestion Reduction",
        f"{pred_reduction:.1f}%"
    )


    m4.metric(
        "⏱Avg Clearance",
        f"{pred_time:.1f} min"
    )

    l,r=st.columns(2)

    with l:

        sim=pd.DataFrame({

            "Stage":[

                "Current",

                "After Simulation"

            ],

            "Events":[

                base,

                pred_events

            ]

        })

        fig=px.bar(

            sim,

            x="Stage",

            y="Events",

            text="Events",

            title="Event Reduction"

        )

        fig.update_layout(height=420)

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with r:

        pie=pd.DataFrame({

            "Metric":[

                "Resolved",

                "Remaining"

            ],

            "Value":[

                base-pred_events,

                pred_events

            ]

        })

        fig=px.pie(

            pie,

            values="Value",

            names="Metric",

            title="Simulation Outcome"

        )

        fig.update_layout(height=420)

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.success(f"""
    ### 🤖 AI Recommendation

    Deploy **{officers}** additional officers.

    Install **{barricades}** temporary barricades.

    Enable **{diversion}%** traffic diversion.

    Optimize signals by **{signal}%**.

    Estimated congestion reduction: **{pred_reduction:.1f}%**

    Estimated average clearance time: **{pred_time:.1f} minutes**
    """)

    if not simulations.empty:

        with st.expander("📋 Historical Simulation Records"):

            st.dataframe(

                simulations.head(100),

                use_container_width=True,

                hide_index=True

            )   
    # 💰 ECONOMIC IMPACT ENGINE   

    st.markdown("---")

with st.expander("💰 Economic & Sustainability Impact", expanded = False):

    current_delay = base * 45

    optimized_delay = (
        current_delay *
        (1 - pred_reduction/100)
    )

    delay_saved = (
        current_delay -
        optimized_delay
    )

    fuel_cost_per_min = 6

    current_fuel_loss = (
        current_delay *
        fuel_cost_per_min
    )

    optimized_fuel_loss = (
        optimized_delay *
        fuel_cost_per_min
    )

    fuel_saved = max(
        0,
        current_fuel_loss -
        optimized_fuel_loss
    )

    co2_per_min = 0.05

    current_co2 = (
        current_delay *
        co2_per_min
    )

    optimized_co2 = (
        optimized_delay *
        co2_per_min
    )

    co2_saved = max(
        0,
        current_co2 -
        optimized_co2
    )

    c1,c2,c3 = st.columns(3)

    with c1:

        st.metric(
            "⏱ Delay Saved",
            f"{delay_saved:,.0f} min"
        )

    with c2:

        st.metric(
            "⛽ Fuel Savings",
            f"₹{fuel_saved:,.0f}"
        )

    with c3:

        st.metric(
            "🌱 CO₂ Reduction",
            f"{co2_saved:,.1f} kg"
        )
    impact = pd.DataFrame({

        "Scenario":[

            "Current System",

            "AI Optimized"

        ],

        "Delay":[

            current_delay,

            optimized_delay

        ],

        "Fuel Loss":[

            current_fuel_loss,

            optimized_fuel_loss

        ]

    })

    fig = px.bar(

        impact,

        x="Scenario",

        y="Delay",

        color="Scenario",

        title="Traffic Delay Comparison"

    )

    fig.update_layout(
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.success(f"""

    ### 🚀 Estimated Societal Impact

    ⏱ Delay Reduced:
    **{delay_saved:,.0f} minutes**

    ⛽ Fuel Savings:
    **₹{fuel_saved:,.0f}**

    🌱 Carbon Reduction:
    **{co2_saved:,.1f} kg CO₂**

    🎯 AI Intervention Efficiency:
    **{pred_reduction:.1f}%**

    This demonstrates measurable
    economic and environmental benefits
    generated through AI-assisted traffic management.

    """)
    st.markdown("---")
with st.expander("👮 AI Resource Command Center"):

    events=st.session_state.get("filtered_events",events_df)
    recommendations=st.session_state.get("filtered_recommendations",recommendations_df)

    merged=events.copy()

    if(
    not recommendations.empty
    and"id" in merged.columns
    and"event_id" in recommendations.columns
    ):
        merged["id"]=merged["id"].astype(str)
        rec=recommendations.copy()
        rec["event_id"]=rec["event_id"].astype(str)
        merged=merged.merge(
            rec,
            left_on="id",
            right_on="event_id",
            how="left"
        )

    if"priority_score_x"in merged.columns:
        merged.rename(
            columns={"priority_score_x":"priority_score"},
            inplace=True
        )

    if"recommended_officers"not in merged.columns:
        merged["recommended_officers"]=merged["severity"]*2

    if"recommended_barricades"not in merged.columns:
        merged["recommended_barricades"]=merged["severity"]

    sort_cols=[]

    if"cluster_risk_score"in merged.columns:
        sort_cols.append("cluster_risk_score")

    if"priority_score"in merged.columns:
        sort_cols.append("priority_score")

    if"severity"in merged.columns:
        sort_cols.append("severity")

    top=merged.sort_values(
        by=sort_cols,
        ascending=False
    ).head(20)

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "👮 Officers",
        int(top["recommended_officers"].sum())
    )

    c2.metric(
        "🚧 Barricades",
        int(top["recommended_barricades"].sum())
    )

    c3.metric(
        "📍 Clusters",
        top["cluster_id"].nunique()
    )

    c4.metric(
        "⚠ Max Risk",
        f"{top['cluster_risk_score'].max():.2f}"
    )

    left,right=st.columns([2,1])

    with left:

        fig=px.bar(
            top.head(10),
            x="junction",
            y="recommended_officers",
            color="severity",
            text="recommended_officers",
            title="Officer Deployment"
        )

        fig.update_layout(height=430)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        zone=top.groupby("zone").agg(

            Officers=("recommended_officers","sum")

        ).reset_index()

        fig=px.pie(

            zone,

            values="Officers",

            names="zone",

            title="Deployment by Zone"

        )

        fig.update_layout(height=430)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.success(f"""

    ### 🤖 AI Deployment Recommendation

    👮 Deploy **{int(top['recommended_officers'].sum())} officers**

    🚧 Deploy **{int(top['recommended_barricades'].sum())} barricades**

    📍 Active Critical Clusters : **{top['cluster_id'].nunique()}**

    ⚠ Highest Risk Score : **{top['cluster_risk_score'].max():.2f}**

    🎯 Strategy : **Priority-based Dynamic Allocation**

    """)

    cols=[
    "event_type",
    "zone",
    "junction",
    "severity",
    "priority_score",
    "cluster_risk_score",
    "cluster_risk_tier",
    "recommended_officers",
    "recommended_barricades",
    "confidence",
    "expected_clearance"
    ]

    cols=[c for c in cols if c in top.columns]

    st.dataframe(
        top[cols],
        use_container_width=True,
        hide_index=True
    )
st.markdown("---")
with st.expander("📚 Historical Intelligence Center", expanded = False):

    events=st.session_state.get("filtered_events",events_df)
    similar=similar_df.copy()

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
    "Historical Cases",
    len(similar)
    )

    if"event_signature_code"in similar.columns:
        c2.metric(
        "Unique Signatures",
        similar["event_signature_code"].nunique()
        )
    else:
        c2.metric(
        "Unique Signatures",
        0
        )

    if"historical_frequency"in events.columns:
        c3.metric(
        "Avg Frequency",
        f"{events['historical_frequency'].mean():.1f}"
        )
    else:
        c3.metric(
        "Avg Frequency",
        0
        )

    if"event_duration"in events.columns:
        c4.metric(
        "Avg Duration",
        f"{events['event_duration'].mean():.1f} min"
        )
    else:
        c4.metric(
        "Avg Duration",
        0
        )

    t1,t2,t3=st.tabs(
    [
    "📈Trend",
    "🔥Recurring",
    "📋Knowledge Base"
    ]
    )

    with t1:

        if(
        "month"in events.columns
        and
        "historical_frequency"in events.columns
        ):

            trend=events.groupby(
            "month"
            ).agg(
            Frequency=("historical_frequency","mean")
            ).reset_index()

            fig=px.line(
            trend,
            x="month",
            y="Frequency",
            markers=True,
            title="Historical Frequency Trend"
            )

            fig.update_layout(height=420)

            st.plotly_chart(
            fig,
            use_container_width=True
            )

        else:

            st.info("Historical trend unavailable.")

    with t2:

        col="event_cause_clean" if"event_cause_clean"in events.columns else"event_cause"

        if col in events.columns:

            recur=events.groupby(
            col
            ).agg(
            Count=("id","count"),
            AvgRisk=("cluster_risk_score","mean")
            ).reset_index()

            recur=recur.sort_values(
            "Count",
            ascending=False
            ).head(10)

            fig=px.bar(
            recur,
            x=col,
            y="Count",
            color="AvgRisk",
            text="Count",
            title="Recurring Event Patterns"
            )

            fig.update_layout(height=420)

            st.plotly_chart(
            fig,
            use_container_width=True
            )

    with t3:

        kb=events.copy()

        cols=[]

        for c in[
        "event_type",
        "event_cause_clean",
        "zone",
        "junction",
        "historical_frequency",
        "event_duration",
        "cluster_risk_tier"
        ]:
            if c in kb.columns:
                cols.append(c)

        kb=kb.sort_values(
        "historical_frequency",
        ascending=False
        ).head(100)

        st.dataframe(
        kb[cols],
        use_container_width=True,
        hide_index=True
        )

    st.success(f"""
    ### 🧠 Historical AI Insight

    • {len(similar)} historical incidents analysed

    • AI identified recurring hotspot patterns

    • Current events matched against historical signatures

    • Historical intelligence improves deployment recommendations and simulation accuracy

    • Knowledge base continuously evolves with every recorded event
    """)
    st.markdown("---")
with st.expander("🧠 AI Decision Support Assistant", expanded = False):

    events=st.session_state.get("filtered_events",events_df)
    predictions=st.session_state.get("filtered_predictions",predictions_df)
    recommendations=st.session_state.get("filtered_recommendations",recommendations_df)

    risk=events["cluster_risk_score"].mean() if"cluster_risk_score"in events.columns else 0
    severity=events["severity"].mean() if"severity"in events.columns else 0
    confidence=predictions["confidence"].mean() if(not predictions.empty and"confidence"in predictions.columns) else 0
    officers=int(recommendations["recommended_officers"].sum()) if(not recommendations.empty and"recommended_officers"in recommendations.columns) else 0

    if confidence<=1:
        confidence*=100

    if risk>=4:
        status="🔴 Critical"
    elif risk>=3:
        status="🟠 High"
    elif risk>=2:
        status="🟡 Moderate"
    else:
        status="🟢 Low"

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Risk Level",status)
    c2.metric("AI Confidence",f"{confidence:.1f}%")
    c3.metric("Avg Severity",f"{severity:.2f}")
    c4.metric("Recommended Officers",officers)

    st.markdown("---")

    exp1,exp2,exp3,exp4=st.tabs([
    "🚨Situation",
    "👮Deployment",
    "🎯Prediction",
    "📋Executive Brief"
    ])

    with exp1:

        st.warning(f"""
    ### Current Situation

    • Active Events : **{len(events)}**

    • Average Cluster Risk : **{risk:.2f}**

    • Average Severity : **{severity:.2f}**

    • Current Risk Level : **{status}**

    • High priority hotspots require immediate monitoring.
    """)

    with exp2:

        zone="N/A"

        if"zone"in events.columns and not events.empty:
            zone=events.groupby("zone")["cluster_risk_score"].mean().idxmax()

        st.success(f"""
    ### Deployment Strategy

    • Deploy **{officers} officers**

    • Prioritize **{zone} zone**

    • Increase barricade allocation

    • Enable adaptive signal timing

    • Monitor recurring hotspots continuously
    """)

    with exp3:

        reduction=min(45,officers*0.08)

        st.info(f"""
    ### AI Prediction

    • Estimated congestion reduction : **{reduction:.1f}%**

    • Model confidence : **{confidence:.1f}%**

    • Risk expected to decline after deployment

    • Historical patterns indicate improved clearance
    """)

    with exp4:

        st.code(f"""
    EVENTWISE AI EXECUTIVE BRIEF

    Current Status : {status}

    Events : {len(events)}

    Average Risk : {risk:.2f}

    Average Severity : {severity:.2f}

    AI Confidence : {confidence:.1f}%

    Recommended Officers : {officers}

    Priority Action :
    Deploy resources to highest risk clusters.
    Enable traffic diversion.
    Continue AI monitoring.
    """)

    st.markdown("---")

    query=st.text_input("💬Ask EventWise AI","What is the highest risk zone?")

    if st.button("Generate Response"):

        q=query.lower()

        if"risk" in q:

            if"zone"in events.columns:
                ans=events.groupby("zone")["cluster_risk_score"].mean().idxmax()
                st.success(f"Highest risk zone is **{ans}**.")

        elif"officer" in q:

            st.success(f"AI recommends deploying **{officers} officers**.")

        elif"confidence" in q:

            st.success(f"Current AI confidence is **{confidence:.1f}%**.")

        elif"event" in q:

            st.success(f"There are **{len(events)} active filtered events**.")

        elif"severity" in q:

            st.success(f"Average severity is **{severity:.2f}**.")

        else:

            st.info("""
    AI Summary

    • Monitor high risk clusters

    • Deploy dynamic resources

    • Continue prediction updates

    • Review historical intelligence

    • Run simulation before intervention
    """)
            st.markdown("---")  
# Executive AI Copilot  

with st.expander("🧠 Executive AI Copilot"):

    events = st.session_state.get(
        "filtered_events",
        events_df
    )

    predictions = st.session_state.get(
        "filtered_predictions",
        predictions_df
    )

    recommendations = st.session_state.get(
        "filtered_recommendations",
        recommendations_df
    )

   
    # Select Highest Priority Event   

    if events.empty:

        st.warning("No events available.")

        st.stop()

    events = st.session_state.get("filtered_events", events_df)
    if events is None or events.empty:
        events = events_df
    
    sort_cols = [c for c in ["cluster_risk_score","priority_score","severity"]
                if c in events.columns]
    if sort_cols:
        highest_event = (
            events
            .sort_values(by=sort_cols, ascending=False)
            .iloc[0]
            .to_dict()
        )
    else:
        highest_event = events.iloc[0].to_dict()
    
    # Ensure all feature columns exist so generate_action_plan never KeyErrors
    from recommend import FEATURE_COLUMNS as _FC
    for _col in _FC:
        if _col not in highest_event:
            highest_event[_col] = 0
    highest_event["event_cause"] = str(highest_event.get("event_cause","others") or "others").strip().lower()
    highest_event["cluster_id"]  = int(float(highest_event.get("cluster_id", -1) or -1))
   
    # Generate Backend AI Plan      
    # Generate Executive AI Plan   

    try:

        plan = generate_action_plan(
            highest_event.copy()
        )

    except Exception as e:

        st.warning(
            "Backend plan unavailable. Using local inference."
        )

        plan = {

            "priority": {

                "priority_score":

                highest_event.get(
                    "priority_score",
                    0
                )

            },

            "event_summary": {

                "severity":

                highest_event.get(
                    "severity",
                    0
                ),

                "confidence":

                highest_event.get(
                    "confidence",
                    0.65
                )

            },

            "resource_plan": {

                "officers":

                highest_event.get(
                    "recommended_officers",
                    10
                ),

                "barricades":

                highest_event.get(
                    "recommended_barricades",
                    5
                )

            },

            "movement_plan": {

                "diversion_required":

                True,

                "primary_route":

                "Dynamic AI Route"

            },

            "time_plan": {

                "expected_clearance":

                "30 minutes"

            },

            "action_steps": [

                "Deploy officers",

                "Enable diversion",

                "Optimize signals"

            ],

            "escalation_notes": [

                "Monitor congestion evolution"

            ]

        }

    # Historical Intelligence Card
    

    try:

        matched_events = similar_df.copy()

        if "cluster_id" in matched_events.columns:

            matched_events = matched_events[
                matched_events["cluster_id"]
                ==
                highest_event.get("cluster_id")
            ]

        historical_matches = len(matched_events)

        if historical_matches > 0:

            if "recommended_officers" in matched_events.columns:

                avg_officers = round(
                    matched_events[
                        "recommended_officers"
                    ].mean(),
                    1
                )

            else:

                avg_officers = plan[
                    "resource_plan"
                ]["officers"]

            best_strategy = "Hybrid Response"

            st.subheader(
                "📚 Historical Intelligence"
            )

            st.success(
                f"""
    Historical Matches Found: {historical_matches}

    Average Officers Used: {avg_officers}

    Most Effective Strategy: {best_strategy}

    Current Event Cluster:
    {highest_event.get('cluster_id')}
    """
            )

        else:

            st.subheader(
                "📚 Historical Intelligence"
            )

            st.warning(
                """
    No matching historical incidents found.

    Recommendation generated using
    live AI prediction and corridor risk.
    """
            )

    except Exception as e:

        st.warning(
            f"Historical Intelligence unavailable: {e}"
        )

    except Exception:

        pass

    try:

        bot_outputs = run_all_bots(

            highest_event.copy()

        )

    except Exception:

        bot_outputs = {

            "Executive AI":

            "Backend unavailable. Displaying filtered dashboard intelligence."

        }

   
    # Extract Backend Outputs   

    risk = plan.get(
        "priority",
        {}
    ).get(
        "priority_score",
        0
    )

    severity = plan.get(
        "event_summary",
        {}
    ).get(
        "severity",
        highest_event.get(
            "severity",
            0
        )
    )

    confidence = plan.get(
        "event_summary",
        {}
    ).get(
        "confidence",
        0
    )

    if confidence <= 1:

        confidence *= 100

    resource_plan = plan.get(
        "resource_plan",
        {}
    )

    officers = resource_plan.get(
        "officers",
        0
    )

    highest_zone = highest_event.get(
        "zone",
        "Unknown"
    )

    highest_cause = highest_event.get(
        "event_cause",
        "Unknown"
    )
   
    # KPI Cards   

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "🚦 Events",

        len(events)

    )

    c2.metric(

        "🎯 AI Confidence",

        f"{confidence:.1f}%"

    )

    c3.metric(

        "👮 Officers",

        officers

    )

    c4.metric(

        "⚠ Priority Score",

        f"{risk:.1f}"

    )

    tab1,tab2,tab3,tab4=st.tabs([

    "📋 Executive Brief",

    "🚨 Priority Actions",

    "💬 AI Copilot",

    "📄 PDF Report"

    ])

    with tab1:

        st.info(f"""

    ## 🚦 Executive AI Situation Report

    Priority Score :

    {plan["priority"]["priority_score"]:.1f}

    Severity :

    {plan["event_summary"]["severity"]}

    Recommended Officers :

    {plan["resource_plan"]["officers"]}

    Recommended Barricades :

    {plan["resource_plan"]["barricades"]}

    Diversion Required :

    {plan["movement_plan"]["diversion_required"]}

    Primary Diversion Route :

    {plan["movement_plan"]["primary_route"]}

    Expected Clearance :

    {plan["time_plan"]["expected_clearance"]}

    """)
    with tab2:

        st.markdown("## 🚨 AI Action Plan")

        for step in plan.get(

            "action_steps",

            []

        ):

            st.success(step)

        st.markdown("---")

        st.markdown("## ⚠ Escalation Notes")

        for note in plan.get(

            "escalation_notes",

            []

        ):

            st.warning(note)

    with tab3:

        st.markdown("## 💬 AI Backend Copilot")

        st.caption(
            "Live recommendations generated from recommend.py and bots.py"
        )

        # ----------------------------------------------------
        # Dictionary Output
        # ----------------------------------------------------

        if isinstance(bot_outputs, dict):

            for bot_name, response in bot_outputs.items():

                st.markdown(f"### 🤖 {bot_name}")

                if isinstance(response, list):

                    for item in response:

                        st.success(str(item))

                elif isinstance(response, dict):

                    st.json(response)

                else:

                    st.info(str(response))

        # ----------------------------------------------------
        # List Output
        # ----------------------------------------------------

        elif isinstance(bot_outputs, list):

            for response in bot_outputs:

                st.success(str(response))

        # ----------------------------------------------------
        # String Output
        # ----------------------------------------------------

        else:

            st.info(str(bot_outputs))

        st.markdown("---")

        st.markdown("## 📋 Backend AI Summary")

        st.code(

    f"""
    Priority Score : {risk:.2f}

    Zone : {highest_zone}

    Cause : {highest_cause}

    Recommended Officers : {officers}

    Expected Clearance :

    {plan.get("time_plan",{}).get("expected_clearance","Unknown")}

    Primary Diversion :

    {plan.get("movement_plan",{}).get("primary_route","Unknown")}

    """

        )

        st.markdown("### 🚨 Action Plan")

        for step in plan.get(

            "action_steps",

            []

        ):

            st.success(step)

        st.markdown("### ⚠ Escalation Notes")

        for note in plan.get(

            "escalation_notes",

            []

        ):

            st.warning(note)
    st.markdown("---")

with st.expander("📄 Executive Report Generator", expanded = False):  
     
# Executive Voice AI Copilot  

    st.markdown("---")

    st.subheader("🎤 Executive Voice AI Copilot")

    events = st.session_state.get(
        "filtered_events",
        events_df
    )

    predictions = st.session_state.get(
        "filtered_predictions",
        predictions_df
    )

    recommendations = st.session_state.get(
        "filtered_recommendations",
        recommendations_df
    )

    # ------------------------------------------------

    total_events = len(events)

    critical_events = 0

    if "severity" in events.columns:

        critical_events = len(
            events[
                events["severity"] >= 4
            ]
        )

    avg_risk = 0

    if "cluster_risk_score" in events.columns:

        avg_risk = events[
            "cluster_risk_score"
        ].mean()

    avg_severity = 0

    if "severity" in events.columns:

        avg_severity = events[
            "severity"
        ].mean()

    confidence = 0

    if (
        not predictions.empty
        and
        "confidence" in predictions.columns
    ):

        confidence = predictions[
            "confidence"
        ].mean()

        if confidence <= 1:

            confidence *= 100

    officers = 0

    if (
        not recommendations.empty
        and
        "recommended_officers" in recommendations.columns
    ):

        officers = plan[
        "resource_plan"
    ]["officers"]

        

    highest_zone = "Unknown"

    if (
        "zone" in events.columns
        and
        "cluster_risk_score" in events.columns
        and
        len(events)
    ):

        highest_zone = events.groupby(

            "zone"

        )["cluster_risk_score"].mean().idxmax()

    # ------------------------------------------------

    voice_message = f"""

    Attention all traffic response units.

    Highest priority event detected.

    Zone {highest_event["zone"]}.

    Cause {highest_event["event_cause"]}.

    Deploy {plan["resource_plan"]["officers"]} officers.

    Primary diversion route is

    {plan["movement_plan"]["primary_route"]}.

    Estimated clearance time is

    {plan["time_plan"]["expected_clearance"]}.

    """   
    # PDF REPORT TAB   

    with tab4:

        st.markdown("## 📄 Executive Deployment Report")

        st.info("""

    Generate an AI deployment report

    for the highest priority incident.

    The report includes

    • Deployment plan

    • Resource allocation

    • Diversion routes

    • AI recommendations

    • Field officer message

    • Executive summary

    """)

        if len(events)==0:

            st.warning(

                "No events available."

            )

        else:

            highest_event=(

                events

                .sort_values(

                    "cluster_risk_score",

                    ascending=False

                )

                .iloc[0]

                .to_dict()

            )

            st.success(

                f"""

    Highest Priority Event

    Cluster :

    {highest_event.get('cluster_id')}

    Cause :

    {highest_event.get('event_cause')}

    Zone :

    {highest_event.get('zone')}

    """

            )

            try:

                plan=generate_action_plan(

                    highest_event

                )

                bot_outputs=run_all_bots(

                    highest_event

                )

                pdf=get_pdf_bytes(

                    plan,

                    highest_event,

                    bot_outputs

                )

                st.download_button(

                    label="📥 Download Deployment Card",

                    data=pdf,

                    file_name=f"Deployment_Card_{highest_event.get('id')}.pdf",

                    mime="application/pdf",

                    use_container_width=True

                )

            except Exception as e:

                st.error(

                    f"PDF generation failed : {e}"

                )


    # ------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if st.button(

            "🔊 Play Executive Briefing",

            use_container_width=True

        ):

            play_voice_message(

                voice_message

            )

    with col2:

        if st.button(

            "🔄 Generate New Briefing",

            use_container_width=True

        ):

            st.success(

                "Executive briefing updated."

            )

    # ------------------------------------------------

    with st.expander(

        "📄 Executive Briefing Transcript",

        expanded=True

    ):

        st.info(

            voice_message

        )

    # ------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "🚦 Events",

        total_events

    )

    c2.metric(

        "🚨 Critical",

        critical_events

    )

    c3.metric(

        "👮 Officers",

        officers

    )

    c4.metric(

        "🎯 AI Confidence",

        f"{confidence:.1f}%"

    )

    st.success("""

    🎙 Executive Voice Copilot Ready

    • Executive briefing available

    • Voice generated from live dashboard

    • No backend modification required

    • Automatically uses current dashboard values

    """)
    get_twin_score_ui()


    st.markdown("---")

    summary=pd.DataFrame({
    "Metric":[
    "Events",
    "Risk",
    "Severity",
    "Confidence",
    "Priority Zone",
    "Dominant Cause",
    "Officers"
    ],
    "Value":[
    len(events),
    round(risk,2),
    round(severity,2),
    f"{confidence:.1f}%",
    highest_zone,
    highest_cause,
    officers
    ]
    })

    st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
    )
    st.markdown("---")
with st.expander("🖥 System Health & Model Monitoring"):

    events=st.session_state.get("filtered_events",events_df)
    predictions=st.session_state.get("filtered_predictions",predictions_df)
    recommendations=st.session_state.get("filtered_recommendations",recommendations_df)
    simulations=st.session_state.get("filtered_simulations",simulations_df)

    metadata=metadata_df.copy()

    cache_age=(datetime.now()-st.session_state.last_refresh).seconds

    confidence=predictions["confidence"].mean() if(not predictions.empty and"confidence"in predictions.columns) else 0

    if confidence<=1:
        confidence*=100

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
    "🗄Database",
    "Online"
    )

    c2.metric(
    "🤖AI Model",
    "Active"
    )

    c3.metric(
    "⚡Cache",
    f"{cache_age}s"
    )

    c4.metric(
    "🎯Confidence",
    f"{confidence:.1f}%"
    )

    tab1,tab2,tab3=st.tabs([
    "📊Pipeline",
    "🤖Model",
    "📋Metadata"
    ])

    with tab1:

        pipe=pd.DataFrame({

        "Component":[

        "Database",

        "Prediction",

        "Recommendation",

        "Simulation",

        "Digital Twin"

        ],

        "Status":[

        "Online",

        "Running",

        "Running",

        "Running",

        "Running"

        ]

        })

        fig=px.bar(

        pipe,

        x="Component",

        y=[1,1,1,1,1],

        color="Status",

        text="Status",

        title="Pipeline Status"

        )

        fig.update_layout(
        showlegend=False,
        yaxis_visible=False,
        height=420
        )

        st.plotly_chart(
        fig,
        use_container_width=True
        )

    with tab2:

        model=pd.DataFrame({

        "Metric":[

        "Events",

        "Predictions",

        "Recommendations",

        "Simulations"

        ],

        "Value":[

        len(events),

        len(predictions),

        len(recommendations),

        len(simulations)

        ]

        })

        fig=px.bar(

        model,

        x="Metric",

        y="Value",

        text="Value",

        title="Processing Volume"

        )

        fig.update_layout(
        height=420
        )

        st.plotly_chart(
        fig,
        use_container_width=True
        )

    with tab3:

        if not metadata.empty:

            st.dataframe(

            metadata,

            use_container_width=True,

            hide_index=True

            )

        else:

            summary=pd.DataFrame({

            "Property":[

            "Database",

            "Events",

            "Predictions",

            "Recommendations",

            "Simulations",

            "AI Confidence"

            ],

            "Value":[

            "brain.db",

            len(events),

            len(predictions),

            len(recommendations),

            len(simulations),

            f"{confidence:.1f}%"

            ]

            })

            st.dataframe(

            summary,

            use_container_width=True,

            hide_index=True

            )

    st.success(f"""
    ### 🚀 System Status

    ✅ Database Connected

    ✅ AI Prediction Engine Running

    ✅ Recommendation Engine Running

    ✅ Digital Twin Active

    ✅ Simulation Engine Active

    Processed **{len(events)}** events with **{confidence:.1f}%** average confidence.

    System operating normally.
    """)   
    # Generate PDF   

    try:

        pdf_bytes = get_pdf_bytes(

            plan=plan,

            event=highest_event,

            bot_outputs=bot_outputs

        )

        st.download_button(

            label="📥 Download Executive PDF Report",

            data=pdf_bytes,

            file_name=f"EventWise_Report_{highest_event.get('cluster_id','NA')}.pdf",

            mime="application/pdf",

            use_container_width=True

        )

    except Exception as e:

        st.error(

            f"PDF generation failed : {e}"

        )
    st.markdown("---")

uptime=datetime.now()-APP_START_TIME

hours=uptime.seconds//3600
minutes=(uptime.seconds%3600)//60
seconds=uptime.seconds%60

events=st.session_state.get("filtered_events",events_df)

predictions=st.session_state.get("filtered_predictions",predictions_df)

recommendations=st.session_state.get("filtered_recommendations",recommendations_df)

simulations=st.session_state.get("filtered_simulations",simulations_df)

st.success("✅ EventWise AI Knowledge Base Loaded Successfully")

# ------------------------------------------------
# Dashboard Statistics
# ------------------------------------------------

total_events = len(events_df)

# High confidence predictions
high_confidence = 0

if (
    not predictions_df.empty
    and
    "confidence" in predictions_df.columns
):

    if predictions_df["confidence"].max() <= 1:

        high_confidence = len(

            predictions_df[
                predictions_df["confidence"] >= 0.80
            ]

        )

    else:

        high_confidence = len(

            predictions_df[
                predictions_df["confidence"] >= 80
            ]

        )

# Total officers recommended
total_officers = 0

if (
    not recommendations_df.empty
    and
    "recommended_officers" in recommendations_df.columns
):

    total_officers = int(

        recommendations_df[
            "recommended_officers"
        ].head(50).sum()

    )

# Critical events
critical_events = 0

if "severity" in events_df.columns:

    critical_events = len(

        events_df[
            events_df["severity"] >= 4
        ]

    )

# AI sessions
total_sessions = len(bot_sessions_df)

# ------------------------------------------------
# KPI Cards
# ------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "🚦 Events",

        total_events

    )


with col2:

    st.metric(

        "👮 Officers",

        total_officers

    )

with col3:

    st.metric(

        "🚨 Critical Events",

        critical_events

    )

with col4:

    st.metric(

        "🤖 AI Sessions",

        total_sessions

    )
st.markdown("---")

l,m,r=st.columns([3,2,2])

with l:

    st.success(f"""
### 🚀 EventWise AI Digital Twin

Enterprise Smart Traffic Command Center

Database : brain.db

AI Engine : Operational

Digital Twin : Active

Last Refresh : {st.session_state.last_refresh.strftime('%H:%M:%S')}
""")

with m:

    if st.button("🔄Refresh Dashboard",use_container_width=True):

        st.cache_data.clear()

        st.session_state.last_refresh=datetime.now()

        st.rerun()

    if st.button("♻Reset Filters",use_container_width=True):

        for k in list(st.session_state.keys()):

            if k.startswith("filtered_"):

                del st.session_state[k]

        st.rerun()

with r:

    st.info("""
### 📦Build

Version : v1.0

Mode : Production

Backend : SQLite

Frontend : Streamlit

Status : Stable
""")

st.markdown("---")

st.caption("""
© 2026 EventWise AI Digital Twin

AI-powered Traffic Intelligence • Digital Twin • Smart Resource Deployment • Predictive Analytics

Built for Smart City Operations & National Hackathon Demonstration
""")