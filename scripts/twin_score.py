#!/usr/bin/env python
# coding: utf-8
"""
twin_score.py — TrafficTwin City Health Score Engine
Reads  : brain.db (events, predictions, recommendations, simulation_logs)
Writes : brain.db twin_score_log table
Used by: app.py sidebar KPI gauge + Twin Score tab
Place  : scripts/twin_score.py
Run    : python scripts/twin_score.py
"""

import sys
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime
import corridor_watch

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from pathlib import Path

# ── path so app.py can do: from twin_score import get_twin_score_ui ──
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# ─────────────────────────────────────────────────────────────
# Config — identical paths to app.py
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "brain.db"

# Score weights (must sum to 1.0)
WEIGHTS = {
    "severity"   : 0.25,   # average event severity (lower = healthier)
    "critical"   : 0.20,   # fraction of critical events (lower = healthier)
    "confidence" : 0.15,   # average prediction confidence (higher = healthier)
    "simulation" : 0.15,   # average simulation improvement (higher = healthier)
    "cluster"    : 0.10,   # average cluster risk (lower = healthier)
    "corridor" : 0.15
}


SCORE_BANDS = [
    (85, 100, "EXCELLENT",  "#2ECC71", "🟢"),
    (70,  85, "GOOD",       "#27AE60", "🟢"),
    (55,  70, "MODERATE",   "#F1C40F", "🟡"),
    (40,  55, "CONCERNING", "#E67E22", "🟠"),
    (20,  40, "CRITICAL",   "#E74C3C", "🔴"),
    ( 0,  20, "EMERGENCY",  "#8B0000", "⛔"),
]


# ─────────────────────────────────────────────────────────────
# CELL 1 — Data loader
# ─────────────────────────────────────────────────────────────
def _load(conn, table):
    try:
        return pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def load_twin_data():
    conn = sqlite3.connect(DB_PATH)
    data = {k: _load(conn, t) for k, t in [
        ("events",      "events"),
        ("predictions", "predictions"),
        ("recommendations", "recommendations"),
        ("simulations", "simulation_logs"),
    ]}
    conn.close()
    return data


# ─────────────────────────────────────────────────────────────
# CELL 2 — Score computation
# ─────────────────────────────────────────────────────────────
def _norm(value, lo, hi, invert=False):
    """Normalise value to 0-100; invert=True means lower raw → higher score."""
    if hi == lo:
        return 50.0
    n = (value - lo) / (hi - lo) * 100
    n = float(np.clip(n, 0, 100))
    return round(100 - n if invert else n, 2)


def compute_twin_score(data: dict) -> dict:
    events      = data["events"]
    predictions = data["predictions"]
    simulations = data["simulations"]

    components = {}
    corridor_health = get_corridor_health()

    # ── Severity component (1-5 scale; 5=worst → invert)
    if not events.empty and "severity" in events.columns:
        avg_sev = float(events["severity"].mean())
        components["severity"] = _norm(avg_sev, 1, 5, invert=True)
    else:
        components["severity"] = 50.0

    # ── Critical fraction (fraction of sev>=4; higher fraction = worse)
    if not events.empty and "severity" in events.columns:
        crit_frac = float((events["severity"] >= 4).mean())
        components["critical"] = _norm(crit_frac, 0, 1, invert=True)
    else:
        components["critical"] = 50.0

    # ── Confidence component (0-1 or 0-100; higher = better)
    if not predictions.empty and "confidence" in predictions.columns:
        conf = predictions["confidence"].copy()
        if conf.max() <= 1.0:
            conf = conf * 100
        avg_conf = float(conf.mean())
        components["confidence"] = _norm(avg_conf, 0, 100)
    else:
        components["confidence"] = 50.0

    # ── Simulation improvement (0-60%; higher = better)
    if not simulations.empty and "improvement" in simulations.columns:
        avg_imp = float(simulations["improvement"].mean())
        components["simulation"] = _norm(avg_imp, 0, 60)
    elif not simulations.empty and "expected_reduction" in simulations.columns:

        avg_imp = simulations["expected_reduction"].fillna(0)

        avg_imp = float(avg_imp.mean())

        if np.isnan(avg_imp):

              avg_imp = 0

        components["simulation"] = _norm(avg_imp, 0, 60)
    else:
        components["simulation"] = 50.0

    # ── Cluster risk (0-5 scale; lower = better → invert)
    if not events.empty and "cluster_risk_score" in events.columns:
        avg_cr = float(events["cluster_risk_score"].mean())
        components["cluster"] = _norm(avg_cr, 0, 5, invert=True)
        components["corridor"] = corridor_health
    else:
        components["cluster"] = 50.0

    # ── Weighted composite
    for k in components:

        if pd.isna(components[k]):

            components[k] = 50.0

    score = sum(

    components[k] * WEIGHTS[k]

    for k in WEIGHTS

)
    score = round(float(np.clip(score, 0, 100)), 2)
    

    # ── Band classification
    band_label = "UNKNOWN"
    band_color = "#888780"
    band_emoji = "⚪"
    for lo, hi, label, color, emoji in SCORE_BANDS:
        if lo <= score <= hi:
            band_label = label
            band_color = color
            band_emoji = emoji
            break

    # ── Ancillary stats for UI
    total_events    = len(events)
    critical_events = int((events["severity"] >= 4).sum()) \
        if not events.empty and "severity" in events.columns else 0
    avg_confidence  = round(components["confidence"], 1)
    avg_improvement = round(components["simulation"], 1)
    n_clusters      = int(events["cluster_id"].nunique()) \
        if not events.empty and "cluster_id" in events.columns else 0
    city_resilience = round(
    (
        0.4 * score
        +
        0.4 * corridor_health
        +
        0.2 * avg_confidence
    ),
    1
)
    return {
        "score"          : score,
        "band_label"     : band_label,
        "band_color"     : band_color,
        "band_emoji"     : band_emoji,
        "components"     : components,
        "total_events"   : total_events,
        "critical_events": critical_events,
        "avg_confidence" : avg_confidence,
        "avg_improvement": avg_improvement,
        "n_clusters"     : n_clusters,
        "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "corridor_health" : corridor_health,
        "city_resilience": city_resilience,
    }
def get_corridor_health():

    try:

        corridor_df = corridor_watch.run_corridor_watch()

        if corridor_df.empty:
            return 70.0

        avg_watch = corridor_df["watch_score"].mean()

        corridor_health = max(
            0,
            min(
                100,
                100 - avg_watch
            )
        )

        return round(corridor_health, 2)

    except Exception:

        return 70.0


# ─────────────────────────────────────────────────────────────
# CELL 3 — Persist to twin_score_log
# ─────────────────────────────────────────────────────────────
def save_twin_score(result: dict):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS twin_score_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            score            REAL,
            band_label       TEXT,
            total_events     INTEGER,
            critical_events  INTEGER,
            severity_score   REAL,
            critical_score   REAL,
            confidence_score REAL,
            simulation_score REAL,
            cluster_score    REAL,
            avg_confidence   REAL,
            avg_improvement  REAL,
            n_clusters       INTEGER,
            city_resilience REAL,
            corridor_health REAL,
            logged_at        TEXT
        )
    """)
    c = result["components"]
    cur.execute("""
        INSERT INTO twin_score_log
(
score,
band_label,
total_events,
critical_events,
severity_score,
critical_score,
confidence_score,
simulation_score,
cluster_score,
avg_confidence,
avg_improvement,
n_clusters,
city_resilience,
corridor_health,
logged_at
)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(
result["score"],
result["band_label"],
result["total_events"],
result["critical_events"],
c["severity"],
c["critical"],
c["confidence"],
c["simulation"],
c["cluster"],
result["avg_confidence"],
result["avg_improvement"],
result["n_clusters"],
result["city_resilience"],
result["corridor_health"],
result["timestamp"]
) )
    conn.commit()
    conn.close()


def load_score_history(limit: int = 30) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            f"SELECT * FROM twin_score_log ORDER BY id DESC LIMIT {limit}",
            conn
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df.sort_values("id") if not df.empty else df


# ─────────────────────────────────────────────────────────────
# CELL 4 — Streamlit UI  (called from app.py)
# Usage in app.py:
#   from twin_score import get_twin_score_ui
#   get_twin_score_ui()
# ─────────────────────────────────────────────────────────────
def get_twin_score_ui():
    """
    Full Streamlit UI for the Twin Score section.
    Call this from any tab or sidebar section inside app.py.
    Requires: streamlit, plotly already imported in app.py.
    """
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown("## 🛰 Digital Twin Health Score")
    st.caption("Composite 0–100 city traffic health index — updates on each run")

    # ── Load and compute ──────────────────────────────────────
    with st.spinner("Computing Twin Score..."):
        data   = load_twin_data()
        result = compute_twin_score(data)
        save_twin_score(result)

    score  = result["score"]
    color  = result["band_color"]
    label  = result["band_label"]
    emoji  = result["band_emoji"]
    comps  = result["components"]

    # ── Gauge ─────────────────────────────────────────────────
    col_gauge, col_kpi = st.columns([1, 1])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = score,
            delta = {"reference": 70, "increasing": {"color": "#2ECC71"},
                     "decreasing": {"color": "#E74C3C"}},
            title = {"text": f"{emoji} City Health Score<br><span style='font-size:14px'>{label}</span>"},
            gauge = {
                "axis"     : {"range": [0, 100], "tickwidth": 1},
                "bar"      : {"color": color, "thickness": 0.25},
                "bgcolor"  : "white",
                "borderwidth": 2,
                "steps"    : [
                    {"range": [0,  20], "color": "#FDECEA"},
                    {"range": [20, 40], "color": "#FEE8D6"},
                    {"range": [40, 55], "color": "#FEF9E7"},
                    {"range": [55, 70], "color": "#EAFAF1"},
                    {"range": [70, 85], "color": "#D5F5E3"},
                    {"range": [85,100], "color": "#C8F7C5"},
                ],
                "threshold": {
                    "line" : {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": score
                }
            },
            number = {"suffix": "/100", "font": {"size": 36}}
        ))
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col_kpi:
        st.markdown(f"""
        <div style='background:{color}18;border-left:4px solid {color};
        border-radius:8px;padding:14px 18px;margin-bottom:10px'>
        <h3 style='margin:0;color:{color}'>{emoji} {label}</h3>
        <p style='margin:4px 0;font-size:13px'>Score: <b>{score}/100</b> &nbsp;|&nbsp;
        Updated: {result['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("🚦 Total Events",    f"{result['total_events']:,}")
        m2.metric("🚨 Critical Events", f"{result['critical_events']:,}")
        m3, m4 = st.columns(2)
        m3.metric("🎯 Avg Confidence",  f"{result['avg_confidence']:.1f}%")
        m4.metric("📉 Sim Improvement", f"{result['avg_improvement']:.1f}%")
        m5, m6 = st.columns(2)

        m5.metric(
    "📍 Clusters",
    f"{result['n_clusters']:,}"
)

        m6.metric(
    "🛰 Corridor Health",
    f"{result['corridor_health']:.1f}/100"
)
        city_resilience = round(

    (
        0.4 * score
        +
        0.4 * result["corridor_health"]
        +
        0.2 * result["avg_confidence"]
    ),

    1
)

    st.metric(
    "🏙 City Resilience",
    f"{city_resilience}/100"
)

    # ── Component breakdown ───────────────────────────────────
    st.markdown("### 🔬 Score Component Breakdown")
    comp_df = pd.DataFrame([
        {"Component": k.capitalize(), "Score (0-100)": v,
         "Weight": f"{int(WEIGHTS[k]*100)}%"}
        for k, v in comps.items()
    ])
    col_bar, col_tbl = st.columns([2, 1])
    with col_bar:
        fig2 = px.bar(
            comp_df, x="Component", y="Score (0-100)",
            color="Score (0-100)",
            color_continuous_scale=["#E74C3C","#E67E22","#F1C40F","#2ECC71"],
            range_color=[0,100],
            text="Score (0-100)",
            title="Component Scores"
        )
        fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig2.update_layout(height=280, showlegend=False,
                           margin=dict(t=40,b=20,l=10,r=10),
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    with col_tbl:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Score history trend ───────────────────────────────────
    history = load_score_history(limit=30)
    if not history.empty and len(history) >= 2:
        st.markdown("### 📈 Score History Trend")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=history["logged_at"], y=history["score"],
            mode="lines+markers",
            line=dict(color="#378ADD", width=2),
            marker=dict(size=6),
            name="City Health Score",
            fill="tozeroy", fillcolor="rgba(55,138,221,0.08)"
        ))
        fig3.add_trace(
    go.Scatter(
        x=history["logged_at"],
        y=history["city_resilience"],
        mode="lines+markers",
        name="City Resilience"
    )
)
        fig3.add_hline(y=70, line_dash="dash",
                       line_color="#2ECC71", annotation_text="Good threshold")
        fig3.add_hline(y=40, line_dash="dash",
                       line_color="#E74C3C", annotation_text="Critical threshold")
        fig3.update_layout(
            height=220, xaxis_title="Time", yaxis_title="Score",
            yaxis=dict(range=[0,100]),
            margin=dict(t=20,b=30,l=40,r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Run the simulator multiple times to see the score trend.")

    # ── Recommendations ───────────────────────────────────────
    st.markdown("### 💡 AI Recommendations")
    recs = []
    if comps["severity"] < 50:
        recs.append("🔴 High average severity — increase patrol frequency in critical clusters.")
    if comps["critical"] < 50:
        recs.append("🔴 High fraction of critical events — pre-deploy resources to top 5 clusters.")
    if comps["confidence"] < 60:
        recs.append("🟡 Low AI confidence — collect more post-event feedback to retrain the model.")
    if comps["simulation"] < 50:
        recs.append("🟡 Low simulation gain — consider enabling diversion on high-severity corridors.")
    if comps["cluster"] < 50:
        recs.append("🟠 High cluster risk scores — prioritise Cluster 146 and Cluster 154 (highest density).")
    if score >= 70:
        recs.append("🟢 System health is good — maintain current deployment strategy.")
    for rec in recs:
        st.markdown(f"- {rec}")

    return result


# ─────────────────────────────────────────────────────────────
# CELL 5 — Sidebar mini-widget (for app.py sidebar)
# Usage:
#   from twin_score import get_sidebar_score_widget
#   get_sidebar_score_widget()
# ─────────────────────────────────────────────────────────────
def get_sidebar_score_widget():
    import streamlit as st
    data   = load_twin_data()
    result = compute_twin_score(data)
    score  = result["score"]
    color  = result["band_color"]
    label  = result["band_label"]
    emoji  = result["band_emoji"]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    <div style='background:{color}22;border-left:3px solid {color};
    border-radius:6px;padding:8px 12px;margin:4px 0'>
    <b style='font-size:13px'>{emoji} City Health</b><br>
    <span style='font-size:22px;font-weight:bold;color:{color}'>{score}</span>
    <span style='font-size:11px;color:{color}'>/100 — {label}</span>
    </div>
    """, unsafe_allow_html=True)
    return result


# ─────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  TrafficTwin — Twin Score Engine")
    print("=" * 50)
    data   = load_twin_data()
    result = compute_twin_score(data)
    save_twin_score(result)

    print(f"\n  City Health Score : {result['score']}/100")
    print(f"  Band              : {result['band_emoji']} {result['band_label']}")
    print(f"  Timestamp         : {result['timestamp']}")
    print(f"\n  Component Breakdown:")
    for k, v in result["components"].items():
        print(f"    {k.capitalize():<15}: {v:.1f}/100  (weight {int(WEIGHTS[k]*100)}%)")
    print(f"\n  Total Events      : {result['total_events']}")
    print(f"  Critical Events   : {result['critical_events']}")
    print(f"  Avg Confidence    : {result['avg_confidence']:.1f}%")
    print(f"  Avg Sim Gain      : {result['avg_improvement']:.1f}%")
    print(f"  Clusters          : {result['n_clusters']}")
    print("\n  ✓ Saved to twin_score_log in brain.db")
    print("=" * 50)
