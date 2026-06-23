import sys
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

      
# CELL 1 — Configuration
      
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "database" / "brain.db"

# Alert thresholds
THRESHOLDS = {
    "recent_activity_pct"   : 0.70,   # cluster with top 30% recent activity
    "risk_score_pct"        : 0.80,   # cluster with top 20% risk score
    "hotspot_density_pct"   : 0.75,   # cluster with top 25% density
    "avg_severity_critical" : 3.5,    # avg severity >= this = watch
    "road_closure_pct"      : 0.60,   # high road-closure ratio
    "confidence_low"        : 0.65,   # predictions < this = uncertain
}

WATCH_LEVELS = {
    "CRITICAL" : {"color": "#E74C3C", "emoji": "🔴", "min_score": 60},
    "HIGH"     : {"color": "#E67E22", "emoji": "🟠", "min_score": 45},
    "MODERATE" : {"color": "#F1C40F", "emoji": "🟡", "min_score": 25},
    "LOW"      : {"color": "#2ECC71", "emoji": "🟢", "min_score":  0},
}

CAUSE_PRIORITY = {
    "accident"         : 5,
    "vehicle_breakdown": 3,
    "water_logging"    : 4,
    "tree_fall"        : 2,
    "public_event"     : 3,
    "others"           : 1,
    "unknown"          : 1,
}


      
# CELL 2 — Data loader
      
def _load(conn, table):
    try:
        return pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def load_corridor_data() -> dict:
    conn = sqlite3.connect(DB_PATH)
    data = {k: _load(conn, t) for k, t in [
        ("events",      "events"),
        ("hotspots",    "hotspot_statistics"),
        ("predictions", "predictions"),
        ("recommendations", "recommendations"),
        ("simulations", "simulation_logs"),
    ]}
    conn.close()
    return data


      
# CELL 3 — Corridor scoring engine
      
def _pct_rank(series: pd.Series, value: float) -> float:
    """Return percentile rank of value in series (0-1)."""
    if series.empty or series.std() == 0:
        return 0.5
    return float((series <= value).mean())


def _cause_weight(dominant_cause: str) -> float:
    cause = str(dominant_cause).lower().strip()
    return CAUSE_PRIORITY.get(cause, 1) / 5.0


def score_corridor(cluster_id, events_df, hotspots_df,
                   predictions_df, recommendations_df, simulations_df) -> dict:
    """Compute a 0-100 watch score for one cluster."""
    mask   = events_df["cluster_id"] == cluster_id
    c_evts = events_df[mask]
    if c_evts.empty:
        return None

    hs = hotspots_df[hotspots_df["cluster_id"] == cluster_id]

    # ── Component 1: recent activity (normalised across all clusters)
    recent    = float(c_evts["recent_cluster_activity"].mean() or 0)
    rec_rank  = _pct_rank(events_df["recent_cluster_activity"], recent)
    c_recent  = rec_rank * 25                              # max 25

    # ── Component 2: average severity
    avg_sev   = float(c_evts["severity"].mean() or 1)
    c_sev     = ((avg_sev - 1) / 4.0) * 25                # max 25

    # ── Component 3: cluster risk score
    avg_risk  = float(c_evts["cluster_risk_score"].mean() or 0)
    risk_rank = _pct_rank(events_df["cluster_risk_score"], avg_risk)
    c_risk    = risk_rank * 20                             # max 20

    # ── Component 4: road closure ratio
    closure   = float(c_evts["road_closure"].mean() or 0)
    c_closure = closure * 15                               # max 15

    # ── Component 5: dominant cause priority
    if not hs.empty and "dominant_cause" in hs.columns:
        dom_cause = str(hs["dominant_cause"].iloc[0])
    else:
        dom_cause = c_evts["event_cause"].mode().iloc[0] \
                    if not c_evts.empty else "others"
    c_cause   = _cause_weight(dom_cause) * 10             # max 10

    # ── Component 6: prediction uncertainty bonus
    c_ids = c_evts["id"].tolist()
    conf_penalty = 0.0
    if not predictions_df.empty and "confidence" in predictions_df.columns:
        c_preds = predictions_df[predictions_df["event_id"].isin(c_ids)]
        if not c_preds.empty:
            avg_conf = float(c_preds["confidence"].mean() or 0.75)
            if avg_conf <= 1.0:
                avg_conf *= 100
            # low confidence → higher watch score
            conf_penalty = max(0, (75 - avg_conf) / 75 * 5)  # max 5
    c_uncertainty = conf_penalty                          # max 5

    watch_score = round(min(
        c_recent + c_sev + c_risk + c_closure + c_cause + c_uncertainty,
        100.0
    ), 2)

    # ── Watch level
    watch_level = "LOW"
    for level, cfg in WATCH_LEVELS.items():
        if watch_score >= cfg["min_score"]:
            watch_level = level
            break

    # ── Recommended action
    if watch_level == "CRITICAL":
        action = "Immediate deployment — pre-position 8+ officers now."
    elif watch_level == "HIGH":
        action = "Alert zone — pre-position 4 officers + barricades on standby."
    elif watch_level == "MODERATE":
        action = "Monitor closely — check again in 30 minutes."
    else:
        action = "Routine monitoring. No immediate action needed."

    # ── Simulation gain for this cluster
    sim_gain = 0.0
    if not simulations_df.empty and "event_id" in simulations_df.columns:
        c_sims = simulations_df[simulations_df["event_id"].isin(c_ids)]
        if not c_sims.empty:
            col = "improvement" if "improvement" in c_sims.columns \
                  else "expected_reduction"
            sim_gain = round(float(c_sims[col].mean() or 0), 2)

    # ── Officers recommended
    officers = 0
    if not recommendations_df.empty:
        c_recs = recommendations_df[recommendations_df["event_id"].isin(c_ids)]
        if not c_recs.empty and "recommended_officers" in c_recs.columns:
            officers = int(c_recs["recommended_officers"].mean() or 0)

    lat_center = float(c_evts["cluster_lat_center"].iloc[0]
                       if "cluster_lat_center" in c_evts.columns
                       else c_evts["latitude"].mean())
    lon_center = float(c_evts["cluster_lon_center"].iloc[0]
                       if "cluster_lon_center" in c_evts.columns
                       else c_evts["longitude"].mean())

    return {
        "cluster_id"      : int(cluster_id),
        "watch_score"     : watch_score,
        "watch_level"     : watch_level,
        "avg_severity"    : round(avg_sev, 2),
        "avg_risk"        : round(avg_risk, 2),
        "event_count"     : int(len(c_evts)),
        "recent_activity" : round(recent, 1),
        "road_closure_pct": round(closure * 100, 1),
        "dominant_cause"  : dom_cause,
        "recommended_officers": officers,
        "sim_gain_pct"    : sim_gain,
        "recommended_action": action,
        "lat_center"      : round(lat_center, 6),
        "lon_center"      : round(lon_center, 6),
        "components": {
            "recent_activity"  : round(c_recent, 2),
            "severity"         : round(c_sev, 2),
            "risk"             : round(c_risk, 2),
            "road_closure"     : round(c_closure, 2),
            "cause_priority"   : round(c_cause, 2),
            "uncertainty"      : round(c_uncertainty, 2),
        },
        "scored_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


      
# CELL 4 — Run all corridors
      
def run_corridor_watch() -> pd.DataFrame:
    data   = load_corridor_data()
    events = data["events"]
    if events.empty or "cluster_id" not in events.columns:
        return pd.DataFrame()

    cluster_ids = sorted(events["cluster_id"].dropna().unique().tolist())
    results     = []

    for cid in cluster_ids:
        if cid == -1:
            continue
        r = score_corridor(
            cid, events,
            data["hotspots"], data["predictions"],
            data["recommendations"], data["simulations"]
        )
        if r:
            results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        return df

# Corridor Score Diagnostics


    print("\n" + "=" * 70)
    print("CORRIDOR WATCH SCORE ANALYSIS")
    print("=" * 70)

    print("\nTop 20 Corridor Scores")

    top_scores = (
        df[
        [
            "cluster_id",
            "watch_score",
            "watch_level"
        ]
    ]
    .sort_values(
        "watch_score",
        ascending=False
    )
    .head(20)
)

    print(top_scores)

    print("\n")

    print("Watch Level Distribution")

    print(
    df["watch_level"]
    .value_counts()
)

    print("\n")

    print("Score Statistics")

    print(
    df["watch_score"]
    .describe()
)

    print("\n")

    print("Maximum Score")

    print(
    df["watch_score"]
    .max()
)

    print("\n")

    print("Minimum Score")

    print(
    df["watch_score"]
    .min()
)

    print("\n")

    print("Average Score")

    print(
    df["watch_score"]
    .mean()
)

    print("=" * 70)

    df = df.sort_values("watch_score", ascending=False).reset_index(drop=True)
    return df


      
# CELL 5 — Persist corridor_alerts to brain.db
      
def save_corridor_alerts(df: pd.DataFrame):
    if df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS corridor_alerts (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id            INTEGER,
            watch_score           REAL,
            watch_level           TEXT,
            avg_severity          REAL,
            avg_risk              REAL,
            event_count           INTEGER,
            recent_activity       REAL,
            road_closure_pct      REAL,
            dominant_cause        TEXT,
            recommended_officers  INTEGER,
            sim_gain_pct          REAL,
            recommended_action    TEXT,
            lat_center            REAL,
            lon_center            REAL,
            scored_at             TEXT
        )
    """)
    cur.execute("DELETE FROM corridor_alerts")
    save_cols = [
        "cluster_id","watch_score","watch_level","avg_severity",
        "avg_risk","event_count","recent_activity","road_closure_pct",
        "dominant_cause","recommended_officers","sim_gain_pct",
        "recommended_action","lat_center","lon_center","scored_at"
    ]
    rows = [tuple(row[c] for c in save_cols) for _, row in df.iterrows()]
    cur.executemany(
        f"INSERT INTO corridor_alerts ({','.join(save_cols)}) "
        f"VALUES ({','.join(['?']*len(save_cols))})",
        rows
    )
    conn.commit()
    conn.close()


def load_corridor_alerts(limit: int = 50) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            f"SELECT * FROM corridor_alerts "
            f"ORDER BY watch_score DESC LIMIT {limit}",
            conn
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


      
# CELL 6 — Streamlit full-page UI
# Usage in app.py:
#   from corridor_watch import get_corridor_watch_ui
#   get_corridor_watch_ui()
      
def get_corridor_watch_ui():
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import folium
    from streamlit_folium import st_folium

    with st.expander("## 🛰 Proactive Corridor Watch", expanded= False):
        st.caption("Surveillance engine — fires alerts before congestion becomes critical")

        col_run, col_info = st.columns([1, 3])
        with col_run:
            run_now = st.button("🔄 Run Watch Scan", type="primary",
                                use_container_width=True)
        with col_info:
            st.info("Scores every cluster on 6 signals: "
                    "recent activity · severity · risk · road closure · "
                    "cause priority · prediction uncertainty")

        # ── Load or compute                      ────
        if run_now or "corridor_df" not in st.session_state:
            with st.spinner("Scanning all corridors..."):
                corridor_df = run_corridor_watch()
                save_corridor_alerts(corridor_df)
                st.session_state["corridor_df"] = corridor_df
        else:
            corridor_df = st.session_state.get("corridor_df", pd.DataFrame())
            if corridor_df.empty:
                corridor_df = load_corridor_alerts(limit=244)

        if corridor_df.empty:
            st.warning("No corridor data available. Run the scan first.")
            return

        # ── Summary KPI row                      ────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📍 Corridors Scanned",
                f"{len(corridor_df):,}")
        critical_n = int((corridor_df["watch_level"] == "CRITICAL").sum())
        high_n     = int((corridor_df["watch_level"] == "HIGH").sum())
        c2.metric("🔴 Critical",  f"{critical_n}",
                delta=f"+{critical_n}" if critical_n > 0 else None,
                delta_color="inverse")
        c3.metric("🟠 High",      f"{high_n}")
        c4.metric("📊 Avg Score",
                f"{corridor_df['watch_score'].mean():.1f}/100")
        c5.metric("👮 Total Officers Needed",
                f"{int(corridor_df['recommended_officers'].sum()):,}")

        st.markdown("---")

    # ── Tab layout                        
    tab_list, tab_map, tab_detail, tab_chart = st.tabs([
        "📋 Watch List", "🗺 Corridor Map",
        "🔬 Cluster Detail", "📊 Score Analysis"
    ])

    # ── TAB 1: Watch List                    
    with tab_list:
        st.markdown("### 🚨 Active Corridor Alerts")
        level_filter = st.selectbox(
            "Filter by watch level",
            ["All", "CRITICAL", "HIGH", "MODERATE", "LOW"],
            key="cw_level_filter"
        )
        display_df = corridor_df.copy()
        if level_filter != "All":
            display_df = display_df[display_df["watch_level"] == level_filter]

        display_df["emoji"] = display_df["watch_level"].map(
            {k: v["emoji"] for k, v in WATCH_LEVELS.items()}
        )
        show_cols = [
            "emoji", "cluster_id", "watch_level", "watch_score",
            "avg_severity", "avg_risk", "event_count",
            "dominant_cause", "recommended_officers",
            "sim_gain_pct", "recommended_action"
        ]
        show_cols = [c for c in show_cols if c in display_df.columns]

        def _highlight(row):
            color_map = {
                "CRITICAL": "background-color:#FDECEA",
                "HIGH"    : "background-color:#FEF3E2",
                "MODERATE": "background-color:#FEFDE7",
                "LOW"     : "background-color:#EAF7EA",
            }
            return [color_map.get(row.get("watch_level",""), "")] * len(row)

        st.dataframe(
            display_df[show_cols].style.apply(_highlight, axis=1),
            use_container_width=True, hide_index=True
        )
        st.caption(f"Showing {len(display_df)} of {len(corridor_df)} corridors")

    # ── TAB 2: Corridor Map                      
    with tab_map:
        st.markdown("### 🗺 Corridor Risk Map")
        if "lat_center" not in corridor_df.columns:
            st.warning("Coordinate data missing from corridor alerts.")
        else:
            valid = corridor_df.dropna(subset=["lat_center","lon_center"])
            m = folium.Map(
                location=[valid["lat_center"].mean(),
                          valid["lon_center"].mean()],
                zoom_start=12,
                tiles="cartodbpositron"
            )
            color_map = {
                "CRITICAL": "red", "HIGH": "orange",
                "MODERATE": "blue", "LOW": "green"
            }
            for _, row in valid.iterrows():
                score  = float(row["watch_score"])
                level  = str(row["watch_level"])
                radius = max(300, min(1400, score * 14))
                folium.Circle(
                    location=[row["lat_center"], row["lon_center"]],
                    radius=radius,
                    color=color_map.get(level, "gray"),
                    fill=True, fill_opacity=0.18, weight=2.5,
                    popup=folium.Popup(
                        f"<b>Cluster {int(row['cluster_id'])}</b><br>"
                        f"Level: {level}<br>"
                        f"Score: {score:.1f}/100<br>"
                        f"Cause: {row.get('dominant_cause','?')}<br>"
                        f"Officers: {int(row.get('recommended_officers',0))}<br>"
                        f"Action: {str(row.get('recommended_action',''))[:60]}",
                        max_width=300
                    ),
                    tooltip=f"{WATCH_LEVELS[level]['emoji']} "
                            f"C{int(row['cluster_id'])} — {level} ({score:.0f})"
                ).add_to(m)

            critical_rows = valid[valid["watch_level"] == "CRITICAL"]
            for _, row in critical_rows.iterrows():
                folium.Marker(
                    [row["lat_center"], row["lon_center"]],
                    icon=folium.DivIcon(html=
                        f"<div style='background:red;color:white;"
                        f"border-radius:4px;padding:2px 5px;"
                        f"font-size:10px;font-weight:bold;'>"
                        f"⚠ C{int(row['cluster_id'])}</div>")
                ).add_to(m)

            st_folium(m, width=None, height=480, returned_objects=[])

    # ── TAB 3: Cluster Detail                   ───
    with tab_detail:
        st.markdown("### 🔬 Cluster Deep Dive")
        cluster_ids = sorted(corridor_df["cluster_id"].tolist())
        sel_cluster = st.selectbox(
            "Select Cluster", cluster_ids, key="cw_cluster_sel"
        )
        row = corridor_df[corridor_df["cluster_id"] == sel_cluster]
        if row.empty:
            st.warning("No data for selected cluster.")
        else:
            row = row.iloc[0]
            level  = str(row["watch_level"])
            color  = WATCH_LEVELS[level]["color"]
            emoji  = WATCH_LEVELS[level]["emoji"]

            st.markdown(f"""
            <div style='background:{color}18;border-left:4px solid {color};
            border-radius:8px;padding:12px 18px;margin-bottom:12px'>
            <h3 style='margin:0;color:{color}'>
            {emoji} Cluster {int(row['cluster_id'])} — {level}</h3>
            <p style='margin:4px 0;font-size:13px'>
            Watch Score: <b>{row['watch_score']:.1f}/100</b> &nbsp;|&nbsp;
            Events: <b>{int(row['event_count'])}</b> &nbsp;|&nbsp;
            Dominant: <b>{row['dominant_cause']}</b></p>
            </div>
            """, unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("⚠ Avg Severity",    f"{row['avg_severity']:.2f}")
            m2.metric("📊 Avg Risk",        f"{row['avg_risk']:.2f}")
            m3.metric("🚧 Road Closure",    f"{row['road_closure_pct']:.1f}%")
            m4.metric("👮 Officers Needed", f"{int(row['recommended_officers'])}")

            st.markdown(f"**🎯 Recommended Action:** {row['recommended_action']}")
            st.markdown(f"**📉 Simulation Gain:** {row['sim_gain_pct']:.1f}% congestion reduction")

            if isinstance(row.get("components"), dict):
                comps = row["components"]
                comp_df = pd.DataFrame([
                    {"Signal": k.replace("_"," ").title(),
                     "Score": v, "Max": [25,25,20,15,10,5][i]}
                    for i, (k, v) in enumerate(comps.items())
                ])
                fig = go.Figure(go.Bar(
                    x=comp_df["Signal"], y=comp_df["Score"],
                    marker_color=color,
                    text=comp_df["Score"].round(1),
                    textposition="outside"
                ))
                fig.update_layout(
                    title="Score Component Breakdown",
                    height=280, margin=dict(t=40,b=20,l=10,r=10),
                    yaxis=dict(range=[0,30]),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: Score Analysis                   ───
    with tab_chart:
        st.markdown("### 📊 Score Distribution & Analysis")

        c_left, c_right = st.columns(2)

        with c_left:
            level_counts = corridor_df["watch_level"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=level_counts.index.tolist(),
                values=level_counts.values.tolist(),
                marker_colors=[
                    WATCH_LEVELS.get(l, {}).get("color","gray")
                    for l in level_counts.index
                ],
                hole=0.4
            ))
            fig_pie.update_layout(
                title="Watch Level Distribution",
                height=280, margin=dict(t=40,b=10,l=10,r=10),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_right:
            fig_hist = px.histogram(
                corridor_df, x="watch_score", nbins=20,
                color_discrete_sequence=["#378ADD"],
                title="Watch Score Distribution"
            )
            fig_hist.update_layout(
                height=280, margin=dict(t=40,b=20,l=10,r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### 🏆 Top 15 Corridors by Watch Score")
        top15 = corridor_df.head(15)[
            ["cluster_id","watch_level","watch_score",
             "avg_severity","dominant_cause","recommended_officers"]
        ]
        fig_bar = px.bar(
            top15, x="cluster_id", y="watch_score",
            color="watch_level",
            color_discrete_map={
                "CRITICAL":"#E74C3C","HIGH":"#E67E22",
                "MODERATE":"#F1C40F","LOW":"#2ECC71"
            },
            text="watch_score",
            title="Top 15 Corridors — Watch Score"
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar.update_layout(
            height=320, margin=dict(t=40,b=30,l=10,r=10),
            xaxis_title="Cluster ID", yaxis_title="Score",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    return corridor_df


      
# CELL 7 — Sidebar mini-widget
# Usage in app.py:
#   from corridor_watch import get_corridor_sidebar_widget
#   get_corridor_sidebar_widget()
      
def get_corridor_sidebar_widget():
    import streamlit as st
    alerts = load_corridor_alerts(limit=5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🛰 Corridor Watch**")
    if alerts.empty:
        st.sidebar.caption("Run corridor_watch.py first.")
        return
    critical = alerts[alerts["watch_level"] == "CRITICAL"]
    high     = alerts[alerts["watch_level"] == "HIGH"]
    c1, c2   = st.sidebar.columns(2)
    c1.metric("🔴 Critical", len(critical))
    c2.metric("🟠 High",     len(high))
    for _, row in alerts.head(3).iterrows():
        level = str(row["watch_level"])
        color = WATCH_LEVELS.get(level, {}).get("color","gray")
        emoji = WATCH_LEVELS.get(level, {}).get("emoji","⚪")
        st.sidebar.markdown(f"""
        <div style='background:{color}18;border-left:3px solid {color};
        border-radius:5px;padding:5px 8px;margin:3px 0;font-size:12px'>
        {emoji} <b>C{int(row['cluster_id'])}</b> — {level}
        <br>Score: {row['watch_score']:.1f} | {row['dominant_cause']}
        </div>
        """, unsafe_allow_html=True)



# Standalone run
  
if __name__ == "__main__":
    print("=" * 55)
    print("  TrafficTwin — Corridor Watch Engine")
    print("=" * 55)
    df = run_corridor_watch()
    save_corridor_alerts(df)
    print(f"\n  Corridors scanned : {len(df)}")
    for level in ["CRITICAL","HIGH","MODERATE","LOW"]:
        n = int((df["watch_level"] == level).sum())
        print(f"  {WATCH_LEVELS[level]['emoji']} {level:<10}: {n}")
    print(f"\n  Top 5 corridors:")
    for _, row in df.head(5).iterrows():
        e = WATCH_LEVELS[row['watch_level']]['emoji']
        print(f"    {e} Cluster {int(row['cluster_id']):>4} | "
              f"Score {row['watch_score']:>5.1f} | "
              f"{row['watch_level']:<9} | {row['dominant_cause']}")
    print(f"\n  ✓ Saved to corridor_alerts in brain.db")
    print("=" * 55)