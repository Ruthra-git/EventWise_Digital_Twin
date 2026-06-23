

import sqlite3, json, warnings, os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from datetime import datetime
warnings.filterwarnings('ignore')

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "outputs" / "engineered_events.csv"
DB_PATH = BASE_DIR / "database" / "brain.db"

 
# STEP 1 — Load engineered_events.csv (NOT raw CSV)
 
def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df)} rows from engineered_events.csv")

    # Validate that ingest was actually run (not raw CSV)
    expected = ['severity','event_duration','cause_code',
                'is_peak_hour','priority_score','location_frequency']
    missing  = [c for c in expected if c not in df.columns]
    if missing:
        print(f"\n  ⚠  WARNING: These columns are missing: {missing}")
        print("     This looks like raw CSV, not engineered_events.csv")
        print("     Run 01_ingest.py first, then re-run this script.\n")

    # Normalise lat/lon column names
    for raw, clean in [('latitude','lat'),('longitude','lon')]:
        if raw in df.columns and clean not in df.columns:
            df[clean] = pd.to_numeric(df[raw], errors='coerce')
    df['lat'] = pd.to_numeric(df.get('lat', np.nan), errors='coerce')
    df['lon'] = pd.to_numeric(df.get('lon', np.nan), errors='coerce')
    before = len(df)
    df = df.dropna(subset=['lat','lon'])
    if before - len(df):
        print(f"  Dropped {before-len(df)} rows with missing lat/lon")

    # Only add defaults for columns that genuinely don't exist
    # (should be zero after running 01_ingest.py correctly)
    defaults = {
        'event_cause':'others', 'event_type':'unplanned',
        'severity':2, 'priority_score':2, 'event_duration':2.0,
        'hour':8, 'weekday':0, 'month':1, 'is_peak_hour':0,
        'cause_code':0, 'historical_frequency':1, 'location_frequency':1,
        'is_planned':0, 'road_closure':0, 'zone_code':0,
        'is_peak_hour':0, 'is_night':0, 'is_weekend':0,
        'minutes_since_midnight':480, 'is_route_event':0,
    }
    added = []
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
            added.append(col)
    if added:
        print(f"  ⚠  Added missing columns (run 01_ingest.py to fix): {added}")
    else:
        print(f"  ✓  All engineered features present — 01_ingest.py ran correctly")

    # Validate feature quality
    print("\n── Feature quality check ──")
    check_cols = ['severity','event_duration','cause_code','priority_score']
    for col in check_cols:
        if col not in df.columns:
            continue
        top_pct = df[col].value_counts().iloc[0] / len(df) * 100
        n_unique = df[col].nunique()
        flag = "  ⚠ NEAR-CONSTANT — re-run 01_ingest.py" if top_pct > 85 else "  ✓"
        print(f"  {col:<25}: {n_unique:>4} unique values, "
              f"top={top_pct:.1f}%{flag}")

    print(f"\n  Severity distribution: "
          f"{df['severity'].value_counts().sort_index().to_dict()}")
    print(f"  Duration std: {df['event_duration'].std():.3f} hrs | "
          f"unique: {df['event_duration'].nunique()}")
    return df


 
# STEP 2 — Three-pass DBSCAN (proven to work on your data)
# Same logic that gave you 244 clusters / 4.37% largest
 
def multi_resolution_dbscan(df):
    coords_rad = np.radians(df[['lat','lon']].values)
    R = 6371.0

    # ── Pass 1: city-level 800m ──
    labels = DBSCAN(eps=0.8/R, min_samples=10,
                    metric='haversine', algorithm='ball_tree'
                    ).fit_predict(coords_rad)

    u, c = np.unique(labels, return_counts=True)
    print("\nPass 1 top clusters:")
    for uid, cnt in sorted(zip(u,c), key=lambda x:-x[1])[:10]:
        print(f"  {'NOISE' if uid==-1 else f'Cluster {uid}':>12}: {cnt}")

    # ── Pass 2: sub-cluster anything >150 events at 250m ──
    next_id = int(labels.max()) + 1
    for cid, cnt in zip(u, c):
        if cid == -1 or cnt < 150:
            continue
        print(f"\n  Subclustering Cluster {cid} ({cnt} events) at 250m…")
        idx = np.where(labels == cid)[0]
        sub = DBSCAN(eps=0.25/R, min_samples=6,
                     metric='haversine', algorithm='ball_tree'
                     ).fit_predict(coords_rad[idx])
        for s in np.unique(sub):
            mask = sub == s
            labels[idx[mask]] = -1 if s == -1 else next_id
            if s != -1:
                next_id += 1

    # ── Pass 3: break any remaining large/spread clusters at 150m ──
    vc = pd.Series(labels).value_counts()
    for cid, size in vc.items():
        if cid == -1:
            continue
        idx = np.where(labels == cid)[0]
        tmp = df.iloc[idx]
        lat_span = (tmp['lat'].max() - tmp['lat'].min()) * 111
        lon_span = (tmp['lon'].max() - tmp['lon'].min()) * 111
        if size < 80 and max(lat_span, lon_span) < 1.2:
            continue
        sub = DBSCAN(eps=0.15/R, min_samples=3,
                     metric='haversine', algorithm='ball_tree'
                     ).fit_predict(coords_rad[idx])
        for s in np.unique(sub):
            mask = sub == s
            labels[idx[mask]] = -1 if s == -1 else next_id
            if s != -1:
                next_id += 1

    # ── Remove micro-clusters (<5 events) ──
    vc2 = pd.Series(labels).value_counts()
    for cid, size in vc2.items():
        if cid != -1 and size < 5:
            labels[labels == cid] = -1

    # ── Final stats ──
    n_clust = len(set(labels[labels >= 0]))
    noise_p = (labels == -1).mean() * 100
    top_p   = pd.Series(labels[labels >= 0]).value_counts().iloc[0] / len(labels) * 100
    print(f"\n  Clusters : {n_clust}")
    print(f"  Noise    : {noise_p:.2f}%")
    print(f"  Largest  : {top_p:.2f}%")
    if top_p > 40:
        print("  ⚠  Largest still >40% — data is genuinely dense in one corridor")
    else:
        print("  ✓  Distribution healthy")
    return labels


 
# STEP 3 — Enrich with cluster-level features
# Uses real engineered features from 01_ingest.py
 
def enrich_cluster_features(df):
    df = df.copy()

    # ── Cluster-level aggregations ──
    agg = df.groupby('cluster_id').agg(
        hotspot_density      = ('cluster_id',     'count'),
        cluster_risk_score   = ('severity',        'mean'),
        planned_ratio        = ('is_planned',      'mean'),
        cluster_lat_center   = ('lat',             'mean'),
        cluster_lon_center   = ('lon',             'mean'),
        avg_cluster_duration = ('event_duration',  'mean'),
        peak_hour_ratio      = ('is_peak_hour',    'mean'),  # NEW
        night_ratio          = ('is_night',        'mean'),  # NEW — uses 01_ingest feature
        road_closure_ratio   = ('road_closure',    'mean'),  # NEW — uses 01_ingest feature
        dominant_cause       = ('event_cause',     lambda x: x.mode()[0] if len(x) else 'others'),
    ).reset_index()

    # ── Recent activity (uses start_datetime if available) ──
    date_col = next((c for c in ['start_datetime','start_dt','start_date']
                     if c in df.columns), None)
    if date_col:
        df['_dt']  = pd.to_datetime(df[date_col], errors='coerce')
        cutoff     = df['_dt'].quantile(0.70)
        recent_cnt = (df[df['_dt'] >= cutoff]
                      .groupby('cluster_id').size()
                      .rename('recent_cluster_activity'))
    else:
        recent_cnt = df.groupby('cluster_id').size().rename('recent_cluster_activity')

    agg = agg.merge(recent_cnt, on='cluster_id', how='left')
    agg['recent_cluster_activity'] = agg['recent_cluster_activity'].fillna(0)

    df = df.merge(agg, on='cluster_id', how='left')

    # ── Distance from cluster centre ──
    df['dist_from_center'] = (np.sqrt(
        (df['lat'] - df['cluster_lat_center'])**2 +
        (df['lon'] - df['cluster_lon_center'])**2
    ) * 111).round(4)

    # ── Cluster risk tier (categorical) ──
    df['cluster_risk_tier'] = pd.cut(
        df['cluster_risk_score'],
        bins=[0, 1.5, 2.5, 3.5, 5.1],
        labels=[1, 2, 3, 4]           # 1=LOW … 4=CRITICAL
    ).astype(float).fillna(1)

    # ── Fill noise cluster rows ──
    noise_mask = df['cluster_id'] == -1
    fill_cols  = ['hotspot_density','cluster_risk_score','planned_ratio',
                  'avg_cluster_duration','recent_cluster_activity',
                  'dist_from_center','cluster_risk_tier',
                  'peak_hour_ratio','night_ratio','road_closure_ratio']
    for c in fill_cols:
        df.loc[noise_mask, c] = df.loc[noise_mask, c].fillna(0)

    print(f"\n✓ Cluster features enriched — {len(df)} events")
    print(f"  New cluster features: peak_hour_ratio, night_ratio, "
          f"road_closure_ratio, cluster_risk_tier, dist_from_center")
    return df, agg


 
# STEP 4 — Similar events lookup
 
def build_similar_events(df):
    rows = []
    sig_cols = ['event_cause', 'is_planned', 'cluster_id']
    for sig, grp in df.groupby(sig_cols, dropna=False):
        sig_str = (f"{sig[0]}|"
                   f"{'planned' if sig[1] else 'unplanned'}|"
                   f"cluster_{int(sig[2])}")
        for _, row in grp.nlargest(5, 'severity').iterrows():
            rows.append({
                'event_signature': sig_str,
                'event_cause'    : str(row.get('event_cause', '')),
                'is_planned'     : int(row.get('is_planned', 0)),
                'cluster_id'     : int(row.get('cluster_id', -1)),
                'severity'       : int(row.get('severity', 1)),
                'event_duration' : round(float(row.get('event_duration', 2)), 2),
                'hour'           : int(row.get('hour', 8)),
                'weekday'        : int(row.get('weekday', 0)),
                'is_peak_hour'   : int(row.get('is_peak_hour', 0)),
                'road_closure'   : int(row.get('road_closure', 0)),
                'priority_score' : int(row.get('priority_score', 1)),
                'address'        : str(row.get('address', '')),
                'lat'            : round(float(row.get('lat', 0)), 6),
                'lon'            : round(float(row.get('lon', 0)), 6),
            })
    out = pd.DataFrame(rows)
    n_sigs = df.groupby(sig_cols, dropna=False).ngroups
    print(f"✓ similar_events: {len(out)} records across {n_sigs} signatures")
    return out


 
# STEP 5 — Write brain.db
 
def write_brain_db(df, agg, similar_df):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # Final ML feature list — updated to include new ingest features
    FEATURES = [
        # temporal (from 01_ingest)
        'hour', 'weekday', 'month', 'is_peak_hour', 'is_night',
        'is_weekend', 'minutes_since_midnight',
        # event properties (from 01_ingest)
        'event_duration', 'is_planned', 'road_closure',
        'cause_code', 'priority_score', 'is_route_event',
        # frequency features (from 01_ingest)
        'historical_frequency', 'location_frequency',
        'zone_frequency', 'junction_frequency', 'police_frequency',
        # spatial cluster features (from 02_cluster)
        'cluster_id', 'hotspot_density', 'cluster_risk_score',
        'cluster_risk_tier', 'avg_cluster_duration', 'planned_ratio',
        'recent_cluster_activity', 'dist_from_center',
        'peak_hour_ratio', 'night_ratio', 'road_closure_ratio',
        # zone
        'zone_code',
    ]
    # Only keep features that actually exist in df
    FEATURES = [f for f in FEATURES if f in df.columns]

    # ── Table 1: events ──
    df.to_sql('events', conn, if_exists='replace', index=False)
    print(f"\n  ✓ events             : {len(df):>5} rows | "
          f"{len(df.columns)} columns")

    # ── Table 2: hotspot_statistics ──
    hs = df[df['cluster_id'] >= 0].groupby('cluster_id').agg(
        event_count      = ('cluster_id',              'count'),
        avg_severity     = ('severity',                 'mean'),
        max_severity     = ('severity',                 'max'),
        lat_center       = ('lat',                      'mean'),
        lon_center       = ('lon',                      'mean'),
        planned_ratio    = ('is_planned',               'mean'),
        avg_duration_hrs = ('event_duration',           'mean'),
        peak_hour_ratio  = ('is_peak_hour',             'mean'),
        night_ratio      = ('is_night',                 'mean'),
        road_closure_pct = ('road_closure',             'mean'),
        recent_activity  = ('recent_cluster_activity',  'first'),
        dominant_cause   = ('event_cause',
                            lambda x: x.mode()[0] if len(x) else 'others'),
    ).reset_index()
    hs['risk_label'] = pd.cut(
        hs['avg_severity'], bins=[0,1.5,2.5,3.5,5.1],
        labels=['LOW','MODERATE','HIGH','CRITICAL']
    )
    hs.to_sql('hotspot_statistics', conn, if_exists='replace', index=False)
    print(f"  ✓ hotspot_statistics : {len(hs):>5} rows")

    # ── Table 3: similar_events ──
    similar_df.to_sql('similar_events', conn, if_exists='replace', index=False)
    print(f"  ✓ similar_events     : {len(similar_df):>5} rows")

    # ── Table 4: deployment_logs ──
    conn.execute("""CREATE TABLE IF NOT EXISTS deployment_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT, event_signature TEXT,
        predicted_severity INTEGER, personnel_deployed INTEGER,
        barricades_placed INTEGER, diversion_activated INTEGER,
        response_time_min REAL, outcome TEXT, officer_id TEXT,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    print(f"  ✓ deployment_logs    :     0 rows (runtime)")

    # ── Table 5: feedback ──
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT, event_signature TEXT,
        predicted_severity INTEGER, actual_severity INTEGER,
        predicted_duration REAL, actual_duration REAL,
        personnel_adequate INTEGER, notes TEXT, officer_id TEXT,
        logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    print(f"  ✓ feedback           :     0 rows (runtime)")

    # ── Table 6: model_metadata ──
    conn.execute("""CREATE TABLE IF NOT EXISTS model_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT, model_version TEXT, training_date TEXT,
        features_used TEXT, n_features INTEGER,
        weighted_f1 REAL, accuracy REAL, best_params TEXT,
        training_rows INTEGER, notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")

    if conn.execute("SELECT COUNT(*) FROM model_metadata").fetchone()[0] == 0:
        conn.execute("""INSERT INTO model_metadata
            (model_name,model_version,training_date,features_used,
             n_features,weighted_f1,accuracy,best_params,training_rows,notes)
            VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            'LightGBM', 'v0.0-untrained',
            datetime.now().strftime('%Y-%m-%d'),
            json.dumps(FEATURES), len(FEATURES),
            0.0, 0.0, json.dumps({}), len(df),
            f'Seeded by 02_cluster.py with {len(FEATURES)} features — '
            f'overwritten after 03_train.py'
        ))
    print(f"  ✓ model_metadata     :     1 row  ({len(FEATURES)} features seeded)")

    conn.commit()

    # ── Final summary ──
    print("\n── brain.db table summary ──")
    for (tbl,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall():
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:<25}: {n:>5} rows")

    # ── Cluster span report ──
    print("\n── Top 20 clusters by event count ──")
    summary = df[df.cluster_id >= 0].groupby('cluster_id').agg(
        events        = ('cluster_id', 'count'),
        avg_severity  = ('severity',    'mean'),
        dominant_cause= ('event_cause', lambda x: x.mode()[0]),
        lat_span_km   = ('lat', lambda x: (x.max()-x.min())*111),
        lon_span_km   = ('lon', lambda x: (x.max()-x.min())*111),
    )
    summary['max_span_km'] = summary[['lat_span_km','lon_span_km']].max(axis=1)
    print(summary.sort_values('events', ascending=False)
          .head(20).round(2).to_string())

    conn.close()
    print(f"\n✓ brain.db ready → {DB_PATH}")
    print(f"  Features in model_metadata: {len(FEATURES)}")
    print(f"  Next → python scripts/03_train.py")


 
# MAIN
 
if __name__ == '__main__':
    print("=" * 55)
    print("  02_cluster.py — Building the TrafficTwin Brain")
    print("=" * 55 + "\n")

    df             = load_csv(CSV_PATH)
    labels         = multi_resolution_dbscan(df)
    df['cluster_id'] = labels
    df, agg        = enrich_cluster_features(df)
    similar_df     = build_similar_events(df)
    write_brain_db(df, agg, similar_df)

