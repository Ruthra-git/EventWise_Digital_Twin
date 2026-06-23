
# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — Imports
# ───────────────────────────────────────PROJECT_ROOT = Path.cwd()──────────────────────────────────────
import os
import json
import sqlite3
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

print("=" * 60)
print("TrafficTwin — Recommendation Engine")
print("=" * 60)
print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 — Configuration
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR          = PROJECT_ROOT / "database"
DATABASE_PATH         = DATABASE_DIR / "brain.db"
MODEL_DIR             = PROJECT_ROOT / "models"
BEST_MODEL_PATH       = MODEL_DIR / "best_model.pkl"
FEATURE_COLUMNS_PATH  = MODEL_DIR / "feature_columns.pkl"
OUTPUT_DIR            = PROJECT_ROOT / "outputs"
REPORT_DIR            = OUTPUT_DIR / "reports"
LOG_DIR               = OUTPUT_DIR / "logs"
BASE_DIR              = Path(__file__).resolve().parent
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

HIGH_CONFIDENCE   = 0.80
MEDIUM_CONFIDENCE = 0.60
LOW_CONFIDENCE    = 0.40
CRITICAL_RISK     = 90
HIGH_RISK         = 75
MODERATE_RISK     = 50
LOW_RISK          = 25

OFFICER_RULES       = {1: 2, 2: 4, 3: 6, 4: 8, 5: 12}
BARRICADE_RULES     = {1: 1, 2: 2, 3: 3, 4: 5, 5: 6}
CONE_RULES          = {1: 10, 2: 15, 3: 20, 4: 25, 5: 30}
WARNING_BOARD_RULES = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

RESOURCE_PRIORITY_WEIGHTS = {
    "severity": 0.40, "cluster_risk": 0.25,
    "road_closure": 0.15, "duration": 0.10, "historical_frequency": 0.10
}

MAX_SIMILAR_EVENTS        = 10
MIN_SIMILAR_EVENTS        = 3
DEFAULT_DIVERSION_RADIUS_KM = 2.0
MAX_DIVERSION_OPTIONS     = 3
ENGINE_NAME               = "TrafficTwin AI Decision Engine"
ENGINE_VERSION            = "1.0"

print("=" * 60)
print("Configuration Loaded Successfully")
print("=" * 60)
print(f"Database      : {DATABASE_PATH}")
print(f"Model         : {BEST_MODEL_PATH}")
print(f"Feature File  : {FEATURE_COLUMNS_PATH}")
print(f"Output Folder : {OUTPUT_DIR}")
print(f"Engine        : {ENGINE_NAME}")
print(f"Version       : {ENGINE_VERSION}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 — Load AI assets
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading AI assets...")

try:
    BEST_MODEL = joblib.load(BEST_MODEL_PATH)
    print("✓ best_model.pkl loaded")
except Exception as e:
    BEST_MODEL = None
    print(f"✗ Failed to load model : {e}")

try:
    FEATURE_COLUMNS = joblib.load(FEATURE_COLUMNS_PATH)
    print(f"✓ feature_columns.pkl loaded ({len(FEATURE_COLUMNS)} features)")
except Exception as e:
    FEATURE_COLUMNS = []
    print(f"✗ Failed to load feature columns : {e}")

print("\nAI Asset Summary")
print("-" * 40)
print("Model Loaded       :", BEST_MODEL is not None)
print("Features Loaded    :", len(FEATURE_COLUMNS))
print("-" * 40)


# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 — Database connection
# ─────────────────────────────────────────────────────────────────────────────
print("\nConnecting to brain.db...")
import threading
_conn_local = threading.local()
 
def get_conn():
    """
    Returns a per-thread SQLite connection.
    Safe for Streamlit's multi-threaded callbacks.
    check_same_thread=False is NOT used — instead we give
    each thread its own connection, which is the correct fix.
    """
    if not hasattr(_conn_local, "conn") or _conn_local.conn is None:
        _conn_local.conn = sqlite3.connect(DATABASE_PATH)
    return _conn_local.conn

try:
    conn   = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    print("✓ Database connected")
except Exception as e:
    conn   = None
    cursor = None
    raise RuntimeError(f"Cannot connect to database : {e}")

tables           = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn)
available_tables = tables["name"].tolist()
print("\nAvailable Tables")
for table in available_tables:
    print(f"  ✓ {table}")

REQUIRED_TABLES = ["events", "hotspot_statistics", "similar_events",
                   "deployment_logs", "feedback", "model_metadata"]
missing_tables  = [t for t in REQUIRED_TABLES if t not in available_tables]
if len(missing_tables) == 0:
    print("\n✓ Knowledge base validation passed")
else:
    print("\n✗ Missing Tables:", missing_tables)

print("\nKnowledge Base Summary")
print("-" * 45)
for table in REQUIRED_TABLES:
    try:
        rows = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table:<25} {rows:>8} rows")
    except:
        pass
print("-" * 45)

try:
    metadata = pd.read_sql("SELECT * FROM model_metadata LIMIT 1", conn)
    if len(metadata):
        print("\nCurrent AI Model")
        print(metadata.iloc[0].to_dict())
except:
    print("\nNo model metadata available.")


# ─────────────────────────────────────────────────────────────────────────────
# Cell 6 — Persistence helpers
# FIX 1  : save_prediction duplicate removed — single implementation only
# FIX 2  : save_recommendation schema now matches inserted values exactly
# FIX 3  : save_simulation schema matches heatmap Cell 11 exactly
# FIX 7  : all values explicitly cast to correct SQLite types
# FIX 8  : DROP + recreate if schema is stale (migration logic)
# FIX 11 : all functions use try/finally conn.close()
# FIX 12 : no dead code after return / conn.close()
# ─────────────────────────────────────────────────────────────────────────────

def _migrate_table(cursor, table_name: str, create_sql: str):
    """
    Drop and recreate the table if it exists with an old schema.
    Safe because predictions/recommendations/simulation_logs are
    fully regenerated each run from brain.db events.
    """
    existing_cols = set()
    try:
        rows = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_cols = {r[1] for r in rows}
    except:
        pass

    required_cols = {
        w.strip().split()[0].lower()
        for w in create_sql.split("(", 1)[1].rsplit(")", 1)[0].split(",")
        if w.strip() and not w.strip().upper().startswith("PRIMARY")
    }

    if existing_cols and not required_cols.issubset(existing_cols):
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    cursor.execute(create_sql)

def save_prediction(event_id, prediction: dict):
    conn_local = sqlite3.connect(DATABASE_PATH)
    try:
        cur = conn_local.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                event_id              TEXT PRIMARY KEY,
                severity              INTEGER,
                confidence            REAL,
                uncertainty           REAL,
                historical_confidence REAL,
                combined_reliability  REAL,
                prediction_time       TEXT
            )
        """)
        cur.execute("DELETE FROM predictions WHERE event_id=?", (str(event_id),))
        cur.execute("""
            INSERT INTO predictions
                (event_id, severity, confidence, uncertainty,
                 historical_confidence, combined_reliability, prediction_time)
            VALUES (?,?,?,?,?,?,?)
        """, (
            str(event_id),
            int(prediction.get("severity", 0)),
            float(prediction.get("confidence", 0.0)),
            float(prediction.get("uncertainty", 0.0)),
            float(prediction.get("historical_confidence") or 0.0),
            float(prediction.get("combined_reliability") or 0.0),
            str(prediction.get("prediction_time", "")),
        ))
        conn_local.commit()
    finally:
        conn_local.close()


def save_recommendation(event_id, action_plan: dict):
    conn_local = sqlite3.connect(DATABASE_PATH)
    try:
        cur = conn_local.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                event_id                TEXT PRIMARY KEY,
                recommended_officers    INTEGER,
                recommended_barricades  INTEGER,
                priority_score          REAL,
                confidence              REAL,
                diversion_required      INTEGER,
                expected_clearance      TEXT
            )
        """)
        cur.execute("DELETE FROM recommendations WHERE event_id=?", (str(event_id),))
        cur.execute("""
            INSERT INTO recommendations
                (event_id, recommended_officers, recommended_barricades,
                 priority_score, confidence, diversion_required, expected_clearance)
            VALUES (?,?,?,?,?,?,?)
        """, (
            str(event_id),
            int(action_plan["resource_plan"]["officers"]),
            int(action_plan["resource_plan"]["barricades"]),
            float(action_plan["priority"]["priority_score"]),
            float(action_plan["event_summary"]["confidence"]),
            int(bool(action_plan["movement_plan"]["diversion_required"])),
            str(action_plan["time_plan"]["expected_clearance"]),
        ))
        conn_local.commit()
    finally:
        conn_local.close()


def save_simulation(event_id, simulation: dict):
    conn_local = sqlite3.connect(DATABASE_PATH)
    try:
        cur = conn_local.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS simulation_logs (
                event_id           TEXT PRIMARY KEY,
                strategy           TEXT,
                officers           INTEGER,
                barricades         INTEGER,
                diversion          INTEGER,
                expected_reduction REAL,
                created_time       TEXT
            )
        """)
        cur.execute("DELETE FROM simulation_logs WHERE event_id=?", (str(event_id),))
        best = simulation.get("recommended_strategy", {})
        cur.execute("""
            INSERT INTO simulation_logs
                (event_id, strategy, officers, barricades,
                 diversion, expected_reduction, created_time)
            VALUES (?,?,?,?,?,?,?)
        """, (
            str(event_id),
            str(best.get("strategy", "")),
            int(best.get("officers", 0)),
            int(best.get("barricades", 0)),
            int(bool(best.get("diversion", False))),
            float(best.get("expected_congestion_reduction", 0.0)),
            str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ))
        conn_local.commit()
    finally:
        conn_local.close()



# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 — Enhanced AI Prediction Engine
# ─────────────────────────────────────────────────────────────────────────────

def predict_congestion(event: dict):
    if BEST_MODEL is None:
        raise RuntimeError("Model not loaded.")

    event_df = pd.DataFrame([event])
    for col in FEATURE_COLUMNS:
        if col not in event_df.columns:
            event_df[col] = 0
    X = event_df[FEATURE_COLUMNS]

    severity      = int(BEST_MODEL.predict(X)[0])
    probabilities = BEST_MODEL.predict_proba(X)[0]
    sorted_probs  = np.sort(probabilities)[::-1]
    confidence_gap = float(sorted_probs[0] - sorted_probs[1])

    # ── NEW confidence formula ──────────────────────────────
    # Base from model probability gap (0-1 range)
    base_conf = 0.55 + confidence_gap * 0.20          # max base = 0.75

    # Severity penalty: higher severity → more uncertainty
    sev_penalty = float(severity - 1) * 0.030          # sev5 → -0.12

    # Cluster risk penalty: dense hotspots are harder to predict
    cluster_risk = float(event.get("cluster_risk_score", 2.5) or 2.5)
    risk_penalty = (cluster_risk / 5.0) * 0.040        # max → -0.04

    # Historical frequency boost: recurring events are more predictable
    hist_freq = float(event.get("historical_frequency", 1) or 1)
    hist_boost = min(hist_freq / 500.0, 0.06)          # max +0.06

    # Road closure uncertainty
    closure_penalty = 0.025 if event.get("road_closure", 0) else 0.0

    ai_confidence = (
        base_conf
        - sev_penalty
        - risk_penalty
        + hist_boost
        - closure_penalty
    )

    # Realistic band: 0.52 – 0.88 (never saturates to 0.92)
    ai_confidence = round(float(np.clip(ai_confidence, 0.52, 0.88)), 3)
    uncertainty   = round(1.0 - ai_confidence, 3)

    if severity >= 5:
        risk = "CRITICAL"
    elif severity == 4:
        risk = "HIGH"
    elif severity == 3:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return {
        "severity"            : severity,
        "risk"                : risk,
        "confidence"          : ai_confidence,
        "ai_confidence"       : ai_confidence,
        "confidence_gap"      : round(confidence_gap, 3),
        "uncertainty"         : uncertainty,
        "historical_confidence": None,
        "combined_reliability": None,
        "probabilities"       : probabilities.tolist(),
        "prediction_time"     : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell 8 — Historical Intelligence Engine
# FIX 4 : empty-df path now caches and returns immediately
# FIX 5 : cache key includes limit → (cause, cluster, limit)
# FIX 6 : cause normalised to lowercase stripped string
# ─────────────────────────────────────────────────────────────────────────────

SIMILAR_CACHE: dict = {}

def find_similar_event(event: dict, limit: int = MAX_SIMILAR_EVENTS):
    cause   = str(event.get("event_cause", "others")).strip().lower()
    cluster = int(event.get("cluster_id", -1))
    key     = (cause, cluster, limit)
 
    if key in SIMILAR_CACHE:
        return SIMILAR_CACHE[key]
 
    _c = get_conn()   # ← thread-local connection
 
    df = pd.read_sql(
        "SELECT * FROM similar_events WHERE event_cause=? AND cluster_id=? LIMIT ?",
        _c, params=[cause, cluster, limit]
    )
 
    if len(df) == 0:
        df = pd.read_sql(
            "SELECT * FROM similar_events WHERE event_cause=? LIMIT ?",
            _c, params=[cause, limit]
        )
 
    if len(df) == 0:
        result = {
            "matches"              : df,
            "historical_confidence": 0.0,
            "average_duration"     : 0.0,
            "average_officers"     : 0.0,
            "average_barricades"   : 0.0,
        }
        SIMILAR_CACHE[key] = result
        return result
 
    historical_confidence = min(len(df) / MAX_SIMILAR_EVENTS, 1.0)
    avg_duration   = df["event_duration"].mean()    if "event_duration"          in df.columns else 0.0
    avg_officers   = df["recommended_officers"].mean()  if "recommended_officers"    in df.columns else 0.0
    avg_barricades = df["recommended_barricades"].mean() if "recommended_barricades"  in df.columns else 0.0
 
    result = {
        "matches"              : df,
        "historical_confidence": round(historical_confidence, 2),
        "average_duration"     : round(avg_duration,   1),
        "average_officers"     : round(avg_officers,   1),
        "average_barricades"   : round(avg_barricades, 1),
    }
    SIMILAR_CACHE[key] = result
    return result



# ─────────────────────────────────────────────────────────────────────────────
# Cell 9 — Adaptive Officer Recommendation
# ─────────────────────────────────────────────────────────────────────────────

def recommend_officers(event: dict, prediction: dict = None, history: dict = None):
    if prediction is None:
        prediction = predict_congestion(event)
    if history is None:
        history = find_similar_event(event)

    severity = prediction["severity"]
    base     = OFFICER_RULES.get(severity, 2)

    if history["average_officers"] > 0:
        optimal = int(round((base + history["average_officers"]) / 2))
    else:
        optimal = base

    minimum    = max(2, optimal - 2)
    maximum    = optimal + 3
    flexibility = (maximum - minimum) / maximum

    return {
        "minimum_officers"    : minimum,
        "optimal_officers"    : optimal,
        "recommended_officers": optimal,
        "maximum_officers"    : maximum,
        "surge_capacity"      : maximum + 2,
        "flexibility_score"   : round(flexibility, 2),
        "deployment_strategy" : "Adaptive Allocation",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell 10 — Recommend Barricades
# ─────────────────────────────────────────────────────────────────────────────

def recommend_barricades(event: dict, prediction: dict = None, history: dict = None):
    if prediction is None:
        prediction = predict_congestion(event)
    if history is None:
        history = find_similar_event(event)

    severity   = prediction["severity"]
    barricades = BARRICADE_RULES.get(severity, 1)
    cones      = CONE_RULES.get(severity, 10)
    boards     = WARNING_BOARD_RULES.get(severity, 1)

    if event.get("road_closure", 0):
        barricades += 2
        cones      += 10
        boards     += 1

    similar = history["matches"]
    if len(similar) > 0 and "recommended_barricades" in similar.columns:
        hist       = int(round(similar["recommended_barricades"].mean()))
        barricades = int(round((barricades + hist) / 2))

    if float(event.get("hotspot_density", 0)) > 100:
        barricades += 1
        cones      += 5

    return {
        "recommended_barricades": barricades,
        "traffic_cones"         : cones,
        "warning_boards"        : boards,
        "confidence"            : prediction["confidence"],
        "reason"                : "Severity, road closure and hotspot intelligence considered",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell 11 — Suggest Diversion Route
# ─────────────────────────────────────────────────────────────────────────────

def suggest_diversion(event: dict, prediction: dict = None):
    if prediction is None:
        prediction = predict_congestion(event)

    severity        = prediction["severity"]
    road_closure    = event.get("road_closure", 0)
    cluster         = event.get("cluster_id", -1)
    hotspot_density = float(event.get("hotspot_density", 0))

    diversion_needed = severity >= 4 or road_closure or hotspot_density > 75

    if diversion_needed:
        primary   = f"Alternate Route around Cluster {cluster}"
        secondary = f"Peripheral Road near Cluster {cluster}"
        message   = "Diversion strongly recommended."
    else:
        primary   = "No diversion required"
        secondary = None
        message   = "Traffic expected to flow normally."

    return {
        "diversion_required": diversion_needed,
        "primary_route"     : primary,
        "secondary_route"   : secondary,
        "message"           : message,
        "confidence"        : prediction["confidence"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell 12 — Estimate Clearance Time
# ─────────────────────────────────────────────────────────────────────────────

def estimate_clearance_time(event: dict, prediction: dict = None, history: dict = None):
    if prediction is None:
        prediction = predict_congestion(event)
    if history is None:
        history = find_similar_event(event)

    similar = history["matches"]
    if len(similar) > 0 and "event_duration" in similar.columns:
        duration = int(round(similar["event_duration"].mean()))
    else:
        duration = prediction["severity"] * 30

    clearance = datetime.now() + timedelta(minutes=duration)

    return {
        "estimated_duration_minutes": duration,
        "expected_clearance"        : clearance.strftime("%Y-%m-%d %H:%M"),
        "confidence"                : prediction["confidence"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell 13 — Compute Adaptive Resource Priority
# ─────────────────────────────────────────────────────────────────────────────

def compute_resource_priority(event: dict, prediction: dict = None, history: dict = None):
    if prediction is None:
        prediction = predict_congestion(event)
    if history is None:
        history = find_similar_event(event)

    severity        = int(prediction["severity"])
    confidence      = float(prediction["confidence"])
    uncertainty     = 1.0 - confidence
    hotspot_density = float(event.get("hotspot_density", 0))
    cluster_risk    = float(event.get("cluster_risk_score", 0))
    road_closure    = int(event.get("road_closure", 0))

    similar       = history["matches"]
    similar_count = len(similar)

    if similar_count > 0 and "event_duration" in similar.columns:
        avg_duration   = float(similar["event_duration"].mean())
        duration_spread = float(similar["event_duration"].std() or 0.0)
    else:
        avg_duration   = severity * 30.0
        duration_spread = 0.0

    severity_component    = (severity / 5.0) * 35.0
    hotspot_component     = min(hotspot_density / 150.0, 1.0) * 20.0
    risk_component        = min(cluster_risk, 1.0) * 15.0
    closure_component     = 10.0 if road_closure else 0.0
    uncertainty_component = min(uncertainty, 1.0) * 10.0
    similarity_component  = min(similar_count / 10.0, 1.0) * 5.0
    duration_component    = min(avg_duration / 180.0, 1.0) * 5.0
    volatility_component  = min(duration_spread / 60.0, 1.0) * 5.0

    priority_score = round(min(
        severity_component + hotspot_component + risk_component +
        closure_component + uncertainty_component + similarity_component +
        duration_component + volatility_component,
        100.0
    ), 2)

    if priority_score >= 85:
        priority_label = "CRITICAL"
    elif priority_score >= 70:
        priority_label = "HIGH"
    elif priority_score >= 50:
        priority_label = "MODERATE"
    else:
        priority_label = "LOW"

    return {
        "priority_score"  : priority_score,
        "priority_label"  : priority_label,
        "severity"        : severity,
        "confidence"      : round(confidence, 3),
        "uncertainty"     : round(uncertainty, 3),
        "historical_matches"                   : similar_count,
        "average_historical_duration_minutes"  : round(avg_duration, 1),
        "duration_volatility_minutes"          : round(duration_spread, 1),
        "reason": "Adaptive score from severity, uncertainty, hotspot recurrence, and historical similarity.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cell 14 — Generate Unified Action Plan
# FIX 13 : all helpers defined above — no forward references
# predict_congestion + find_similar_event called ONCE, passed downstream
# save_prediction + save_recommendation called here after plan is built
# ─────────────────────────────────────────────────────────────────────────────

def generate_action_plan(event: dict):
    prediction = predict_congestion(event)
    history    = find_similar_event(event)

    # Enrich prediction with historical signals
    prediction["historical_confidence"] = history["historical_confidence"]

    hist_conf = float(history["historical_confidence"] or 0.0)
    ai_conf   = float(prediction["confidence"])
    # Weight: if we have good historical data, trust it more
    hist_weight = min(hist_conf * 0.5, 0.35)   # 0 when no history, max 0.35
    ai_weight   = 1.0 - hist_weight
    combined    = round(ai_conf * ai_weight + hist_conf * hist_weight, 3)
    # Apply mild density penalty so crowded hotspots stay uncertain
    density_pen = min(float(event.get("hotspot_density", 0) or 0) / 2000.0, 0.05)
    prediction["combined_reliability"] = round(
        float(np.clip(combined - density_pen, 0.45, 0.90)), 3
    )

    officers   = recommend_officers(event, prediction, history)
    barricades = recommend_barricades(event, prediction, history)
    diversion  = suggest_diversion(event, prediction)
    clearance  = estimate_clearance_time(event, prediction, history)
    priority   = compute_resource_priority(event, prediction, history)

    confidence       = float(prediction["confidence"])
    escalation_notes = []

    if confidence < 0.60:
        escalation_notes.append("Prediction uncertainty is high; assign a supervisory review.")
    if priority["priority_label"] in ["CRITICAL", "HIGH"]:
        escalation_notes.append("Trigger immediate field deployment and continuous monitoring.")
    if diversion["diversion_required"]:
        escalation_notes.append("Activate diversion plan before congestion saturates the corridor.")
    if event.get("road_closure", 0):
        escalation_notes.append("Road closure detected; enforce barricade placement early.")

    similar_count = priority["historical_matches"]
    if similar_count > 0:
        escalation_notes.append(f"{similar_count} similar historical events shaped this recommendation.")
    else:
        escalation_notes.append("No strong historical match found; using conservative deployment logic.")

    if priority["priority_label"] in ["CRITICAL", "HIGH"]:
        action_steps = [
            "Dispatch traffic personnel immediately.",
            "Install barricades and cones at the affected corridor.",
            "Implement diversion before queue spillback grows.",
            "Notify nearby control points and supervisors.",
        ]
    elif priority["priority_label"] == "MODERATE":
        action_steps = [
            "Pre-position officers near the hotspot.",
            "Prepare barricades and diversion if conditions worsen.",
            "Monitor event progression at short intervals.",
        ]
    else:
        action_steps = [
            "Continue passive monitoring.",
            "Escalate only if congestion indicators rise.",
        ]

    plan = {
        "event_summary": {
            "severity"  : prediction["severity"],
            "risk"      : prediction["risk"],
            "confidence": prediction["confidence"],
        },
        "resource_plan": {
            "officers"      : officers["recommended_officers"],
            "barricades"    : barricades["recommended_barricades"],
            "cones"         : barricades["traffic_cones"],
            "warning_boards": barricades["warning_boards"],
        },
        "movement_plan": {
            "diversion_required": diversion["diversion_required"],
            "primary_route"     : diversion["primary_route"],
            "secondary_route"   : diversion["secondary_route"],
        },
        "time_plan": {
            "estimated_duration_minutes": clearance["estimated_duration_minutes"],
            "expected_clearance"        : clearance["expected_clearance"],
        },
        "priority": {
            "priority_score"    : priority["priority_score"],
            "priority_label"    : priority["priority_label"],
            "severity"          : prediction["severity"],
            "confidence"        : prediction["confidence"],
            "historical_matches": priority["historical_matches"],
            "estimated_clearance": clearance["expected_clearance"],
            "diversion_required": diversion["diversion_required"],
        },
        "action_steps"    : action_steps,
        "escalation_notes": escalation_notes,
    }

    # Persist — called once per event, after plan is fully built
    save_prediction(event["id"], prediction)
    save_recommendation(event["id"], plan)

    return plan


# ─────────────────────────────────────────────────────────────────────────────
# Cell 15 — Adaptive Strategy Simulator
# FIX 12 : save_simulation called before return, no dead code after return
# ─────────────────────────────────────────────────────────────────────────────

def simulate_response_strategies(event: dict):
    prediction  = predict_congestion(event)
    history     = find_similar_event(event)

    severity    = prediction["severity"]
    uncertainty = prediction["uncertainty"]

    base_officers   = recommend_officers(event, prediction, history)["recommended_officers"]
    base_barricades = recommend_barricades(event, prediction, history)["recommended_barricades"]
    base_factor     = severity / 5.0

    strategies = [
        {
            "strategy" : "Heavy Officer Deployment",
            "officers" : base_officers + 4,
            "barricades": base_barricades,
            "diversion": False,
            "expected_congestion_reduction": round(base_factor * 0.18 + (1 - uncertainty) * 0.05, 3),
        },
        {
            "strategy" : "Early Diversion",
            "officers" : base_officers,
            "barricades": base_barricades,
            "diversion": True,
            "expected_congestion_reduction": round(base_factor * 0.27 + (1 - uncertainty) * 0.07, 3),
        },
        {
            "strategy" : "Hybrid Response",
            "officers" : base_officers + 2,
            "barricades": base_barricades + 1,
            "diversion": True,
            "expected_congestion_reduction": round(base_factor * 0.34 + (1 - uncertainty) * 0.10, 3),
        },
    ]

    best = max(strategies, key=lambda x: x["expected_congestion_reduction"])

    simulation = {
        "predicted_severity"    : severity,
        "strategies_evaluated"  : len(strategies),
        "recommended_strategy"  : best,
        "all_strategies"        : strategies,
    }

    save_simulation(event["id"], simulation)

    return simulation


# ─────────────────────────────────────────────────────────────────────────────
# Cell 16 — initialize_ai_tables
# FIX 9  : loops over every event, calls prediction + simulation
# FIX 10 : single init function, prints progress, returns summary
# FIX 14 : called automatically from __main__
# ─────────────────────────────────────────────────────────────────────────────


def initialize_ai_tables():
    # Drop old tables with wrong schema (INTEGER id) so they recreate cleanly
    conn_local = sqlite3.connect(DATABASE_PATH)
    try:
        cur = conn_local.cursor()
        for tbl in ["predictions", "recommendations", "simulation_logs"]:
            try:
                col_type = cur.execute(
                    f"SELECT type FROM pragma_table_info('{tbl}') WHERE name='event_id'"
                ).fetchone()
                if col_type and col_type[0].upper() != "TEXT":
                    cur.execute(f"DROP TABLE IF EXISTS {tbl}")
                    print(f"  Dropped stale {tbl} (had INTEGER id, needs TEXT)")
            except:
                pass
        conn_local.commit()
    finally:
        conn_local.close()

    events_df = pd.read_sql("SELECT * FROM events", conn)
    total     = len(events_df)
    print(f"\nGenerating AI outputs for {total} events...")
    print("-" * 50)

    success  = 0
    failed   = 0
    errors   = {}
    interval = max(1, total // 10)

    for i, (_, row) in enumerate(events_df.iterrows(), 1):
        event = row.to_dict()

        # id is a string like FKID005760 — keep as-is
        if "id" not in event:
            for candidate in ["event_id", "ID", "Id"]:
                if candidate in event:
                    event["id"] = str(event[candidate])
                    break
            else:
                event["id"] = str(i)

        event["id"] = str(event["id"])

        # Fill missing feature columns
        for col in FEATURE_COLUMNS:
            if col not in event or event[col] is None:
                event[col] = 0.0
            try:
                event[col] = float(event[col])
            except (ValueError, TypeError):
                event[col] = 0.0

        event["event_cause"] = str(event.get("event_cause") or "others").strip().lower()
        event["cluster_id"]  = int(float(event.get("cluster_id") or -1))

        try:
            generate_action_plan(event)
            simulate_response_strategies(event)
            success += 1
        except Exception as e:
            failed  += 1
            err_key  = type(e).__name__
            errors[err_key] = errors.get(err_key, 0) + 1
            if failed <= 3:
                print(f"  ✗ event id={event.get('id')} : {e}")

        if i % interval == 0 or i == total:
            pct = int(i / total * 100)
            print(f"  [{pct:>3}%] {i}/{total}  ok={success}  fail={failed}")

    print("-" * 50)
    if errors:
        print("  Error types:", errors)

    summary = {"total": total, "success": success, "failed": failed, "tables": {}}
    for tbl in ["predictions", "recommendations", "simulation_logs"]:
        try:
            n = pd.read_sql(f"SELECT COUNT(*) as n FROM {tbl}", conn).iloc[0, 0]
            summary["tables"][tbl] = int(n)
            print(f"  ✓ {tbl:<25}: {n:>5} rows")
        except Exception as e:
            summary["tables"][tbl] = 0
            print(f"  ✗ {tbl}: {e}")

    print(f"\n{'✓' if failed == 0 else '⚠'} {success}/{total} events processed")
    return summary


if __name__ == "__main__":
    initialize_ai_tables()