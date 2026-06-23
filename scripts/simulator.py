#!/usr/bin/env python
# coding: utf-8

# ============================================================
# CELL 1 — Configuration
# ============================================================

import sqlite3
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("Simulator")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "database" / "brain.db"

VERSION       = "1.0"
BATCH_LOG_INTERVAL = 1000

# Intervention impact weights (deterministic heuristics — no ML)
OFFICER_REDUCTION_PER_EXTRA   = 0.025   # each extra officer → -2.5% congestion
BARRICADE_REDUCTION_PER_EXTRA = 0.040   # each extra barricade → -4.0% congestion
DIVERSION_REDUCTION           = 0.120   # diversion enabled   → -12.0% congestion
MAX_REDUCTION                 = 0.60    # cap at 60% reduction

OFFICER_CONFIDENCE_BOOST      = 0.008   # per extra officer
BARRICADE_CONFIDENCE_BOOST    = 0.010
DIVERSION_CONFIDENCE_BOOST    = 0.040
MAX_CONFIDENCE_BOOST          = 0.25

CLEARANCE_REDUCTION_FACTOR    = 0.80    # clearance improves proportionally to congestion

logger.info("Simulation Engine Initialized | Version %s", VERSION)
logger.info("Database : %s", DATABASE_PATH)


# ============================================================
# CELL 2 — Load AI State
# ============================================================

def load_simulation_state():
    conn  = sqlite3.connect(DATABASE_PATH)
    state = {}
    for tbl, key in [
        ("events",         "events"),
        ("predictions",    "predictions"),
        ("recommendations","recommendations"),
    ]:
        try:
            state[key] = pd.read_sql(f"SELECT * FROM {tbl}", conn)
            logger.info("Loaded %d rows from %s.", len(state[key]), tbl)
        except Exception as e:
            logger.warning("%s not found: %s", tbl, e)
            state[key] = pd.DataFrame()
    conn.close()

    # Merge into one flat table so each row has event + prediction + recommendation
    base = state["events"].copy()

    if not state["predictions"].empty:
        base = base.merge(
            state["predictions"].add_suffix("_pred")
            .rename(columns={"event_id_pred": "id"}),
            on="id", how="left"
        )

    if not state["recommendations"].empty:
        base = base.merge(
            state["recommendations"].add_suffix("_rec")
            .rename(columns={"event_id_rec": "id"}),
            on="id", how="left"
        )

    # Resolve column name collisions from suffix merge
    def _col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    base["_severity"]    = pd.to_numeric(base.get(_col(base, ["severity_pred","severity"]) or "severity"), errors="coerce").fillna(2)
    base["_confidence"]  = pd.to_numeric(base.get(_col(base, ["confidence_pred","confidence"]) or "confidence"), errors="coerce").fillna(0.75)
    if "estimated_duration_minutes_pred" in base.columns:

        clearance = base["estimated_duration_minutes_pred"]

    elif "estimated_duration_minutes" in base.columns:

        clearance = base["estimated_duration_minutes"]

    else:

        clearance = pd.Series(

        base["_severity"] * 30,

        index=base.index

    )

    base["_clearance"] = pd.to_numeric(

    clearance,

    errors="coerce"

).fillna(

    base["_severity"] * 30

)
    base["_officers"]    = pd.to_numeric(base.get(_col(base, ["recommended_officers_rec","recommended_officers"]) or "recommended_officers"), errors="coerce").fillna(4)
    base["_barricades"]  = pd.to_numeric(base.get(_col(base, ["recommended_barricades_rec","recommended_barricades"]) or "recommended_barricades"), errors="coerce").fillna(2)
    base["_diversion"]   = pd.to_numeric(base.get(_col(base, ["diversion_required_rec","diversion_required"]) or "diversion_required"), errors="coerce").fillna(0)
    base["_priority"]    = pd.to_numeric(base.get(_col(base, ["priority_score_rec","priority_score"]) or "priority_score"), errors="coerce").fillna(50)

    # Normalise confidence to 0-1
    if base["_confidence"].max() > 1.0:
        base["_confidence"] = base["_confidence"] / 100.0

    state["merged"] = base.reset_index(drop=True)
    logger.info("Simulation state built: %d events ready.", len(base))
    return state

SIMULATION_STATE = load_simulation_state()


# ============================================================
# CELL 3 — What-if Engine
# Deterministic heuristic — no ML, no DB calls
# ============================================================
def apply_intervention(current_severity, current_confidence,
                       current_clearance, base_officers,
                       base_barricades, base_diversion,
                       extra_officers=0, extra_barricades=0,
                       enable_diversion=False):
    severity   = float(current_severity)
    confidence = float(current_confidence)
    clearance  = float(current_clearance)

    old_congestion = (severity / 5.0) * 100.0

    reduction  = 0.0
    reduction += extra_officers   * OFFICER_REDUCTION_PER_EXTRA
    reduction += extra_barricades * BARRICADE_REDUCTION_PER_EXTRA
    if enable_diversion and not base_diversion:
        reduction += DIVERSION_REDUCTION
    reduction = min(reduction, MAX_REDUCTION)

    new_congestion = round(old_congestion * (1.0 - reduction), 2)
    improvement    = round(reduction * 100, 2)

    # FIX 1 — cap confidence at 0.95, scale gain from improvement
    # ----------------------------------------------------
# Confidence calibration
# Keeps AI confidence realistic for decision support
# ----------------------------------------------------

    confidence_gain = improvement * 0.0015

    new_confidence = min(
    0.92,
    confidence + confidence_gain
)

    new_confidence = round(
    new_confidence,
    4
)

    # FIX 2 — clearance reduces proportional to improvement, floor 20 min
    # ----------------------------------------------------
# Clearance estimation
# Non-linear improvement model
# ----------------------------------------------------

    clearance_factor = 1 - (
    improvement / 120
)

    new_clearance = max(
    20,
    clearance * clearance_factor
)

    new_clearance = round(
    new_clearance,
    1
)

    return {
        "old_congestion"  : round(old_congestion, 2),
        "new_congestion"  : new_congestion,
        "improvement"     : improvement,
        "old_confidence"  : round(confidence * 100, 2),
        "new_confidence"  : round(new_confidence * 100, 2),
        "old_clearance"   : round(clearance, 1),
        "new_clearance"   : new_clearance,
    }


    # Compute confidence boost
    


# ============================================================
# CELL 4 — Strategy Selector
# Picks best strategy per event using 3 scenarios
# ============================================================

def select_best_strategy(row):
    s   = float(row["_severity"])
    c   = float(row["_confidence"])
    cl  = float(row["_clearance"])
    o   = float(row["_officers"])
    b   = float(row["_barricades"])
    div = int(row["_diversion"])

    scenarios = [
        {"strategy": "Heavy Officer Deployment",  "extra_officers": 4, "extra_barricades": 0, "enable_diversion": False},
        {"strategy": "Early Diversion",           "extra_officers": 0, "extra_barricades": 0, "enable_diversion": True},
        {"strategy": "Hybrid Response",           "extra_officers": 2, "extra_barricades": 1, "enable_diversion": True},
    ]

    best       = None
    best_score = -1
    best_scen  = scenarios[0]

    for scen in scenarios:
        result = apply_intervention(
            s, c, cl, o, b, div,
            extra_officers   = scen["extra_officers"],
            extra_barricades = scen["extra_barricades"],
            enable_diversion = scen["enable_diversion"]
        )
        score = result["improvement"] + (result["new_confidence"] - result["old_confidence"])
        if score > best_score:
            best_score = score
            best       = result
            best_scen  = scen

    priority = float(row["_priority"])

    if priority >= 90:

        strategy = "Heavy Officer Deployment"

    elif priority >= 70:

        strategy = "Targeted Diversion"

    elif priority >= 50:

        strategy = "Dynamic Signal Timing"

    else:

        strategy = "Routine Monitoring"

    return {

    "strategy": strategy,

    "officers": int(
        o + best_scen["extra_officers"]
    ),

    "barricades": int(
        b + best_scen["extra_barricades"]
    ),

    "diversion": int(
        best_scen["enable_diversion"]
    ),

    "expected_reduction": round(
        best["improvement"],
        2
    ),

    **best

}


# ============================================================
# CELL 5 — Save simulation_logs
# Schema matches 06_heatmap.py Cell 11 exactly
# ============================================================

def save_simulation_logs(records: list):
    conn = sqlite3.connect(DATABASE_PATH)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS simulation_logs")
    cur.execute("""
        CREATE TABLE simulation_logs (
            event_id          TEXT PRIMARY KEY,
            strategy          TEXT,
            officers          INTEGER,
            barricades        INTEGER,
            diversion         INTEGER,
            expected_reduction REAL,
            old_congestion    REAL,
            new_congestion    REAL,
            improvement       REAL,
            old_confidence    REAL,
            new_confidence    REAL,
            old_clearance     REAL,
            new_clearance     REAL,
            created_time      TEXT
        )
    """)
    cur.executemany("""
        INSERT OR REPLACE INTO simulation_logs VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        (
            str(r["event_id"]),
            str(r["strategy"]),
            int(r["officers"]),
            int(r["barricades"]),
            int(r["diversion"]),
            float(r["expected_reduction"]),
            float(r["old_congestion"]),
            float(r["new_congestion"]),
            float(r["improvement"]),
            float(r["old_confidence"]),
            float(r["new_confidence"]),
            float(r["old_clearance"]),
            float(r["new_clearance"]),
            str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        for r in records
    ])
    conn.commit()
    conn.close()
    logger.info("Saved %d rows to simulation_logs.", len(records))


# ============================================================
# CELL 6 — Batch Simulation
# ============================================================

def run_batch_simulation():
    merged  = SIMULATION_STATE["merged"]
    total   = len(merged)
    records = []
    failed  = 0

    logger.info("Starting batch simulation for %d events...", total)

    for i, (_, row) in enumerate(merged.iterrows(), 1):
        try:
            result = select_best_strategy(row)
            if i <= 5:
               logger.info(
        "Event=%s Improvement=%.2f Strategy=%s",
        row["id"],
        result["improvement"],
        result["strategy"]
    )
            result["event_id"] = str(row["id"])
            records.append(result)
        except Exception as e:
            failed += 1
            if failed <= 3:
                logger.warning("Row %d failed: %s", i, e)

        if i % BATCH_LOG_INTERVAL == 0 or i == total:
            logger.info("[%3d%%] %d/%d  ok=%d  fail=%d",
                        int(i/total*100), i, total, len(records), failed)

    save_simulation_logs(records)
    return records

SIMULATION_RECORDS = run_batch_simulation()


# ============================================================
# CELL 7 — Summary Dashboard
# ============================================================

def print_simulation_summary(records):
    if not records:
        print("No simulation records.")
        return

    improvements  = [r["improvement"]   for r in records]
    confidences   = [r["new_confidence"] for r in records]
    clearances    = [r["new_clearance"]  for r in records]
    strategies    = [r["strategy"]       for r in records]

    from collections import Counter
    top_strategy = Counter(strategies).most_common(1)[0][0]

    print()
    print("=" * 50)
    print("  SIMULATION SUMMARY")
    print("=" * 50)
    print(f"  Events Simulated    : {len(records)}")
    print(f"  Average Improvement : {round(np.mean(improvements),2)} %")
    print(f"  Best Improvement    : {round(max(improvements),2)} %")
    print(f"  Average Confidence  : {round(np.mean(confidences),2)} %")
    print(f"  Average Clearance   : {round(np.mean(clearances),1)} min")
    print(f"  Top Strategy        : {top_strategy}")
    print("=" * 50)

print_simulation_summary(SIMULATION_RECORDS)


# ============================================================
# CELL 8 — Export Confirmation for Digital Twin
# ============================================================

def verify_export():
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        n = pd.read_sql("SELECT COUNT(*) as n FROM simulation_logs", conn).iloc[0,0]
        conn.close()
        print()
        print("=" * 50)
        print(f"  ✓ Simulation Completed")
        print(f"  ✓ simulation_logs updated : {n} rows")
        print(f"  ✓ Ready for 06_heatmap.py Cell 11")
        print(f"  ✓ Ready for Digital Twin Visualization")
        print("=" * 50)
    except Exception as e:
        conn.close()
        print(f"  ✗ Export verification failed: {e}")

verify_export()

if __name__ == "__main__":
    pass
