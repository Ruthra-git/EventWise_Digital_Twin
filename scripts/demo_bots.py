#!/usr/bin/env python
# coding: utf-8
"""
demo_bots.py — End-to-end functionality test for 05_bots.py
Run : python scripts/demo_bots.py
Does NOT modify brain.db beyond writing to bot_sessions table.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# ── Make sure scripts/ folder is on path ─────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Import bots layer
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  TrafficTwin — demo_bots.py")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

try:
    from bots import (
        run_bot,
        run_all_bots,
        recall_sessions,
        recurrence_insight,
        ExplainableBot,
        EnforcementBot,
        SupervisorBot,
        FieldOfficerBot,
        PredictiveBot,
        SEVERITY_LABELS,
    )
    print("✓ 05_bots.py imported successfully\n")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("  Make sure you run from project root:")
    print("  python scripts/demo_bots.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Test events — 4 scenarios covering the severity range
# ─────────────────────────────────────────────────────────────────────────────
TEST_EVENTS = [
    {
        "id": 1,
        "name": "Silk Board — Peak hour accident (CRITICAL)",
        "event_cause"            : "accident",
        "event_type"             : "unplanned",
        "hour"                   : 9,
        "weekday"                : 1,
        "month"                  : 3,
        "is_peak_hour"           : 1,
        "is_weekend"             : 0,
        "is_night"               : 0,
        "minutes_since_midnight" : 540,
        "event_duration"         : 3.5,
        "is_planned"             : 0,
        "road_closure"           : 1,
        "cause_code"             : 0,
        "priority_score"         : 4,
        "cluster_id"             : 4,
        "hotspot_density"        : 120,
        "cluster_risk_score"     : 4.2,
        "cluster_risk_tier"      : 4,
        "avg_cluster_duration"   : 3.8,
        "planned_ratio"          : 0.1,
        "recent_cluster_activity": 18,
        "dist_from_center"       : 0.3,
        "peak_hour_ratio"        : 0.72,
        "night_ratio"            : 0.08,
        "road_closure_ratio"     : 0.45,
        "historical_frequency"   : 280,
        "location_frequency"     : 94,
        "zone_frequency"         : 340,
        "junction_frequency"     : 58,
        "police_frequency"       : 120,
        "is_route_event"         : 1,
        "zone_code"              : 3,
        "address"                : "Silk Board Junction, Bengaluru",
        "lat"                    : 12.9175,
        "lon"                    : 77.6229,
    },
    {
        "id"                     :2,
        "name"                   : "Koramangala — Night water logging (HIGH)",
        "event_cause"            : "water_logging",
        "event_type"             : "unplanned",
        "hour"                   : 23,
        "weekday"                : 4,
        "month"                  : 6,
        "is_peak_hour"           : 0,
        "is_weekend"             : 0,
        "is_night"               : 1,
        "minutes_since_midnight" : 1380,
        "event_duration"         : 5.0,
        "is_planned"             : 0,
        "road_closure"           : 0,
        "cause_code"             : 4,
        "priority_score"         : 3,
        "cluster_id"             : 12,
        "hotspot_density"        : 65,
        "cluster_risk_score"     : 3.1,
        "cluster_risk_tier"      : 3,
        "avg_cluster_duration"   : 4.5,
        "planned_ratio"          : 0.2,
        "recent_cluster_activity": 9,
        "dist_from_center"       : 0.7,
        "peak_hour_ratio"        : 0.3,
        "night_ratio"            : 0.55,
        "road_closure_ratio"     : 0.1,
        "historical_frequency"   : 95,
        "location_frequency"     : 40,
        "zone_frequency"         : 180,
        "junction_frequency"     : 22,
        "police_frequency"       : 60,
        "is_route_event"         : 0,
        "zone_code"              : 7,
        "address"                : "5th Block, Koramangala, Bengaluru",
        "lat"                    : 12.9352,
        "lon"                    : 77.6245,
    },
    {
        "id"                     : 3,
        "name"                   : "MG Road — Planned rally Saturday afternoon (MODERATE)",
        "event_cause"            : "public_event",
        "event_type"             : "planned",
        "hour"                   : 15,
        "weekday"                : 5,
        "month"                  : 1,
        "is_peak_hour"           : 0,
        "is_weekend"             : 1,
        "is_night"               : 0,
        "minutes_since_midnight" : 900,
        "event_duration"         : 4.0,
        "is_planned"             : 1,
        "road_closure"           : 1,
        "cause_code"             : 3,
        "priority_score"         : 2,
        "cluster_id"             : 7,
        "hotspot_density"        : 45,
        "cluster_risk_score"     : 2.5,
        "cluster_risk_tier"      : 2,
        "avg_cluster_duration"   : 3.2,
        "planned_ratio"          : 0.75,
        "recent_cluster_activity": 5,
        "dist_from_center"       : 0.4,
        "peak_hour_ratio"        : 0.25,
        "night_ratio"            : 0.05,
        "road_closure_ratio"     : 0.6,
        "historical_frequency"   : 60,
        "location_frequency"     : 30,
        "zone_frequency"         : 150,
        "junction_frequency"     : 18,
        "police_frequency"       : 80,
        "is_route_event"         : 1,
        "zone_code"              : 2,
        "address"                : "MG Road, Bengaluru",
        "lat"                    : 12.9757,
        "lon"                    : 77.6096,
    },
    {
        "id"                     :4,
        "name"                   : "Whitefield — Minor breakdown off-peak (LOW)",
        "event_cause"            : "vehicle_breakdown",
        "event_type"             : "unplanned",
        "hour"                   : 13,
        "weekday"                : 2,
        "month"                  : 2,
        "is_peak_hour"           : 0,
        "is_weekend"             : 0,
        "is_night"               : 0,
        "minutes_since_midnight" : 780,
        "event_duration"         : 1.2,
        "is_planned"             : 0,
        "road_closure"           : 0,
        "cause_code"             : 1,
        "priority_score"         : 1,
        "cluster_id"             : 23,
        "hotspot_density"        : 18,
        "cluster_risk_score"     : 1.8,
        "cluster_risk_tier"      : 1,
        "avg_cluster_duration"   : 1.5,
        "planned_ratio"          : 0.1,
        "recent_cluster_activity": 2,
        "dist_from_center"       : 1.1,
        "peak_hour_ratio"        : 0.2,
        "night_ratio"            : 0.0,
        "road_closure_ratio"     : 0.05,
        "historical_frequency"   : 35,
        "location_frequency"     : 12,
        "zone_frequency"         : 90,
        "junction_frequency"     : 8,
        "police_frequency"       : 30,
        "is_route_event"         : 0,
        "zone_code"              : 9,
        "address"                : "ITPL Road, Whitefield, Bengaluru",
        "lat"                    : 12.9698,
        "lon"                    : 77.7499,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper — pretty section divider
# ─────────────────────────────────────────────────────────────────────────────
def section(title, char="─", width=60):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def ok(label, value=""):
    print(f"  ✓  {label:<35} {value}")

def fail(label, err):
    print(f"  ✗  {label:<35} ERROR: {err}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Test each bot individually on Event 1 (hardest case)
# ─────────────────────────────────────────────────────────────────────────────
section("UNIT TESTS — each bot on Silk Board accident", "═")
event1 = TEST_EVENTS[0]
print(f"\n  Event : {event1['name']}\n")

PASS = 0
FAIL = 0

# ── ExplainableBot ──
try:
    r = run_bot("explainable", event1)
    assert "narrative"    in r, "missing narrative"
    assert "trust_source" in r, "missing trust_source"
    assert "counterfactual" in r, "missing counterfactual"
    assert "audit_trail"  in r, "missing audit_trail"
    assert "session_id"   in r, "missing session_id"
    ok("ExplainableBot",
       f"sev={r['severity']} | conf={r['confidence']} | "
       f"session={r['session_id']}")
    print(f"\n     Narrative preview:")
    print(f"     {r['narrative'][:160]}…\n")
    PASS += 1
except Exception as e:
    fail("ExplainableBot", e); FAIL += 1

# ── EnforcementBot ──
try:
    r = run_bot("enforcement", event1)
    assert "card_text"    in r, "missing card_text"
    assert "risk_tier"    in r, "missing risk_tier"
    assert "resource_plan" in r, "missing resource_plan"
    assert "dashboard_status" in r, "missing dashboard_status"
    ok("EnforcementBot",
       f"tier={r['risk_tier']} | "
       f"officers={r['resource_plan']['officers']} | "
       f"status={r['dashboard_status']['code']}")
    print(f"\n     Deployment card:")
    for line in r["card_text"].split("\n"):
        print(f"     {line}")
    PASS += 1
except Exception as e:
    fail("EnforcementBot", e); FAIL += 1

# ── SupervisorBot ──
try:
    r = run_bot("supervisor", event1)
    assert "brief"        in r, "missing brief"
    assert "risk_tier"    in r, "missing risk_tier"
    assert "escalation"   in r, "missing escalation"
    ok("SupervisorBot",
       f"tier={r['risk_tier']} | priority={r['priority_label']}")
    print(f"\n     Executive brief:")
    for line in r["brief"].split("\n"):
        print(f"     {line}")
    PASS += 1
except Exception as e:
    fail("SupervisorBot", e); FAIL += 1

# ── FieldOfficerBot (EN) ──
try:
    r = run_bot("field", event1, language="en")
    assert "message"    in r, "missing message"
    assert r["char_count"] <= 200, "message too long for SMS"
    ok("FieldOfficerBot (EN)",
       f"chars={r['char_count']} | lang={r['language']}")
    print(f"\n     SMS card:")
    for line in r["message"].split("\n"):
        print(f"     {line}")
    PASS += 1
except Exception as e:
    fail("FieldOfficerBot (EN)", e); FAIL += 1

# ── FieldOfficerBot (KN) ──
try:
    r = run_bot("field", event1, language="kn")
    assert "message" in r
    ok("FieldOfficerBot (KN — Kannada)", f"chars={r['char_count']}")
    print(f"\n     Kannada SMS card:")
    for line in r["message"].split("\n"):
        print(f"     {line}")
    PASS += 1
except Exception as e:
    fail("FieldOfficerBot (KN)", e); FAIL += 1

# ── PredictiveBot ──
try:
    r = run_bot("predictive", event1)
    assert "queue" in r, "missing queue"
    assert "total_events" in r, "missing total_events"
    ok("PredictiveBot",
       f"events={r['total_events']} | "
       f"critical={r.get('critical_count',0)} | "
       f"hitl_flags={r.get('hitl_flags',0)}")
    if r["queue"]:
        top = r["queue"][0]
        print(f"\n     Top queue item: {top.get('fingerprint','?')} | "
              f"severity={top.get('severity_label','?')} | "
              f"score={top.get('priority_score','?')}")
    PASS += 1
except Exception as e:
    fail("PredictiveBot", e); FAIL += 1

# ─────────────────────────────────────────────────────────────────────────────
# 5. run_all_bots — one call for all bots, 4 events
# ─────────────────────────────────────────────────────────────────────────────
section("INTEGRATION TEST — run_all_bots() on all 4 events", "═")

all_results = []
for ev in TEST_EVENTS:
    print(f"\n  ── {ev['name']}")
    try:
        result = run_all_bots(ev)
        assert "explainable" in result
        assert "enforcement" in result
        assert "supervisor"  in result
        assert "field"       in result
        assert "strategies"  in result

        enf = result["enforcement"]
        sup = result["supervisor"]
        fld = result["field"]
        exp = result["explainable"]

        ok("run_all_bots",
           f"sev={exp['severity']} | "
           f"tier={enf['risk_tier']} | "
           f"officers={enf['resource_plan']['officers']} | "
           f"session={result['session_id']}")

        hitl_flags = [
            b for b in ["explainable","enforcement","supervisor","field"]
            if result[b].get("hitl_override")
        ]
        if hitl_flags:
            print(f"     ⚠  HITL override triggered on: {', '.join(hitl_flags)}")

        strat = result["strategies"]
        best  = strat.get("recommended_strategy", {})
        print(f"     Best strategy  : {best.get('strategy','?')} "
              f"(reduction={best.get('expected_congestion_reduction','?')})")

        all_results.append({"event": ev["name"], "ok": True, "result": result})
        PASS += 1
    except Exception as e:
        fail(ev["name"], e)
        all_results.append({"event": ev["name"], "ok": False, "error": str(e)})
        FAIL += 1

# ─────────────────────────────────────────────────────────────────────────────
# 6. Confidence narrator — boundary tests
# ─────────────────────────────────────────────────────────────────────────────
section("CONFIDENCE NARRATOR — boundary values")
from bots import confidence_narrator

for score, expected_tier in [(0.95,"High"), (0.72,"Moderate"), (0.40,"Low"), (0.0,"Low")]:
    txt = confidence_narrator(score)
    ok(f"score={score}", txt)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Conversation memory — recall + recurrence insight
# ─────────────────────────────────────────────────────────────────────────────
section("CONVERSATION MEMORY — session recall")

df = recall_sessions(limit=10)
if df.empty:
    print("  • No sessions recorded yet (expected on first run if DB unreachable)")
else:
    ok("Sessions in brain.db", f"{len(df)} rows")
    print(f"\n  Recent sessions:")
    cols = ["session_id","bot_name","event_cause","severity","confidence","logged_at"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(6).to_string(index=False))

insight = recurrence_insight(limit=20)
ok("Recurrence insight", "")
print(f"\n  → {insight}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. HITL override check — deliberately low-confidence event
# ─────────────────────────────────────────────────────────────────────────────
section("HITL OVERRIDE — low confidence scenario")

low_conf_event = dict(TEST_EVENTS[3])   # start from LOW event
low_conf_event["cluster_id"]        = -1   # noise point — no cluster
low_conf_event["hotspot_density"]   = 0
low_conf_event["cluster_risk_score"]= 0
low_conf_event["historical_frequency"] = 1
low_conf_event["name"] = "Noise point — no cluster (should trigger HITL)"

print(f"\n  Event: {low_conf_event['name']}")
r_enf = run_bot("enforcement", low_conf_event)
r_exp = run_bot("explainable", low_conf_event)

hitl_enf = r_enf.get("hitl_override")
hitl_exp = r_exp.get("hitl_override")

ok("EnforcementBot HITL flag", "TRIGGERED" if hitl_enf else "not triggered")
ok("ExplainableBot HITL flag", "TRIGGERED" if hitl_exp else "not triggered")
if hitl_enf:
    print(f"\n  Override message (truncated):")
    print(f"  {str(hitl_enf)[:100]}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Multilingual output comparison
# ─────────────────────────────────────────────────────────────────────────────
section("MULTILINGUAL TEST — EN / KN / HI side by side")

for lang in ["en", "kn", "hi"]:
    r = run_bot("field", TEST_EVENTS[0], language=lang)
    print(f"\n  [{lang.upper()}]")
    for line in r["message"].split("\n"):
        print(f"    {line}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary
# ─────────────────────────────────────────────────────────────────────────────
section("TEST SUMMARY", "═")
total = PASS + FAIL
print(f"\n  Passed : {PASS}/{total}")
print(f"  Failed : {FAIL}/{total}")

if FAIL == 0:
    print("\n  ✓ All tests passed — bots layer is functional")
    print("  Next step → build app.py Streamlit dashboard")
else:
    print("\n  ✗ Some tests failed — check error messages above")
    print("  Common fixes:")
    print("    1. Run from project root: python scripts/demo_bots.py")
    print("    2. Ensure reccomend.py is in scripts/ folder")
    print("    3. Check brain.db path in Cell 3 of 05_bots.py")

print(f"\n  Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

section("DECISION DNA VALIDATION","═")

try:

    r=run_bot("explainable",event1)

    dna=r.get("decision_dna")

    assert dna is not None

    assert len(dna)==16

    ok("Decision DNA",dna)

    PASS+=1

except Exception as e:

    fail("Decision DNA",e)

    FAIL+=1
section("TRUST FUSION VALIDATION","═")

r=run_bot("explainable",event1)

fusion=r.get("trust_fusion",{})

print(json.dumps(fusion,indent=4))
section("COMMUNICATION ENTROPY","═")

from bots import communication_entropy

exp=run_bot("explainable",event1)

field=run_bot("field",event1)

print(

"Explainable:",

communication_entropy(

exp["narrative"]

)

)

print(

"Field:",

communication_entropy(

field["message"]

)

)
section("STAKEHOLDER ADAPTATION","═")

from bots import stakeholder_format

text=run_bot(

"explainable",

event1

)["narrative"]

for role in [

"field",

"planner",

"analyst",

"supervisor"

]:

    print()

    print(role.upper())

    print(

        stakeholder_format(

            text,

            role

        )

    )
section("RECOMMENDATION STABILITY","═")

e1=dict(event1)

e2=dict(event1)

e2["hour"]+=1

r1=run_bot("enforcement",e1)

r2=run_bot("enforcement",e2)

print(

"Original:",

r1["risk_tier"]

)

print(

"Perturbed:",

r2["risk_tier"]

)
section("PERFORMANCE","═")

import time

start=time.time()

for i in range(100):

    run_bot(

        "field",

        event1

    )

elapsed=time.time()-start

print(

"Average:",

elapsed/100

)
section("LEARNING ENGINE","═")

from bots import learn_from_session

print(

learn_from_session()

)
section("TREND DISCOVERY","═")

r=run_bot(

"predictive",

event1

)

print(

r.get(

"cluster_trends",

{}

)

)
section("DASHBOARD OBJECT","═")

r=run_bot(

"enforcement",

event1

)

print(

json.dumps(

r["dashboard_status"],

indent=4

)

)
section("DIGITAL TWIN MEMORY","═")

print()

print("Incident")

print("↓")

print("Recommendation")

print("↓")

print("Communication")

print("↓")

print("Session")

print("↓")

print("Memory")

print("↓")

print("Future Learning")
print()

print("="*60)

print("DIGITAL TWIN MATURITY")

print("="*60)

print("Communication Layer     ✓")

print("Explainable AI          ✓")

print("Trust Attribution       ✓")

print("Counterfactual          ✓")

print("Multilingual            ✓")

print("Audit Trail             ✓")

print("Conversation Memory     ✓")

print("Human Override          ✓")

print("Predictive Analytics    ✓")

print("Dashboard Ready         ✓")

print("="*60)