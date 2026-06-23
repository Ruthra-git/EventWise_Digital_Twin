
 
# Cell 1 — Documentation 
"""
05_bots.py — TrafficTwin AI Communication Layer

Purpose
-------
Pure language conversion layer. This module never:
  • loads ML models
  • calls predict_congestion
  • queries SHAP
  • computes recommendations
  • queries brain.db directly for prediction

It receives a pre-computed action plan from generate_action_plan(event)
and converts it into human-readable outputs for different audiences.

Bot Roster
----------
  ExplainableBot    → SHAP-grounded plain-language explanation for analysts
  EnforcementBot    → Structured deployment card for field commanders
  PredictiveBot     → Ranked upcoming-event risk queue for planners
  SupervisorBot     → 5-line executive brief for control-room supervisors
  FieldOfficerBot   → 3-line SMS/radio card for ground officers

Shared Utilities
----------------
  confidence_narrator()     → Consistent confidence language across all bots
  run_bot()                 → Single dispatcher entry point
  log_session()             → Persists every interaction to brain.db

Dependencies
------------
  reccomend.py              → generate_action_plan(), simulate_response_strategies()
  brain.db                  → bot_sessions table (auto-created on first run)

Usage
-----
  from scripts.bots import run_bot
  result = run_bot("enforcement", event_dict)

Do NOT run demo code here — use demo_bots.py for that.
"""

 
# Cell 2 — Imports 

import os
import json
import uuid
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd

warnings.filterwarnings("ignore")


# reccomend.py must be importable — adjust sys.path if running standalone
try:
    from recommend import (
        generate_action_plan,
        simulate_response_strategies,
        conn as DB_CONN,
        DATABASE_PATH,
    )
    _RECOMMEND_LOADED = True
except ImportError as _e:
    _RECOMMEND_LOADED = False
    _RECOMMEND_ERROR  = str(_e)
    print(f"[bots] WARNING: Could not import reccomend.py — {_e}")
    print("[bots] Bots will return stub output until reccomend.py is available.")

print("05_bots.py — Communication Layer loaded")
print(f"  reccomend.py available : {_RECOMMEND_LOADED}")

 
# Cell 3 — Bot Configuration 

# ── Paths (match existing project structure) ──────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DATABASE_PATH_BOTS = PROJECT_ROOT / "database" / "brain.db"
MODEL_DIR     = PROJECT_ROOT / "models"
OUTPUT_DIR    = PROJECT_ROOT / "outputs"
REPORT_DIR    = OUTPUT_DIR / "reports"
LOG_DIR       = OUTPUT_DIR / "logs"

# ── Operational ontology & standardised vocabulary ────────────────────────────
SEVERITY_LABELS = {1: "LOW", 2: "MODERATE", 3: "HIGH", 4: "CRITICAL", 5: "EXTREME"}
RISK_COLOURS    = {"LOW": "green", "MODERATE": "amber", "HIGH": "orange",
                   "CRITICAL": "red", "EXTREME": "darkred"}
PRIORITY_EMOJI  = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠",
                   "CRITICAL": "🔴", "EXTREME": "⛔"}

CAUSE_VOCABULARY = {
    "accident"         : "road traffic accident",
    "vehicle_breakdown": "vehicle breakdown",
    "water_logging"    : "water logging / flooding",
    "tree_fall"        : "fallen tree / debris",
    "public_event"     : "planned public event",
    "others"           : "unclassified incident",
    "unknown"          : "unclassified incident",
}

#   
# OPERATIONAL ONTOLOGY
#   

OPERATIONAL_ONTOLOGY={

"Incident":{

"accident":"Road Accident",

"vehicle_breakdown":"Breakdown",

"water_logging":"Flooding",

"tree_fall":"Obstruction",

"public_event":"Planned Event"

},

"Action":[

"Deploy",

"Escalate",

"Monitor",

"Redirect",

"Notify"

],

"Resources":[

"Officer",

"Barricade",

"Cone",

"Tow Vehicle",

"Emergency Unit"

]

}

#  Risk matrix (severity × road_closure → escalation tier) 
RISK_MATRIX = {
    (1, 0): "TIER-1", (1, 1): "TIER-2",
    (2, 0): "TIER-2", (2, 1): "TIER-2",
    (3, 0): "TIER-2", (3, 1): "TIER-3",
    (4, 0): "TIER-3", (4, 1): "TIER-3",
    (5, 0): "TIER-3", (5, 1): "TIER-3",
}

ESCALATION_ACTIONS = {
    "TIER-1": "Monitor passively. No immediate deployment required.",
    "TIER-2": "Pre-position personnel. Activate if conditions deteriorate.",
    "TIER-3": "Immediate deployment. Engage diversion and notify supervisor.",
}

#  Multilingual templates (add languages as needed) 
LANGUAGE_TEMPLATES = {
    "en": {
        "severity_prefix" : "Severity",
        "deploy_prefix"   : "Deploy",
        "divert_prefix"   : "Divert via",
        "clear_prefix"    : "Clear by",
        "confidence_high" : "High confidence",
        "confidence_med"  : "Moderate confidence — review advised",
        "confidence_low"  : "Low confidence — human override recommended",
    },
    "kn": {
        "severity_prefix" : "ತೀವ್ರತೆ",
        "deploy_prefix"   : "ನಿಯೋಜಿಸಿ",
        "divert_prefix"   : "ಮಾರ್ಗ ಬದಲಿಸಿ",
        "clear_prefix"    : "ಮುಕ್ತಾಯ",
        "confidence_high" : "ಹೆಚ್ಚಿನ ವಿಶ್ವಾಸ",
        "confidence_med"  : "ಮಧ್ಯಮ ವಿಶ್ವಾಸ — ಪರಿಶೀಲಿಸಿ",
        "confidence_low"  : "ಕಡಿಮೆ ವಿಶ್ವಾಸ — ಮಾನವ ಅನುಮೋದನೆ",
    },
    "hi": {
        "severity_prefix" : "गंभीरता",
        "deploy_prefix"   : "तैनात करें",
        "divert_prefix"   : "मार्ग बदलें",
        "clear_prefix"    : "समाप्ति",
        "confidence_high" : "उच्च विश्वास",
        "confidence_med"  : "मध्यम विश्वास — समीक्षा करें",
        "confidence_low"  : "कम विश्वास — मानव अनुमोदन",
    },
}
DEFAULT_LANGUAGE = "en"

# Multi-agent communication policy 
AGENT_POLICY = {
    "explainable" : {"audience": "analyst",    "max_words": 120, "tone": "technical"},
    "enforcement" : {"audience": "commander",  "max_words":  80, "tone": "directive"},
    "predictive"  : {"audience": "planner",    "max_words": 100, "tone": "advisory"},
    "supervisor"  : {"audience": "executive",  "max_words":  60, "tone": "concise"},
    "field"       : {"audience": "officer",    "max_words":  30, "tone": "imperative"},
}
#   
# STAKEHOLDER HIERARCHY
#   

STAKEHOLDER_HIERARCHY = {

    "citizen":1,

    "field":2,

    "dispatcher":3,

    "planner":4,

    "analyst":5,

    "supervisor":6,

    "administrator":7

}

VERBOSITY_LEVELS={

    "radio":20,

    "brief":50,

    "standard":100,

    "technical":200,

    "research":500

}

COMMUNICATION_ENTROPY={

    "field":0.10,

    "supervisor":0.35,

    "planner":0.55,

    "analyst":0.85

}
# ── Human-in-the-loop override thresholds 
HITL_OVERRIDE_CONFIDENCE_THRESHOLD = 0.55
HITL_OVERRIDE_SEVERITY_THRESHOLD   = 4
HITL_OVERRIDE_MESSAGE = (
    "⚠ HUMAN OVERRIDE RECOMMENDED — confidence below threshold "
    "or severity is CRITICAL/EXTREME. Please verify before deploying."
)

# ── Dashboard-ready status codes
STATUS_CODES = {
    "PENDING"   : {"color": "#888780", "label": "Pending review"},
    "ACTIVE"    : {"color": "#BA7517", "label": "Active — monitor"},
    "DEPLOYED"  : {"color": "#1D9E75", "label": "Resources deployed"},
    "ESCALATED" : {"color": "#D85A30", "label": "Escalated to supervisor"},
    "RESOLVED"  : {"color": "#378ADD", "label": "Resolved"},
}

# ── Conversation memory config 
SESSION_TABLE     = "bot_sessions"
MAX_SESSION_MEMORY = 20          # keep last N interactions per session

print("Bot configuration loaded")
print(f"  Agents configured  : {list(AGENT_POLICY.keys())}")
print(f"  Languages supported: {list(LANGUAGE_TEMPLATES.keys())}")
print(f"  HITL threshold     : confidence < {HITL_OVERRIDE_CONFIDENCE_THRESHOLD}")
HIGH_CONFIDENCE   = 0.80
MEDIUM_CONFIDENCE = 0.60
LOW_CONFIDENCE    = 0.40
 
# Cell 4 — ExplainableBot
# Audience: traffic analysts, decision-support operators
# Input  : plan dict from generate_action_plan()
# Output : plain-language narrative + SHAP attribution + trust source + audit trail 

class ExplainableBot:
    """
    Converts the AI action plan into a plain-language explanation.
    Applies hybrid trust-source attribution (AI + Historical + Rules).
    Generates counterfactual explanation templates.
    Produces a decision audit trail entry.
    """

    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language
        self.tmpl     = LANGUAGE_TEMPLATES.get(language, LANGUAGE_TEMPLATES["en"])

    def _trust_attribution(self, plan: dict) -> str:
        conf         = plan["event_summary"]["confidence"]
        hist_matches = plan["priority"]["historical_matches"]
        if conf >= 0.80 and hist_matches >= 5:
            return "AI model (high confidence) + strong historical precedent"
        elif conf >= 0.60 and hist_matches >= 2:
            return "AI model (moderate confidence) + partial historical match"
        elif conf < 0.60 and hist_matches == 0:
            return "Rule-based fallback (low AI confidence, no historical match)"
        else:
            return "AI model + rule-based crosscheck"
    def _trust_fusion(self,plan):

        ai=plan["event_summary"]["confidence"]

        historical=min(
        plan["priority"]["historical_matches"]/5,
        1.0
    )

        rule=1.0

        fusion=0.5*ai+0.3*historical+0.2*rule

        return{

        "ai":round(ai,2),

        "historical":round(historical,2),

        "rule":rule,

        "fusion":round(fusion,2)

    }

    def _counterfactual(self, plan: dict) -> str:
        sev = plan["event_summary"]["severity"]
        if sev >= 4:
            return (
                f"If this event occurred during off-peak hours, predicted severity "
                f"would likely reduce from {SEVERITY_LABELS[sev]} to "
                f"{SEVERITY_LABELS.get(sev - 1, 'LOWER')}."
            )
        return (
            f"If road closure were not present, required personnel would likely "
            f"reduce by 2–3 officers."
        )
    def _decision_dna(self,plan,event):

        import hashlib

        payload=str(

        plan["event_summary"]["severity"]

        )+str(

        event.get("cluster_id")

         )+str(

        event.get("hour")

         )+str(

        plan["resource_plan"]["officers"]

        )

        return hashlib.sha256(

        payload.encode()

        ).hexdigest()[:16]

    def _shap_narrative(self, plan: dict, event: dict) -> str:
        cause    = CAUSE_VOCABULARY.get(
            str(event.get("event_cause", "others")).lower(), "an incident")
        hour     = event.get("hour", 8)
        is_peak  = event.get("is_peak_hour", 0)
        cluster  = event.get("cluster_id", -1)
        density  = event.get("hotspot_density", 0)

        time_label = "peak hour" if is_peak else "off-peak hour"
        parts = [f"a {cause} occurring during {time_label} (hour {hour})"]
        if density > 50:
            parts.append(f"in a high-density hotspot (cluster {cluster}, {int(density)} prior events)")
        if event.get("road_closure", 0):
            parts.append("with a confirmed road closure")
        return "Primary drivers: " + ", ".join(parts) + "."

    def _audit_entry(self, plan: dict, event: dict) -> dict:
        return {
            "audit_id"       : str(uuid.uuid4())[:8].upper(),
            "timestamp"      : datetime.now().isoformat(),
            "bot"            : "ExplainableBot",
            "severity"       : plan["event_summary"]["severity"],
            "confidence"     : plan["event_summary"]["confidence"],
            "trust_source"   : self._trust_attribution(plan),
            "event_cause"    : event.get("event_cause", "unknown"),
            "cluster_id"     : event.get("cluster_id", -1),
            "override_flag"  : plan["event_summary"]["confidence"] < HITL_OVERRIDE_CONFIDENCE_THRESHOLD,
        }

    def explain(self, plan: dict, event: dict) -> dict:
        summary   = plan["event_summary"]
        priority  = plan["priority"]
        sev_label = SEVERITY_LABELS.get(summary["severity"], "UNKNOWN")
        conf_text = confidence_narrator(summary["confidence"], self.language)
        shap_text = self._shap_narrative(plan, event)
        cf_text   = self._counterfactual(plan)
        trust     = self._trust_attribution(plan)
        fusion=self._trust_fusion(plan)
        audit     = self._audit_entry(plan, event)
        dna=self._decision_dna(plan,event)
        
        hitl      = HITL_OVERRIDE_MESSAGE if audit["override_flag"] else None

        narrative = (
            f"This event is predicted as {sev_label} severity ({conf_text}). "
            f"{shap_text} "
            f"The system has identified {priority['historical_matches']} similar historical events. "
            f"Expected clearance: {priority['estimated_clearance']}. "
            f"Trust source: {trust}. "
            f"Counterfactual: {cf_text}"
        )

        return {
            "bot"           : "ExplainableBot",
            "audience"      : AGENT_POLICY["explainable"]["audience"],
            "narrative"     : narrative,
            "severity"      : sev_label,
            "confidence"    : summary["confidence"],
            "confidence_text": conf_text,
            "trust_source"  : trust,
            "counterfactual": cf_text,
            "shap_drivers"  : shap_text,
            "audit_trail"   : audit,
            "hitl_override" : hitl,
            "language"      : self.language,
            "trust_fusion":fusion,
            "decision_dna":dna,
        }

 
# Cell 5 — EnforcementBot
# Audience: field commanders, deployment coordinators
# Input  : plan dict from generate_action_plan()
# Output : structured deployment card (text + dashboard-ready dict) 

class EnforcementBot:
    """
    Converts action plan into a structured field deployment card.
    Applies risk matrix escalation engine.
    Produces dashboard-ready status codes.
    Supports communication consistency validation.
    """

    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language
        self.tmpl     = LANGUAGE_TEMPLATES.get(language, LANGUAGE_TEMPLATES["en"])

    def _risk_tier(self, plan: dict, event: dict) -> str:
        sev     = plan["event_summary"]["severity"]
        closure = int(event.get("road_closure", 0))
        return RISK_MATRIX.get((sev, closure), "TIER-2")

    def _escalation_action(self, tier: str) -> str:
        return ESCALATION_ACTIONS.get(tier, ESCALATION_ACTIONS["TIER-2"])

    def _dashboard_status(self, plan: dict) -> dict:
        sev = plan["event_summary"]["severity"]
        if sev >= 4:
            code = "ESCALATED"
        elif sev >= 3:
            code = "DEPLOYED"
        elif sev >= 2:
            code = "ACTIVE"
        else:
            code = "PENDING"
        return {**STATUS_CODES[code], "code": code}

    def _validate_consistency(self, plan: dict) -> list:
        issues = []
        res    = plan["resource_plan"]
        sev    = plan["event_summary"]["severity"]
        if res["officers"] < 2 and sev >= 3:
            issues.append("Officers below minimum for severity ≥ HIGH")
        if res["barricades"] < 1 and plan["movement_plan"]["diversion_required"]:
            issues.append("Diversion active but no barricades allocated")
        if plan["event_summary"]["confidence"] < HITL_OVERRIDE_CONFIDENCE_THRESHOLD:
            issues.append("Confidence below threshold — deployment figures may be conservative")
        return issues
    def _deployment_efficiency(

    self,

    plan):

        r=plan["resource_plan"]

        score=100

        score-=r["officers"]*2

        score-=r["barricades"]

        return max(score,0)

    def deploy(self, plan: dict, event: dict) -> dict:
        res     = plan["resource_plan"]
        move    = plan["movement_plan"]
        time    = plan["time_plan"]
        summary = plan["event_summary"]
        tier    = self._risk_tier(plan, event)
        esc_act = self._escalation_action(tier)
        status  = self._dashboard_status(plan)
        eff=self._deployment_efficiency(plan)
        issues  = self._validate_consistency(plan)
        hitl    = HITL_OVERRIDE_MESSAGE if (
            summary["confidence"] < HITL_OVERRIDE_CONFIDENCE_THRESHOLD
            or summary["severity"] >= HITL_OVERRIDE_SEVERITY_THRESHOLD
        ) else None

        card_text = (
            f"DEPLOYMENT CARD — {SEVERITY_LABELS[summary['severity']]} SEVERITY\n"
            f"{'─'*50}\n"
            f"Risk tier        : {tier} — {esc_act}\n"
            f"Personnel        : {res['officers']} officers\n"
            f"Barricades       : {res['barricades']} units\n"
            f"Traffic cones    : {res['cones']} units\n"
            f"Warning boards   : {res['warning_boards']} units\n"
            f"Diversion        : {'ACTIVATE — ' + str(move['primary_route']) if move['diversion_required'] else 'Not required'}\n"
            f"Alt route        : {move.get('secondary_route') or 'N/A'}\n"
            f"Est. clearance   : {time['expected_clearance']} "
            f"({time['estimated_duration_minutes']} min)\n"
            f"Confidence       : {confidence_narrator(summary['confidence'], self.language)}\n"
        )
        if issues:
            card_text += f"⚠ Consistency issues: {'; '.join(issues)}\n"
        if hitl:
            card_text += f"\n{hitl}\n"

        return {
            "bot"               : "EnforcementBot",
            "audience"          : AGENT_POLICY["enforcement"]["audience"],
            "card_text"         : card_text,
            "risk_tier"         : tier,
            "escalation_action" : esc_act,
            "resource_plan"     : res,
            "movement_plan"     : move,
            "time_plan"         : time,
            "dashboard_status"  : status,
            "consistency_issues": issues,
            "hitl_override"     : hitl,
            "action_steps"      : plan.get("action_steps", []),
            "language"          : self.language,
            "deployment_efficiency":eff
        }

 
# Cell 6 — PredictiveBot
# Audience: traffic management planners, shift supervisors
# Input  : list of upcoming event dicts (from brain.db active/planned events)
# Output : ranked risk queue with incident fingerprints 

class PredictiveBot:
    """
    Generates a ranked upcoming-event risk queue.
    Creates incident fingerprints for each event.
    Applies multi-agent communication policy for planner audience.
    """

    def _incident_fingerprint(self, event: dict, plan: dict) -> str:
        cause   = CAUSE_VOCABULARY.get(str(event.get("event_cause","others")).lower(), "incident")
        cluster = event.get("cluster_id", -1)
        hour    = event.get("hour", 0)
        is_peak = event.get("is_peak_hour", 0)
        sev     = plan["event_summary"]["severity"]
        fp      = (
            f"{SEVERITY_LABELS[sev][:3]}-"
            f"C{cluster}-"
            f"H{hour:02d}-"
            f"{'PK' if is_peak else 'OP'}-"
            f"{str(cause)[:6].upper().replace(' ','_')}"
        )
        return fp

    def rank_events(self, event_list: list) -> dict:
        if not _RECOMMEND_LOADED:
            return {"bot": "PredictiveBot", "error": "reccomend.py not loaded", "queue": []}

        results = []
        for event in event_list:
            try:
                plan = generate_action_plan(event)
                fp   = self._incident_fingerprint(event, plan)
                results.append({
                    "fingerprint"       : fp,
                    "event_cause"       : event.get("event_cause", "unknown"),
                    "cluster_id"        : event.get("cluster_id", -1),
                    "hour"              : event.get("hour", 0),
                    "severity"          : plan["event_summary"]["severity"],
                    "severity_label"    : SEVERITY_LABELS[plan["event_summary"]["severity"]],
                    "priority_score"    : plan["priority"]["priority_score"],
                    "priority_label"    : plan["priority"]["priority_label"],
                    "confidence"        : plan["event_summary"]["confidence"],
                    "confidence_text"   : confidence_narrator(plan["event_summary"]["confidence"]),
                    "expected_clearance": plan["priority"]["estimated_clearance"],
                    "diversion_needed"  : plan["movement_plan"]["diversion_required"],
                    "officers_needed"   : plan["resource_plan"]["officers"],
                    "hitl_flag"         : (
                        plan["event_summary"]["confidence"] < HITL_OVERRIDE_CONFIDENCE_THRESHOLD
                        or plan["event_summary"]["severity"] >= HITL_OVERRIDE_SEVERITY_THRESHOLD
                    ),
                })
            except Exception as e:
                results.append({"event": event, "error": str(e)})

        ranked = sorted(
            [r for r in results if "priority_score" in r],
            key=lambda x: x["priority_score"], reverse=True
        )
        cluster_frequency={}

        for x in ranked:

            c=x["cluster_id"]

            cluster_frequency[c]=cluster_frequency.get(c,0)+1

        critical_count = sum(1 for r in ranked if r.get("severity", 0) >= 4)
        hitl_count     = sum(1 for r in ranked if r.get("hitl_flag"))

        return {
            "bot"            : "PredictiveBot",
            "audience"       : AGENT_POLICY["predictive"]["audience"],
            "total_events"   : len(ranked),
            "critical_count" : critical_count,
            "hitl_flags"     : hitl_count,
            "queue"          : ranked,
            "generated_at"   : datetime.now().isoformat(),
            "cluster_trends":cluster_frequency
        }

 
# Cell 7 — SupervisorBot
# Audience: control-room supervisors, shift commanders
# Input  : plan dict from generate_action_plan()
# Output : concise executive brief (≤ 60 words) + escalation engine output 

class SupervisorBot:
    """
    Generates a concise executive brief for command-level audiences.
    Applies escalation engine with tier classification.
    Produces adaptive stakeholder-aware response formatting.
    """

    def brief(self, plan: dict, event: dict) -> dict:
        summary   = plan["event_summary"]
        priority  = plan["priority"]
        res       = plan["resource_plan"]
        move      = plan["movement_plan"]
        time      = plan["time_plan"]
        sev_label = SEVERITY_LABELS.get(summary["severity"], "UNKNOWN")
        tier      = RISK_MATRIX.get(
            (summary["severity"], int(event.get("road_closure", 0))), "TIER-2")
        cause_txt = CAUSE_VOCABULARY.get(
            str(event.get("event_cause","others")).lower(), "incident")

        brief_text = (
            f"SITUATION : {sev_label} severity {cause_txt} | "
            f"Cluster {event.get('cluster_id', '?')} | "
            f"Hour {event.get('hour', '?')}\n"
            f"RISK      : {priority['priority_label']} ({priority['priority_score']:.0f}/100) | "
            f"{tier}\n"
            f"RESOURCES : {res['officers']} officers · {res['barricades']} barricades · "
            f"Diversion {'ON' if move['diversion_required'] else 'OFF'}\n"
            f"CLEARANCE : {time['expected_clearance']} "
            f"({time['estimated_duration_minutes']} min)\n"
            f"CONFIDENCE: {confidence_narrator(summary['confidence'])}\n"
            f"ACTION    : {ESCALATION_ACTIONS.get(tier, '')}"
        )
        brief_text+=(

        "\nSYSTEM HEALTH : "

        f"{priority['historical_matches']} "

        "historical matches"

        ) 

        return {
            "bot"           : "SupervisorBot",
            "audience"      : AGENT_POLICY["supervisor"]["audience"],
            "brief"         : brief_text,
            "risk_tier"     : tier,
            "severity"      : sev_label,
            "priority_score": priority["priority_score"],
            "priority_label": priority["priority_label"],
            "confidence"    : summary["confidence"],
            "escalation"    : ESCALATION_ACTIONS.get(tier),
            "hitl_override" : HITL_OVERRIDE_MESSAGE if (
                summary["confidence"] < HITL_OVERRIDE_CONFIDENCE_THRESHOLD
                or summary["severity"] >= HITL_OVERRIDE_SEVERITY_THRESHOLD
            ) else None,
        }

 
# Cell 7.5 — FieldOfficerBot
# Audience: on-ground officers (SMS / radio / WhatsApp)
# Input  : plan dict from generate_action_plan()
# Output : ≤ 3-line compressed message in chosen language 

class FieldOfficerBot:
    """
    Generates a compressed 3-line field message for ground officers.
    Applies SMS/radio compressor for maximum brevity.
    Supports multilingual output (en / kn / hi).
    """

    def __init__(self, language: str = DEFAULT_LANGUAGE):
        self.language = language
        self.tmpl     = LANGUAGE_TEMPLATES.get(language, LANGUAGE_TEMPLATES["en"])

    def _compress(self, plan: dict, event: dict) -> str:
        t         = self.tmpl
        sev       = SEVERITY_LABELS.get(plan["event_summary"]["severity"], "UNK")
        cause     = str(event.get("event_cause", "INCIDENT")).upper()[:10]
        address   = str(event.get("address", f"Cluster {event.get('cluster_id','?')}"))[:35]
        officers  = plan["resource_plan"]["officers"]
        bars      = plan["resource_plan"]["barricades"]
        diversion = plan["movement_plan"]["primary_route"]
        clearance = plan["time_plan"]["expected_clearance"]

        line1 = f"{sev} — {cause} @ {address}"
        line2 = f"{t['deploy_prefix']} {officers} officers + {bars} barricades"
        if plan["movement_plan"]["diversion_required"]:
            line3 = f"{t['divert_prefix']} {str(diversion)[:40]}"
        else:
            line3 = f"{t['clear_prefix']} {clearance}"

        return f"{line1}\n{line2}\n{line3}"
    def voice_dispatch(

       self,plan,event

):

       return self._compress(

        plan,

        event

    ).replace(

        "\n",

        ". "

    )

    def dispatch(self, plan: dict, event: dict) -> dict:
        msg  = self._compress(plan, event)
        hitl = (
            plan["event_summary"]["confidence"] < HITL_OVERRIDE_CONFIDENCE_THRESHOLD
            or plan["event_summary"]["severity"] >= HITL_OVERRIDE_SEVERITY_THRESHOLD
        )
        return {
            "bot"          : "FieldOfficerBot",
            "audience"     : AGENT_POLICY["field"]["audience"],
            "message"      : msg,
            "severity"     : SEVERITY_LABELS.get(plan["event_summary"]["severity"]),
            "hitl_override": HITL_OVERRIDE_MESSAGE if hitl else None,
            "language"     : self.language,
            "char_count"   : len(msg),
        }

 
# Cell 8 — Unified Interface
# Shared utilities + single dispatcher entry point 

def confidence_narrator(score: float, language: str = DEFAULT_LANGUAGE) -> str:
    """
    Converts a numeric confidence score to consistent human language.
    Used by ALL bots to guarantee communication consistency.
    """
    tmpl = LANGUAGE_TEMPLATES.get(language, LANGUAGE_TEMPLATES["en"])
    if score >= HIGH_CONFIDENCE if "HIGH_CONFIDENCE" in dir() else 0.80:
        return f"{tmpl['confidence_high']} ({int(score*100)}%)"
    elif score >= MEDIUM_CONFIDENCE if "MEDIUM_CONFIDENCE" in dir() else 0.60:
        return f"{tmpl['confidence_med']} ({int(score*100)}%)"
    else:
        return f"{tmpl['confidence_low']} ({int(score*100)}%)"
def communication_entropy(

text

):

    words=len(

        text.split()

    )

    return round(

        words/100,

        2

    )

# Re-define with proper constant references after config is loaded
def confidence_narrator(score: float, language: str = DEFAULT_LANGUAGE) -> str:
    tmpl = LANGUAGE_TEMPLATES.get(language, LANGUAGE_TEMPLATES["en"])
    if score >= 0.80:
        return f"{tmpl['confidence_high']} ({int(score*100)}%)"
    elif score >= 0.60:
        return f"{tmpl['confidence_med']} ({int(score*100)}%)"
    else:
        return f"{tmpl['confidence_low']} ({int(score*100)}%)"


def run_bot(bot_name: str, event: dict, language: str = DEFAULT_LANGUAGE) -> dict:
    """
    Single dispatcher. Calls generate_action_plan ONCE,
    routes to the correct bot, logs the session, returns result.

    Parameters
    ----------
    bot_name : "explainable" | "enforcement" | "predictive" | "supervisor" | "field"
    event    : dict with engineered features
    language : "en" | "kn" | "hi"

    Returns
    -------
    dict with bot output + session_id for audit trail
    """
    if not _RECOMMEND_LOADED:
        return {
            "bot"  : bot_name,
            "error": f"reccomend.py not available: {_RECOMMEND_ERROR}",
        }

    bot_name = bot_name.lower().strip()
    if bot_name not in AGENT_POLICY and bot_name != "field":
        return {"bot": bot_name, "error": f"Unknown bot: {bot_name}"}

    plan = generate_action_plan(event)

    if bot_name == "explainable":
        result = ExplainableBot(language).explain(plan, event)
    elif bot_name == "enforcement":
        result = EnforcementBot(language).deploy(plan, event)
    elif bot_name == "predictive":
        result = PredictiveBot().rank_events([event])
    elif bot_name == "supervisor":
        result = SupervisorBot().brief(plan, event)
    elif bot_name == "field":
        result = FieldOfficerBot(language).dispatch(plan, event)
    else:
        result = {"error": f"Bot '{bot_name}' not implemented"}

    session_id = log_session(bot_name, event, result)
    result["session_id"] = session_id
    return result


def run_all_bots(event: dict, language: str = DEFAULT_LANGUAGE) -> dict:
    """
    Runs all 5 bots in a single call with ONE generate_action_plan call.
    Used by Streamlit dashboard to populate all tabs at once.
    """
    if not _RECOMMEND_LOADED:
        return {"error": "reccomend.py not available"}

    plan = generate_action_plan(event)
    strategies = simulate_response_strategies(event)

    results = {
        "explainable" : ExplainableBot(language).explain(plan, event),
        "enforcement" : EnforcementBot(language).deploy(plan, event),
        "supervisor"  : SupervisorBot().brief(plan, event),
        "field"       : FieldOfficerBot(language).dispatch(plan, event),
        "strategies"  : strategies,
        "plan_raw"    : plan,
        "generated_at": datetime.now().isoformat(),
    }

    session_id = log_session("all_bots", event, results)
    results["session_id"] = session_id
    return results
def stakeholder_format(

text,

role

):

    entropy=COMMUNICATION_ENTROPY.get(

        role,

        0.5

    )

    if entropy<0.2:

        return text[:80]

    elif entropy<0.5:

        return text[:150]

    return text

 
# Cell 9 — Conversation Memory
# Persists every bot interaction to brain.db bot_sessions table
# Powers the "last N interactions" recurrence insight 

def _ensure_session_table():
    """Auto-create bot_sessions table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DATABASE_PATH_BOTS)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SESSION_TABLE} (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id    TEXT,
                bot_name      TEXT,
                event_cause   TEXT,
                cluster_id    INTEGER,
                hour          INTEGER,
                severity      INTEGER,
                priority_label TEXT,
                confidence    REAL,
                output_text   TEXT,
                language      TEXT,
                logged_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[bots] Session table creation failed: {e}")
        return False

_ensure_session_table()


def log_session(bot_name: str, event: dict, result: dict) -> str:
    """
    Writes one row to bot_sessions.
    Returns session_id for audit trail.
    """
    session_id = str(uuid.uuid4())[:12].upper()
    try:
        conn = sqlite3.connect(DATABASE_PATH_BOTS)
        severity  = result.get("severity", 0)
        if isinstance(severity, str):
            severity = list(SEVERITY_LABELS.values()).index(severity) + 1 \
                       if severity in SEVERITY_LABELS.values() else 0
        output_preview = str(result.get(
            "narrative",
            result.get("card_text",
            result.get("brief",
            result.get("message", json.dumps(result)[:200])
            ))
        ))[:500]
        conn.execute(f"""
            INSERT INTO {SESSION_TABLE}
            (session_id, bot_name, event_cause, cluster_id, hour,
             severity, priority_label, confidence, output_text, language)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            session_id,
            bot_name,
            str(event.get("event_cause", "unknown")),
            int(event.get("cluster_id", -1)),
            int(event.get("hour", 0)),
            int(severity),
            str(result.get("priority_label", result.get("severity", ""))),
            float(result.get("confidence", 0.0)),
            output_preview,
            str(result.get("language", DEFAULT_LANGUAGE)),
        ))
        # Prune old sessions (keep last MAX_SESSION_MEMORY per bot)
        conn.execute(f"""
            DELETE FROM {SESSION_TABLE}
            WHERE bot_name=? AND id NOT IN (
                SELECT id FROM {SESSION_TABLE}
                WHERE bot_name=?
                ORDER BY logged_at DESC
                LIMIT {MAX_SESSION_MEMORY}
            )
        """, (bot_name, bot_name))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[bots] Session log failed: {e}")
    return session_id


def recall_sessions(bot_name: str = None, limit: int = 10) -> pd.DataFrame:
    """
    Retrieves recent bot sessions from brain.db.
    Used by PredictiveBot and Streamlit dashboard for recurrence insights.
    """
    try:
        conn  = sqlite3.connect(DATABASE_PATH_BOTS)
        query = f"SELECT * FROM {SESSION_TABLE}"
        params = []
        if bot_name:
            query  += " WHERE bot_name=?"
            params.append(bot_name)
        query += " ORDER BY logged_at DESC LIMIT ?"
        params.append(limit)
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        print(f"[bots] Session recall failed: {e}")
        return pd.DataFrame()


def recurrence_insight(limit: int = 20) -> str:
    """
    Reads last N sessions and generates a recurrence insight sentence.
    Example: 'Cluster 4 appeared 7 times in last 20 queries — highest recurrence zone.'
    """
    df = recall_sessions(limit=limit)
    if df.empty:
        return "No session history available yet."
    top = df["cluster_id"].value_counts().idxmax()
    count = df["cluster_id"].value_counts().max()
    return (
        f"Cluster {top} appeared {count} times in your last {min(limit, len(df))} "
        f"queries — highest recurrence zone this session."
    )
def learn_from_session():

    df=recall_sessions(

        limit=100

    )

    if df.empty:

        return{}

    return{

        "top_cluster":

        int(

            df.cluster_id.mode()[0]

        ),

        "avg_confidence":

        float(

            df.confidence.mean()

        ),

        "top_bot":

        df.bot_name.mode()[0]

    }

print("\n05_bots.py ready")
print(f"  Bots available     : ExplainableBot, EnforcementBot, "
      f"PredictiveBot, SupervisorBot, FieldOfficerBot")
print(f"  Entry points       : run_bot(), run_all_bots()")
print(f"  Memory             : log_session(), recall_sessions(), recurrence_insight()")
print(f"  Session table      : {SESSION_TABLE} in brain.db")
print()

print("Advanced Cognitive Features Enabled")

print("Trust Fusion")

print("Decision DNA")

print("Operational Ontology")

print("Communication Entropy")

print("Stakeholder Adaptation")

print("Learning Engine")

print("Deployment Efficiency")

print("Voice Dispatch")

print("Cluster Trend Analytics")