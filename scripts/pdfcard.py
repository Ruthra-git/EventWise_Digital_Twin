import os
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
from reportlab.lib.pagesizes import A5
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Config — paths match existing project structure
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "database" / "brain.db"
OUTPUT_DIR = BASE_DIR / "outputs" / "pdf_cards"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BRAND_NAME    = "EventWise AI — TrafficTwin"
BRAND_COLOR   = colors.HexColor("#1E293B")
ACCENT_MAP    = {
    "LOW"     : colors.HexColor("#2ECC71"),
    "MODERATE": colors.HexColor("#F1C40F"),
    "HIGH"    : colors.HexColor("#E67E22"),
    "CRITICAL": colors.HexColor("#E74C3C"),
    "EXTREME" : colors.HexColor("#8B0000"),
}
TIER_COLOR = {
    "TIER-1": colors.HexColor("#2ECC71"),
    "TIER-2": colors.HexColor("#E67E22"),
    "TIER-3": colors.HexColor("#E74C3C"),
}

# ─────────────────────────────────────────────────────────────
# Helper — safe float/int formatting
# ─────────────────────────────────────────────────────────────
def _s(val, fmt="{}", fallback="N/A"):
    try:
        return fmt.format(val)
    except:
        return str(fallback)


# ─────────────────────────────────────────────────────────────
# Core generator
# ─────────────────────────────────────────────────────────────
def generate_pdf_card(
    plan: dict,
    event: dict,
    bot_outputs: dict = None,
    filename: str = None,
) -> str:
    """
    Parameters
    ----------
    plan        : output of generate_action_plan(event)
    event       : raw event dict with at least address, event_cause, cluster_id
    bot_outputs : optional dict — output of run_all_bots(event)
                  uses EnforcementBot card_text, ExplainableBot narrative,
                  FieldOfficerBot message if available
    filename    : custom filename; auto-generated if None

    Returns
    -------
    str — absolute path to saved PDF
    """
    if filename is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        ev_id    = str(event.get("id", "unknown"))[:12]
        filename = f"deployment_card_{ev_id}_{ts}.pdf"

    filepath = OUTPUT_DIR / filename
    W, H     = A5                       # 148 × 210 mm
    c        = canvas.Canvas(str(filepath), pagesize=A5)

    # ── Pull data ────────────────────────────────────────────
    summary   = plan.get("event_summary", {})
    resources = plan.get("resource_plan", {})
    movement  = plan.get("movement_plan", {})
    timeplan  = plan.get("time_plan", {})
    priority  = plan.get("priority", {})
    steps     = plan.get("action_steps", [])
    notes     = plan.get("escalation_notes", [])

    sev_label = {1:"LOW",2:"MODERATE",3:"HIGH",4:"CRITICAL",5:"EXTREME"}.get(
        int(summary.get("severity", 2)), "MODERATE")
    risk_tier = "TIER-3" if sev_label in ("CRITICAL","EXTREME") else \
                "TIER-2" if sev_label in ("HIGH","MODERATE") else "TIER-1"
    accent    = ACCENT_MAP.get(sev_label, colors.orange)
    tier_col  = TIER_COLOR.get(risk_tier, colors.orange)
    conf_pct  = int(float(summary.get("confidence", 0.75)) * 100
                    if float(summary.get("confidence", 0.75)) <= 1.0
                    else float(summary.get("confidence", 75)))

    cause   = str(event.get("event_cause", "unknown")).replace("_", " ").title()
    address = str(event.get("address", f"Cluster {event.get('cluster_id','?')}"))[:60]
    cluster = event.get("cluster_id", "?")
    hour    = event.get("hour", "?")
    zone    = event.get("zone", "N/A")

    # field bot message (3-line SMS card)
    field_msg = ""
    explain   = ""
    if bot_outputs:
        field_msg = bot_outputs.get("field", {}).get("message", "")
        explain   = bot_outputs.get("explainable", {}).get("narrative", "")

    # ── HEADER BAND ──────────────────────────────────────────
    c.setFillColor(BRAND_COLOR)
    c.rect(0, H - 22*mm, W, 22*mm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(6*mm, H - 10*mm, BRAND_COLOR and BRAND_NAME)
    c.setFont("Helvetica", 8)
    c.drawString(6*mm, H - 16*mm,
                 f"Generated : {datetime.now().strftime('%d %b %Y  %H:%M')}  |  "
                 f"Cluster : {cluster}  |  Zone : {zone}")

    # ── SEVERITY BADGE ───────────────────────────────────────
    c.setFillColor(accent)
    c.roundRect(W - 42*mm, H - 20*mm, 38*mm, 16*mm, 3*mm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(W - 23*mm, H - 14*mm, sev_label)
    c.setFont("Helvetica", 7)
    c.drawCentredString(W - 23*mm, H - 18.5*mm, f"{risk_tier}  |  {conf_pct}% conf")

    y = H - 28*mm

    # ── EVENT DETAILS ────────────────────────────────────────
    c.setFillColor(BRAND_COLOR)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(6*mm, y, "INCIDENT DETAILS")
    y -= 5*mm
    c.setStrokeColor(accent)
    c.setLineWidth(1.5)
    c.line(6*mm, y, W - 6*mm, y)
    y -= 5*mm

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(BRAND_COLOR)
    c.drawString(6*mm, y, f"Cause  :  {cause}")
    y -= 5*mm
    c.setFont("Helvetica", 8)
    c.drawString(6*mm, y, f"Location : {address[:58]}")
    y -= 4.5*mm
    c.drawString(6*mm, y,
                 f"Hour : {hour}:00   |   Planned : "
                 f"{'Yes' if event.get('is_planned') else 'No'}   |   "
                 f"Road Closure : {'Yes' if event.get('road_closure') else 'No'}")
    y -= 7*mm

    # ── DEPLOYMENT TABLE ─────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(BRAND_COLOR)
    c.drawString(6*mm, y, "DEPLOYMENT PLAN")
    y -= 5*mm
    c.setStrokeColor(accent)
    c.line(6*mm, y, W - 6*mm, y)
    y -= 2*mm

    table_data = [
        ["Resource",      "Count", "Resource",     "Count"],
        ["👮 Officers",   str(resources.get("officers", 0)),
         "🚧 Barricades", str(resources.get("barricades", 0))],
        ["🔶 Cones",      str(resources.get("cones", 0)),
         "⚠ Warn Boards", str(resources.get("warning_boards", 0))],
        ["⏱ Response",   str(timeplan.get("estimated_duration_minutes","?")),
         "🕐 Clearance",  str(timeplan.get("expected_clearance","?"))],
        ["📊 Priority",   f"{priority.get('priority_score',0):.0f}/100",
         "🎯 Confidence", f"{conf_pct}%"],
    ]

    tbl = Table(table_data,
                colWidths=[32*mm, 18*mm, 32*mm, 18*mm],
                rowHeights=6*mm)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), BRAND_COLOR),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F8FAFC"),
                                          colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.lightgrey),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",       (1,0), (1,-1), "CENTER"),
        ("ALIGN",       (3,0), (3,-1), "CENTER"),
    ]))
    tbl_w, tbl_h = tbl.wrapOn(c, W - 12*mm, 60*mm)
    tbl.drawOn(c, 6*mm, y - tbl_h)
    y -= tbl_h + 5*mm

    # ── DIVERSION ────────────────────────────────────────────
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(BRAND_COLOR)
    c.drawString(6*mm, y, "MOVEMENT PLAN")
    y -= 5*mm
    c.setStrokeColor(accent)
    c.line(6*mm, y, W - 6*mm, y)
    y -= 5*mm
    c.setFont("Helvetica", 8)
    divert = movement.get("diversion_required", False)
    c.setFillColor(colors.red if divert else colors.HexColor("#2ECC71"))
    c.drawString(6*mm, y,
                 f"Diversion : {'ACTIVATE' if divert else 'NOT REQUIRED'}")
    y -= 4.5*mm
    c.setFillColor(BRAND_COLOR)
    if divert:
        c.drawString(6*mm, y,
                     f"Primary  : {str(movement.get('primary_route',''))[:55]}")
        y -= 4.5*mm
        sec = movement.get("secondary_route")
        if sec:
            c.drawString(6*mm, y, f"Secondary: {str(sec)[:55]}")
            y -= 4.5*mm
    y -= 2*mm

    # ── ACTION STEPS ─────────────────────────────────────────
    if steps:
        c.setFont("Helvetica-Bold", 9)
        c.drawString(6*mm, y, "ACTION STEPS")
        y -= 5*mm
        c.setStrokeColor(accent)
        c.line(6*mm, y, W - 6*mm, y)
        y -= 5*mm
        c.setFont("Helvetica", 7.5)
        for i, step in enumerate(steps[:4], 1):
            c.setFillColor(accent)
            c.circle(8.5*mm, y + 1.5*mm, 1.5*mm, fill=1, stroke=0)
            c.setFillColor(BRAND_COLOR)
            c.drawString(11*mm, y, f"{step[:70]}")
            y -= 5*mm

    # ── FIELD SMS CARD ───────────────────────────────────────
    if field_msg:
        y -= 2*mm
        c.setFillColor(colors.HexColor("#F1F5F9"))
        c.roundRect(5*mm, y - 16*mm, W - 10*mm, 18*mm, 2*mm, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(BRAND_COLOR)
        c.drawString(8*mm, y, "📱 FIELD SMS")
        y -= 5*mm
        c.setFont("Courier", 7.5)
        for line in field_msg.split("\n")[:3]:
            c.drawString(8*mm, y, line[:72])
            y -= 4.5*mm
        y -= 2*mm

    # ── ESCALATION NOTE ──────────────────────────────────────
    if notes:
        top_note = str(notes[0])[:90]
        c.setFont("Helvetica-Oblique", 7)
        c.setFillColor(colors.grey)
        c.drawString(6*mm, max(y - 3*mm, 12*mm), f"Note: {top_note}")

    # ── FOOTER ───────────────────────────────────────────────
    c.setFillColor(BRAND_COLOR)
    c.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 6.5)
    c.drawString(6*mm, 3.5*mm,
                 "EventWise AI — TrafficTwin  |  Bengaluru Traffic Management  |  CONFIDENTIAL")
    c.setFont("Helvetica-Bold", 6.5)
    c.drawRightString(W - 6*mm, 3.5*mm,
                      f"Priority Score : {priority.get('priority_score',0):.0f}/100")

    c.save()
    print(f"✓ PDF saved → {filepath}")
    return str(filepath)


# ─────────────────────────────────────────────────────────────
# Batch generator — generate cards for top-N critical events
# from brain.db (used by app.py batch export button)
# ─────────────────────────────────────────────────────────────
def generate_batch_cards(top_n: int = 10) -> list:
    """
    Reads top-N events by cluster_risk_score from brain.db,
    builds minimal action plans from stored recommendations,
    generates one PDF per event.
    Returns list of saved file paths.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        events = pd.read_sql("""
            SELECT e.*, r.recommended_officers, r.recommended_barricades,
                   r.priority_score, r.confidence, r.diversion_required,
                   r.expected_clearance,
                   p.severity as pred_severity, p.uncertainty
            FROM events e
            LEFT JOIN recommendations r ON e.id = r.event_id
            LEFT JOIN predictions     p ON e.id = p.event_id
            ORDER BY r.priority_score DESC
            LIMIT ?
        """, conn, params=[top_n])
    finally:
        conn.close()

    if events.empty:
        print("No events found in brain.db for batch card generation.")
        return []

    paths = []
    for _, row in events.iterrows():
        event = row.to_dict()
        sev   = int(float(event.get("pred_severity", event.get("severity", 2)) or 2))
        conf  = float(event.get("confidence", 0.75) or 0.75)
        conf  = conf / 100 if conf > 1.0 else conf

        # Build minimal plan from stored recommendation data
        plan = {
            "event_summary": {
                "severity"  : sev,
                "risk"      : {1:"LOW",2:"MODERATE",3:"HIGH",4:"CRITICAL",5:"EXTREME"}.get(sev,"HIGH"),
                "confidence": conf,
            },
            "resource_plan": {
                "officers"      : int(float(event.get("recommended_officers", 4) or 4)),
                "barricades"    : int(float(event.get("recommended_barricades", 2) or 2)),
                "cones"         : int(float(event.get("recommended_officers", 4) or 4)) * 3,
                "warning_boards": max(1, sev - 1),
            },
            "movement_plan": {
                "diversion_required": bool(int(float(event.get("diversion_required", 0) or 0))),
                "primary_route"     : f"Alternate route — Cluster {event.get('cluster_id','?')}",
                "secondary_route"   : None,
            },
            "time_plan": {
                "estimated_duration_minutes": sev * 30,
                "expected_clearance"        : str(event.get("expected_clearance", "N/A")),
            },
            "priority": {
                "priority_score": float(event.get("priority_score", 50) or 50),
                "priority_label": "HIGH" if sev >= 4 else "MODERATE",
                "historical_matches": 0,
                "estimated_clearance": str(event.get("expected_clearance", "N/A")),
                "diversion_required": bool(int(float(event.get("diversion_required", 0) or 0))),
            },
            "action_steps": [
                "Dispatch personnel to location immediately.",
                "Set up barricades and manage traffic flow.",
                "Monitor clearance and update control room.",
            ],
            "escalation_notes": [
                f"Priority score: {float(event.get('priority_score',50) or 50):.0f}/100"
            ],
        }

        try:
            path = generate_pdf_card(plan, event)
            paths.append(path)
        except Exception as e:
            print(f"  ✗ Card failed for event {event.get('id')}: {e}")

    print(f"\n✓ Batch complete — {len(paths)}/{top_n} cards generated")
    print(f"  Saved to: {OUTPUT_DIR}")
    return paths


# ─────────────────────────────────────────────────────────────
# Streamlit helper — returns bytes for st.download_button
# ─────────────────────────────────────────────────────────────
def get_pdf_bytes(plan: dict, event: dict, bot_outputs: dict = None) -> bytes:
    """Call from app.py Tab 2 download button."""
    path = generate_pdf_card(plan, event, bot_outputs)
    with open(path, "rb") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating batch PDF cards for top 10 critical events...")
    paths = generate_batch_cards(top_n=10)
    print(f"\nGenerated {len(paths)} cards:")
    for p in paths:
        print(f"  {p}")
