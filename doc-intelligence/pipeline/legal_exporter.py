"""
Legal Export Module.
Generates court-ready output files:
  - legal_brief.md   — structured brief with agreed facts, disputed facts, timeline, contradictions
  - evidence_index.json — per-event source index with file hashes
  - run_config.json  — full pipeline configuration for reproducibility

All outputs are deterministic given the same inputs and config.
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

LEGAL_BRIEF_PATH = Path("data/outputs/legal_brief.md")
EVIDENCE_INDEX_PATH = Path("data/outputs/evidence_index.json")
RUN_CONFIG_PATH = Path("data/outputs/run_config.json")
UNRESOLVED_PATH = Path("data/outputs/unresolved_issues.json")


def save_run_config(
    case_name: str,
    model: str,
    date_window_days: int,
    sim_threshold: float,
    extra: Optional[dict] = None,
) -> dict:
    config = {
        "case_name": case_name,
        "model": model,
        "date_window_days": date_window_days,
        "sim_threshold": sim_threshold,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "source_weights": {"high": 1.0, "medium": 0.70, "low": 0.40},
        "confidence_thresholds": {
            "high": 0.75,
            "medium": 0.50,
            "low": 0.0,
        },
        "contradiction_min_similarity": 0.40,
        "narrative_language_rules": {
            "high": "The records confirm",
            "medium": "The documents indicate",
            "low": "There is an indication",
        },
        "timestamp": time.time(),
        "timestamp_human": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    if extra:
        config.update(extra)

    RUN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info("Run config saved to run_config.json")
    return config


def load_run_config() -> dict:
    if not RUN_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_evidence_index(
    merged_events: List[dict],
    hash_lookup: Optional[Dict[str, str]] = None,
) -> List[dict]:
    hash_lookup = hash_lookup or {}
    index = []
    for ev in merged_events:
        eid = ev.get("merged_event_id", "")
        sources = []
        for cit in ev.get("citations", []):
            sources.append({
                "file": cit.get("source_file", ""),
                "page": cit.get("page", ""),
                "hash": hash_lookup.get(eid, ""),
                "source_reliability": cit.get("source_reliability", "medium"),
                "source_weight": cit.get("source_weight", 0.7),
            })
        index.append({
            "event_id": eid,
            "date": ev.get("date_normalized", ""),
            "event_type": ev.get("event_type", ""),
            "confidence": ev.get("confidence", 0),
            "conflict": ev.get("has_conflict", False),
            "sources": sources,
        })

    EVIDENCE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVIDENCE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    logger.info(f"Evidence index saved: {len(index)} events")
    return index


def collect_unresolved_issues(
    merged_events: List[dict],
    contradictions: List[dict],
    confidence_threshold: float = 0.5,
) -> List[dict]:
    issues = []

    for ev in merged_events:
        conf = ev.get("confidence", 0)
        date = ev.get("date_normalized", "")
        has_high_conflict = ev.get("has_conflict", False)

        if conf < confidence_threshold:
            issues.append({
                "issue_type": "low_confidence",
                "event_id": ev.get("merged_event_id", ""),
                "date": date,
                "confidence": conf,
                "description": ev.get("description", "")[:150],
                "reason": f"Confidence {conf:.0%} below threshold {confidence_threshold:.0%}",
                "action": "Manual review required",
            })
        if not date:
            issues.append({
                "issue_type": "missing_date",
                "event_id": ev.get("merged_event_id", ""),
                "date": None,
                "confidence": conf,
                "description": ev.get("description", "")[:150],
                "reason": "No date could be extracted or normalized",
                "action": "Manual date assignment required",
            })

    for c in contradictions:
        ev_a = c.get("event_a", {})
        ev_b = c.get("event_b", {})
        rel_a = ev_a.get("reliability", "medium")
        rel_b = ev_b.get("reliability", "medium")
        if rel_a == "high" and rel_b == "high":
            issues.append({
                "issue_type": "high_weight_conflict",
                "event_ids": [ev_a.get("merged_event_id",""), ev_b.get("merged_event_id","")],
                "severity": c.get("severity","?"),
                "description": c.get("description","")[:150],
                "reason": "Conflict between two high-reliability sources — cannot auto-resolve",
                "action": "Priority legal review required",
            })

    UNRESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(UNRESOLVED_PATH, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2, ensure_ascii=False)
    logger.info(f"Unresolved issues flagged: {len(issues)}")
    return issues


def load_unresolved_issues() -> List[dict]:
    if not UNRESOLVED_PATH.exists():
        return []
    try:
        return json.loads(UNRESOLVED_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def generate_legal_brief(
    case_name: str,
    merged_events: List[dict],
    contradictions: List[dict],
    unresolved: List[dict],
    run_config: Optional[dict] = None,
    hash_lookup: Optional[Dict[str, str]] = None,
) -> str:
    hash_lookup = hash_lookup or {}
    run_config = run_config or {}

    high_conf = [e for e in merged_events if e.get("confidence", 0) >= 0.75]
    disputed = [e for e in merged_events if e.get("has_conflict") or e.get("confidence", 0) < 0.5]
    with_date = [e for e in merged_events if e.get("date_normalized")]
    critical_c = [c for c in contradictions if c.get("impact_label") == "CRITICAL"]
    significant_c = [c for c in contradictions if c.get("impact_label") == "SIGNIFICANT"]
    minor_c = [c for c in contradictions if c.get("impact_label") == "MINOR"]

    ts = run_config.get("timestamp_human", time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    sim_thresh = run_config.get("sim_threshold", 0.65)
    date_win = run_config.get("date_window_days", 3)

    lines = []
    lines.append(f"# LEGAL INTELLIGENCE BRIEF")
    lines.append(f"## {case_name}")
    lines.append(f"\n*Generated: {ts}*")
    lines.append(f"*Similarity threshold: {sim_thresh:.0%} | Date window: ±{date_win} days*")
    lines.append(f"*All claims are traceable to source documents via event IDs.*\n")
    lines.append("---\n")

    lines.append("## 1. Case Overview\n")
    dates = [e.get("date_normalized","") for e in with_date]
    date_range = f"{min(dates)} → {max(dates)}" if dates else "Unknown period"
    total_src = sum(e.get("source_count",1) for e in merged_events)
    high_rel = sum(1 for e in merged_events if e.get("source_reliability")=="high")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Period | {date_range} |")
    lines.append(f"| Total Events (merged) | {len(merged_events)} |")
    lines.append(f"| Total Source Citations | {total_src} |")
    lines.append(f"| High-Reliability Documents | {high_rel} |")
    lines.append(f"| High-Confidence Events | {len(high_conf)} |")
    lines.append(f"| Disputed / Uncertain Events | {len(disputed)} |")
    lines.append(f"| Contradictions | {len(contradictions)} (Critical: {len(critical_c)}) |")
    lines.append(f"| Unresolved Issues | {len(unresolved)} |\n")
    lines.append("---\n")

    lines.append("## 2. Agreed Facts\n")
    lines.append("*High-confidence events (≥75%) from verified sources. These may be stated definitively.*\n")
    if not high_conf:
        lines.append("*No events meet the high-confidence threshold.*\n")
    for ev in high_conf[:30]:
        eid = ev.get("merged_event_id","")
        date = ev.get("date_normalized","?")
        conf = ev.get("confidence",0)
        desc = ev.get("description","")[:200]
        rel = ev.get("source_reliability","?")
        ev_hash = hash_lookup.get(eid,"")
        cits = " | ".join(
            f"{c.get('source_file','?')} p.{c.get('page','?')}"
            for c in ev.get("citations",[])[:3]
        )
        lines.append(f"**[{eid}]** `{date}` — {desc}")
        lines.append(f"  - Confidence: {conf:.0%} | Reliability: {rel} | Sources: {cits}")
        if ev_hash:
            lines.append(f"  - Hash: `{ev_hash[:16]}...`")
        lines.append("")
    if len(high_conf) > 30:
        lines.append(f"*...and {len(high_conf)-30} more agreed facts. See evidence_index.json.*\n")
    lines.append("---\n")

    lines.append("## 3. Disputed Facts\n")
    lines.append("*Events with conflicts or below-threshold confidence. Cannot be stated definitively.*\n")
    if not disputed:
        lines.append("*No disputed facts identified.*\n")
    for ev in disputed[:20]:
        eid = ev.get("merged_event_id","")
        date = ev.get("date_normalized","?")
        conf = ev.get("confidence",0)
        desc = ev.get("description","")[:200]
        conflict_flag = " ⚠ CONFLICT" if ev.get("has_conflict") else ""
        lines.append(f"**[{eid}]** `{date}` — {desc}{conflict_flag}")
        lines.append(f"  - Confidence: {conf:.0%} (flagged: low confidence or conflicting sources)")
        lines.append("")
    if len(disputed) > 20:
        lines.append(f"*...and {len(disputed)-20} more disputed items.*\n")
    lines.append("---\n")

    lines.append("## 4. Chronological Timeline\n")
    sorted_ev = sorted(with_date, key=lambda e: e.get("date_normalized",""))
    for i, ev in enumerate(sorted_ev[:50], 1):
        date = ev.get("date_normalized","?")
        etype = ev.get("event_type","general").upper()
        desc = ev.get("description","")[:180]
        conf = ev.get("confidence",0)
        conf_label = "HIGH" if conf >= 0.75 else ("MEDIUM" if conf >= 0.5 else "⚠ LOW")
        eid = ev.get("merged_event_id","")
        lines.append(f"{i}. **{date}** [{etype}] — {desc}")
        lines.append(f"   `[{eid}]` Confidence: {conf_label} ({conf:.0%})")
        lines.append("")
    if len(sorted_ev) > 50:
        lines.append(f"*...{len(sorted_ev)-50} more events. See timeline_legal.json.*\n")
    lines.append("---\n")

    lines.append("## 5. Key Contradictions (Ranked by Impact)\n")
    lines.append("*Contradictions ranked by: severity × source weight × semantic similarity*\n")
    if not contradictions:
        lines.append("*No contradictions detected.*\n")
    for i, c in enumerate(contradictions[:20], 1):
        label = c.get("impact_label","?")
        score = c.get("impact_score",0)
        sim = c.get("semantic_similarity",0)
        sev = c.get("severity","?").upper()
        desc = c.get("description","")
        ev_a = c.get("event_a",{})
        ev_b = c.get("event_b",{})
        lines.append(f"### [{label}] Contradiction {i} — Impact Score: {score:.3f}")
        lines.append(f"**Severity:** {sev} | **Semantic similarity:** {sim:.0%}")
        lines.append(f"**Issue:** {desc}")
        lines.append(f"**Source A** `[{ev_a.get('merged_event_id','')}]` ({ev_a.get('date','?')}, "
                     f"reliability: {ev_a.get('reliability','?')}):")
        lines.append(f"> {ev_a.get('description','')[:200]}")
        lines.append(f"**Source B** `[{ev_b.get('merged_event_id','')}]` ({ev_b.get('date','?')}, "
                     f"reliability: {ev_b.get('reliability','?')}):")
        lines.append(f"> {ev_b.get('description','')[:200]}")
        lines.append("")
    if len(contradictions) > 20:
        lines.append(f"*...{len(contradictions)-20} more contradictions. See contradictions.json.*\n")
    lines.append("---\n")

    if unresolved:
        lines.append("## 6. Unresolved Issues (Requires Manual Review)\n")
        for u in unresolved[:20]:
            utype = u.get("issue_type","?").replace("_"," ").upper()
            reason = u.get("reason","?")
            action = u.get("action","?")
            ids = u.get("event_id") or ", ".join(u.get("event_ids",[]))
            lines.append(f"- **[{utype}]** Event(s): `{ids}`")
            lines.append(f"  Reason: {reason}")
            lines.append(f"  Required action: **{action}**")
            lines.append("")
        if len(unresolved) > 20:
            lines.append(f"*...{len(unresolved)-20} more unresolved issues.*\n")
        lines.append("---\n")

    lines.append("## Evidence Appendix\n")
    lines.append("*Complete source citations per event. See evidence_index.json for machine-readable form.*\n")
    for ev in merged_events[:20]:
        eid = ev.get("merged_event_id","")
        cits = ev.get("citations",[])
        if not cits:
            continue
        lines.append(f"**[{eid}]** {ev.get('date_normalized','?')} — {ev.get('event_type','?').upper()}")
        for c in cits[:5]:
            rel_mark = "★" if c.get("source_reliability")=="high" else ("◆" if c.get("source_reliability")=="medium" else "◇")
            lines.append(f"  {rel_mark} {c.get('source_file','?')}, page {c.get('page','?')}, "
                         f"weight: {c.get('source_weight',0.7):.2f}")
        lines.append("")
    if len(merged_events) > 20:
        lines.append(f"*...{len(merged_events)-20} more events. See evidence_index.json.*")

    content = "\n".join(lines)
    LEGAL_BRIEF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEGAL_BRIEF_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Legal brief generated")
    return content


def load_legal_brief() -> str:
    if not LEGAL_BRIEF_PATH.exists():
        return ""
    return LEGAL_BRIEF_PATH.read_text(encoding="utf-8")


def load_evidence_index() -> List[dict]:
    if not EVIDENCE_INDEX_PATH.exists():
        return []
    try:
        return json.loads(EVIDENCE_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
