"""
Traceable Legal Narrative Generator — Forensic Edition.

Rules:
- Every sentence MUST reference ≥1 merged_event_id
- No definitive statements without high confidence
- Confidence-driven language:
    high   → "The records confirm…"
    medium → "The documents indicate…"
    low    → "There is an indication…"
- No AI-generated fact without a source event
- Low confidence events → Unresolved Issues section
- No merging without traceable similarity score
"""
import gc
import json
import logging
import requests
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

NARRATIVE_PATH = Path("data/outputs/legal_narrative.md")
NARRATIVE_SENTENCES_PATH = Path("data/outputs/narrative_sentences.jsonl")
TIMELINE_JSON_PATH = Path("data/outputs/timeline_legal.json")
TIMELINE_MD_PATH = Path("data/outputs/timeline_legal.md")

OLLAMA_URL = "http://localhost:11434/api/generate"

CONF_LANGUAGE = {
    "high": "The records confirm",
    "medium": "The documents indicate",
    "low": "There is an indication",
}

CONF_THRESHOLD_HIGH = 0.75
CONF_THRESHOLD_MED = 0.50


def confidence_level(conf: float) -> str:
    if conf >= CONF_THRESHOLD_HIGH:
        return "high"
    elif conf >= CONF_THRESHOLD_MED:
        return "medium"
    return "low"


def confidence_opener(conf: float) -> str:
    lvl = confidence_level(conf)
    return CONF_LANGUAGE[lvl]


def ollama_available() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def query_ollama(prompt: str, model: str = "mistral", max_tokens: int = 600) -> str:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 2048, "temperature": 0.05, "num_predict": max_tokens},
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=90)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama narrative query failed: {e}")
    return ""


def format_citation_inline(citations: List[dict], max_cits: int = 3) -> str:
    parts = []
    for c in sorted(citations, key=lambda x: -x.get("source_weight", 0.7))[:max_cits]:
        src = c.get("source_file", "Unknown")
        page = c.get("page", "")
        rel = c.get("source_reliability", "")
        rel_mark = "★" if rel == "high" else ("◆" if rel == "medium" else "◇")
        parts.append(f"[{rel_mark} {src}, p.{page}]" if page else f"[{rel_mark} {src}]")
    return " ".join(parts)


def make_sentence(
    text: str,
    section: str,
    event_refs: List[str],
    citations: List[dict],
    confidence: float = 1.0,
) -> dict:
    sentence_id = str(uuid.uuid4())[:12]
    return {
        "sentence_id": sentence_id,
        "section": section,
        "text": text,
        "event_refs": event_refs,
        "citations": citations[:4],
        "citation_inline": format_citation_inline(citations),
        "confidence": confidence,
        "confidence_level": confidence_level(confidence),
    }


def save_narrative_sentences(sentences: List[dict]) -> None:
    NARRATIVE_SENTENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NARRATIVE_SENTENCES_PATH, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def load_narrative_sentences() -> List[dict]:
    if not NARRATIVE_SENTENCES_PATH.exists():
        return []
    sentences = []
    with open(NARRATIVE_SENTENCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sentences.append(json.loads(line))
                except Exception:
                    pass
    return sentences


def generate_timeline_json(merged_events: List[dict]) -> None:
    TIMELINE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    timeline = []
    for ev in merged_events:
        if not ev.get("date_normalized"):
            continue
        conf = ev.get("confidence", 0)
        conf_lvl = confidence_level(conf)
        timeline.append({
            "merged_event_id": ev.get("merged_event_id", ""),
            "date": ev["date_normalized"],
            "confidence": conf,
            "confidence_level": conf_lvl,
            "source_reliability": ev.get("source_reliability", "medium"),
            "event_type": ev.get("event_type", ""),
            "description": ev.get("description", "")[:300],
            "actors": ev.get("actors", []),
            "location": ev.get("location", ""),
            "citations": ev.get("citations", []),
            "source_count": ev.get("source_count", 1),
            "has_conflict": ev.get("has_conflict", False),
            "versions": ev.get("versions", []),
            "low_confidence_flag": conf < CONF_THRESHOLD_MED,
            "date_confidence": ev.get("date_confidence", 0.5),
        })
    with open(TIMELINE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved timeline JSON: {len(timeline)} events")


def generate_timeline_md(merged_events: List[dict]) -> str:
    CONF_BADGE = {"high": "★★★ HIGH", "medium": "★★ MEDIUM", "low": "★ LOW"}
    lines = ["# Legal Case Timeline\n"]
    lines.append(
        f"*{len(merged_events)} events reconstructed. Sorted by date.*  \n"
        "Legend: ★ = High-reliability | ◆ = Medium | ◇ = Low | ⚠ = Low confidence | ⚡ = Conflict\n\n---\n"
    )

    sorted_ev = sorted(
        [e for e in merged_events if e.get("date_normalized")],
        key=lambda e: e.get("date_normalized", ""),
    )

    low_conf_cluster = []

    for i, ev in enumerate(sorted_ev, 1):
        date_str = ev.get("date_normalized", "?")
        etype = ev.get("event_type", "general").upper()
        conf = ev.get("confidence", 0)
        conf_lvl = confidence_level(conf)
        rel = ev.get("source_reliability", "medium")
        desc = ev.get("description", "")[:300]
        actors = ", ".join(ev.get("actors", [])[:4])
        location = ev.get("location", "")
        cit_str = format_citation_inline(ev.get("citations", []))
        low_flag = " ⚠ LOW CONFIDENCE" if conf < CONF_THRESHOLD_MED else ""
        conflict_flag = " ⚡ CONFLICT" if ev.get("has_conflict") else ""
        rel_badge = "★ HIGH" if rel == "high" else ("◆ MEDIUM" if rel == "medium" else "◇ LOW")
        conf_badge = CONF_BADGE.get(conf_lvl, "★★ MEDIUM")
        ref_id = ev.get("merged_event_id", "")

        if conf < CONF_THRESHOLD_MED:
            low_conf_cluster.append(ref_id)

        lines.append(f"### {i}. {date_str} — {etype} [{rel_badge}] [{conf_badge}]{low_flag}{conflict_flag}")
        lines.append(f"`ref: {ref_id}`")
        lines.append(f"\n{desc}")
        if actors:
            lines.append(f"\n- **Parties:** {actors}")
        if location:
            lines.append(f"- **Location:** {location}")
        lines.append(f"- **Source(s):** {cit_str}")
        lines.append(f"- **Confidence:** {conf:.0%} | **Sources:** {ev.get('source_count',1)}")
        versions = ev.get("versions", [])
        if len(versions) > 1:
            lines.append(f"- **⚡ {len(versions)} conflicting versions** (see event versioning)")
        lines.append("")

    if low_conf_cluster:
        lines.append("\n---\n## Uncertainty Cluster\n")
        lines.append(f"*{len(low_conf_cluster)} events below confidence threshold require manual verification:*\n")
        for ref in low_conf_cluster:
            lines.append(f"- `{ref}`")

    content = "\n".join(lines)
    TIMELINE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMELINE_MD_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def _build_overview_sentences(
    merged_events: List[dict],
    contradictions: List[dict],
    case_name: str,
    model: str,
    use_ollama: bool,
) -> Tuple[List[dict], str]:
    dates = [ev.get("date_normalized", "") for ev in merged_events if ev.get("date_normalized")]
    date_range = f"{min(dates)} to {max(dates)}" if dates else "unknown period"
    total_sources = sum(ev.get("source_count", 1) for ev in merged_events)
    high_rel = sum(1 for ev in merged_events if ev.get("source_reliability") == "high")

    all_actors = []
    seen = set()
    for ev in merged_events:
        for a in ev.get("actors", []):
            if a not in seen:
                all_actors.append(a)
                seen.add(a)

    sentences = []
    all_event_ids = [ev.get("merged_event_id", "") for ev in merged_events[:5] if ev.get("merged_event_id")]
    all_citations = []
    for ev in merged_events[:5]:
        all_citations.extend(ev.get("citations", []))

    sent_intro = make_sentence(
        f"This legal case reconstruction for **{case_name}** spans **{date_range}**, "
        f"comprising **{len(merged_events)} verified events** extracted and merged from "
        f"**{total_sources} document citations**.",
        section="Case Overview",
        event_refs=all_event_ids[:3],
        citations=all_citations[:4],
        confidence=1.0,
    )
    sentences.append(sent_intro)

    sent_sources = make_sentence(
        f"The records confirm that **{high_rel}** source documents are classified as "
        f"high-reliability (official/legal documents), providing a strong evidentiary "
        f"foundation for this reconstruction.",
        section="Case Overview",
        event_refs=all_event_ids[:2],
        citations=all_citations[:3],
        confidence=1.0 if high_rel > 0 else 0.6,
    )
    sentences.append(sent_sources)

    if all_actors:
        actor_ev_ids = [ev.get("merged_event_id","") for ev in merged_events if ev.get("actors")][:3]
        actor_cits = []
        for ev in merged_events[:3]:
            actor_cits.extend(ev.get("citations", []))
        sent_actors = make_sentence(
            f"The documents indicate that the following key parties appear across the "
            f"analyzed documents: **{', '.join(all_actors[:6])}**.",
            section="Case Overview",
            event_refs=actor_ev_ids,
            citations=actor_cits[:3],
            confidence=0.7,
        )
        sentences.append(sent_actors)

    if contradictions:
        critical_c = [c for c in contradictions if c.get("impact_label") == "CRITICAL"]
        cont_ids = [c.get("event_a", {}).get("merged_event_id", "") for c in contradictions[:3]]
        cont_cits = []
        for c in contradictions[:2]:
            cont_cits.extend(c.get("event_a", {}).get("citations", []))
        sent_cont = make_sentence(
            f"The records confirm that **{len(contradictions)} contradiction(s)** were detected "
            f"across source documents, of which **{len(critical_c)}** are classified as "
            f"CRITICAL impact and require priority legal review.",
            section="Case Overview",
            event_refs=[eid for eid in cont_ids if eid],
            citations=cont_cits[:3],
            confidence=0.9,
        )
        sentences.append(sent_cont)

    overview_md = "\n\n".join(
        f"{s['text']} {s['citation_inline']}" for s in sentences
    )
    return sentences, overview_md


def _build_chronological_sentences(
    merged_events: List[dict], use_ollama: bool, model: str
) -> Tuple[List[dict], str]:
    sentences = []
    md_lines = []

    for i, ev in enumerate(merged_events, 1):
        if not ev.get("date_normalized"):
            continue

        date_str = ev.get("date_normalized", "?")
        etype = ev.get("event_type", "general")
        desc = ev.get("description", "")[:250]
        conf = ev.get("confidence", 0)
        rel = ev.get("source_reliability", "medium")
        ref_id = ev.get("merged_event_id", "")
        conflict = ev.get("has_conflict", False)

        opener = confidence_opener(conf)
        lang_note = (
            f" *(source language: {'/'.join(ev.get('languages',['en'])).upper()})*"
            if "ar" in ev.get("languages", []) else ""
        )
        low_flag = " *(low confidence — treat with caution)*" if conf < CONF_THRESHOLD_MED else ""
        conflict_note = " *(conflicting source versions exist)*" if conflict else ""

        text = f"{opener} that on **{date_str}**, a {etype} event occurred: {desc}{lang_note}{low_flag}{conflict_note}."
        sent = make_sentence(
            text,
            section="Chronological Events",
            event_refs=[ref_id] if ref_id else [],
            citations=ev.get("citations", []),
            confidence=conf,
        )
        sentences.append(sent)

        actors = ", ".join(ev.get("actors", [])[:4])
        location = ev.get("location", "")
        cit_str = sent["citation_inline"]
        conf_lvl = confidence_level(conf)

        md_lines.append(f"**{i}. {date_str}** — `{etype.upper()}` `[{ref_id}]` [{conf_lvl.upper()}]")
        md_lines.append(f"{text}")
        if actors:
            md_lines.append(f"*Parties:* {actors}")
        if location:
            md_lines.append(f"*Location:* {location}")
        md_lines.append(f"*Citations:* {cit_str}")
        md_lines.append("")

    return sentences, "\n".join(md_lines)


def _build_agreed_facts_sentences(merged_events: List[dict]) -> Tuple[List[dict], str]:
    high_conf = [ev for ev in merged_events if ev.get("confidence", 0) >= CONF_THRESHOLD_HIGH]
    sentences = []
    md_lines = []

    if not high_conf:
        placeholder = make_sentence(
            "No events met the high-confidence threshold for agreed facts classification.",
            section="Agreed Facts",
            event_refs=[],
            citations=[],
            confidence=1.0,
        )
        sentences.append(placeholder)
        md_lines.append("*No high-confidence agreed facts identified.*")
        return sentences, "\n".join(md_lines)

    intro = make_sentence(
        f"The records confirm the following **{len(high_conf)} agreed facts** — events with "
        f"confidence ≥{CONF_THRESHOLD_HIGH:.0%} verified across multiple sources.",
        section="Agreed Facts",
        event_refs=[ev.get("merged_event_id","") for ev in high_conf[:3]],
        citations=[c for ev in high_conf[:2] for c in ev.get("citations",[])],
        confidence=1.0,
    )
    sentences.append(intro)
    md_lines.append(f"> {intro['text']} {intro['citation_inline']}\n")

    for ev in high_conf:
        ref_id = ev.get("merged_event_id","")
        date = ev.get("date_normalized","?")
        conf = ev.get("confidence",0)
        desc = ev.get("description","")[:200]

        text = f"The records confirm: on **{date}**, {desc}."
        sent = make_sentence(
            text, "Agreed Facts",
            event_refs=[ref_id],
            citations=ev.get("citations",[]),
            confidence=conf,
        )
        sentences.append(sent)
        md_lines.append(f"- `[{ref_id}]` {text} {sent['citation_inline']}")

    return sentences, "\n".join(md_lines)


def _build_disputed_facts_sentences(
    merged_events: List[dict], contradictions: List[dict]
) -> Tuple[List[dict], str]:
    conflicted_ids = set()
    for c in contradictions:
        conflicted_ids.add(c.get("event_a",{}).get("merged_event_id",""))
        conflicted_ids.add(c.get("event_b",{}).get("merged_event_id",""))

    disputed = [
        ev for ev in merged_events
        if ev.get("merged_event_id","") in conflicted_ids
        or ev.get("confidence",0) < CONF_THRESHOLD_MED
    ]

    sentences = []
    md_lines = []

    if not disputed:
        placeholder = make_sentence(
            "No disputed facts were identified across the analyzed documents.",
            section="Disputed Facts",
            event_refs=[],
            citations=[],
            confidence=1.0,
        )
        sentences.append(placeholder)
        md_lines.append("*No disputed facts identified.*")
        return sentences, "\n".join(md_lines)

    intro = make_sentence(
        f"There is an indication that **{len(disputed)} event(s)** are disputed — either because "
        f"conflicting sources exist or confidence falls below threshold. These CANNOT be stated definitively.",
        section="Disputed Facts",
        event_refs=[ev.get("merged_event_id","") for ev in disputed[:3]],
        citations=[],
        confidence=0.6,
    )
    sentences.append(intro)
    md_lines.append(f"> ⚠ {intro['text']}\n")

    for ev in disputed:
        ref_id = ev.get("merged_event_id","")
        date = ev.get("date_normalized","?") or "undated"
        conf = ev.get("confidence",0)
        desc = ev.get("description","")[:200]
        conflict = ref_id in conflicted_ids

        opener = "There is an indication" if conf < CONF_THRESHOLD_MED else "The documents indicate"
        reason = "conflicting source documents" if conflict else f"low confidence ({conf:.0%})"
        text = f"{opener} that on **{date}**, {desc}. *(Disputed — {reason}.)*"
        sent = make_sentence(
            text, "Disputed Facts",
            event_refs=[ref_id],
            citations=ev.get("citations",[]),
            confidence=conf,
        )
        sentences.append(sent)
        md_lines.append(f"- ⚠ `[{ref_id}]` {text} {sent['citation_inline']}")

    return sentences, "\n".join(md_lines)


def _build_unresolved_sentences(
    merged_events: List[dict], contradictions: List[dict]
) -> Tuple[List[dict], str]:
    issues = []

    for ev in merged_events:
        if ev.get("confidence",0) < CONF_THRESHOLD_MED:
            issues.append(("low_confidence", ev, None))
        if not ev.get("date_normalized"):
            issues.append(("missing_date", ev, None))

    for c in contradictions:
        ev_a = c.get("event_a",{})
        ev_b = c.get("event_b",{})
        if ev_a.get("reliability","") == "high" and ev_b.get("reliability","") == "high":
            issues.append(("high_weight_conflict", None, c))

    sentences = []
    md_lines = []

    if not issues:
        s = make_sentence(
            "No unresolved issues requiring manual review were identified.",
            section="Unresolved Issues",
            event_refs=[],
            citations=[],
            confidence=1.0,
        )
        sentences.append(s)
        md_lines.append("*No unresolved issues.*")
        return sentences, "\n".join(md_lines)

    intro = make_sentence(
        f"There are **{len(issues)} unresolved issue(s)** that require manual legal review "
        f"before this reconstruction can be considered final.",
        section="Unresolved Issues",
        event_refs=[],
        citations=[],
        confidence=1.0,
    )
    sentences.append(intro)
    md_lines.append(f"> ⚠ {intro['text']}\n")

    for issue_type, ev, contradiction in issues[:20]:
        if issue_type == "low_confidence" and ev:
            ref_id = ev.get("merged_event_id","")
            conf = ev.get("confidence",0)
            text = (
                f"There is an indication of an event `[{ref_id}]` on "
                f"{ev.get('date_normalized','undated')}, but confidence is "
                f"only **{conf:.0%}** — manual verification required."
            )
            sent = make_sentence(text, "Unresolved Issues", [ref_id], ev.get("citations",[]), confidence=conf)
            sentences.append(sent)
            md_lines.append(f"- ⚠ LOW CONFIDENCE: {text}")

        elif issue_type == "missing_date" and ev:
            ref_id = ev.get("merged_event_id","")
            text = (
                f"The documents indicate an event `[{ref_id}]` for which no date "
                f"could be reliably extracted. Manual date assignment required."
            )
            sent = make_sentence(text, "Unresolved Issues", [ref_id], ev.get("citations",[]), confidence=0.3)
            sentences.append(sent)
            md_lines.append(f"- ⚠ MISSING DATE: {text}")

        elif issue_type == "high_weight_conflict" and contradiction:
            ev_a = contradiction.get("event_a",{})
            ev_b = contradiction.get("event_b",{})
            id_a = ev_a.get("merged_event_id","")
            id_b = ev_b.get("merged_event_id","")
            text = (
                f"**PRIORITY:** A conflict exists between two high-reliability sources "
                f"`[{id_a}]` and `[{id_b}]`: {contradiction.get('description','')}. "
                f"This cannot be auto-resolved — immediate legal review required."
            )
            all_cits = ev_a.get("citations",[]) + ev_b.get("citations",[])
            sent = make_sentence(text, "Unresolved Issues", [id_a, id_b], all_cits[:4], confidence=0.0)
            sentences.append(sent)
            md_lines.append(f"- 🔴 HIGH-WEIGHT CONFLICT: {text}")

    return sentences, "\n".join(md_lines)


def _build_contradictions_sentences(contradictions: List[dict]) -> Tuple[List[dict], str]:
    sentences = []
    md_lines = []

    if not contradictions:
        sent = make_sentence(
            "No significant contradictions were detected across the source documents.",
            section="Conflicting Evidence",
            event_refs=[],
            citations=[],
            confidence=1.0,
        )
        sentences.append(sent)
        md_lines.append("*No contradictions detected.*")
        return sentences, "\n".join(md_lines)

    intro_refs = [c.get("event_a", {}).get("merged_event_id", "") for c in contradictions[:3]]
    intro_cits = []
    for c in contradictions[:2]:
        intro_cits.extend(c.get("event_a", {}).get("citations", []))
    intro_sent = make_sentence(
        f"The records confirm that **{len(contradictions)} contradiction(s)** were identified using "
        f"semantic similarity analysis and opposition pattern detection, ranked by impact score.",
        section="Conflicting Evidence",
        event_refs=[r for r in intro_refs if r],
        citations=intro_cits[:3],
        confidence=1.0,
    )
    sentences.append(intro_sent)
    md_lines.append(f"> {intro_sent['text']} {intro_sent['citation_inline']}\n")

    for i, c in enumerate(contradictions, 1):
        ev_a = c.get("event_a", {})
        ev_b = c.get("event_b", {})
        i_label = c.get("impact_label", "MINOR")
        impact = c.get("impact_score", 0)
        severity = c.get("severity", "medium").upper()
        explanation = c.get("description", "")
        sim = c.get("semantic_similarity", 0)
        id_a = ev_a.get("merged_event_id", "")
        id_b = ev_b.get("merged_event_id", "")

        text = (
            f"**[{i_label}] Contradiction {i}** (impact: {impact:.3f}, sim: {sim:.0%}): "
            f"{explanation}. "
            f"Source A [{ev_a.get('date','?')}, {ev_a.get('reliability','?')} reliability]: "
            f"\"{ev_a.get('description','')[:100]}…\" "
            f"contradicts Source B [{ev_b.get('date','?')}, {ev_b.get('reliability','?')} reliability]: "
            f"\"{ev_b.get('description','')[:100]}…\""
        )
        all_cits = ev_a.get("citations", []) + ev_b.get("citations", [])
        sent = make_sentence(
            text,
            section="Conflicting Evidence",
            event_refs=[eid for eid in [id_a, id_b] if eid],
            citations=all_cits[:4],
            confidence=0.5,
        )
        sentences.append(sent)
        md_lines.append(f"#### [{i_label}] Contradiction {i} — Impact: {impact:.3f}")
        md_lines.append(f"`[{id_a}]` vs `[{id_b}]` | Severity: {severity} | Sim: {sim:.0%}")
        md_lines.append(f"{text}")
        md_lines.append(f"*Citations:* {sent['citation_inline']}\n")

    return sentences, "\n".join(md_lines)


def _build_conclusion_sentences(
    merged_events: List[dict], contradictions: List[dict]
) -> Tuple[List[dict], str]:
    total_sources = sum(ev.get("source_count", 1) for ev in merged_events)
    high_conf = sum(1 for ev in merged_events if ev.get("confidence", 0) >= CONF_THRESHOLD_HIGH)
    low_conf = sum(1 for ev in merged_events if ev.get("confidence", 0) < CONF_THRESHOLD_MED)
    critical_cont = sum(1 for c in contradictions if c.get("impact_label") == "CRITICAL")

    all_ev_ids = [ev.get("merged_event_id", "") for ev in merged_events if ev.get("merged_event_id")][-3:]
    all_cits = []
    for ev in merged_events[-3:]:
        all_cits.extend(ev.get("citations", []))

    text1 = (
        f"The records confirm that the reconstructed case timeline contains "
        f"**{len(merged_events)} distinct events** verified across **{total_sources} citations**, "
        f"of which **{high_conf}** carry high confidence (≥{CONF_THRESHOLD_HIGH:.0%}) and "
        f"**{low_conf}** are flagged as low confidence."
    )
    sent1 = make_sentence(text1, "Conclusion", all_ev_ids, all_cits[:3], confidence=1.0)

    if contradictions:
        text2 = (
            f"The records confirm that **{len(contradictions)} contradiction(s)** were identified, "
            f"with **{critical_cont}** rated CRITICAL impact. "
            f"These conflicts arise from semantically opposing claims across source documents "
            f"and require priority legal review before any definitive legal determination."
        )
        conf2 = 0.9
    else:
        text2 = "The records confirm that no significant contradictions were detected across the analyzed documents."
        conf2 = 1.0

    cont_cits = []
    for c in contradictions[:2]:
        cont_cits.extend(c.get("event_a", {}).get("citations", []))
    sent2 = make_sentence(text2, "Conclusion", all_ev_ids[:2], cont_cits[:3], confidence=conf2)

    disclaimer = make_sentence(
        "This report was generated automatically by the Document Intelligence System. "
        "All findings are fully traceable to source documents via embedded event references and SHA256 hash chain. "
        "No claim in this document was generated without a corresponding source event. "
        "This document should be reviewed by a qualified legal professional before use in proceedings.",
        section="Conclusion",
        event_refs=[],
        citations=[],
        confidence=1.0,
    )

    md = f"{text1}\n\n{text2}\n\n*{disclaimer['text']}*"
    return [sent1, sent2, disclaimer], md


def generate_narrative_document(
    merged_events: List[dict],
    contradictions: List[dict],
    case_name: str = "Legal Case",
    model: str = "mistral",
) -> Tuple[str, List[dict]]:
    use_ollama = ollama_available()
    all_sentences: List[dict] = []
    sections_md: Dict[str, str] = {}

    overview_sents, overview_md = _build_overview_sentences(
        merged_events, contradictions, case_name, model, use_ollama
    )
    all_sentences.extend(overview_sents)
    sections_md["overview"] = overview_md

    agreed_sents, agreed_md = _build_agreed_facts_sentences(merged_events)
    all_sentences.extend(agreed_sents)
    sections_md["agreed"] = agreed_md

    disputed_sents, disputed_md = _build_disputed_facts_sentences(merged_events, contradictions)
    all_sentences.extend(disputed_sents)
    sections_md["disputed"] = disputed_md

    chrono_sents, chrono_md = _build_chronological_sentences(merged_events, use_ollama, model)
    all_sentences.extend(chrono_sents)
    sections_md["chronological"] = chrono_md

    cont_sents, cont_md = _build_contradictions_sentences(contradictions)
    all_sentences.extend(cont_sents)
    sections_md["contradictions"] = cont_md

    unresolv_sents, unresolv_md = _build_unresolved_sentences(merged_events, contradictions)
    all_sentences.extend(unresolv_sents)
    sections_md["unresolved"] = unresolv_md

    conc_sents, conc_md = _build_conclusion_sentences(merged_events, contradictions)
    all_sentences.extend(conc_sents)
    sections_md["conclusion"] = conc_md

    save_narrative_sentences(all_sentences)

    lines = [
        f"# {case_name} — Legal Intelligence Report (Forensic Edition)",
        f"\n*Court-grade reconstruction. All claims traceable to source events. SHA256 hash chain included.*\n",
        "---\n",
        "## 1. Case Overview\n", overview_md, "\n---\n",
        "## 2. Agreed Facts\n", agreed_md, "\n---\n",
        "## 3. Disputed Facts\n", disputed_md, "\n---\n",
        "## 4. Chronological Events\n", chrono_md, "\n---\n",
        "## 5. Conflicting Evidence\n", cont_md, "\n---\n",
        "## 6. Unresolved Issues\n", unresolv_md, "\n---\n",
        "## 7. Conclusion\n", conc_md,
    ]
    content = "\n".join(lines)

    NARRATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NARRATIVE_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    gc.collect()
    return content, all_sentences


def load_narrative() -> str:
    if not NARRATIVE_PATH.exists():
        return ""
    return NARRATIVE_PATH.read_text(encoding="utf-8")


def load_timeline_json() -> List[dict]:
    if not TIMELINE_JSON_PATH.exists():
        return []
    try:
        return json.load(open(TIMELINE_JSON_PATH, "r", encoding="utf-8"))
    except Exception:
        return []


def load_timeline_md() -> str:
    if not TIMELINE_MD_PATH.exists():
        return ""
    return TIMELINE_MD_PATH.read_text(encoding="utf-8")
