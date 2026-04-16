"""
Advanced Contradiction Detection Engine — Forensic Edition.

Detection methods:
1. Embedding cosine similarity (topic proximity)
2. Opposition keyword pairs (Arabic + English)
3. Date conflict detection
4. Negation pattern analysis

Impact scoring formula:
  impact_score = severity_num × max_source_weight × frequency_factor × semantic_similarity

Impact labels: CRITICAL / SIGNIFICANT / MINOR
Output ranked by impact_score descending.
"""
import gc
import json
import logging
import numpy as np
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CONTRADICTIONS_PATH = Path("data/outputs/contradictions.json")

DATE_CONFLICT_MIN_DAYS = 5
SIM_LIKELY_SAME_TOPIC = 0.40
SIM_HIGH_SIMILARITY = 0.70

SEVERITY_SCORE = {"high": 3, "medium": 2, "low": 1}
WEIGHT_MAP = {"high": 1.0, "medium": 0.70, "low": 0.40}


def parse_iso(iso: str):
    try:
        from datetime import date
        parts = iso.split("-")
        if len(parts) >= 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        elif len(parts) == 1:
            return date(int(parts[0]), 1, 1)
    except Exception:
        pass
    return None


def dates_conflict(d1: str, d2: str, min_days: int = DATE_CONFLICT_MIN_DAYS) -> bool:
    p1 = parse_iso(d1)
    p2 = parse_iso(d2)
    if p1 is None or p2 is None:
        return False
    return abs((p1 - p2).days) >= min_days


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


NEGATION_PREFIXES = [
    "not", "no", "never", "didn't", "did not", "wasn't", "was not",
    "couldn't", "could not", "hasn't", "has not", "haven't", "have not",
    "لم", "لا", "لن", "ليس", "ليست", "غير", "عدم",
]

OPPOSITION_PAIRS = [
    (["paid", "payment made", "transferred", "دفع", "سداد", "تحويل"],
     ["unpaid", "not paid", "failed to pay", "لم يدفع", "عدم الدفع", "رفض الدفع"]),
    (["signed", "executed", "موقع", "توقيع", "وقّع"],
     ["unsigned", "not signed", "refused to sign", "لم يوقع", "رفض التوقيع"]),
    (["delivered", "تسليم", "تم التسليم", "سلّم"],
     ["not delivered", "failed to deliver", "لم يتم التسليم", "رفض التسليم"]),
    (["agreed", "approved", "وافق", "اتفق", "قبل", "موافقة"],
     ["disagreed", "rejected", "refused", "رفض", "لم يوافق", "اعترض"]),
    (["present", "attended", "حضر", "حضور"],
     ["absent", "did not attend", "غاب", "غياب", "لم يحضر"]),
    (["completed", "finished", "أكمل", "انتهى", "أنجز"],
     ["incomplete", "unfinished", "لم يكتمل", "لم ينته", "فشل"]),
    (["valid", "legal", "صالح", "قانوني", "نافذ"],
     ["invalid", "illegal", "void", "باطل", "لاغ", "غير قانوني"]),
    (["received", "accepted", "استلم", "قبل", "وصل"],
     ["rejected", "returned", "refused", "رُفض", "لم يستلم", "أعاد"]),
    (["authorized", "permitted", "مرخص", "مسموح", "مأذون"],
     ["unauthorized", "prohibited", "غير مرخص", "محظور", "ممنوع"]),
    (["present", "existing", "موجود", "قائم"],
     ["absent", "missing", "none", "غائب", "مفقود", "لا يوجد"]),
]


def detect_opposition_keywords(t1: str, t2: str) -> Optional[str]:
    l1, l2 = t1.lower(), t2.lower()
    for positives, negatives in OPPOSITION_PAIRS:
        pos_in_1 = any(kw in l1 for kw in positives)
        neg_in_1 = any(kw in l1 for kw in negatives)
        pos_in_2 = any(kw in l2 for kw in positives)
        neg_in_2 = any(kw in l2 for kw in negatives)
        if (pos_in_1 and neg_in_2) or (neg_in_1 and pos_in_2):
            pos_word = next((kw for kw in positives if kw in l1 or kw in l2), positives[0])
            neg_word = next((kw for kw in negatives if kw in l1 or kw in l2), negatives[0])
            return f"Opposing claims: '{pos_word}' vs '{neg_word}'"
    return None


def has_negation(text: str) -> bool:
    tl = text.lower()
    return any(f" {neg} " in tl or tl.startswith(neg + " ") for neg in NEGATION_PREFIXES)


def compute_contradiction_severity(
    sim: float,
    d1: Optional[str],
    d2: Optional[str],
    rel1: str,
    rel2: str,
    has_keyword_conflict: bool,
) -> str:
    base = 0
    if has_keyword_conflict:
        base += 3
    if sim >= SIM_HIGH_SIMILARITY:
        base += 2
    elif sim >= SIM_LIKELY_SAME_TOPIC:
        base += 1
    if rel1 == "high" or rel2 == "high":
        base += 2
    if d1 and d2:
        p1, p2 = parse_iso(d1), parse_iso(d2)
        if p1 and p2:
            gap = abs((p1 - p2).days)
            if gap > 30:
                base += 1
    if base >= 5:
        return "high"
    elif base >= 3:
        return "medium"
    return "low"


def compute_impact_score(
    severity: str,
    rel1: str,
    rel2: str,
    sim: float,
    frequency: int = 1,
) -> float:
    """
    impact_score = severity_num × max_source_weight × frequency_factor × semantic_similarity
    """
    sev_num = SEVERITY_SCORE.get(severity, 1)
    max_weight = max(WEIGHT_MAP.get(rel1, 0.7), WEIGHT_MAP.get(rel2, 0.7))
    freq_factor = min(1.0 + (frequency - 1) * 0.1, 2.0)
    return round(sev_num * max_weight * freq_factor * max(sim, 0.01), 4)


def impact_label_from_score(score: float) -> str:
    if score >= 1.5:
        return "CRITICAL"
    elif score >= 0.8:
        return "SIGNIFICANT"
    return "MINOR"


def embed_descriptions_for_contradiction(descriptions: List[str], model) -> np.ndarray:
    if not descriptions:
        return np.array([])
    embeddings = model.encode(
        descriptions,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype="float32")


def find_contradictions_semantic(
    merged_events: List[dict],
    model=None,
    sim_topic_threshold: float = SIM_LIKELY_SAME_TOPIC,
) -> List[dict]:
    """
    Detect contradictions across merged events using:
    1. Embedding cosine similarity (topic proximity)
    2. Opposition keyword detection (Arabic + English)
    3. Date conflict detection
    4. Negation pattern analysis

    Output sorted by impact_score descending (CRITICAL first).
    """
    contradictions = []

    descriptions = [ev.get("description", "") for ev in merged_events]
    embeddings = None

    if model is not None and descriptions:
        try:
            embeddings = embed_descriptions_for_contradiction(descriptions, model)
        except Exception as e:
            logger.warning(f"Embedding for contradiction failed: {e}")
            embeddings = None

    same_type_groups: Dict[str, List[int]] = {}
    for i, ev in enumerate(merged_events):
        t = ev.get("event_type", "general")
        same_type_groups.setdefault(t, []).append(i)

    checked_pairs = set()

    for etype, indices in same_type_groups.items():
        for a_pos, i in enumerate(indices):
            for j in indices[a_pos + 1:]:
                pair_key = (min(i, j), max(i, j))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                ev1, ev2 = merged_events[i], merged_events[j]
                d1 = ev1.get("date_normalized", "")
                d2 = ev2.get("date_normalized", "")
                desc1 = ev1.get("description", "")
                desc2 = ev2.get("description", "")
                rel1 = ev1.get("source_reliability", "medium")
                rel2 = ev2.get("source_reliability", "medium")

                sim = 0.0
                if embeddings is not None and i < len(embeddings) and j < len(embeddings):
                    sim = cosine_sim(embeddings[i], embeddings[j])

                if sim < sim_topic_threshold:
                    continue

                keyword_conflict = detect_opposition_keywords(desc1, desc2)
                date_conflict = dates_conflict(d1, d2) if d1 and d2 else False

                negation_conflict = (
                    (has_negation(desc1) and not has_negation(desc2)) or
                    (has_negation(desc2) and not has_negation(desc1))
                ) and sim >= 0.55

                if not (keyword_conflict or date_conflict or negation_conflict):
                    continue

                explanation_parts = []
                if keyword_conflict:
                    explanation_parts.append(keyword_conflict)
                if date_conflict:
                    explanation_parts.append(f"Date conflict: {d1} vs {d2}")
                if negation_conflict:
                    explanation_parts.append("Negation pattern between related statements")

                explanation = "; ".join(explanation_parts)
                severity = compute_contradiction_severity(
                    sim, d1, d2, rel1, rel2,
                    has_keyword_conflict=bool(keyword_conflict),
                )
                impact_score = compute_impact_score(severity, rel1, rel2, sim)
                i_label = impact_label_from_score(impact_score)

                contradiction = {
                    "type": "date_conflict" if date_conflict and not keyword_conflict else "semantic_contradiction",
                    "event_type": etype,
                    "description": explanation,
                    "semantic_similarity": round(sim, 3),
                    "severity": severity,
                    "impact_score": impact_score,
                    "impact_label": i_label,
                    "event_a": {
                        "merged_event_id": ev1.get("merged_event_id", ""),
                        "date": d1,
                        "description": desc1[:250],
                        "reliability": rel1,
                        "confidence": ev1.get("confidence", 0),
                        "source_weight": WEIGHT_MAP.get(rel1, 0.7),
                        "citations": ev1.get("citations", []),
                    },
                    "event_b": {
                        "merged_event_id": ev2.get("merged_event_id", ""),
                        "date": d2,
                        "description": desc2[:250],
                        "reliability": rel2,
                        "confidence": ev2.get("confidence", 0),
                        "source_weight": WEIGHT_MAP.get(rel2, 0.7),
                        "citations": ev2.get("citations", []),
                    },
                }
                contradictions.append(contradiction)

    if embeddings is not None:
        del embeddings
        gc.collect()

    contradictions.sort(key=lambda c: -c.get("impact_score", 0))
    return contradictions


def save_contradictions(contradictions: List[dict]) -> None:
    CONTRADICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTRADICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(contradictions, f, indent=2, ensure_ascii=False)


def load_contradictions() -> List[dict]:
    if not CONTRADICTIONS_PATH.exists():
        return []
    try:
        return json.load(open(CONTRADICTIONS_PATH, "r", encoding="utf-8"))
    except Exception:
        return []
