"""
Source Reliability Weighting.
Assigns reliability levels to documents based on filename patterns and content signals.
Propagates weights into events for timeline prioritization and contradiction scoring.
"""
import gc
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

RELIABILITY_PATH = Path("data/outputs/reliability.json")

LEVEL_WEIGHTS = {"high": 1.0, "medium": 0.70, "low": 0.40}

HIGH_SIGNALS = [
    r"court", r"judgment", r"ruling", r"verdict", r"decree", r"order",
    r"notary", r"official", r"ministry", r"tribunal", r"arbitration",
    r"sworn", r"authenticated", r"certified", r"محكمة", r"حكم", r"قرار",
    r"وزارة", r"رسمي", r"موثق", r"توثيق", r"قضاء",
    r"contract", r"agreement", r"deed", r"عقد", r"اتفاقية",
    r"affidavit", r"testimony", r"شهادة",
]

MEDIUM_SIGNALS = [
    r"email", r"letter", r"notice", r"report", r"invoice",
    r"memo", r"correspondence", r"notification",
    r"رسالة", r"إشعار", r"تقرير", r"فاتورة", r"مذكرة",
]

LOW_SIGNALS = [
    r"note", r"draft", r"temp", r"copy", r"scan", r"photo",
    r"screenshot", r"whatsapp", r"chat", r"message",
    r"ملاحظة", r"مسودة", r"صورة",
]

HIGH_EXTENSIONS = {".pdf", ".docx", ".doc"}
MEDIUM_EXTENSIONS = {".txt", ".xlsx", ".xls", ".csv", ".eml"}
LOW_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _match_signals(text: str, patterns: list) -> int:
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower, re.IGNORECASE))


def classify_reliability(file_name: str, file_ext: str = "", content_preview: str = "") -> dict:
    """
    Classify a document's reliability and return a record with level, weight, and reasons.
    """
    combined = f"{file_name} {content_preview[:300]}"

    high_score = _match_signals(combined, HIGH_SIGNALS)
    medium_score = _match_signals(combined, MEDIUM_SIGNALS)
    low_score = _match_signals(combined, LOW_SIGNALS)

    ext = file_ext.lower()
    if ext in HIGH_EXTENSIONS:
        high_score += 2
    elif ext in MEDIUM_EXTENSIONS:
        medium_score += 1
    elif ext in LOW_EXTENSIONS:
        low_score += 2

    reasons = []
    if high_score >= medium_score and high_score >= low_score and high_score > 0:
        level = "high"
        reasons.append(f"{high_score} official/legal signal(s) detected")
    elif medium_score >= low_score and medium_score > 0:
        level = "medium"
        reasons.append(f"{medium_score} communication/report signal(s) detected")
    elif low_score > 0:
        level = "low"
        reasons.append(f"{low_score} informal/unclear signal(s) detected")
    else:
        level = "medium"
        reasons.append("No specific signals; defaulting to medium")

    return {
        "file_name": file_name,
        "level": level,
        "weight": LEVEL_WEIGHTS[level],
        "reasons": reasons,
        "scores": {"high": high_score, "medium": medium_score, "low": low_score},
    }


def build_reliability_map(queue_items: list) -> Dict[str, dict]:
    """Build and persist reliability map for all queued files."""
    reliability = {}
    for item in queue_items:
        if item.get("status") != "done":
            continue
        file_id = item["id"]
        file_name = item.get("name", "")
        ext = item.get("ext", "")
        rec = classify_reliability(file_name, ext)
        reliability[file_id] = rec

    RELIABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RELIABILITY_PATH, "w", encoding="utf-8") as f:
        json.dump(reliability, f, indent=2, ensure_ascii=False)

    logger.info(f"Classified {len(reliability)} documents for reliability")
    return reliability


def load_reliability_map() -> Dict[str, dict]:
    if not RELIABILITY_PATH.exists():
        return {}
    with open(RELIABILITY_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def get_weight(file_id: str, reliability_map: Dict[str, dict]) -> float:
    rec = reliability_map.get(file_id)
    if rec:
        return rec.get("weight", LEVEL_WEIGHTS["medium"])
    return LEVEL_WEIGHTS["medium"]


def get_level(file_id: str, reliability_map: Dict[str, dict]) -> str:
    rec = reliability_map.get(file_id)
    if rec:
        return rec.get("level", "medium")
    return "medium"


def apply_reliability_to_event(event: dict, reliability_map: Dict[str, dict]) -> dict:
    """Inject reliability weight and level into an event dict."""
    file_id = event.get("file_id", "")
    weight = get_weight(file_id, reliability_map)
    level = get_level(file_id, reliability_map)
    event["source_reliability"] = level
    event["source_weight"] = weight
    event["confidence"] = round(event.get("confidence", 0.5) * (0.5 + 0.5 * weight), 3)
    return event
