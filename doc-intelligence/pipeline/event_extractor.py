"""
Event Extraction Engine.
Extracts structured legal events from text chunks.
"""
import re
import gc
import json
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict

from pipeline.date_normalizer import extract_all_dates, normalize_date

logger = logging.getLogger(__name__)

EVENTS_DIR = Path("data/outputs/events")

EVENT_TYPES = {
    "contract": [
        "contract", "agreement", "عقد", "اتفاقية", "اتفاق", "signed", "executed",
        "موقع", "توقيع", "مبرم", "مبرمة",
    ],
    "payment": [
        "payment", "paid", "invoice", "amount", "دفع", "مبلغ", "فاتورة", "سداد",
        "تحويل", "transfer", "wire", "cheque", "check", "شيك",
    ],
    "meeting": [
        "meeting", "session", "اجتماع", "جلسة", "لقاء", "conference", "مؤتمر",
        "attended", "حضر", "discussed", "ناقش",
    ],
    "incident": [
        "incident", "accident", "حادث", "واقعة", "حادثة", "occurred", "وقع",
        "happened", "breach", "violation", "انتهاك", "خرق",
    ],
    "claim": [
        "claim", "alleged", "ادعاء", "مطالبة", "يدعي", "زعم", "claimed",
        "lawsuit", "دعوى", "complaint", "شكوى",
    ],
    "decision": [
        "decision", "ruling", "judgment", "حكم", "قرار", "order", "أمر",
        "decree", "مرسوم", "verdict", "حكم",
    ],
    "correspondence": [
        "letter", "email", "notice", "رسالة", "إشعار", "خطاب", "notification",
        "correspondence", "مراسلة", "sent", "received", "أرسل", "استلم",
    ],
}

ACTOR_PATTERNS = [
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b",
    r"(?:Mr\.?|Mrs\.?|Dr\.?|Prof\.?|Eng\.?)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
    r"(?:السيد|السيدة|الدكتور|المهندس|الأستاذ)\s+([\u0600-\u06FF\s]{3,30})",
]

LOCATION_PATTERNS = [
    r"\bin\s+([A-Z][a-zA-Z\s,]+(?:City|Town|Court|Office|Center|Building))\b",
    r"\bat\s+([A-Z][a-zA-Z\s,]+)\b",
    r"(?:في|بـ|ب)\s+([\u0600-\u06FF\s]{2,20})",
]


def detect_language(text: str) -> str:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    total_alpha = len(re.findall(r"[a-zA-Z\u0600-\u06FF]", text))
    if total_alpha == 0:
        return "en"
    ratio = arabic_chars / total_alpha
    if ratio > 0.6:
        return "ar"
    elif ratio > 0.3:
        return "mixed"
    return "en"


def detect_event_type(text: str) -> str:
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for etype, keywords in EVENT_TYPES.items():
        count = sum(1 for kw in keywords if kw.lower() in text_lower)
        if count > 0:
            scores[etype] = count
    if not scores:
        return "general"
    return max(scores, key=lambda k: scores[k])


def extract_actors(text: str) -> List[str]:
    actors = []
    seen = set()
    for pattern in ACTOR_PATTERNS:
        for m in re.finditer(pattern, text):
            actor = m.group(1).strip() if m.lastindex else m.group(0).strip()
            actor = " ".join(actor.split())
            if actor and actor not in seen and len(actor) >= 3:
                actors.append(actor)
                seen.add(actor)
    return actors[:8]


def extract_location(text: str) -> str:
    for pattern in LOCATION_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
    return ""


def build_description(text: str, max_len: int = 300) -> str:
    sentences = re.split(r"[.!?\n؟،]", text)
    desc_parts = []
    total = 0
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        if total + len(sent) > max_len:
            break
        desc_parts.append(sent)
        total += len(sent)
    return ". ".join(desc_parts) if desc_parts else text[:max_len]


def extract_events_from_chunk(chunk: dict) -> List[dict]:
    text = chunk.get("text", "")
    if not text or len(text) < 30:
        return []

    dates = extract_all_dates(text)
    if not dates:
        return []

    language = detect_language(text)
    event_type = detect_event_type(text)
    actors = extract_actors(text)
    location = extract_location(text)
    description = build_description(text)

    events = []
    # Group text by sentences to find which date belongs to which part of the text
    sentences = re.split(r"[.!?\n؟،]", text)
    
    for date_info in dates:
        # Find the sentence containing this date to build a better description
        relevant_desc = description
        for sent in sentences:
            if date_info["raw"] in sent:
                if len(sent) > 30:
                    relevant_desc = sent.strip()
                break
                
        event_id = str(uuid.uuid4())[:8]
        event = {
            "event_id": event_id,
            "date_raw": date_info["raw"],
            "date_normalized": date_info["iso"],
            "event_type": event_type,
            "description": relevant_desc,
            "actors": actors,
            "location": location,
            "source_file": chunk.get("file_name", ""),
            "file_id": chunk.get("file_id", ""),
            "page": str(chunk.get("page", "")),
            "chunk_id": chunk.get("chunk_id", ""),
            "language": language,
            "confidence": round(date_info["confidence"] * 0.9, 2),
        }
        events.append(event)

    return events


def save_events_for_file(file_id: str, events: List[dict]) -> None:
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVENTS_DIR / f"{file_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def load_events_for_file(file_id: str) -> List[dict]:
    path = EVENTS_DIR / f"{file_id}.jsonl"
    if not path.exists():
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return events


def load_all_events() -> List[dict]:
    if not EVENTS_DIR.exists():
        return []
    events = []
    for path in sorted(EVENTS_DIR.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
    return events


def extract_all_file_events(file_id: str, chunks: List[dict]) -> List[dict]:
    all_events = []
    for chunk in chunks:
        evs = extract_events_from_chunk(chunk)
        all_events.extend(evs)
        gc.collect()
    save_events_for_file(file_id, all_events)
    return all_events
