import gc
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path("data/outputs")
SUMMARIES_DIR = OUTPUTS_DIR / "summaries"
ENTITIES_DIR = OUTPUTS_DIR / "entities"
TIMELINE_DIR = OUTPUTS_DIR / "timeline"


def extract_dates(text: str) -> List[str]:
    date_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}\b",
    ]
    found = []
    for pat in date_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        found.extend(matches)
    return list(set(found))


def extract_basic_entities(text: str) -> dict:
    entities = {"emails": [], "urls": [], "numbers": [], "capitalized_phrases": []}

    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    entities["emails"] = list(set(emails))[:10]

    urls = re.findall(r"https?://[^\s]+", text)
    entities["urls"] = list(set(urls))[:10]

    cap_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b", text)
    phrase_counts: Dict[str, int] = {}
    for p in cap_phrases:
        phrase_counts[p] = phrase_counts.get(p, 0) + 1
    top_phrases = sorted(phrase_counts.items(), key=lambda x: -x[1])[:20]
    entities["capitalized_phrases"] = [{"phrase": p, "count": c} for p, c in top_phrases]

    return entities


def generate_file_summary(file_id: str, file_name: str, chunks: List[dict]) -> dict:
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARIES_DIR / f"{file_id}.json"

    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)

    full_text = " ".join(c["text"] for c in chunks[:10])

    summary = {
        "file_id": file_id,
        "file_name": file_name,
        "chunk_count": len(chunks),
        "total_words": sum(c.get("word_count", 0) for c in chunks),
        "pages": sorted(set(c["page"] for c in chunks)),
        "preview": full_text[:500] if full_text else "",
        "dates": extract_dates(full_text),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    del full_text
    gc.collect()
    return summary


def generate_entities_report(file_id: str, file_name: str, chunks: List[dict]) -> dict:
    ENTITIES_DIR.mkdir(parents=True, exist_ok=True)
    entities_path = ENTITIES_DIR / f"{file_id}.json"

    if entities_path.exists():
        with open(entities_path, "r") as f:
            return json.load(f)

    text_sample = " ".join(c["text"] for c in chunks[:5])
    entities = extract_basic_entities(text_sample)
    entities["file_id"] = file_id
    entities["file_name"] = file_name

    with open(entities_path, "w") as f:
        json.dump(entities, f, indent=2)

    del text_sample
    gc.collect()
    return entities


def generate_global_timeline() -> List[dict]:
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
    timeline_path = TIMELINE_DIR / "timeline.json"

    from pipeline.chunker import CHUNKS_DIR
    import os

    events = []
    if not CHUNKS_DIR.exists():
        return events

    for chunk_file in sorted(CHUNKS_DIR.glob("*.jsonl")):
        file_id = chunk_file.stem
        chunks = []
        with open(chunk_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except Exception:
                        pass
        if not chunks:
            continue

        file_name = chunks[0].get("file_name", file_id)
        for chunk in chunks[:3]:
            dates = extract_dates(chunk.get("text", ""))
            for d in dates[:3]:
                events.append({"date": d, "file_name": file_name, "file_id": file_id, "page": chunk.get("page", 0)})

        del chunks
        gc.collect()

    events.sort(key=lambda e: e["date"])

    with open(timeline_path, "w") as f:
        json.dump(events[:200], f, indent=2)

    return events[:200]


def load_global_timeline() -> List[dict]:
    timeline_path = TIMELINE_DIR / "timeline.json"
    if not timeline_path.exists():
        return []
    with open(timeline_path, "r") as f:
        return json.load(f)


def load_all_summaries() -> List[dict]:
    if not SUMMARIES_DIR.exists():
        return []
    summaries = []
    for p in sorted(SUMMARIES_DIR.glob("*.json")):
        with open(p, "r") as f:
            try:
                summaries.append(json.load(f))
            except Exception:
                pass
    return summaries


def load_all_entities() -> List[dict]:
    if not ENTITIES_DIR.exists():
        return []
    entities = []
    for p in sorted(ENTITIES_DIR.glob("*.json")):
        with open(p, "r") as f:
            try:
                entities.append(json.load(f))
            except Exception:
                pass
    return entities
