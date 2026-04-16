"""
Event Merging Engine.
Groups events by date proximity and semantic similarity, merging duplicates
while preserving all source citations.
"""
import gc
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import date, timedelta

logger = logging.getLogger(__name__)

MERGED_EVENTS_PATH = Path("data/outputs/merged_events.jsonl")
DATE_PROXIMITY_DAYS = 3


def parse_iso(iso: str) -> Optional[date]:
    try:
        parts = iso.split("-")
        if len(parts) == 1:
            return date(int(parts[0]), 1, 1)
        elif len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        else:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def dates_are_close(d1: str, d2: str, max_days: int = DATE_PROXIMITY_DAYS) -> bool:
    p1 = parse_iso(d1)
    p2 = parse_iso(d2)
    if p1 is None or p2 is None:
        return False
    return abs((p1 - p2).days) <= max_days


def keyword_overlap(desc1: str, desc2: str, threshold: float = 0.25) -> float:
    words1 = set(desc1.lower().split())
    words2 = set(desc2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)


def events_should_merge(ev1: dict, ev2: dict) -> bool:
    d1 = ev1.get("date_normalized", "")
    d2 = ev2.get("date_normalized", "")

    if not d1 or not d2:
        return False

    if d1 == d2 or dates_are_close(d1, d2):
        if ev1.get("event_type") == ev2.get("event_type"):
            return True
        overlap = keyword_overlap(
            ev1.get("description", ""), ev2.get("description", "")
        )
        if overlap >= threshold_for_types(ev1.get("event_type"), ev2.get("event_type")):
            return True

    return False


def threshold_for_types(t1: Optional[str], t2: Optional[str]) -> float:
    if t1 == t2:
        return 0.20
    return 0.35


def merge_event_group(group: List[dict]) -> dict:
    group_sorted = sorted(group, key=lambda e: e.get("confidence", 0), reverse=True)
    primary = group_sorted[0]

    all_actors = []
    seen_actors = set()
    for ev in group:
        for actor in ev.get("actors", []):
            if actor not in seen_actors:
                all_actors.append(actor)
                seen_actors.add(actor)

    citations = []
    seen_citations = set()
    for ev in group:
        key = (ev.get("source_file", ""), str(ev.get("page", "")))
        if key not in seen_citations:
            citations.append(
                {
                    "source_file": ev.get("source_file", ""),
                    "file_id": ev.get("file_id", ""),
                    "page": ev.get("page", ""),
                    "chunk_id": ev.get("chunk_id", ""),
                    "language": ev.get("language", "en"),
                    "confidence": ev.get("confidence", 0.0),
                }
            )
            seen_citations.add(key)

    descriptions = list(
        dict.fromkeys(ev.get("description", "") for ev in group if ev.get("description"))
    )

    locations = list(
        dict.fromkeys(ev.get("location", "") for ev in group if ev.get("location"))
    )

    merged = {
        "merged_event_id": primary["event_id"],
        "date_normalized": primary["date_normalized"],
        "date_raw": primary["date_raw"],
        "event_type": primary["event_type"],
        "description": descriptions[0] if descriptions else "",
        "all_descriptions": descriptions[:3],
        "actors": all_actors,
        "location": locations[0] if locations else "",
        "citations": citations,
        "source_count": len(citations),
        "confidence": round(max(ev.get("confidence", 0) for ev in group), 2),
        "languages": list(set(ev.get("language", "en") for ev in group)),
    }
    return merged


def cluster_events(events: List[dict]) -> List[List[dict]]:
    if not events:
        return []

    clusters: List[List[dict]] = []
    assigned = [False] * len(events)

    for i, ev in enumerate(events):
        if assigned[i]:
            continue
        cluster = [ev]
        assigned[i] = True
        for j, other in enumerate(events):
            if assigned[j] or j == i:
                continue
            if events_should_merge(ev, other):
                cluster.append(other)
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def merge_all_events(events: List[dict]) -> List[dict]:
    dated = [ev for ev in events if ev.get("date_normalized")]
    undated = [ev for ev in events if not ev.get("date_normalized")]

    dated_sorted = sorted(dated, key=lambda e: e.get("date_normalized", ""))

    clusters = cluster_events(dated_sorted)
    merged = [merge_event_group(cluster) for cluster in clusters]

    for ev in undated:
        merged.append(merge_event_group([ev]))

    merged.sort(key=lambda e: e.get("date_normalized", "9999"))

    del dated, undated, dated_sorted, clusters
    gc.collect()
    return merged


def save_merged_events(merged: List[dict]) -> None:
    MERGED_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MERGED_EVENTS_PATH, "w", encoding="utf-8") as f:
        for ev in merged:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def load_merged_events() -> List[dict]:
    if not MERGED_EVENTS_PATH.exists():
        return []
    events = []
    with open(MERGED_EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return events
