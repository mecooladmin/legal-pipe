"""
Semantic Event Merging using embedding cosine similarity.
Replaces keyword-overlap merging with vector similarity.
Merges events only when:
  - date proximity <= configurable window (days)
  - semantic similarity >= configurable threshold (cosine)
"""
import gc
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

DEFAULT_DATE_WINDOW_DAYS = 3
DEFAULT_SIM_THRESHOLD = 0.65
MAX_SAME_TYPE_SIM = 0.50


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def embed_descriptions(descriptions: List[str], model) -> np.ndarray:
    """Embed a list of descriptions, batch_size=8 for low RAM."""
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


def parse_iso_to_int(iso: str) -> Optional[int]:
    """Convert ISO date string to integer days for proximity comparison."""
    try:
        from datetime import date
        parts = iso.split("-")
        if len(parts) == 3:
            d = date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            d = date(int(parts[0]), int(parts[1]), 1)
        elif len(parts) == 1:
            d = date(int(parts[0]), 1, 1)
        else:
            return None
        return d.toordinal()
    except Exception:
        return None


def dates_within_window(d1: str, d2: str, window_days: int) -> bool:
    ord1 = parse_iso_to_int(d1)
    ord2 = parse_iso_to_int(d2)
    if ord1 is None or ord2 is None:
        return False
    return abs(ord1 - ord2) <= window_days


def semantic_cluster_events(
    events: List[dict],
    embeddings: np.ndarray,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
) -> List[List[int]]:
    """
    Cluster event indices. Two events merge if:
      1. Both have valid normalized dates within date_window_days, OR same exact date
      2. Cosine similarity of descriptions >= sim_threshold
         OR same event_type AND similarity >= MAX_SAME_TYPE_SIM

    Returns list of clusters (each cluster is list of event indices).
    """
    n = len(events)
    assigned = [-1] * n
    clusters: List[List[int]] = []
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0:
            continue
        assigned[i] = cluster_id
        cluster = [i]

        d1 = events[i].get("date_normalized", "")
        emb_i = embeddings[i] if len(embeddings) > i else None

        for j in range(i + 1, n):
            if assigned[j] >= 0:
                continue

            d2 = events[j].get("date_normalized", "")

            if not d1 or not d2:
                continue

            if not dates_within_window(d1, d2, date_window_days):
                continue

            emb_j = embeddings[j] if len(embeddings) > j else None
            if emb_i is None or emb_j is None:
                continue
            sim = cosine_similarity(emb_i, emb_j)
            
            # If dates are exactly the same, we still require some semantic similarity
            # to avoid merging completely different events on the same day.
            # If dates are within a window but not same, we require higher similarity.
            is_same_date = d1 == d2
            
            same_type = events[i].get("event_type") == events[j].get("event_type")
            
            if is_same_date:
                # Same day: allow lower threshold if same type, but still need 0.4+
                threshold = 0.40 if same_type else sim_threshold
            else:
                # Different day (within window): require standard threshold
                threshold = sim_threshold
                
            if sim >= threshold:
                assigned[j] = cluster_id
                cluster.append(j)
        clusters.append(cluster)
        cluster_id += 1

    for i in range(n):
        if assigned[i] < 0:
            clusters.append([i])

    return clusters


def merge_event_group_semantic(
    group_events: List[dict],
    group_embeddings: Optional[np.ndarray],
    reliability_map: Optional[dict] = None,
) -> dict:
    """
    Merge a group of events into one, preserving all citations.
    Primary event = highest reliability weight * confidence.
    """
    from pipeline.source_reliability import get_weight

    def event_score(ev: dict) -> float:
        weight = ev.get("source_weight", 0.7)
        conf = ev.get("confidence", 0.5)
        return weight * conf

    group_sorted = sorted(group_events, key=event_score, reverse=True)
    primary = group_sorted[0]

    all_actors = []
    seen_actors = set()
    for ev in group_events:
        for actor in ev.get("actors", []):
            if actor not in seen_actors:
                all_actors.append(actor)
                seen_actors.add(actor)

    citations = []
    seen_citations = set()
    for ev in group_events:
        key = (ev.get("source_file", ""), str(ev.get("page", "")))
        if key not in seen_citations:
            citations.append({
                "source_file": ev.get("source_file", ""),
                "file_id": ev.get("file_id", ""),
                "page": ev.get("page", ""),
                "chunk_id": ev.get("chunk_id", ""),
                "language": ev.get("language", "en"),
                "confidence": ev.get("confidence", 0.0),
                "source_reliability": ev.get("source_reliability", "medium"),
                "source_weight": ev.get("source_weight", 0.7),
                "event_id": ev.get("event_id", ""),
            })
            seen_citations.add(key)

    citations_sorted = sorted(citations, key=lambda c: -c.get("source_weight", 0.7))

    descriptions = list(dict.fromkeys(
        ev.get("description", "") for ev in group_sorted if ev.get("description")
    ))

    locations = list(dict.fromkeys(
        ev.get("location", "") for ev in group_events if ev.get("location")
    ))

    avg_sim = 0.0
    if group_embeddings is not None and len(group_embeddings) > 1:
        sims = []
        for a in range(len(group_embeddings)):
            for b in range(a + 1, len(group_embeddings)):
                sims.append(cosine_similarity(group_embeddings[a], group_embeddings[b]))
        avg_sim = float(np.mean(sims)) if sims else 0.0

    merged_conf = float(np.mean([ev.get("confidence", 0.5) for ev in group_events]))
    reliability_levels = list(dict.fromkeys(ev.get("source_reliability", "medium") for ev in group_events))
    best_reliability = "high" if "high" in reliability_levels else ("medium" if "medium" in reliability_levels else "low")

    source_ids = [ev.get("event_id", "") for ev in group_events if ev.get("event_id")]

    versions = [
        {
            "event_id": ev.get("event_id", ""),
            "date": ev.get("date_normalized", ""),
            "date_raw": ev.get("date_raw", ""),
            "source_file": ev.get("source_file", ""),
            "page": ev.get("page", ""),
            "source_weight": ev.get("source_weight", 0.7),
            "source_reliability": ev.get("source_reliability", "medium"),
            "confidence": ev.get("confidence", 0.0),
            "description": ev.get("description", "")[:200],
            "language": ev.get("language", "en"),
        }
        for ev in group_sorted
    ]

    dates_in_group = list(dict.fromkeys(
        ev.get("date_normalized", "") for ev in group_events if ev.get("date_normalized")
    ))
    has_conflict = len(dates_in_group) > 1 or len(set(
        ev.get("source_reliability","") for ev in group_events
    )) > 1 and len(group_events) > 1

    descriptions_unique = list(dict.fromkeys(
        ev.get("description", "") for ev in group_sorted if ev.get("description")
    ))

    merged = {
        "merged_event_id": primary["event_id"],
        "source_event_ids": source_ids,
        "canonical_event": {
            "resolved_date": primary.get("date_normalized", ""),
            "confidence": round(merged_conf, 3),
            "primary_source": primary.get("source_file", ""),
        },
        "versions": versions,
        "has_conflict": has_conflict,
        "date_normalized": primary["date_normalized"],
        "date_raw": primary["date_raw"],
        "date_confidence": primary.get("date_confidence", 0.7),
        "event_type": primary["event_type"],
        "description": descriptions_unique[0] if descriptions_unique else "",
        "all_descriptions": descriptions_unique[:3],
        "actors": all_actors,
        "location": locations[0] if locations else "",
        "citations": citations_sorted,
        "source_count": len(citations_sorted),
        "confidence": round(merged_conf, 3),
        "confidence_level": "high" if merged_conf >= 0.75 else ("medium" if merged_conf >= 0.50 else "low"),
        "source_reliability": best_reliability,
        "avg_semantic_similarity": round(avg_sim, 3),
        "languages": list(dict.fromkeys(ev.get("language", "en") for ev in group_events)),
    }
    return merged


def run_semantic_merge(
    events: List[dict],
    model,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    reliability_map: Optional[dict] = None,
) -> List[dict]:
    """
    Full semantic merge pipeline:
    1. Apply reliability weights to events
    2. Embed all descriptions
    3. Cluster by date proximity + cosine similarity
    4. Merge each cluster
    """
    from pipeline.source_reliability import apply_reliability_to_event

    if reliability_map:
        for ev in events:
            apply_reliability_to_event(ev, reliability_map)

    dated = [ev for ev in events if ev.get("date_normalized")]
    undated = [ev for ev in events if not ev.get("date_normalized")]

    dated_sorted = sorted(dated, key=lambda e: e.get("date_normalized", ""))

    descriptions = [ev.get("description", "") or ev.get("date_raw", "") for ev in dated_sorted]

    logger.info(f"Embedding {len(descriptions)} event descriptions for semantic merge...")
    if descriptions:
        embeddings = embed_descriptions(descriptions, model)
    else:
        embeddings = np.array([])

    clusters = semantic_cluster_events(
        dated_sorted, embeddings,
        date_window_days=date_window_days,
        sim_threshold=sim_threshold,
    )

    merged = []
    for cluster_indices in clusters:
        group_events = [dated_sorted[i] for i in cluster_indices]
        if len(embeddings) > 0:
            group_embs = np.array([embeddings[i] for i in cluster_indices if i < len(embeddings)])
        else:
            group_embs = None
        merged_ev = merge_event_group_semantic(group_events, group_embs, reliability_map)
        merged.append(merged_ev)

    for ev in undated:
        if reliability_map:
            apply_reliability_to_event(ev, reliability_map)
        merged.append(merge_event_group_semantic([ev], None, reliability_map))

    merged.sort(key=lambda e: (e.get("date_normalized", "9999"), -e.get("confidence", 0)))

    del embeddings, dated, undated, dated_sorted
    gc.collect()

    logger.info(f"Semantic merge: {len(events)} events → {len(merged)} merged events")
    return merged
