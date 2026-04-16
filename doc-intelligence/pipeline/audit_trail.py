"""
Audit Trail System.
Maintains full provenance: raw text → chunk → event → merged event → narrative sentence.
Enables complete drill-down from any narrative claim back to the source document.
"""
import gc
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

AUDIT_PATH = Path("data/outputs/audit_trail.jsonl")
AUDIT_INDEX_PATH = Path("data/outputs/audit_index.json")


def build_chunk_record(chunk: dict) -> dict:
    return {
        "chunk_id": chunk.get("chunk_id", ""),
        "file_id": chunk.get("file_id", ""),
        "file_name": chunk.get("file_name", ""),
        "page": chunk.get("page", ""),
        "word_count": chunk.get("word_count", 0),
        "text_preview": chunk.get("text", "")[:300],
    }


def build_event_record(event: dict) -> dict:
    return {
        "event_id": event.get("event_id", ""),
        "chunk_id": event.get("chunk_id", ""),
        "file_id": event.get("file_id", ""),
        "file_name": event.get("source_file", ""),
        "page": event.get("page", ""),
        "date_raw": event.get("date_raw", ""),
        "date_normalized": event.get("date_normalized", ""),
        "event_type": event.get("event_type", ""),
        "language": event.get("language", "en"),
        "confidence": event.get("confidence", 0),
        "source_reliability": event.get("source_reliability", "medium"),
        "source_weight": event.get("source_weight", 0.7),
        "description_preview": event.get("description", "")[:200],
    }


def build_merged_record(merged: dict) -> dict:
    return {
        "merged_event_id": merged.get("merged_event_id", ""),
        "source_event_ids": merged.get("source_event_ids", []),
        "date_normalized": merged.get("date_normalized", ""),
        "event_type": merged.get("event_type", ""),
        "confidence": merged.get("confidence", 0),
        "confidence_level": merged.get("confidence_level", "medium"),
        "source_reliability": merged.get("source_reliability", "medium"),
        "avg_semantic_similarity": merged.get("avg_semantic_similarity", 0),
        "source_count": merged.get("source_count", 1),
        "description_preview": merged.get("description", "")[:200],
        "citations": merged.get("citations", []),
    }


def write_audit_record(record: dict) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_audit_trail(
    all_chunks_by_id: Dict[str, dict],
    all_events_by_id: Dict[str, dict],
    merged_events: List[dict],
    narrative_sentences: List[dict],
) -> None:
    """
    Write complete audit trail linking:
    narrative_sentence → merged_event → raw_events → chunks → raw text
    """
    if AUDIT_PATH.exists():
        AUDIT_PATH.unlink()

    merged_by_id = {ev.get("merged_event_id", ""): ev for ev in merged_events}

    audit_records = []
    for sentence in narrative_sentences:
        event_refs = sentence.get("event_refs", [])
        record = {
            "sentence_id": sentence.get("sentence_id", ""),
            "section": sentence.get("section", ""),
            "text": sentence.get("text", ""),
            "event_refs": event_refs,
            "provenance": [],
        }

        for ref_id in event_refs:
            merged = merged_by_id.get(ref_id)
            if not merged:
                continue
            prov = build_merged_record(merged)
            prov["raw_events"] = []

            for src_event_id in merged.get("source_event_ids", []):
                raw_ev = all_events_by_id.get(src_event_id)
                if not raw_ev:
                    continue
                ev_record = build_event_record(raw_ev)

                chunk = all_chunks_by_id.get(raw_ev.get("chunk_id", ""))
                if chunk:
                    ev_record["chunk"] = build_chunk_record(chunk)

                prov["raw_events"].append(ev_record)

            record["provenance"].append(prov)

        audit_records.append(record)

    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        for r in audit_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    index = {
        r["sentence_id"]: {
            "section": r["section"],
            "text_preview": r["text"][:100],
            "event_refs": r["event_refs"],
        }
        for r in audit_records
        if r.get("sentence_id")
    }
    with open(AUDIT_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    logger.info(f"Audit trail written: {len(audit_records)} sentence records")
    gc.collect()


def load_audit_trail() -> List[dict]:
    if not AUDIT_PATH.exists():
        return []
    records = []
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def load_audit_index() -> dict:
    if not AUDIT_INDEX_PATH.exists():
        return {}
    with open(AUDIT_INDEX_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def get_audit_record_by_sentence(sentence_id: str) -> Optional[dict]:
    for record in load_audit_trail():
        if record.get("sentence_id") == sentence_id:
            return record
    return None


def load_all_chunks_indexed() -> Dict[str, dict]:
    from pipeline.chunker import CHUNKS_DIR
    import json
    chunks = {}
    if not CHUNKS_DIR.exists():
        return chunks
    for chunk_file in CHUNKS_DIR.glob("*.jsonl"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        c = json.loads(line)
                        chunk_id = c.get("chunk_id", "")
                        if chunk_id:
                            chunks[chunk_id] = c
                    except Exception:
                        pass
    return chunks


def load_all_events_indexed() -> Dict[str, dict]:
    from pipeline.event_extractor import load_all_events
    events = load_all_events()
    return {ev.get("event_id", ""): ev for ev in events if ev.get("event_id")}
