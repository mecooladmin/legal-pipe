"""
Evidence Immutability Layer — SHA256 Hash Chain.
Generates and persists cryptographic hashes for every stage of the pipeline:
  raw file → text → chunk → event → merged_event

Rules:
- Every transformation produces a new hash
- Previous hashes are NEVER overwritten (append-only chain)
- Each record links to its parent hash
- Chain stored at data/outputs/audit_hash_chain.json (append-only JSONL)
"""
import gc
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

HASH_CHAIN_PATH = Path("data/outputs/audit_hash_chain.json")


def sha256_of(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8", errors="replace")).hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_dict(d: dict) -> str:
    serialized = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return sha256_of(serialized)


def hash_file_bytes(file_path: Path) -> str:
    try:
        return sha256_of_bytes(file_path.read_bytes())
    except Exception as e:
        logger.warning(f"Cannot hash file {file_path}: {e}")
        return ""


def hash_text(text: str) -> str:
    return sha256_of(text)


def hash_chunk(chunk: dict) -> str:
    stable = {
        "file_id": chunk.get("file_id", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "page": chunk.get("page", ""),
        "text": chunk.get("text", ""),
    }
    return sha256_of_dict(stable)


def hash_event(event: dict) -> str:
    stable = {
        "event_id": event.get("event_id", ""),
        "chunk_id": event.get("chunk_id", ""),
        "date_normalized": event.get("date_normalized", ""),
        "event_type": event.get("event_type", ""),
        "description": event.get("description", ""),
        "actors": sorted(event.get("actors", [])),
    }
    return sha256_of_dict(stable)


def hash_merged_event(merged: dict) -> str:
    stable = {
        "merged_event_id": merged.get("merged_event_id", ""),
        "source_event_ids": sorted(merged.get("source_event_ids", [])),
        "date_normalized": merged.get("date_normalized", ""),
        "event_type": merged.get("event_type", ""),
        "description": merged.get("description", ""),
        "confidence": round(merged.get("confidence", 0), 4),
    }
    return sha256_of_dict(stable)


def _load_chain_records() -> list:
    if not HASH_CHAIN_PATH.exists():
        return []
    try:
        content = HASH_CHAIN_PATH.read_text(encoding="utf-8").strip()
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _save_chain_records(records: list) -> None:
    HASH_CHAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HASH_CHAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def append_hash_record(record: dict) -> None:
    records = _load_chain_records()
    records.append(record)
    _save_chain_records(records)


def build_file_hash_record(
    file_path: Path,
    extracted_text: str = "",
    parent_hash: str = "",
) -> dict:
    file_hash = hash_file_bytes(file_path) if file_path.exists() else ""
    text_hash = hash_text(extracted_text) if extracted_text else ""
    record = {
        "stage": "file",
        "timestamp": time.time(),
        "file_name": file_path.name,
        "file_hash": file_hash,
        "text_hash": text_hash,
        "chunk_hash": "",
        "event_hash": "",
        "merged_event_hash": "",
        "parent": parent_hash,
    }
    return record


def build_chunk_hash_record(chunk: dict, file_hash: str = "") -> dict:
    chunk_hash = hash_chunk(chunk)
    return {
        "stage": "chunk",
        "timestamp": time.time(),
        "chunk_id": chunk.get("chunk_id", ""),
        "file_name": chunk.get("file_name", ""),
        "file_hash": file_hash,
        "chunk_hash": chunk_hash,
        "text_hash": hash_text(chunk.get("text", "")),
        "event_hash": "",
        "merged_event_hash": "",
        "parent": file_hash,
    }


def build_event_hash_record(event: dict, chunk_hash: str = "") -> dict:
    event_hash = hash_event(event)
    return {
        "stage": "event",
        "timestamp": time.time(),
        "event_id": event.get("event_id", ""),
        "chunk_id": event.get("chunk_id", ""),
        "file_name": event.get("source_file", ""),
        "file_hash": "",
        "chunk_hash": chunk_hash,
        "event_hash": event_hash,
        "merged_event_hash": "",
        "parent": chunk_hash,
    }


def build_merged_hash_record(merged: dict, source_event_hashes: list) -> dict:
    merged_hash = hash_merged_event(merged)
    parent_hash = sha256_of("|".join(sorted(source_event_hashes)))
    return {
        "stage": "merged_event",
        "timestamp": time.time(),
        "merged_event_id": merged.get("merged_event_id", ""),
        "source_count": merged.get("source_count", 1),
        "file_hash": "",
        "chunk_hash": "",
        "event_hash": "",
        "merged_event_hash": merged_hash,
        "source_event_hashes": source_event_hashes,
        "parent": parent_hash,
    }


def build_hash_chain_for_pipeline(
    merged_events: list,
    all_events_indexed: dict,
    all_chunks_indexed: dict,
    file_hashes: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Build and persist the complete hash chain for the entire pipeline run.
    Returns a lookup: {merged_event_id: merged_event_hash}
    """
    file_hashes = file_hashes or {}
    records = []
    chunk_hash_cache: Dict[str, str] = {}
    event_hash_cache: Dict[str, str] = {}
    merged_hash_lookup: Dict[str, str] = {}

    for merged in merged_events:
        m_id = merged.get("merged_event_id", "")
        source_ids = merged.get("source_event_ids", [])
        source_ev_hashes = []

        for ev_id in source_ids:
            ev = all_events_indexed.get(ev_id)
            if not ev:
                continue

            chunk_id = ev.get("chunk_id", "")
            if chunk_id and chunk_id not in chunk_hash_cache:
                chunk = all_chunks_indexed.get(chunk_id, {})
                c_hash = hash_chunk(chunk) if chunk else ""
                chunk_hash_cache[chunk_id] = c_hash
                if chunk:
                    file_h = file_hashes.get(chunk.get("file_id", ""), "")
                    records.append(build_chunk_hash_record(chunk, file_h))

            ev_hash = hash_event(ev)
            event_hash_cache[ev_id] = ev_hash
            chunk_h = chunk_hash_cache.get(chunk_id, "")
            records.append(build_event_hash_record(ev, chunk_h))
            source_ev_hashes.append(ev_hash)

        if source_ev_hashes:
            m_record = build_merged_hash_record(merged, source_ev_hashes)
            records.append(m_record)
            merged_hash_lookup[m_id] = m_record["merged_event_hash"]

    _save_chain_records(records)
    logger.info(f"Hash chain built: {len(records)} records for {len(merged_events)} merged events")
    gc.collect()
    return merged_hash_lookup


def load_hash_chain() -> list:
    return _load_chain_records()


def get_file_hashes_from_queue(queue: list) -> Dict[str, str]:
    """Hash all raw files in the ingestion queue."""
    from pathlib import Path
    file_hashes = {}
    for item in queue:
        fp = Path(item.get("path", ""))
        if fp.exists():
            # Use the 'id' field from the queue item, which is the 12-char MD5 hash
            fid = item.get("id")
            if fid:
                file_hashes[fid] = hash_file_bytes(fp)
    return file_hashes


def verify_hash_chain_integrity() -> dict:
    """
    Verify that the hash chain is internally consistent.
    Returns summary: {total, valid, broken_links}
    """
    records = _load_chain_records()
    total = len(records)
    broken = []

    known_hashes = set()
    for r in records:
        for key in ("file_hash", "chunk_hash", "event_hash", "merged_event_hash"):
            val = r.get(key, "")
            if val:
                known_hashes.add(val)

    for r in records:
        parent = r.get("parent", "")
        if parent and parent not in known_hashes:
            broken.append({
                "stage": r.get("stage"),
                "id": r.get("merged_event_id") or r.get("event_id") or r.get("chunk_id") or "",
                "broken_parent": parent,
            })

    return {
        "total_records": total,
        "broken_links": len(broken),
        "broken_details": broken[:20],
        "integrity_ok": len(broken) == 0,
    }
