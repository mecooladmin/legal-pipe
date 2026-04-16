import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

QUEUE_PATH = Path("data/queue.jsonl")
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".xlsx", ".xls", ".csv", ".txt", ".md"}


def get_file_id(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()[:12]


def build_queue(input_dir: str) -> None:
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.warning(f"Input directory not found: {input_dir}")
        return

    existing_ids = set()
    if QUEUE_PATH.exists():
        with open(QUEUE_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        existing_ids.add(item["id"])
                    except Exception:
                        pass

    new_count = 0
    with open(QUEUE_PATH, "a") as f:
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in sorted(input_path.rglob(f"*{ext}")):
                file_id = get_file_id(str(file_path))
                if file_id not in existing_ids:
                    item = {
                        "id": file_id,
                        "path": str(file_path),
                        "name": file_path.name,
                        "ext": ext,
                        "status": "pending",
                        "pages": 0,
                        "chunks": 0,
                        "error": None,
                    }
                    f.write(json.dumps(item) + "\n")
                    existing_ids.add(file_id)
                    new_count += 1

    logger.info(f"Added {new_count} new files to queue")


def load_queue() -> list:
    if not QUEUE_PATH.exists():
        return []
    items = []
    with open(QUEUE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    return items


def update_queue_item(file_id: str, **kwargs) -> None:
    if not QUEUE_PATH.exists():
        return
    lines = []
    with open(QUEUE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item["id"] == file_id:
                    item.update(kwargs)
                lines.append(json.dumps(item))
            except Exception:
                lines.append(line)

    with open(QUEUE_PATH, "w") as f:
        for line in lines:
            f.write(line + "\n")


def get_pending_files() -> Iterator[dict]:
    for item in load_queue():
        if item.get("status") == "pending":
            yield item


def get_queue_stats() -> dict:
    items = load_queue()
    stats = {"total": len(items), "pending": 0, "done": 0, "error": 0, "total_chunks": 0}
    for item in items:
        status = item.get("status", "pending")
        if status == "done":
            stats["done"] += 1
            stats["total_chunks"] += item.get("chunks", 0)
        elif status == "error":
            stats["error"] += 1
        else:
            stats["pending"] += 1
    return stats
