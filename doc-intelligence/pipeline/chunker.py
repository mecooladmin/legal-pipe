import re
import gc
import json
import logging
from pathlib import Path
from typing import Iterator, List

logger = logging.getLogger(__name__)

CHUNKS_DIR = Path("data/processed/chunks")
RAW_TEXT_DIR = Path("data/processed/raw_text")
MIN_CHUNK_WORDS = 50
TARGET_CHUNK_WORDS = 600
MAX_CHUNK_WORDS = 800


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    # Allow Arabic characters and other non-ASCII printable characters
    text = re.sub(r"[^\x20-\x7E\n\u0600-\u06FF]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def split_into_chunks(text: str, file_id: str, file_name: str, page_num: int = 0) -> List[dict]:
    words = text.split()
    chunks = []
    chunk_idx = 0
    i = 0
    while i < len(words):
        chunk_words = words[i : i + TARGET_CHUNK_WORDS]
        chunk_text = " ".join(chunk_words)
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunks.append(
                {
                    "chunk_id": f"{file_id}_p{page_num}_c{chunk_idx}",
                    "file_id": file_id,
                    "file_name": file_name,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                }
            )
            chunk_idx += 1
        i += TARGET_CHUNK_WORDS
    return chunks


def save_raw_text(file_id: str, file_name: str, pages: List) -> None:
    RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_TEXT_DIR / f"{file_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for page_num, text in pages:
            record = {"file_id": file_id, "file_name": file_name, "page": page_num, "text": text}
            f.write(json.dumps(record) + "\n")


def save_chunks(file_id: str, chunks: List[dict]) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / f"{file_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


def load_chunks_for_file(file_id: str) -> List[dict]:
    chunk_path = CHUNKS_DIR / f"{file_id}.jsonl"
    if not chunk_path.exists():
        return []
    chunks = []
    with open(chunk_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except Exception:
                    pass
    return chunks


def iter_all_chunks() -> Iterator[dict]:
    if not CHUNKS_DIR.exists():
        return
    for chunk_file in sorted(CHUNKS_DIR.glob("*.jsonl")):
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass


def process_pages_to_chunks(file_id: str, file_name: str, pages: List) -> List[dict]:
    all_chunks = []
    for page_num, raw_text in pages:
        text = normalize_text(raw_text)
        if not text:
            continue
        chunks = split_into_chunks(text, file_id, file_name, page_num)
        all_chunks.extend(chunks)
        del text
        gc.collect()
    return all_chunks
