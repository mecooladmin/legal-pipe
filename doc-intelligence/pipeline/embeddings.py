import gc
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

VECTOR_STORE_DIR = Path("data/vector_store")
INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
META_PATH = VECTOR_STORE_DIR / "meta.jsonl"
EMBEDDED_IDS_PATH = VECTOR_STORE_DIR / "embedded_ids.txt"
BATCH_SIZE = 8
PERSIST_EVERY = 25
EMBEDDING_DIM = 384


def load_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return model


def load_embedded_ids() -> set:
    if not EMBEDDED_IDS_PATH.exists():
        return set()
    with open(EMBEDDED_IDS_PATH, "r") as f:
        return set(line.strip() for line in f if line.strip())


def save_embedded_id(chunk_id: str) -> None:
    EMBEDDED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDED_IDS_PATH, "a") as f:
        f.write(chunk_id + "\n")


def load_or_create_index():
    import faiss

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    else:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        logger.info("Created new FAISS index")
    return index


def save_index(index) -> None:
    import faiss

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))


def append_meta(chunk: dict) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "chunk_id": chunk["chunk_id"],
        "file_id": chunk["file_id"],
        "file_name": chunk["file_name"],
        "page": chunk["page"],
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"][:500],
    }
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_all_meta() -> List[dict]:
    if not META_PATH.exists():
        return []
    meta = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    meta.append(json.loads(line))
                except Exception:
                    pass
    return meta


def embed_chunks(chunks: List[dict], model) -> None:
    import faiss
    import numpy as np

    embedded_ids = load_embedded_ids()
    index = load_or_create_index()

    pending = [c for c in chunks if c["chunk_id"] not in embedded_ids]
    if not pending:
        return

    count = 0
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        try:
            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embeddings = np.array(embeddings, dtype="float32")
            index.add(embeddings)
            for j, chunk in enumerate(batch):
                append_meta(chunk)
                save_embedded_id(chunk["chunk_id"])
            count += len(batch)
            del embeddings
            gc.collect()

            if count % PERSIST_EVERY == 0:
                save_index(index)
                logger.info(f"Checkpoint: embedded {count} new chunks")
        except Exception as e:
            logger.error(f"Embedding batch failed: {e}")

    save_index(index)
    del index
    gc.collect()
    logger.info(f"Embedded {count} new chunks total")


def search_index(query: str, model, top_k: int = 5) -> List[dict]:
    import faiss
    import numpy as np

    if not INDEX_PATH.exists():
        return []

    index = load_or_create_index()
    if index.ntotal == 0:
        return []

    meta = load_all_meta()

    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(meta):
            result = dict(meta[idx])
            result["score"] = float(dist)
            results.append(result)

    del index
    gc.collect()
    return results
