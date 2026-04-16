import gc
import json
import logging
import requests
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MAX_CONTEXT_CHARS = 3000
DEFAULT_MODEL = "mistral"


def check_ollama_available() -> Tuple[bool, str]:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return True, ", ".join(models) if models else "no models loaded"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


def build_rag_prompt(query: str, context_chunks: List[dict]) -> str:
    context_parts = []
    total_chars = 0
    for i, chunk in enumerate(context_chunks):
        text = chunk.get("text", "")
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            text = text[: MAX_CONTEXT_CHARS - total_chars]
        citation = f"[Source {i+1}: {chunk.get('file_name', 'unknown')}, page {chunk.get('page', '?')}]"
        part = f"{citation}\n{text}"
        context_parts.append(part)
        total_chars += len(text)
        if total_chars >= MAX_CONTEXT_CHARS:
            break

    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        f"You are a document analysis assistant. Answer the question using ONLY the provided document excerpts. "
        f"Be concise and accurate. If the answer is not in the documents, say so clearly.\n\n"
        f"Document excerpts:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return prompt


def query_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        return f"LLM error: HTTP {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return "Ollama not available. Make sure Ollama is running locally with `ollama serve`."
    except Exception as e:
        return f"LLM error: {e}"


def answer_question(query: str, model_fn, model: str = DEFAULT_MODEL, top_k: int = 5) -> dict:
    from pipeline.embeddings import search_index

    chunks = search_index(query, model_fn, top_k=top_k)
    if not chunks:
        return {
            "answer": "No relevant documents found. Please process some documents first.",
            "sources": [],
            "query": query,
        }

    prompt = build_rag_prompt(query, chunks)
    answer = query_ollama(prompt, model=model)

    sources = []
    for chunk in chunks:
        sources.append(
            {
                "file_name": chunk.get("file_name", ""),
                "file_id": chunk.get("file_id", ""),
                "page": chunk.get("page", 0),
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "score": chunk.get("score", 0),
            }
        )

    gc.collect()
    return {"answer": answer, "sources": sources, "query": query}
