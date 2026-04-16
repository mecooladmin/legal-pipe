# Document Intelligence System

A low-RAM, production-ready document intelligence system with OCR, semantic search, and AI Q&A.

## Features

- Processes PDF, images, Excel, CSV, and text files
- Tiered OCR: PyMuPDF embedded text → Tesseract fallback
- Chunking (500–800 words) + FAISS vector embeddings
- RAG Q&A via local Ollama LLM (Mistral or LLaMA3)
- Resume-safe pipeline (skip already-processed files)
- 4-page Streamlit UI: Dashboard, Search, Explorer, Insights

## RAM Design

- Processes ONE file / ONE page at a time
- Batch size ≤ 8 for embeddings
- FAISS persisted every 25 embeddings
- Explicit gc.collect() after each step

## Storage Structure

```
data/
  input/              ← Drop your documents here
  processed/
    raw_text/         ← Extracted text per file
    chunks/           ← Text chunks per file
  vector_store/       ← FAISS index + metadata
  outputs/
    summaries/        ← Per-file summaries
    entities/         ← Per-file entity extraction
    timeline/         ← Global timeline events
  queue.jsonl         ← Resume-safe processing queue
logs/
  app.log             ← All pipeline logs
```

## Setup

```bash
cd doc-intelligence
pip install -r requirements.txt

# Install Tesseract OCR (system dependency)
# On Ubuntu/Debian:  sudo apt-get install tesseract-ocr
# On macOS:          brew install tesseract

# For AI Q&A (optional):
# Install Ollama: https://ollama.ai
# Then: ollama pull mistral
```

## Running

### Start the UI
```bash
streamlit run streamlit_app.py --server.port 5000
```

### CLI processing only
```bash
# Process documents in data/input/
python main.py --input data/input

# Process + run a query
python main.py --input data/input --query "What are the key findings?"

# Generate timeline
python main.py --input data/input --timeline
```

## Usage

1. Drop documents into `data/input/`
2. Open the Dashboard and click "Start / Resume Processing"
3. Use Document Search to ask questions
4. Browse extracted text in Document Explorer
5. View auto-generated insights in the Insights tab

## Resume Support

The pipeline tracks every file in `data/queue.jsonl` with status `pending`, `processing`, `done`, or `error`. If interrupted, re-running the pipeline skips already-processed files automatically.
