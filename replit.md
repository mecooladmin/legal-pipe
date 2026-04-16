# Workspace

## Overview

pnpm workspace monorepo using TypeScript, plus a Python-based Document Intelligence System.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

---

## Document Intelligence System (`doc-intelligence/`)

A low-RAM Python document intelligence pipeline with Streamlit UI.

### Features
- Processes PDF, images, Excel, CSV, and text files (up to 743+ docs)
- Tiered OCR: PyMuPDF → Tesseract fallback
- Semantic search via FAISS + sentence-transformers (all-MiniLM-L6-v2)
- RAG Q&A via Ollama (Mistral/LLaMA3)
- Resume-safe pipeline (queue.jsonl tracking)
- 4-page Streamlit UI: Dashboard, Search, Explorer, Insights

### RAM Design (1-2GB constraint)
- One file / one page at a time
- Batch size ≤ 8 for embeddings
- FAISS persisted every 25 embeddings
- Explicit gc.collect() after each step

### Structure
```
doc-intelligence/
  pipeline/          ← Modular backend
    ingestion.py     ← Queue system (resume-safe)
    ocr.py           ← Tiered OCR (PyMuPDF + Tesseract)
    converter.py     ← Excel/CSV/text extraction
    chunker.py       ← 500-800 word chunks
    embeddings.py    ← FAISS + sentence-transformers
    rag.py           ← Ollama RAG engine
    insights.py      ← Summaries, entities, timeline
    processor.py     ← Main orchestration
  streamlit_app.py   ← 4-page Streamlit UI
  main.py            ← CLI entrypoint
  requirements.txt
  data/
    input/           ← Drop documents here
    queue.jsonl      ← Processing queue
    processed/       ← Extracted text + chunks
    vector_store/    ← FAISS index
    outputs/         ← Summaries, entities, timeline
  logs/app.log
```

### Legal Intelligence Layer (5th page)

Modules in `pipeline/`:
- `date_normalizer.py` — Arabic (١٢ مارس ٢٠٢١) + English date normalization to ISO 8601. Handles Arabic-Indic numerals, Arabic month names, all common formats. Confidence scoring.
- `event_extractor.py` — Extracts structured events from chunks: event_id, date_raw, date_normalized, event_type (contract/payment/meeting/incident/claim/decision/correspondence), description, actors, location, source_file, page, language (ar/en/mixed), confidence
- `event_merger.py` — Groups events by date proximity (3-day window) + semantic keyword overlap. Merges duplicate events across all documents while preserving all citations.
- `contradiction_detector.py` — Detects date conflicts and opposing claims (paid vs. unpaid, signed vs. unsigned, etc.) across source documents. Severity: high/medium.
- `narrative_generator.py` — Generates: timeline.json, timeline_legal.md, and a full legal narrative with sections: Case Overview, Chronological Events, Key Turning Points, Conflicting Evidence, Conclusion. Uses Ollama if available; falls back to template generation.
- `legal_pipeline.py` — Orchestrates the full legal pipeline with progress callbacks.

### Running
```bash
# UI (workflow: "Document Intelligence UI")
cd doc-intelligence && python3 -m streamlit run streamlit_app.py --server.port 5000

# CLI
cd doc-intelligence && python3 main.py --input data/input

# With query
cd doc-intelligence && python3 main.py --input data/input --query "What are the key findings?"
```

### Optional: Ollama for AI Q&A
```bash
# Install Ollama: https://ollama.ai
ollama pull mistral
ollama serve
```
