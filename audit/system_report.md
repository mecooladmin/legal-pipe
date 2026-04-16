# Legal-Pipe System Audit Report (Phase 0)

## Executive Summary
The repository contains a sophisticated legal intelligence system with two distinct processing paths: a **Basic Processor** (`processor.py`) and an **Advanced Forensic Orchestrator** (`legal_pipeline.py`). While the advanced path implements most of the user's requested features (semantic merging, contradiction ranking, hash-based audit trails), it is currently disconnected from the main UI and contains several critical integration bugs.

## Identified Issues

### 1. Orchestration Gap
The primary Streamlit UI is wired to the `Basic Processor`, which lacks the forensic depth requested. The `Advanced Forensic Orchestrator` exists as a separate module but is not invoked by the default pipeline.

### 2. Critical Integration Bugs
*   **Hash Chain ID Mismatch**: `hash_chain.py` uses `item.get("file_id", fp.stem)` for hashing, while the rest of the system uses a 12-character MD5 hash of the file path. This will break the cryptographic link between files and chunks.
*   **Reliability Map Race Condition**: `source_reliability.py` only builds a map for files marked as `done` in the queue. If the forensic pipeline is run before the basic processor, the reliability map will be empty, degrading all downstream legal intelligence.
*   **Memory Management**: While `gc.collect()` is used, the system loads full Excel/CSV files into memory without streaming, posing a risk for large datasets.

### 3. Feature Implementation Status

| Feature | Status | Implementation File |
| :--- | :--- | :--- |
| **Semantic Event Merging** | Implemented | `semantic_merger.py` |
| **Event Versioning** | Implemented | `semantic_merger.py` |
| **Contradiction Ranking** | Implemented | `contradiction_detector.py` |
| **Hash-based Audit Trail** | Implemented | `hash_chain.py` |
| **Arabic/English Support** | Implemented | `date_normalizer.py`, `event_extractor.py` |
| **Deployment Config** | Partial | `.replit` exists; `render.yaml` missing |

## Phase 1 Audit Results (Targeted Audit)

### Findings
*   **Missing Dependencies**: The system was missing `sentence_transformers`, `faiss-cpu`, and other critical libraries. These have been installed.
*   **Normalization Bug**: The `normalize_text` function was stripping Arabic characters, causing empty chunks for Arabic documents. This has been fixed.
*   **Chunking Threshold**: The `MIN_CHUNK_WORDS` was set to 50, which caused small sample files to be ignored. This was temporarily reduced for testing.
*   **Event Extraction Limit**: The extractor was limited to only one event per chunk. This has been improved to allow multiple events per chunk based on detected dates.
*   **Contradiction Detection**: Successfully detected date conflicts in the sample dataset.

### Status of Core Failures
*   [x] Normalization bug (Arabic support)
*   [x] Hash ID mismatch (Fixed in `hash_chain.py`)
*   [x] Reliability map race condition (Fixed in `source_reliability.py`)
*   [x] Missing dependencies (Installed)
*   [x] Event extraction limit (Multiple events per chunk allowed)
*   [x] CSV memory safety (Streaming/chunked reading implemented)
*   [x] UI Integration (Verified advanced path is already present in `streamlit_app.py`)
*   [x] Performance Hardening (Restored production chunking thresholds)

