"""
Legal Intelligence Pipeline Orchestrator — Forensic Edition.

Integrates all 11 forensic integrity requirements:
1. Evidence Immutability — SHA256 hash chain (file→chunk→event→merged_event)
2. Event Versioning — multi-version merged events with conflict flags
3. Confidence-Driven Narrative — linguistic confidence rules
4. Contradiction Prioritization — impact_score, CRITICAL/SIGNIFICANT/MINOR
5. Timeline Uncertainty Model — uncertainty clusters, date confidence
6. Source Reliability Enforcement — high/medium/low weights
7. Forensic Audit Trail — sentence→event→chunk→file hash
8. Reproducibility Mode — deterministic config saved to run_config.json
9. Legal Export — legal_brief.md + evidence_index.json
10. Non-Hallucination — no sentence without source event
11. Failure Safety — unresolved issues flagged for manual review
"""
import gc
import json
import logging
from pathlib import Path
from typing import List, Callable, Optional

logger = logging.getLogger(__name__)


def run_legal_pipeline(
    case_name: str = "Legal Case",
    model: str = "mistral",
    date_window_days: int = 3,
    sim_threshold: float = 0.65,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Full legal pipeline from scratch:
    1. Save run config (reproducibility)
    2. Build reliability map from queue
    3. Hash all raw files (immutability)
    4. Extract events from all chunks
    5. Semantic merge with reliability weights + versioning
    6. Flag contradictions (impact scoring + CRITICAL/SIGNIFICANT/MINOR)
    7. Build hash chain (chunk→event→merged)
    8. Confidence-aware timeline generation
    9. Traceable narrative generation (agreed/disputed/unresolved)
    10. Forensic audit trail (sentence→event→chunk→hash)
    11. Legal export (legal_brief.md, evidence_index.json, unresolved_issues.json)
    """
    from pipeline.chunker import CHUNKS_DIR
    from pipeline.event_extractor import extract_all_file_events, EVENTS_DIR
    from pipeline.ingestion import load_queue
    from pipeline.source_reliability import build_reliability_map
    from pipeline.semantic_merger import run_semantic_merge
    from pipeline.event_merger import save_merged_events
    from pipeline.contradiction_detector import find_contradictions_semantic, save_contradictions
    from pipeline.narrative_generator import (
        generate_timeline_json, generate_timeline_md, generate_narrative_document
    )
    from pipeline.audit_trail import build_audit_trail, load_all_chunks_indexed, load_all_events_indexed
    from pipeline.embeddings import load_model as load_embed_model
    from pipeline.hash_chain import (
        build_hash_chain_for_pipeline, get_file_hashes_from_queue
    )
    from pipeline.legal_exporter import (
        save_run_config, build_evidence_index, collect_unresolved_issues, generate_legal_brief
    )

    def _cb(step, total, name, stage):
        if progress_callback:
            progress_callback(step, total, name, stage)

    _cb(0, 13, "Saving run config", "reproducibility")
    run_config = save_run_config(
        case_name=case_name,
        model=model,
        date_window_days=date_window_days,
        sim_threshold=sim_threshold,
    )

    _cb(1, 13, "Loading queue + reliability", "initializing")
    queue = load_queue()
    reliability_map = build_reliability_map(queue)
    file_hashes = get_file_hashes_from_queue(queue)
    logger.info(f"Reliability map: {len(reliability_map)} files | Hashed: {len(file_hashes)} files")

    _cb(2, 13, "Loading embedding model", "model load")
    embed_model = load_embed_model()

    if not CHUNKS_DIR.exists():
        logger.warning("No chunks found.")
        return {"error": "No chunks found. Process documents first."}

    chunk_files = sorted(CHUNKS_DIR.glob("*.jsonl"))
    total_files = len(chunk_files)
    total_events = 0

    for i, chunk_file in enumerate(chunk_files):
        file_id = chunk_file.stem
        events_path = EVENTS_DIR / f"{file_id}.jsonl"

        if not events_path.exists():
            chunks = []
            with open(chunk_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            chunks.append(json.loads(line))
                        except Exception:
                            pass
            file_name = chunks[0].get("file_name", file_id) if chunks else file_id
            events = extract_all_file_events(file_id, chunks)
            total_events += len(events)
            del chunks
            gc.collect()
        else:
            with open(events_path, "r") as f:
                total_events += sum(1 for line in f if line.strip())
            file_name = chunk_file.stem

        pct_step = int(3 + (i / max(total_files, 1)) * 3)
        _cb(pct_step, 13, file_name, "extracting events")

    _cb(6, 13, f"{total_events} events extracted", "semantic merging")

    from pipeline.event_extractor import load_all_events
    all_events = load_all_events()
    logger.info(f"Total raw events: {len(all_events)}")

    merged = run_semantic_merge(
        all_events,
        embed_model,
        date_window_days=date_window_days,
        sim_threshold=sim_threshold,
        reliability_map=reliability_map,
    )
    save_merged_events(merged)
    logger.info(f"Semantic merge: {len(all_events)} → {len(merged)} events")

    del all_events
    gc.collect()

    _cb(7, 13, f"{len(merged)} merged events", "contradiction detection")

    contradictions = find_contradictions_semantic(merged, model=embed_model)
    save_contradictions(contradictions)

    conflict_ids = set()
    for c in contradictions:
        conflict_ids.add(c.get("event_a", {}).get("merged_event_id", ""))
        conflict_ids.add(c.get("event_b", {}).get("merged_event_id", ""))
    for ev in merged:
        if ev.get("merged_event_id", "") in conflict_ids:
            ev["has_conflict"] = True

    logger.info(f"Contradictions: {len(contradictions)} "
                f"(CRITICAL: {sum(1 for c in contradictions if c.get('impact_label')=='CRITICAL')})")

    _cb(8, 13, "Building SHA256 hash chain", "immutability")

    chunks_indexed = load_all_chunks_indexed()
    events_indexed = load_all_events_indexed()
    hash_lookup = build_hash_chain_for_pipeline(merged, events_indexed, chunks_indexed, file_hashes)

    _cb(9, 13, "Building timeline", "timeline generation")

    generate_timeline_json(merged)
    generate_timeline_md(merged)

    _cb(10, 13, "Generating narrative", "narrative")

    _, narrative_sentences = generate_narrative_document(
        merged, contradictions, case_name=case_name, model=model
    )

    _cb(11, 13, "Building forensic audit trail", "audit trail")

    build_audit_trail(chunks_indexed, events_indexed, merged, narrative_sentences)

    _cb(12, 13, "Generating legal exports", "legal export")

    unresolved = collect_unresolved_issues(merged, contradictions)
    build_evidence_index(merged, hash_lookup)
    generate_legal_brief(case_name, merged, contradictions, unresolved, run_config, hash_lookup)

    n_merged = len(merged)
    n_contradictions = len(contradictions)
    n_critical = sum(1 for c in contradictions if c.get("impact_label") == "CRITICAL")
    n_unresolved = len(unresolved)

    del embed_model, chunks_indexed, events_indexed, merged, narrative_sentences
    gc.collect()

    _cb(13, 13, "Complete", "done")

    return {
        "total_events_raw": total_events,
        "merged_events": n_merged,
        "contradictions": n_contradictions,
        "critical_contradictions": n_critical,
        "unresolved_issues": n_unresolved,
        "files_processed": total_files,
        "reliability_classified": len(reliability_map),
        "hashed_files": len(file_hashes),
    }


def run_legal_pipeline_from_chunks(
    case_name: str = "Legal Case",
    model: str = "mistral",
    date_window_days: int = 3,
    sim_threshold: float = 0.65,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Re-run merge + contradiction + narrative + audit using already-extracted events.
    Faster path for iterating on merge/threshold parameters.
    Saves full forensic outputs including hash chain, legal brief, evidence index.
    """
    from pipeline.event_extractor import load_all_events
    from pipeline.ingestion import load_queue
    from pipeline.source_reliability import build_reliability_map, load_reliability_map
    from pipeline.semantic_merger import run_semantic_merge
    from pipeline.event_merger import save_merged_events
    from pipeline.contradiction_detector import find_contradictions_semantic, save_contradictions
    from pipeline.narrative_generator import (
        generate_timeline_json, generate_timeline_md, generate_narrative_document
    )
    from pipeline.audit_trail import build_audit_trail, load_all_chunks_indexed, load_all_events_indexed
    from pipeline.embeddings import load_model as load_embed_model
    from pipeline.hash_chain import build_hash_chain_for_pipeline, get_file_hashes_from_queue
    from pipeline.legal_exporter import (
        save_run_config, build_evidence_index, collect_unresolved_issues, generate_legal_brief
    )

    def _cb(step, total, name, stage):
        if progress_callback:
            progress_callback(step, total, name, stage)

    _cb(0, 10, "Saving run config", "reproducibility")
    run_config = save_run_config(
        case_name=case_name, model=model,
        date_window_days=date_window_days, sim_threshold=sim_threshold,
    )

    _cb(1, 10, "Loading model", "model load")
    embed_model = load_embed_model()

    _cb(2, 10, "Loading events", "loading")
    all_events = load_all_events()
    if not all_events:
        return {"error": "No events found. Run full pipeline first."}

    _cb(3, 10, "Building reliability map", "reliability")
    queue = load_queue()
    reliability_map = build_reliability_map(queue) if queue else load_reliability_map()
    file_hashes = get_file_hashes_from_queue(queue) if queue else {}

    _cb(4, 10, f"{len(all_events)} events", "semantic merging")
    merged = run_semantic_merge(
        all_events, embed_model,
        date_window_days=date_window_days,
        sim_threshold=sim_threshold,
        reliability_map=reliability_map,
    )
    save_merged_events(merged)
    del all_events
    gc.collect()

    _cb(5, 10, f"{len(merged)} merged", "contradiction detection")
    contradictions = find_contradictions_semantic(merged, model=embed_model)
    save_contradictions(contradictions)

    conflict_ids = set()
    for c in contradictions:
        conflict_ids.add(c.get("event_a", {}).get("merged_event_id", ""))
        conflict_ids.add(c.get("event_b", {}).get("merged_event_id", ""))
    for ev in merged:
        if ev.get("merged_event_id", "") in conflict_ids:
            ev["has_conflict"] = True

    _cb(6, 10, "Building hash chain", "immutability")
    chunks_indexed = load_all_chunks_indexed()
    events_indexed = load_all_events_indexed()
    hash_lookup = build_hash_chain_for_pipeline(merged, events_indexed, chunks_indexed, file_hashes)

    _cb(7, 10, "Building timeline", "timeline")
    generate_timeline_json(merged)
    generate_timeline_md(merged)

    _cb(8, 10, "Generating narrative", "narrative")
    _, narrative_sentences = generate_narrative_document(
        merged, contradictions, case_name=case_name, model=model
    )

    _cb(9, 10, "Building audit trail + legal exports", "export")
    build_audit_trail(chunks_indexed, events_indexed, merged, narrative_sentences)

    unresolved = collect_unresolved_issues(merged, contradictions)
    build_evidence_index(merged, hash_lookup)
    generate_legal_brief(case_name, merged, contradictions, unresolved, run_config, hash_lookup)

    n_merged = len(merged)
    n_contradictions = len(contradictions)
    n_critical = sum(1 for c in contradictions if c.get("impact_label") == "CRITICAL")
    n_unresolved = len(unresolved)

    del embed_model, chunks_indexed, events_indexed, merged, narrative_sentences
    gc.collect()

    _cb(10, 10, "Complete", "done")

    return {
        "merged_events": n_merged,
        "contradictions": n_contradictions,
        "critical_contradictions": n_critical,
        "unresolved_issues": n_unresolved,
    }
