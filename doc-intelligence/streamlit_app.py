"""
Document Intelligence System - Streamlit UI
"""
import os
import sys
import json
import gc
import logging
from pathlib import Path

import streamlit as st

os.chdir(Path(__file__).parent)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("logs/app.log")],
)

DIRS = [
    "data/processed/raw_text",
    "data/processed/chunks",
    "data/vector_store",
    "data/outputs/summaries",
    "data/outputs/entities",
    "data/outputs/timeline",
    "data/outputs/events",
    "data/input",
    "logs",
]
for d in DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = ["Dashboard", "Document Search", "Document Explorer", "Insights", "Legal Intelligence"]

with st.sidebar:
    st.title("Document Intelligence")
    st.caption("Legal Case Reconstruction")
    st.divider()
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()

    from pipeline.ingestion import get_queue_stats
    stats = get_queue_stats()
    st.metric("Total Files", stats["total"])
    st.metric("Processed", stats["done"])
    if stats["total"] > 0:
        pct = int((stats["done"] / stats["total"]) * 100)
        st.progress(pct / 100, text=f"{pct}% complete")


if page == "Dashboard":
    st.title("Dashboard")

    from pipeline.ingestion import get_queue_stats, load_queue
    from pipeline.rag import check_ollama_available
    from pipeline.embeddings import INDEX_PATH, load_all_meta

    col1, col2, col3, col4 = st.columns(4)
    stats = get_queue_stats()

    with col1:
        st.metric("Total Files", stats["total"])
    with col2:
        st.metric("Processed", stats["done"])
    with col3:
        st.metric("Errors", stats["error"])
    with col4:
        st.metric("Total Chunks", stats["total_chunks"])

    st.divider()

    st.subheader("Processing Progress")
    if stats["total"] > 0:
        pct = stats["done"] / stats["total"]
        st.progress(pct, text=f"{stats['done']} / {stats['total']} files processed ({int(pct*100)}%)")
    else:
        st.info("No files in queue yet. Add documents to `data/input/` and click 'Run Pipeline' below.")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("System Status")
        vector_ok = INDEX_PATH.exists()

        st.write("OCR Engine", ":white_check_mark: Tesseract / PyMuPDF" if True else ":x: Not available")
        st.write("Vector Index", ":white_check_mark: FAISS index ready" if vector_ok else ":x: No index yet")

        ollama_ok, ollama_msg = check_ollama_available()
        if ollama_ok:
            st.write(f"LLM (Ollama)", f":white_check_mark: Available — {ollama_msg}")
        else:
            st.write("LLM (Ollama)", f":warning: Not running — {ollama_msg}")
            st.caption("Start Ollama with `ollama serve` and pull a model: `ollama pull mistral`")

    with col_b:
        st.subheader("Run Pipeline")
        input_dir = st.text_input("Input directory", value="data/input")
        if st.button("Start / Resume Processing", type="primary"):
            from pipeline.processor import run_pipeline

            progress_bar = st.progress(0, text="Initializing...")
            status_text = st.empty()

            def update_progress(current, total, name):
                pct = current / total if total > 0 else 0
                progress_bar.progress(pct, text=f"Processing: {name}")
                status_text.text(f"{current}/{total} files processed")

            with st.spinner("Running pipeline..."):
                results = run_pipeline(input_dir=input_dir, progress_callback=update_progress)
                gc.collect()

            st.success(f"Pipeline complete! Processed: {results['processed']}, Errors: {results['errors']}")
            st.rerun()

    st.divider()

    st.subheader("Upload Documents")
    uploaded = st.file_uploader(
        "Upload documents to process", accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "txt", "xlsx", "xls", "csv"]
    )
    if uploaded:
        input_path = Path("data/input")
        input_path.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in uploaded:
            out = input_path / f.name
            with open(out, "wb") as fp:
                fp.write(f.read())
            saved.append(f.name)
        st.success(f"Saved {len(saved)} files to data/input/. Click 'Start / Resume Processing' to process them.")

    st.divider()

    queue = load_queue()
    if queue:
        st.subheader("File Queue")
        error_files = [q for q in queue if q.get("status") == "error"]
        if error_files:
            with st.expander(f"Errors ({len(error_files)})"):
                for item in error_files:
                    st.error(f"{item['name']}: {item.get('error', 'unknown error')}")

        with st.expander(f"All Files ({len(queue)})"):
            for item in queue[:100]:
                status = item.get("status", "pending")
                icon = ":white_check_mark:" if status == "done" else ":x:" if status == "error" else ":clock3:"
                st.write(f"{icon} {item['name']} — {status} ({item.get('chunks', 0)} chunks)")
            if len(queue) > 100:
                st.caption(f"...and {len(queue)-100} more files")


elif page == "Document Search":
    st.title("Document Search")
    st.caption("Ask questions over all processed documents using AI")

    from pipeline.embeddings import INDEX_PATH

    if not INDEX_PATH.exists():
        st.warning("No vector index found. Please process documents first using the Dashboard.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = None

    with st.sidebar:
        st.divider()
        st.subheader("Search Settings")
        top_k = st.slider("Top results to retrieve", 1, 10, 5)
        model_name = st.text_input("Ollama model", value="mistral")

        if st.button("Load Embedding Model"):
            with st.spinner("Loading model..."):
                from pipeline.embeddings import load_model
                st.session_state.embed_model = load_model()
            st.success("Model loaded!")

        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                if msg["sources"]:
                    with st.expander(f"Sources ({len(msg['sources'])})"):
                        for i, src in enumerate(msg["sources"]):
                            st.markdown(f"**Source {i+1}:** {src['file_name']} — Page {src['page']}")
                            st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                            st.divider()

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                if st.session_state.embed_model is None:
                    from pipeline.embeddings import load_model
                    st.session_state.embed_model = load_model()

                from pipeline.rag import answer_question
                result = answer_question(query, st.session_state.embed_model, model=model_name, top_k=top_k)
                gc.collect()

            st.write(result["answer"])

            if result["sources"]:
                with st.expander(f"Sources ({len(result['sources'])})"):
                    for i, src in enumerate(result["sources"]):
                        st.markdown(f"**Source {i+1}:** {src['file_name']} — Page {src['page']}")
                        st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
                        st.divider()

        st.session_state.chat_history.append(
            {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
        )


elif page == "Document Explorer":
    st.title("Document Explorer")
    st.caption("Browse extracted text and chunks from processed documents")

    from pipeline.ingestion import load_queue
    from pipeline.chunker import load_chunks_for_file, CHUNKS_DIR, RAW_TEXT_DIR

    queue = load_queue()
    done_files = [q for q in queue if q.get("status") == "done"]

    if not done_files:
        st.warning("No processed documents yet. Run the pipeline from the Dashboard first.")
        st.stop()

    col_f, col_s = st.columns([1, 3])

    with col_f:
        st.subheader("Filters")
        ext_options = sorted(set(f.get("ext", "") for f in done_files))
        selected_ext = st.multiselect("File type", ext_options, default=ext_options)

        search_name = st.text_input("Search by filename")

    filtered = [f for f in done_files if f.get("ext", "") in selected_ext]
    if search_name:
        filtered = [f for f in filtered if search_name.lower() in f["name"].lower()]

    with col_s:
        st.subheader(f"Documents ({len(filtered)} shown)")

        if not filtered:
            st.info("No documents match the current filters.")
        else:
            file_names = [f["name"] for f in filtered]
            selected_name = st.selectbox("Select document", file_names)
            selected_file = next(f for f in filtered if f["name"] == selected_name)
            file_id = selected_file["id"]

            tab1, tab2 = st.tabs(["Extracted Text", "Chunks"])

            with tab1:
                raw_path = RAW_TEXT_DIR / f"{file_id}.jsonl"
                if raw_path.exists():
                    pages_shown = 0
                    with open(raw_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                with st.expander(f"Page {record.get('page', '?')}"):
                                    st.text(record.get("text", "")[:2000])
                                pages_shown += 1
                                if pages_shown >= 20:
                                    st.caption("(Showing first 20 pages)")
                                    break
                            except Exception:
                                pass

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        if st.button("Download Raw Text (JSONL)"):
                            with open(raw_path, "r") as f:
                                content = f.read()
                            st.download_button(
                                "Download",
                                data=content,
                                file_name=f"{selected_file['name']}_raw.jsonl",
                                mime="application/json",
                            )
                else:
                    st.info("Raw text not available for this file.")

            with tab2:
                chunks = load_chunks_for_file(file_id)
                if chunks:
                    st.write(f"{len(chunks)} chunks")
                    for i, chunk in enumerate(chunks[:30]):
                        with st.expander(f"Chunk {i+1} — Page {chunk.get('page', '?')} ({chunk.get('word_count', 0)} words)"):
                            st.text(chunk.get("text", ""))
                    if len(chunks) > 30:
                        st.caption(f"...and {len(chunks)-30} more chunks. Download to see all.")

                    chunks_json = json.dumps(chunks, indent=2)
                    st.download_button(
                        "Download All Chunks (JSON)",
                        data=chunks_json,
                        file_name=f"{selected_file['name']}_chunks.json",
                        mime="application/json",
                    )
                else:
                    st.info("No chunks available for this file.")


elif page == "Insights":
    st.title("Insights")
    st.caption("Auto-generated summaries, entities, and timeline from all processed documents")

    from pipeline.insights import load_all_summaries, load_all_entities, load_global_timeline, generate_global_timeline

    tab1, tab2, tab3 = st.tabs(["Summaries", "Entities", "Timeline"])

    with tab1:
        summaries = load_all_summaries()
        if not summaries:
            st.warning("No summaries yet. Process documents from the Dashboard first.")
        else:
            st.write(f"{len(summaries)} document summaries")
            search_sum = st.text_input("Search summaries")
            filtered_sum = [s for s in summaries if not search_sum or search_sum.lower() in s.get("file_name", "").lower()]

            for s in filtered_sum[:50]:
                with st.expander(f"{s.get('file_name', 'Unknown')} — {s.get('chunk_count', 0)} chunks, {s.get('total_words', 0)} words"):
                    st.write(s.get("preview", "")[:500])
                    if s.get("dates"):
                        st.caption(f"Dates found: {', '.join(s['dates'][:5])}")

            col_e1, col_e2 = st.columns(2)
            with col_e1:
                summary_json = json.dumps(summaries, indent=2)
                st.download_button("Export Summaries (JSON)", data=summary_json, file_name="summaries.json", mime="application/json")
            with col_e2:
                md_lines = ["# Document Summaries\n"]
                for s in summaries:
                    md_lines.append(f"## {s.get('file_name', 'Unknown')}")
                    md_lines.append(f"- Chunks: {s.get('chunk_count', 0)}")
                    md_lines.append(f"- Words: {s.get('total_words', 0)}")
                    if s.get("dates"):
                        md_lines.append(f"- Dates: {', '.join(s['dates'][:5])}")
                    md_lines.append(f"\n{s.get('preview', '')[:300]}\n")
                st.download_button("Export Summaries (Markdown)", data="\n".join(md_lines), file_name="summaries.md", mime="text/markdown")

    with tab2:
        entities = load_all_entities()
        if not entities:
            st.warning("No entities extracted yet.")
        else:
            st.write(f"{len(entities)} files analyzed")
            all_emails = []
            all_urls = []
            all_phrases = {}

            for e in entities:
                all_emails.extend(e.get("emails", []))
                all_urls.extend(e.get("urls", []))
                for p in e.get("capitalized_phrases", []):
                    phrase = p["phrase"]
                    all_phrases[phrase] = all_phrases.get(phrase, 0) + p["count"]

            col_x, col_y = st.columns(2)
            with col_x:
                st.subheader("Top Phrases")
                top_phrases = sorted(all_phrases.items(), key=lambda x: -x[1])[:25]
                for phrase, count in top_phrases:
                    st.write(f"**{phrase}** ({count})")
            with col_y:
                if all_emails:
                    st.subheader("Emails Found")
                    for email in list(set(all_emails))[:20]:
                        st.write(email)
                if all_urls:
                    st.subheader("URLs Found")
                    for url in list(set(all_urls))[:10]:
                        st.write(url)

            entities_json = json.dumps(entities, indent=2)
            st.download_button("Export Entities (JSON)", data=entities_json, file_name="entities.json", mime="application/json")

    with tab3:
        col_gen, _ = st.columns([1, 3])
        with col_gen:
            if st.button("Regenerate Timeline"):
                with st.spinner("Generating timeline..."):
                    generate_global_timeline()
                st.success("Timeline updated!")
                st.rerun()

        events = load_global_timeline()
        if not events:
            st.warning("No timeline events found. Click 'Regenerate Timeline' after processing documents.")
        else:
            st.write(f"{len(events)} date mentions found across documents")
            search_evt = st.text_input("Filter events")
            filtered_evt = [e for e in events if not search_evt or search_evt.lower() in e.get("date", "").lower() or search_evt.lower() in e.get("file_name", "").lower()]

            for evt in filtered_evt[:100]:
                st.write(f"**{evt.get('date', '?')}** — {evt.get('file_name', '?')} (page {evt.get('page', '?')})")

            timeline_json = json.dumps(events, indent=2)
            st.download_button("Export Timeline (JSON)", data=timeline_json, file_name="timeline.json", mime="application/json")


elif page == "Legal Intelligence":
    st.title("Legal Intelligence")
    st.caption("Court-grade case reconstruction: semantic merging · reliability weighting · contradiction detection · traceable narrative · full audit trail")

    from pipeline.event_extractor import load_all_events, EVENTS_DIR
    from pipeline.event_merger import load_merged_events, MERGED_EVENTS_PATH
    from pipeline.contradiction_detector import load_contradictions, CONTRADICTIONS_PATH
    from pipeline.source_reliability import load_reliability_map
    from pipeline.audit_trail import load_audit_trail, load_audit_index
    from pipeline.narrative_generator import (
        load_narrative, load_timeline_json, load_timeline_md,
        load_narrative_sentences,
        NARRATIVE_PATH, TIMELINE_JSON_PATH, TIMELINE_MD_PATH
    )
    from pipeline.legal_exporter import (
        load_legal_brief, load_evidence_index, load_unresolved_issues, load_run_config
    )
    from pipeline.hash_chain import load_hash_chain, verify_hash_chain_integrity

    raw_events = load_all_events()
    merged_events = load_merged_events()
    contradictions = load_contradictions()
    narrative = load_narrative()
    timeline_json_data = load_timeline_json()
    reliability_map = load_reliability_map()
    narrative_sentences = load_narrative_sentences()
    audit_trail = load_audit_trail()
    legal_brief = load_legal_brief()
    evidence_index = load_evidence_index()
    unresolved_issues = load_unresolved_issues()
    run_config = load_run_config()

    critical_c = [c for c in contradictions if c.get("impact_label") == "CRITICAL"]
    hash_chain = load_hash_chain()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Raw Events", len(raw_events))
    with col2:
        st.metric("Merged Events", len(merged_events))
    with col3:
        c_label = f"⚠ {len(critical_c)} CRITICAL" if critical_c else str(len(contradictions))
        st.metric("Contradictions", c_label)
    with col4:
        high_rel = sum(1 for v in reliability_map.values() if v.get("level") == "high")
        st.metric("High-Reliability Docs", high_rel)
    with col5:
        st.metric("Unresolved Issues", len(unresolved_issues))
    with col6:
        st.metric("Hash Chain Records", len(hash_chain))

    st.divider()

    with st.expander("Pipeline Configuration & Run", expanded=not bool(merged_events)):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            case_name = st.text_input("Case name / title", value="Legal Case")
            ollama_model = st.text_input("Ollama model (for AI narrative)", value="mistral")
        with col_b:
            date_window = st.slider("Date proximity window (days)", 1, 14, 3,
                help="Events within this many days of each other can be merged")
            sim_thresh = st.slider("Semantic similarity threshold", 0.30, 0.90, 0.65, 0.05,
                help="Cosine similarity required to merge two events (higher = stricter)")
        with col_c:
            st.markdown("**Merge logic:**")
            st.markdown(f"- Date proximity ≤ **{date_window} days**")
            st.markdown(f"- Cosine similarity ≥ **{sim_thresh:.0%}**")
            st.markdown("- Source reliability propagated into confidence")
            st.markdown("- Every narrative sentence references event IDs")

        run_col, regen_col = st.columns(2)
        with run_col:
            run_full = st.button("Run Full Legal Pipeline", type="primary",
                help="Extract events + semantic merge + contradictions + timeline + narrative + audit trail")
        with regen_col:
            run_regen = st.button("Re-merge & Regenerate",
                help="Skip event extraction (already done) — re-run merge with new parameters")

        if run_full or run_regen:
            prog_bar = st.progress(0, text="Starting...")
            status_area = st.empty()

            def legal_progress(current, total, name, stage):
                pct = current / max(total, 1)
                prog_bar.progress(pct, text=f"[{stage.upper()}] {name}")
                status_area.text(f"Step {current}/{total}: {stage} — {name}")

            with st.spinner("Running legal intelligence pipeline..."):
                from pipeline.legal_pipeline import run_legal_pipeline, run_legal_pipeline_from_chunks
                gc.collect()
                if run_full:
                    result = run_legal_pipeline(
                        case_name=case_name, model=ollama_model,
                        date_window_days=date_window, sim_threshold=sim_thresh,
                        progress_callback=legal_progress,
                    )
                else:
                    result = run_legal_pipeline_from_chunks(
                        case_name=case_name, model=ollama_model,
                        date_window_days=date_window, sim_threshold=sim_thresh,
                        progress_callback=legal_progress,
                    )
                gc.collect()

            if "error" in result:
                st.error(result["error"])
            else:
                n_crit = result.get("critical_contradictions", 0)
                crit_str = f" · **{n_crit} CRITICAL**" if n_crit else ""
                st.success(
                    f"Done! {result.get('merged_events', 0)} merged events · "
                    f"{result.get('contradictions', 0)} contradictions{crit_str} · "
                    f"{result.get('unresolved_issues', 0)} unresolved issues · "
                    f"Hash chain: {result.get('hashed_files', 0)} files"
                )
                st.rerun()

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Timeline", "Contradictions", "Legal Narrative", "Audit Trail",
        "Legal Export", "Reliability", "Raw Events"
    ])

    with tab1:
        if not timeline_json_data:
            st.warning("No timeline generated yet. Run the Legal Intelligence Pipeline above.")
        else:
            CONF_BADGE = {"high": "🟢 HIGH", "medium": "🟡 MED", "low": "🔴 LOW"}
            REL_BADGE = {"high": "★ Official", "medium": "◆ Report", "low": "◇ Informal"}

            c_f, c_t, c_conf = st.columns(3)
            with c_f:
                search_tl = st.text_input("Filter by keyword / actor / date")
            with c_t:
                event_types_all = sorted(set(e.get("event_type", "") for e in timeline_json_data))
                selected_types = st.multiselect("Event type", event_types_all, default=event_types_all)
            with c_conf:
                conf_filter = st.selectbox("Confidence level", ["All", "High only", "Flag low"])

            filtered_tl = []
            for e in timeline_json_data:
                if e.get("event_type", "") not in selected_types:
                    continue
                if conf_filter == "High only" and e.get("confidence_level") != "high":
                    continue
                if search_tl:
                    haystack = (e.get("description","") + " ".join(e.get("actors",[])) +
                                " ".join(c.get("source_file","") for c in e.get("citations",[])))
                    if search_tl.lower() not in haystack.lower():
                        continue
                filtered_tl.append(e)

            low_count = sum(1 for e in filtered_tl if e.get("low_confidence_flag"))
            st.write(f"**{len(filtered_tl)} events** shown" +
                     (f" · ⚠ {low_count} low-confidence" if low_count else ""))

            for ev in filtered_tl[:100]:
                date_str = ev.get("date", "?")
                etype = ev.get("event_type", "general").upper()
                conf = ev.get("confidence", 0)
                conf_lvl = ev.get("confidence_level", "medium")
                rel = ev.get("source_reliability", "medium")
                low_flag = "⚠ " if ev.get("low_confidence_flag") else ""
                ref_id = ev.get("merged_event_id", "")
                badge = CONF_BADGE.get(conf_lvl, "🟡 MED")
                rel_b = REL_BADGE.get(rel, "◆")
                label = f"{low_flag}{date_str} — {etype} | {badge} | {rel_b} | {conf:.0%}"

                with st.expander(label):
                    st.write(ev.get("description", "")[:300])
                    col_ev1, col_ev2 = st.columns(2)
                    with col_ev1:
                        actors = ", ".join(ev.get("actors", [])[:5])
                        if actors:
                            st.caption(f"**Parties:** {actors}")
                        if ev.get("location"):
                            st.caption(f"**Location:** {ev['location']}")
                    with col_ev2:
                        st.caption(f"**Event ref:** `{ref_id}`")
                        st.caption(f"**Sources verified:** {ev.get('source_count', 1)}")
                        for cit in ev.get("citations", [])[:3]:
                            rel_mark = "★" if cit.get("source_reliability","") == "high" else "◆"
                            st.caption(f"  {rel_mark} {cit.get('source_file','?')} p.{cit.get('page','?')}"
                                       f" (weight: {cit.get('source_weight',0.7):.2f})")

            if len(filtered_tl) > 100:
                st.caption(f"...{len(filtered_tl)-100} more events. Download to see all.")

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                tl_json_str = json.dumps(timeline_json_data, indent=2, ensure_ascii=False)
                st.download_button("Download Timeline (JSON)", data=tl_json_str,
                                   file_name="timeline_legal.json", mime="application/json")
            with col_dl2:
                tl_md = load_timeline_md()
                if tl_md:
                    st.download_button("Download Timeline (Markdown)", data=tl_md,
                                       file_name="timeline_legal.md", mime="text/markdown")

    with tab2:
        if not contradictions:
            if merged_events:
                st.success("No significant contradictions detected across source documents.")
            else:
                st.warning("Run the Legal Intelligence Pipeline to detect contradictions.")
        else:
            crit_grp = [c for c in contradictions if c.get("impact_label") == "CRITICAL"]
            sig_grp = [c for c in contradictions if c.get("impact_label") == "SIGNIFICANT"]
            minor_grp = [c for c in contradictions if c.get("impact_label") == "MINOR"]

            if crit_grp:
                st.error(f"**{len(crit_grp)} CRITICAL contradiction(s)** — highest impact. Priority review required.")
            st.info(
                f"Total: **{len(contradictions)}** | CRITICAL: **{len(crit_grp)}** | "
                f"SIGNIFICANT: **{len(sig_grp)}** | MINOR: **{len(minor_grp)}**"
            )
            st.caption(
                "Impact score = severity × source_weight × semantic_similarity  |  "
                "Ranked by impact score descending  |  "
                "Detection: embedding cosine similarity + opposition patterns + negation analysis"
            )

            for impact_label, group, icon in [
                ("CRITICAL", crit_grp, "🔴"),
                ("SIGNIFICANT", sig_grp, "🟡"),
                ("MINOR", minor_grp, "🔵"),
            ]:
                if not group:
                    continue
                st.subheader(f"{icon} {impact_label} ({len(group)})")
                for i, c in enumerate(group, 1):
                    ev_a = c.get("event_a", {})
                    ev_b = c.get("event_b", {})
                    sim = c.get("semantic_similarity", 0)
                    impact = c.get("impact_score", 0)
                    sev = c.get("severity", "?").upper()
                    ctype = c.get("type", "")
                    rel_a = ev_a.get("reliability", "?")
                    rel_b = ev_b.get("reliability", "?")
                    both_high = rel_a == "high" and rel_b == "high"
                    both_flag = " ⚠ BOTH HIGH-RELIABILITY" if both_high else ""

                    with st.expander(
                        f"{icon} Contradiction {i} | Impact: {impact:.3f} | Sim: {sim:.0%} | {c.get('description','')[:70]}{both_flag}"
                    ):
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Impact Score", f"{impact:.3f}")
                        m2.metric("Semantic Sim", f"{sim:.0%}")
                        m3.metric("Severity", sev)
                        m4.metric("Type", ctype.replace("_"," ").title())

                        col_a, col_b_col = st.columns(2)
                        with col_a:
                            wt_a = ev_a.get("source_weight", 0.7)
                            rel_mark_a = "★" if rel_a == "high" else ("◆" if rel_a == "medium" else "◇")
                            st.markdown(f"**{rel_mark_a} Source A** `[{ev_a.get('merged_event_id','?')}]`")
                            st.caption(f"Date: `{ev_a.get('date','?')}` | Reliability: **{rel_a}** "
                                       f"(weight: {wt_a:.2f}) | Conf: {ev_a.get('confidence',0):.0%}")
                            st.write(ev_a.get("description","")[:280])
                            for cit in ev_a.get("citations", [])[:2]:
                                st.caption(f"  [{cit.get('source_file','?')}, p.{cit.get('page','?')}] "
                                           f"wt:{cit.get('source_weight',0.7):.2f}")
                        with col_b_col:
                            wt_b = ev_b.get("source_weight", 0.7)
                            rel_mark_b = "★" if rel_b == "high" else ("◆" if rel_b == "medium" else "◇")
                            st.markdown(f"**{rel_mark_b} Source B** `[{ev_b.get('merged_event_id','?')}]`")
                            st.caption(f"Date: `{ev_b.get('date','?')}` | Reliability: **{rel_b}** "
                                       f"(weight: {wt_b:.2f}) | Conf: {ev_b.get('confidence',0):.0%}")
                            st.write(ev_b.get("description","")[:280])
                            for cit in ev_b.get("citations", [])[:2]:
                                st.caption(f"  [{cit.get('source_file','?')}, p.{cit.get('page','?')}] "
                                           f"wt:{cit.get('source_weight',0.7):.2f}")

                        if both_high:
                            st.error("Both sources are high-reliability — this conflict cannot be auto-resolved. "
                                     "Manual legal review required.")

            cont_json = json.dumps(contradictions, indent=2, ensure_ascii=False)
            st.download_button("Export All Contradictions (JSON)", data=cont_json,
                               file_name="contradictions.json", mime="application/json")

    with tab3:
        if not narrative:
            st.warning("No legal narrative generated yet. Run the Legal Intelligence Pipeline above.")
        else:
            st.info("Every statement below includes event reference IDs `[EVENT_ID]` for full traceability.")
            if narrative_sentences:
                _last_section_seen = None
                for sent in narrative_sentences:
                    section = sent.get("section", "")
                    if section and section != _last_section_seen:
                        st.subheader(section)
                        _last_section_seen = section
                    refs = sent.get("event_refs", [])
                    cit_inline = sent.get("citation_inline", "")
                    ref_str = " ".join(f"`{r}`" for r in refs) if refs else ""
                    st.markdown(f"{sent.get('text','')}  \n{cit_inline}  \n{ref_str}")
                    st.markdown("---")
            else:
                st.markdown(narrative)

            col_n1, col_n2, col_n3 = st.columns(3)
            with col_n1:
                st.download_button("Download Narrative (Markdown)", data=narrative,
                                   file_name="legal_narrative.md", mime="text/markdown")
            with col_n2:
                st.download_button("Download Narrative (TXT)", data=narrative,
                                   file_name="legal_narrative.txt", mime="text/plain")
            with col_n3:
                if narrative_sentences:
                    sent_json = json.dumps(narrative_sentences, indent=2, ensure_ascii=False)
                    st.download_button("Download Sentences (JSON)", data=sent_json,
                                       file_name="narrative_sentences.json", mime="application/json")

    with tab4:
        st.subheader("Audit Trail — Full Provenance")
        st.caption("Trace any claim: narrative sentence → merged event → raw event → chunk → raw text")

        if not audit_trail:
            st.warning("No audit trail yet. Run the Legal Intelligence Pipeline to build the full provenance chain.")
        else:
            search_audit = st.text_input("Search audit trail by sentence text / event ID / file")
            filtered_audit = [
                r for r in audit_trail
                if not search_audit
                or search_audit.lower() in r.get("text","").lower()
                or search_audit.lower() in " ".join(r.get("event_refs",[])).lower()
                or any(
                    search_audit.lower() in str(p.get("description_preview","")).lower() or
                    any(search_audit.lower() in str(re.get("file_name","")).lower()
                        for re in p.get("raw_events",[]))
                    for p in r.get("provenance",[])
                )
            ]

            st.write(f"**{len(filtered_audit)} audit records** shown")

            for record in filtered_audit[:40]:
                sid = record.get("sentence_id","?")
                text = record.get("text","")[:120]
                section = record.get("section","")
                refs = record.get("event_refs",[])
                prov = record.get("provenance",[])
                ref_str = ", ".join(f"`{r}`" for r in refs) if refs else "*(no event refs)*"

                with st.expander(f"[{section}] {text}..."):
                    st.caption(f"**Sentence ID:** `{sid}` | **Event refs:** {ref_str}")
                    st.write(f"**Full text:** {record.get('text','')}")

                    for p_idx, prov_item in enumerate(prov):
                        st.markdown(f"**Merged Event:** `{prov_item.get('merged_event_id','?')}`")
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.markdown(f"- Date: `{prov_item.get('date_normalized','?')}`")
                            st.markdown(f"- Type: `{prov_item.get('event_type','?')}`")
                            st.markdown(f"- Confidence: {prov_item.get('confidence',0):.0%} "
                                        f"({prov_item.get('confidence_level','?')})")
                        with m_col2:
                            st.markdown(f"- Reliability: **{prov_item.get('source_reliability','?')}**")
                            st.markdown(f"- Source count: {prov_item.get('source_count',1)}")
                            st.markdown(f"- Semantic sim: {prov_item.get('avg_semantic_similarity',0):.0%}")

                        for raw_ev in prov_item.get("raw_events",[])[:3]:
                            with st.expander(
                                f"↳ Raw event `{raw_ev.get('event_id','?')}` | "
                                f"{raw_ev.get('source_file','?')} p.{raw_ev.get('page','?')} "
                                f"[{raw_ev.get('language','?').upper()}]"
                            ):
                                st.write(raw_ev.get("description_preview",""))
                                st.caption(f"Date: {raw_ev.get('date_normalized','?')} "
                                           f"(raw: '{raw_ev.get('date_raw','?')}') | "
                                           f"Confidence: {raw_ev.get('confidence',0):.0%} | "
                                           f"Reliability: {raw_ev.get('source_reliability','?')} "
                                           f"(weight: {raw_ev.get('source_weight',0.7):.2f})")
                                chunk = raw_ev.get("chunk",{})
                                if chunk:
                                    with st.expander(f"↳ Chunk `{chunk.get('chunk_id','?')}` — "
                                                     f"{chunk.get('word_count',0)} words"):
                                        st.text(chunk.get("text_preview",""))
                                        st.caption(f"File: {chunk.get('file_name','?')} | "
                                                   f"Page: {chunk.get('page','?')}")

            if len(filtered_audit) > 40:
                st.caption(f"...{len(filtered_audit)-40} more audit records. Download to view all.")

            audit_json = json.dumps(audit_trail[:200], indent=2, ensure_ascii=False)
            st.download_button("Export Audit Trail (JSON, first 200)",
                               data=audit_json, file_name="audit_trail.json", mime="application/json")

    with tab5:
        st.subheader("Legal Export — Court-Ready Documents")
        st.caption(
            "Structured legal brief, evidence index, SHA256 hash chain, and reproducibility config. "
            "All outputs are deterministic given the same inputs and config."
        )

        if not legal_brief and not evidence_index:
            st.warning("No legal exports yet. Run the Legal Intelligence Pipeline to generate court-ready documents.")
        else:
            e_col1, e_col2, e_col3, e_col4 = st.columns(4)
            with e_col1:
                st.metric("Hash Chain Records", len(hash_chain))
            with e_col2:
                integrity = verify_hash_chain_integrity() if hash_chain else {"integrity_ok": None, "broken_links": 0, "total_records": 0}
                ok = integrity.get("integrity_ok")
                label = "✓ Intact" if ok is True else ("⚠ Broken" if ok is False else "—")
                st.metric("Chain Integrity", label)
            with e_col3:
                st.metric("Evidence Index Entries", len(evidence_index))
            with e_col4:
                st.metric("Unresolved Issues", len(unresolved_issues))

            st.divider()

            sec_brief, sec_evidence, sec_chain, sec_config, sec_unresolved = st.tabs([
                "Legal Brief", "Evidence Index", "Hash Chain", "Run Config", "Unresolved Issues"
            ])

            with sec_brief:
                if legal_brief:
                    st.markdown(legal_brief)
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.download_button("Download legal_brief.md", data=legal_brief,
                                           file_name="legal_brief.md", mime="text/markdown")
                    with col_d2:
                        st.download_button("Download legal_brief.txt", data=legal_brief,
                                           file_name="legal_brief.txt", mime="text/plain")
                else:
                    st.info("Legal brief not yet generated.")

            with sec_evidence:
                if evidence_index:
                    st.write(f"**{len(evidence_index)} events** in evidence index")
                    search_ev = st.text_input("Search evidence index by event ID / file / date", key="ev_search")
                    filtered_ev = [
                        e for e in evidence_index
                        if not search_ev
                        or search_ev.lower() in e.get("event_id","").lower()
                        or search_ev.lower() in e.get("date","").lower()
                        or any(search_ev.lower() in s.get("file","").lower()
                               for s in e.get("sources",[]))
                    ]
                    for ev_idx in filtered_ev[:50]:
                        eid = ev_idx.get("event_id","?")
                        date = ev_idx.get("date","?")
                        etype = ev_idx.get("event_type","?")
                        conf = ev_idx.get("confidence",0)
                        conflict = "⚡ CONFLICT" if ev_idx.get("conflict") else ""
                        srcs = ev_idx.get("sources",[])
                        with st.expander(f"`{eid}` | {date} | {etype} | conf:{conf:.0%} {conflict}"):
                            for s in srcs:
                                rel_mark = "★" if s.get("source_reliability")=="high" else ("◆" if s.get("source_reliability")=="medium" else "◇")
                                h = s.get("hash","")
                                hash_str = f"`{h[:12]}...`" if h else "*(not hashed)*"
                                st.caption(f"{rel_mark} **{s.get('file','?')}** p.{s.get('page','?')} "
                                           f"weight:{s.get('source_weight',0.7):.2f} hash:{hash_str}")
                    if len(filtered_ev) > 50:
                        st.caption(f"...{len(filtered_ev)-50} more entries")
                    ev_json = json.dumps(evidence_index, indent=2, ensure_ascii=False)
                    st.download_button("Download evidence_index.json", data=ev_json,
                                       file_name="evidence_index.json", mime="application/json")
                else:
                    st.info("Evidence index not yet generated.")

            with sec_chain:
                if hash_chain:
                    st.write(f"**{len(hash_chain)} hash records** in SHA256 chain")
                    if ok is True:
                        st.success("Hash chain integrity verified — no broken links detected.")
                    elif ok is False:
                        broken = integrity.get("broken_links", 0)
                        st.error(f"Hash chain has **{broken} broken link(s)**. Evidence may have been modified.")
                        for b in integrity.get("broken_details",[])[:5]:
                            st.caption(f"Broken: {b.get('stage')} `{b.get('id')}` → orphan parent `{b.get('broken_parent','?')[:16]}...`")

                    stage_filter = st.selectbox("Filter by stage", ["All", "file", "chunk", "event", "merged_event"], key="hash_stage")
                    shown = [r for r in hash_chain if stage_filter == "All" or r.get("stage") == stage_filter][:100]
                    for r in shown:
                        stage = r.get("stage","?")
                        rid = r.get("merged_event_id") or r.get("event_id") or r.get("chunk_id") or ""
                        fname = r.get("file_name","")
                        m_hash = r.get("merged_event_hash") or r.get("event_hash") or r.get("chunk_hash") or ""
                        p_hash = r.get("parent","")
                        with st.expander(f"[{stage.upper()}] {rid or fname} — hash:{m_hash[:12] if m_hash else '?'}..."):
                            st.caption(f"File: {fname}")
                            if r.get("file_hash"):
                                st.code(f"file_hash:  {r['file_hash']}", language=None)
                            if r.get("chunk_hash"):
                                st.code(f"chunk_hash: {r['chunk_hash']}", language=None)
                            if r.get("event_hash"):
                                st.code(f"event_hash: {r['event_hash']}", language=None)
                            if r.get("merged_event_hash"):
                                st.code(f"merged_hash:{r['merged_event_hash']}", language=None)
                            if p_hash:
                                st.caption(f"Parent: `{p_hash[:32]}...`")

                    chain_json = json.dumps(hash_chain[:500], indent=2, ensure_ascii=False)
                    st.download_button("Download audit_hash_chain.json (first 500 records)", data=chain_json,
                                       file_name="audit_hash_chain.json", mime="application/json")
                else:
                    st.info("Hash chain not yet generated.")

            with sec_config:
                if run_config:
                    st.success("Reproducibility config saved. Re-running with these exact parameters produces identical results.")
                    cfg_cols = st.columns(2)
                    with cfg_cols[0]:
                        st.markdown(f"**Case:** {run_config.get('case_name','?')}")
                        st.markdown(f"**Generated:** {run_config.get('timestamp_human','?')}")
                        st.markdown(f"**Model:** {run_config.get('model','?')}")
                        st.markdown(f"**Embedding model:** `{run_config.get('embedding_model','?')}`")
                    with cfg_cols[1]:
                        st.markdown(f"**Date window:** ±{run_config.get('date_window_days','?')} days")
                        st.markdown(f"**Similarity threshold:** {run_config.get('sim_threshold',0):.0%}")
                        st.markdown(f"**High confidence threshold:** {run_config.get('confidence_thresholds',{}).get('high',0):.0%}")
                        sw = run_config.get('source_weights',{})
                        st.markdown(f"**Source weights:** high={sw.get('high',1.0)} | medium={sw.get('medium',0.7)} | low={sw.get('low',0.4)}")

                    lang_rules = run_config.get("narrative_language_rules",{})
                    if lang_rules:
                        st.markdown("**Narrative language rules:**")
                        for lvl, phrase in lang_rules.items():
                            st.caption(f"  {lvl} → \"{phrase}\"")

                    cfg_json = json.dumps(run_config, indent=2, ensure_ascii=False)
                    st.download_button("Download run_config.json", data=cfg_json,
                                       file_name="run_config.json", mime="application/json")
                else:
                    st.info("Run config not yet saved. Run the pipeline to generate it.")

            with sec_unresolved:
                if not unresolved_issues:
                    st.success("No unresolved issues requiring manual review.")
                else:
                    st.error(f"**{len(unresolved_issues)} unresolved issue(s)** require manual legal review before this reconstruction can be considered final.")
                    type_counts = {}
                    for u in unresolved_issues:
                        t = u.get("issue_type","?")
                        type_counts[t] = type_counts.get(t, 0) + 1
                    for t, count in type_counts.items():
                        st.caption(f"  - {t.replace('_',' ').title()}: {count}")
                    st.divider()
                    for u in unresolved_issues[:30]:
                        utype = u.get("issue_type","?").replace("_"," ").upper()
                        ids = u.get("event_id") or ", ".join(u.get("event_ids",[]))
                        reason = u.get("reason","?")
                        action = u.get("action","?")
                        icon = "🔴" if "conflict" in utype.lower() else "⚠"
                        with st.expander(f"{icon} [{utype}] Event: `{ids}`"):
                            st.write(f"**Reason:** {reason}")
                            st.write(f"**Required action:** {action}")
                            if u.get("description"):
                                st.caption(f"Preview: {u['description']}")
                            if u.get("confidence") is not None:
                                st.caption(f"Confidence: {u.get('confidence',0):.0%}")

                    unresolv_json = json.dumps(unresolved_issues, indent=2, ensure_ascii=False)
                    st.download_button("Export Unresolved Issues (JSON)", data=unresolv_json,
                                       file_name="unresolved_issues.json", mime="application/json")

    with tab6:
        st.subheader("Source Reliability Classification")
        st.caption("Documents classified as high/medium/low reliability. Weights propagated into event confidence scores.")

        if not reliability_map:
            st.warning("No reliability map yet. Run the Legal Intelligence Pipeline to classify documents.")
        else:
            high_docs = [(fid, r) for fid, r in reliability_map.items() if r.get("level") == "high"]
            med_docs = [(fid, r) for fid, r in reliability_map.items() if r.get("level") == "medium"]
            low_docs = [(fid, r) for fid, r in reliability_map.items() if r.get("level") == "low"]

            c1, c2, c3 = st.columns(3)
            c1.metric("★ High Reliability", len(high_docs))
            c2.metric("◆ Medium Reliability", len(med_docs))
            c3.metric("◇ Low Reliability", len(low_docs))

            for label, group, weight in [
                ("★ High Reliability (weight: 1.0)", high_docs, 1.0),
                ("◆ Medium Reliability (weight: 0.70)", med_docs, 0.70),
                ("◇ Low Reliability (weight: 0.40)", low_docs, 0.40),
            ]:
                if not group:
                    continue
                with st.expander(f"{label} — {len(group)} documents"):
                    for fid, r in group[:50]:
                        st.markdown(f"**{r.get('file_name','?')}**")
                        st.caption(f"Reasons: {', '.join(r.get('reasons',[]))}")
                        scores = r.get("scores", {})
                        st.caption(f"Signal scores — high:{scores.get('high',0)} "
                                   f"medium:{scores.get('medium',0)} low:{scores.get('low',0)}")
                    if len(group) > 50:
                        st.caption(f"...and {len(group)-50} more")

            rel_json = json.dumps(reliability_map, indent=2, ensure_ascii=False)
            st.download_button("Export Reliability Map (JSON)",
                               data=rel_json, file_name="reliability_map.json", mime="application/json")

    with tab7:
        if not raw_events:
            st.warning("No events extracted yet. Run the Legal Intelligence Pipeline or process documents first.")
        else:
            st.write(f"**{len(raw_events)} raw events** extracted from document chunks")
            c_f, c_l = st.columns(2)
            with c_f:
                filter_raw = st.text_input("Filter raw events by keyword / file / date")
            with c_l:
                lang_opts = sorted(set(e.get("language","en") for e in raw_events))
                sel_lang = st.multiselect("Language", lang_opts, default=lang_opts)

            filtered_raw = [
                e for e in raw_events
                if e.get("language","en") in sel_lang
                and (not filter_raw
                     or filter_raw.lower() in e.get("description","").lower()
                     or filter_raw.lower() in e.get("source_file","").lower()
                     or filter_raw.lower() in e.get("date_raw","").lower())
            ]
            st.write(f"{len(filtered_raw)} events match filter")

            for ev in filtered_raw[:60]:
                date_raw = ev.get("date_raw","?")
                date_norm = ev.get("date_normalized","?")
                lang = ev.get("language","?").upper()
                etype = ev.get("event_type","?").upper()
                src = ev.get("source_file","?")
                page = ev.get("page","?")
                conf = ev.get("confidence",0)
                rel = ev.get("source_reliability","medium")
                wt = ev.get("source_weight",0.7)
                rel_mark = "★" if rel == "high" else ("◆" if rel == "medium" else "◇")
                with st.expander(
                    f"[{lang}] {date_raw} → {date_norm} | {etype} | {src} p.{page} "
                    f"| conf:{conf:.0%} | {rel_mark}{rel}"
                ):
                    st.write(ev.get("description",""))
                    c_ev1, c_ev2 = st.columns(2)
                    with c_ev1:
                        if ev.get("actors"):
                            st.caption(f"Actors: {', '.join(ev['actors'])}")
                        if ev.get("location"):
                            st.caption(f"Location: {ev['location']}")
                    with c_ev2:
                        st.caption(f"Event ID: `{ev.get('event_id','?')}`")
                        st.caption(f"Chunk ID: `{ev.get('chunk_id','?')}`")
                        st.caption(f"Source weight: {wt:.2f}")

            if len(filtered_raw) > 60:
                st.caption(f"...{len(filtered_raw)-60} more events")

            raw_json = json.dumps(raw_events[:500], indent=2, ensure_ascii=False)
            st.download_button("Export Raw Events (JSON, first 500)",
                               data=raw_json, file_name="raw_events.json", mime="application/json")
