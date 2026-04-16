#!/usr/bin/env python3
"""
Document Intelligence System - Main CLI Entrypoint

Usage:
    python main.py --input /path/to/documents
    python main.py --input /path/to/documents --query "What are the key findings?"
"""
import os
import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def ensure_dirs():
    dirs = [
        "data/processed/raw_text",
        "data/processed/chunks",
        "data/vector_store",
        "data/outputs/summaries",
        "data/outputs/entities",
        "data/outputs/timeline",
        "logs",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Document Intelligence System")
    parser.add_argument("--input", type=str, default="data/input", help="Input directory with documents")
    parser.add_argument("--query", type=str, default=None, help="Run a RAG query after processing")
    parser.add_argument("--timeline", action="store_true", help="Generate timeline from processed documents")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model to use (default: mistral)")
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=== Document Intelligence System ===")
    logger.info(f"Input directory: {args.input}")

    from pipeline.processor import run_pipeline

    def show_progress(current, total, name):
        pct = int((current / total) * 100) if total > 0 else 0
        logger.info(f"  [{pct:3d}%] ({current}/{total}) {name}")

    results = run_pipeline(input_dir=args.input, progress_callback=show_progress)
    logger.info(f"Pipeline results: {results}")

    if args.timeline:
        from pipeline.insights import generate_global_timeline
        events = generate_global_timeline()
        logger.info(f"Generated timeline with {len(events)} events")

    if args.query:
        from pipeline.embeddings import load_model
        from pipeline.rag import answer_question

        logger.info(f"Running query: {args.query}")
        model = load_model()
        result = answer_question(args.query, model, model_name=args.model)

        print("\n=== ANSWER ===")
        print(result["answer"])
        print("\n=== SOURCES ===")
        for src in result["sources"]:
            print(f"  - {src['file_name']} (page {src['page']})")


if __name__ == "__main__":
    main()
