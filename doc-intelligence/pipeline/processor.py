import gc
import logging
from pathlib import Path

from pipeline.ingestion import update_queue_item, get_pending_files
from pipeline.ocr import extract_pdf_text, ocr_pdf_pages, ocr_image_file
from pipeline.converter import excel_to_text, read_text_file
from pipeline.chunker import process_pages_to_chunks, save_raw_text, save_chunks
from pipeline.embeddings import embed_chunks, load_model
from pipeline.insights import generate_file_summary, generate_entities_report
from pipeline.event_extractor import extract_all_file_events

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
PDF_EXTENSION = ".pdf"
EXCEL_EXTENSIONS = {".xlsx", ".xls"}
CSV_EXTENSION = ".csv"
TEXT_EXTENSIONS = {".txt", ".md"}


def process_file(item: dict, model) -> bool:
    file_id = item["id"]
    file_path = item["path"]
    file_name = item["name"]
    ext = item["ext"].lower()

    logger.info(f"Processing: {file_name} ({file_id})")
    update_queue_item(file_id, status="processing")

    try:
        if ext == PDF_EXTENSION:
            pages = ocr_pdf_pages(file_path, min_text_length=50)
        elif ext in IMAGE_EXTENSIONS:
            text = ocr_image_file(file_path)
            pages = [(1, text)] if text else []
        elif ext in EXCEL_EXTENSIONS or ext == CSV_EXTENSION:
            text = excel_to_text(file_path)
            pages = [(1, text)] if text else []
        elif ext in TEXT_EXTENSIONS:
            text = read_text_file(file_path)
            pages = [(1, text)] if text else []
        else:
            logger.warning(f"Unsupported extension: {ext}")
            update_queue_item(file_id, status="error", error=f"Unsupported: {ext}")
            return False

        if not pages:
            logger.warning(f"No content extracted from {file_name}")
            update_queue_item(file_id, status="done", pages=0, chunks=0)
            return True

        save_raw_text(file_id, file_name, pages)

        chunks = process_pages_to_chunks(file_id, file_name, pages)
        del pages
        gc.collect()

        if chunks:
            save_chunks(file_id, chunks)
            embed_chunks(chunks, model)

            generate_file_summary(file_id, file_name, chunks)
            generate_entities_report(file_id, file_name, chunks)
            extract_all_file_events(file_id, chunks)

        update_queue_item(
            file_id,
            status="done",
            pages=len(set(c["page"] for c in chunks)) if chunks else 0,
            chunks=len(chunks),
        )
        del chunks
        gc.collect()
        logger.info(f"Done: {file_name}")
        return True

    except Exception as e:
        logger.error(f"Failed processing {file_name}: {e}")
        update_queue_item(file_id, status="error", error=str(e))
        return False


def run_pipeline(input_dir: str = "data/input", progress_callback=None) -> dict:
    from pipeline.ingestion import build_queue, get_queue_stats

    build_queue(input_dir)
    stats_before = get_queue_stats()

    logger.info(f"Pipeline start: {stats_before['pending']} pending files")

    model = load_model()
    logger.info("Embedding model loaded")

    processed = 0
    errors = 0

    pending = list(get_pending_files())
    total_pending = len(pending)

    for i, item in enumerate(pending):
        success = process_file(item, model)
        if success:
            processed += 1
        else:
            errors += 1

        if progress_callback:
            progress_callback(i + 1, total_pending, item["name"])

        del item
        gc.collect()

    del model
    gc.collect()

    stats_after = get_queue_stats()
    logger.info(f"Pipeline complete: {processed} processed, {errors} errors")
    return {"processed": processed, "errors": errors, "stats": stats_after}
