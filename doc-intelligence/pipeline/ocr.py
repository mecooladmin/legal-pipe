import gc
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def preprocess_image(image):
    from PIL import Image

    max_width = 1280
    img = image.convert("L")
    if img.width > max_width:
        ratio = max_width / img.width
        new_size = (max_width, int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def extract_pdf_text(pdf_path: str) -> List[Tuple[int, str]]:
    try:
        import fitz

        pages = []
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            text = text.strip()
            pages.append((page_num + 1, text))
            del page
        doc.close()
        del doc
        gc.collect()
        return pages
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        return []


def ocr_image_tesseract(image) -> str:
    try:
        import pytesseract

        text = pytesseract.image_to_string(image, config="--psm 6 --oem 1")
        return text.strip()
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""


def ocr_pdf_pages(pdf_path: str, min_text_length: int = 50) -> List[Tuple[int, str]]:
    import fitz
    from PIL import Image
    import io

    results = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if len(text) >= min_text_length:
                results.append((page_num + 1, text))
                del page
                gc.collect()
                continue

            matrix = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            pix = None

            img = preprocess_image(img)
            ocr_text = ocr_image_tesseract(img)
            del img
            gc.collect()

            results.append((page_num + 1, ocr_text if ocr_text else text))
            del page
            gc.collect()

        doc.close()
        del doc
        gc.collect()
    except Exception as e:
        logger.error(f"OCR PDF processing failed for {pdf_path}: {e}")

    return results


def ocr_image_file(image_path: str) -> str:
    from PIL import Image

    try:
        img = Image.open(image_path)
        img = preprocess_image(img)
        text = ocr_image_tesseract(img)
        del img
        gc.collect()
        return text
    except Exception as e:
        logger.error(f"Image OCR failed for {image_path}: {e}")
        return ""
