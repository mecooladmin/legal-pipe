"""
Date normalization for Arabic and English dates.
Converts all detected dates to ISO 8601 (YYYY-MM-DD).
"""
import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

ARABIC_INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

ARABIC_MONTH_MAP = {
    "يناير": "01", "جانفي": "01", "كانون الثاني": "01",
    "فبراير": "02", "فيفري": "02", "شباط": "02",
    "مارس": "03", "آذار": "03",
    "أبريل": "04", "ابريل": "04", "نيسان": "04",
    "مايو": "05", "أيار": "05",
    "يونيو": "06", "حزيران": "06",
    "يوليو": "07", "تموز": "07",
    "أغسطس": "08", "اغسطس": "08", "آب": "08",
    "سبتمبر": "09", "أيلول": "09",
    "أكتوبر": "10", "اكتوبر": "10", "تشرين الأول": "10",
    "نوفمبر": "11", "نونبر": "11", "تشرين الثاني": "11",
    "ديسمبر": "12", "دجنبر": "12", "كانون الأول": "12",
}

ENGLISH_MONTH_MAP = {
    "january": "01", "jan": "01",
    "february": "02", "feb": "02",
    "march": "03", "mar": "03",
    "april": "04", "apr": "04",
    "may": "05",
    "june": "06", "jun": "06",
    "july": "07", "jul": "07",
    "august": "08", "aug": "08",
    "september": "09", "sep": "09", "sept": "09",
    "october": "10", "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def normalize_arabic_numerals(text: str) -> str:
    return text.translate(ARABIC_INDIC_MAP)


def pad(n: str) -> str:
    return n.zfill(2)


def validate_date(year: str, month: str, day: str) -> Optional[str]:
    try:
        y, m, d = int(year), int(month), int(day)
        if 1900 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        pass
    return None


def validate_year_only(year: str) -> Optional[str]:
    try:
        y = int(year)
        if 1900 <= y <= 2100:
            return f"{y:04d}"
    except Exception:
        pass
    return None


def normalize_date(raw: str) -> Tuple[Optional[str], float]:
    """
    Returns (normalized_iso_date, confidence).
    confidence: 1.0 = full date, 0.6 = month+year, 0.3 = year only.
    """
    text = normalize_arabic_numerals(raw.strip())

    # ISO format: 2021-03-12
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", text)
    if m:
        result = validate_date(m.group(1), m.group(2), m.group(3))
        if result:
            return result, 1.0

    # DD/MM/YYYY or MM/DD/YYYY
    m = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$", text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        result = validate_date(y, mo, d)
        if result:
            return result, 0.9

    # YYYY/MM/DD
    m = re.match(r"^(\d{4})[/\-.](\d{2})[/\-.](\d{2})$", text)
    if m:
        result = validate_date(m.group(1), m.group(2), m.group(3))
        if result:
            return result, 1.0

    # Arabic: ١٢ مارس ٢٠٢١ or 12 مارس 2021
    text_lower = text
    for ar_month, month_num in ARABIC_MONTH_MAP.items():
        pattern = rf"(\d{{1,2}})\s*{re.escape(ar_month)}\s*(\d{{4}})"
        m = re.search(pattern, text_lower)
        if m:
            result = validate_date(m.group(2), month_num, m.group(1))
            if result:
                return result, 1.0
        # Year month day order: 2021 مارس 12
        pattern2 = rf"(\d{{4}})\s*{re.escape(ar_month)}\s*(\d{{1,2}})"
        m = re.search(pattern2, text_lower)
        if m:
            result = validate_date(m.group(1), month_num, m.group(2))
            if result:
                return result, 1.0

    # English: "January 15, 2021" or "15 January 2021" or "Jan 15 2021"
    text_lower = text.lower()
    for en_month, month_num in ENGLISH_MONTH_MAP.items():
        pattern = rf"(\d{{1,2}})\s*{re.escape(en_month)},?\s*(\d{{4}})"
        m = re.search(pattern, text_lower)
        if m:
            result = validate_date(m.group(2), month_num, m.group(1))
            if result:
                return result, 1.0
        pattern2 = rf"{re.escape(en_month)}\s*(\d{{1,2}}),?\s*(\d{{4}})"
        m = re.search(pattern2, text_lower)
        if m:
            result = validate_date(m.group(2), month_num, m.group(1))
            if result:
                return result, 1.0

    # Month/Year only: "March 2021" or "مارس 2021"
    for ar_month, month_num in ARABIC_MONTH_MAP.items():
        pattern = rf"{re.escape(ar_month)}\s*(\d{{4}})"
        m = re.search(pattern, text)
        if m:
            result = validate_date(m.group(1), month_num, "01")
            if result:
                return result, 0.6

    for en_month, month_num in ENGLISH_MONTH_MAP.items():
        pattern = rf"{re.escape(en_month)}\s*(\d{{4}})"
        m = re.search(pattern, text_lower)
        if m:
            result = validate_date(m.group(1), month_num, "01")
            if result:
                return result, 0.6

    # Year only
    m = re.match(r"^(\d{4})$", text.strip())
    if m:
        result = validate_year_only(m.group(1))
        if result:
            return result, 0.3

    return None, 0.0


def extract_all_dates(text: str) -> list:
    """Extract all date strings from text with normalization."""
    text_normalized = normalize_arabic_numerals(text)

    raw_patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}",
        r"\d{4}[/\-.]\d{2}[/\-.]\d{2}",
    ]
    for ar_month in ARABIC_MONTH_MAP:
        raw_patterns.append(rf"\d{{1,2}}\s*{re.escape(ar_month)}\s*\d{{4}}")
        raw_patterns.append(rf"\d{{4}}\s*{re.escape(ar_month)}\s*\d{{1,2}}")
        raw_patterns.append(rf"{re.escape(ar_month)}\s*\d{{4}}")

    for en_month in ENGLISH_MONTH_MAP:
        raw_patterns.append(rf"\d{{1,2}}\s*{re.escape(en_month)},?\s*\d{{4}}")
        raw_patterns.append(rf"{re.escape(en_month)}\s*\d{{1,2}},?\s*\d{{4}}")

    raw_patterns.append(r"\b\d{4}\b")

    found = []
    seen = set()
    for pat in raw_patterns:
        for match in re.finditer(pat, text_normalized, re.IGNORECASE):
            raw = match.group(0).strip()
            if raw in seen:
                continue
            seen.add(raw)
            iso, confidence = normalize_date(raw)
            if iso:
                found.append({"raw": raw, "iso": iso, "confidence": confidence})

    found.sort(key=lambda x: -x["confidence"])
    return found
