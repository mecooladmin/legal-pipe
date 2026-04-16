import gc
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def excel_to_text(file_path: str) -> str:
    try:
        import pandas as pd

        ext = Path(file_path).suffix.lower()
        if ext in (".xlsx", ".xls"):
            xls = pd.ExcelFile(file_path)
            all_text = []
            for sheet_name in xls.sheet_names:
                df = xls.parse(sheet_name, dtype=str, header=None)
                df.fillna("", inplace=True)
                rows = []
                for _, row in df.iterrows():
                    row_str = " | ".join(str(v).strip() for v in row if str(v).strip())
                    if row_str:
                        rows.append(row_str)
                if rows:
                    all_text.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))
                del df
                gc.collect()
            xls.close()
            del xls
            gc.collect()
            return "\n\n".join(all_text)
        elif ext == ".csv":
            df = pd.read_csv(file_path, dtype=str, on_bad_lines="skip")
            df.fillna("", inplace=True)
            rows = []
            for _, row in df.iterrows():
                row_str = " | ".join(str(v).strip() for v in row if str(v).strip())
                if row_str:
                    rows.append(row_str)
            del df
            gc.collect()
            return "\n".join(rows)
    except Exception as e:
        logger.error(f"Excel/CSV conversion failed for {file_path}: {e}")
        return ""


def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Text file read failed for {file_path}: {e}")
        return ""
