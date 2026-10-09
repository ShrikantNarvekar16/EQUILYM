"""CSV loading and dataset inspection helpers (no analytics here)."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("month", "district")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # untrusted input: refuse absurdly large files
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_healthcare_data.csv"


@dataclass
class LoadResult:
    raw: Optional[pd.DataFrame]
    header_duplicates: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.raw is not None and not self.errors


def normalize_column_name(name: Any) -> str:
    """Trim, lower-case and replace runs of spaces/hyphens with '_'."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _clean_cell(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return np.nan if value == "" else value
    return value


def read_csv_content(content: Any) -> LoadResult:
    """Parse CSV bytes / text / file-like into an all-string DataFrame.

    Everything is read as text on purpose so validation can report exactly which
    cells are not numeric instead of letting pandas guess silently. The file is
    only parsed as data - nothing in it is ever evaluated or executed.
    """
    errors: list = []
    notes: list = []
    try:
        if hasattr(content, "read"):
            content = content.read()
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, (bytes, bytearray)):
            return LoadResult(None, errors=["Unsupported input: expected CSV bytes, text or a file."])
        content = bytes(content)
    except Exception as exc:  # pragma: no cover - defensive
        return LoadResult(None, errors=[f"Could not read the file: {exc}"])

    if len(content) > MAX_UPLOAD_BYTES:
        return LoadResult(None, errors=[f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."])
    if not content.strip():
        return LoadResult(None, errors=["The file is empty."])
    if b"\x00" in content:
        return LoadResult(None, errors=["The file contains binary data and does not look like a text CSV."])

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("cp1252", errors="replace")
        notes.append("File is not valid UTF-8; decoded as Windows-1252 (some characters may be replaced).")

    try:
        header = next(csv.reader(io.StringIO(text)), None)
    except csv.Error as exc:
        return LoadResult(None, errors=[f"Malformed CSV: {exc}"])
    if not header or all(not str(h).strip() for h in header):
        return LoadResult(None, errors=["The file has no header row."])

    width = len(header)
    try:
        for lineno, row in enumerate(csv.reader(io.StringIO(text)), start=1):
            if lineno > 1 and len(row) > width and any(str(c).strip() for c in row[width:]):
                return LoadResult(
                    None,
                    errors=[f"Malformed CSV: line {lineno} has {len(row)} fields but the header has {width}. "
                            "Check for unquoted commas or a missing header name."],
                )
    except csv.Error as exc:
        return LoadResult(None, errors=[f"Malformed CSV: {exc}"])

    normalized = [normalize_column_name(h) for h in header]
    header_duplicates = sorted({n for n in normalized if normalized.count(n) > 1})

    try:
        raw = pd.read_csv(io.StringIO(text), dtype=str, skipinitialspace=True, on_bad_lines="error", index_col=False)
    except pd.errors.EmptyDataError:
        return LoadResult(None, errors=["The file is empty or has no parsable columns."])
    except (pd.errors.ParserError, ValueError) as exc:
        return LoadResult(None, errors=[f"Malformed CSV (row/column count mismatch or bad quoting): {exc}"])

    seen: dict = {}
    columns = []
    for col in raw.columns:
        name = normalize_column_name(col)
        if name in seen:
            seen[name] += 1
            name = f"{name}__dup{seen[name]}"
        else:
            seen[name] = 0
        columns.append(name)
    raw.columns = columns
    for col in raw.columns:
        raw[col] = raw[col].map(_clean_cell)
    return LoadResult(raw, header_duplicates=header_duplicates, errors=errors, notes=notes)


def load_sample() -> LoadResult:
    return read_csv_content(SAMPLE_PATH.read_bytes())


def dataframe_info_text(df: pd.DataFrame) -> str:
    """Exactly what ``DataFrame.info()`` prints, captured as text."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def schema_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tabular equivalent of ``DataFrame.info()``."""
    return pd.DataFrame(
        {
            "column": list(df.columns),
            "non_null": [int(df[c].notna().sum()) for c in df.columns],
            "null": [int(df[c].isna().sum()) for c in df.columns],
            "dtype": [str(df[c].dtype) for c in df.columns],
        }
    )


def missing_value_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values for every column (count and share of rows)."""
    n = len(df)
    counts = df.isna().sum()
    return pd.DataFrame(
        {
            "column": list(counts.index),
            "missing": [int(v) for v in counts.values],
            "missing_pct": [round(100.0 * int(v) / n, 2) if n else 0.0 for v in counts.values],
        }
    )


def dataset_overview(clean: pd.DataFrame, indicators: list) -> dict:
    return {
        "rows": int(len(clean)),
        "columns": int(len(clean.columns)),
        "districts": int(clean["district"].nunique()) if "district" in clean else 0,
        "periods": int(clean["month"].nunique()) if "month" in clean else 0,
        "indicators": list(indicators),
    }
