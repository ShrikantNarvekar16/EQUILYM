"""CSV / JSON exports. Every export states its scope in the file name or payload."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .utils import to_python

INSIGHT_COLUMNS = [
    "insight_id", "type", "indicator", "entity", "period", "value", "prev_value", "change", "change_pct",
    "severity", "explanation",
]
CSV_ENCODING = "utf-8-sig"  # BOM so Excel on Windows reads non-ASCII district names correctly
ROUND_DECIMALS = 4
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_cell(value: Any) -> Any:
    """Neutralise spreadsheet formula injection in text taken from untrusted CSV content."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def insights_to_frame(insights: Iterable) -> pd.DataFrame:
    """Insights -> DataFrame with the required column order. N/A fields stay empty (None)."""
    rows = []
    for i in insights:
        d = i.to_dict() if hasattr(i, "to_dict") else dict(i)
        rows.append({c: d.get(c) for c in INSIGHT_COLUMNS})
    frame = pd.DataFrame(rows, columns=INSIGHT_COLUMNS)
    for c in ("value", "prev_value", "change", "change_pct"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float)
    return frame


def frame_to_csv_bytes(frame: pd.DataFrame, index: bool = False) -> bytes:
    safe = frame.copy()
    for col in safe.columns:
        if safe[col].dtype == object or str(safe[col].dtype).startswith(("str", "string")):
            safe[col] = safe[col].map(sanitize_cell)
    numeric = safe.select_dtypes(include=[np.number]).columns
    safe[numeric] = safe[numeric].round(ROUND_DECIMALS)
    if index:
        safe.index = [sanitize_cell(str(i)) for i in safe.index]
        safe.columns = [sanitize_cell(str(c)) for c in safe.columns]
    buf = io.StringIO()
    safe.to_csv(buf, index=index, na_rep="", lineterminator="\n")
    return buf.getvalue().encode(CSV_ENCODING)


def insights_to_csv_bytes(insights: Iterable) -> bytes:
    """Insights CSV: insight_id,type,indicator,entity,period,value,prev_value,change,change_pct,severity,explanation."""
    return frame_to_csv_bytes(insights_to_frame(insights))


def correlation_matrix_to_csv_bytes(matrix: pd.DataFrame) -> bytes:
    """Standard DataFrame.corr() layout: indicator names as row and column labels; undefined = empty."""
    return frame_to_csv_bytes(matrix, index=True)


def _json_default(obj: Any) -> Any:
    value = to_python(obj)
    if value is obj and not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return value


def _scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return to_python(obj)


def insights_to_json(insights: Iterable, scope_label: str = "all findings in the analysed scope", settings: Optional[dict] = None) -> str:
    payload = {
        "scope": scope_label,
        "count": 0,
        "settings": settings or {},
        "insights": [],
    }
    for i in insights:
        payload["insights"].append(_scrub(i.to_dict() if hasattr(i, "to_dict") else dict(i)))
    payload["count"] = len(payload["insights"])
    return json.dumps(_scrub(payload), indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)


def data_quality_report_json(report: Any, overview: dict) -> str:
    payload = {
        "overview": _scrub(overview),
        "can_analyze": bool(report.can_analyze),
        "duplicate_policy": report.duplicate_policy,
        "indicators": list(report.indicators),
        "non_indicator_columns": list(report.categorical_columns),
        "issues": report.to_records(),
        "excluded_row_count": int(len(report.excluded_rows)),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)


def write_all_outputs(results: Any, out_dir: Path, validation_report: Any = None, overview: Optional[dict] = None) -> list:
    """Write the full analysis bundle (all findings in scope) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "insights_all.csv": insights_to_csv_bytes(results.insights),
        "insights_all.json": insights_to_json(results.insights, settings=json.loads(results.settings.to_json())).encode("utf-8"),
        "correlation_matrix.csv": correlation_matrix_to_csv_bytes(results.correlations.matrix),
        "trends_all_comparisons.csv": frame_to_csv_bytes(results.trends),
        "outliers.csv": frame_to_csv_bytes(results.outliers.outliers),
    }
    if validation_report is not None and overview is not None:
        files["data_quality_report.json"] = data_quality_report_json(validation_report, overview).encode("utf-8")
    written = []
    for name, data in files.items():
        path = out_dir / name
        path.write_bytes(data)
        written.append(path)
    return written
