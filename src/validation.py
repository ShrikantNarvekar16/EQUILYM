"""Schema validation and normalisation of the uploaded dataset.

Principles: never silently drop or alter data. Every excluded row, coerced
cell, resolved duplicate and out-of-range value is reported in the returned
``ValidationReport``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from .data_loader import REQUIRED_COLUMNS
from .utils import find_missing_periods, is_percent_indicator, month_to_ord

MONTH_FORMATS = (
    "%Y-%m", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m", "%Y/%m/%d",
    "%b %Y", "%B %Y", "%b-%Y", "%B-%Y", "%m/%Y", "%Y%m",
)
DUPLICATE_POLICIES = ("error", "keep_first", "keep_last", "mean")
INTERNAL_COLUMNS = ("_source_row",)


@dataclass
class Issue:
    level: str  # 'error' | 'warning' | 'info'
    code: str
    message: str
    rows: list = field(default_factory=list)  # 1-based CSV line numbers (header = line 1)


@dataclass
class ValidationReport:
    issues: list = field(default_factory=list)
    indicators: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    excluded_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicate_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicate_policy: str = "error"
    can_analyze: bool = False

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self) -> list:
        return [i for i in self.issues if i.level == "info"]

    def to_records(self) -> list:
        return [
            {"level": i.level, "code": i.code, "message": i.message, "rows": ", ".join(map(str, i.rows[:50]))}
            for i in self.issues
        ]


def parse_month_value(value: Any) -> tuple:
    """Return ('YYYY-MM', format) or (None, None)."""
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    for fmt in MONTH_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if 1900 <= parsed.year <= 2200:
            return f"{parsed.year:04d}-{parsed.month:02d}", fmt
    return None, None


_THOUSANDS = __import__("re").compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def _to_float(value: Any) -> float:
    if value is None or value is pd.NA:
        return np.nan
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else np.nan
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if _THOUSANDS.match(text):
        text = text.replace(",", "")
    if "_" in text or not text:
        return np.nan
    try:
        number = float(text)
    except ValueError:
        return np.nan
    return number if math.isfinite(number) else np.nan


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Strings -> float. Accepts '85', '85%', '1,200'; anything else -> NaN."""
    return series.map(_to_float).astype("float64")


def _fail(report: ValidationReport, code: str, message: str) -> tuple:
    report.issues.append(Issue("error", code, message))
    report.can_analyze = False
    return None, report


def validate_and_normalize(
    raw: Optional[pd.DataFrame],
    header_duplicates: Iterable = (),
    duplicate_policy: str = "error",
    exclude_out_of_range: bool = False,
    signed_indicators: Iterable = (),
    min_numeric_fraction: float = 0.8,
) -> tuple:
    """Validate ``raw`` (all-string frame) and return ``(clean_df | None, report)``.

    clean_df columns: month ('YYYY-MM'), district, numeric indicators, any other
    (categorical) columns, and ``_source_row`` (CSV line number of the record).
    """
    report = ValidationReport(duplicate_policy=duplicate_policy)
    add = lambda level, code, msg, rows=None: report.issues.append(Issue(level, code, msg, list(rows or [])))

    if duplicate_policy not in DUPLICATE_POLICIES:
        return _fail(report, "bad_policy", f"Unknown duplicate policy '{duplicate_policy}'.")
    if raw is None or raw.shape[1] == 0:
        return _fail(report, "empty_file", "The dataset is empty.")
    header_duplicates = list(header_duplicates)
    if header_duplicates:
        return _fail(
            report, "duplicate_columns",
            "Duplicate column names after normalisation: " + ", ".join(header_duplicates)
            + ". Rename or remove the repeated columns and upload again.",
        )
    missing_required = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_required:
        return _fail(
            report, "missing_columns",
            f"Required column(s) missing: {', '.join(missing_required)}. Found columns: {', '.join(map(str, raw.columns))}.",
        )
    if len(raw) == 0:
        return _fail(report, "no_rows", "The file has a header but no data rows.")

    df = raw.copy()
    df["_source_row"] = np.arange(len(df)) + 2

    # --- identifiers ------------------------------------------------------- #
    df["district"] = df["district"].map(lambda v: str(v).strip() if isinstance(v, str) or pd.notna(v) else np.nan)
    parsed = df["month"].map(parse_month_value)
    df["_month_norm"] = parsed.map(lambda t: t[0])
    formats_used = {t[1] for t in parsed if t[1] is not None}

    reason = pd.Series("", index=df.index, dtype=object)
    reason[df["district"].isna()] = "missing district"
    bad_month = df["_month_norm"].isna()
    reason[bad_month & (reason == "")] = "invalid or missing month"
    reason[bad_month & (reason == "missing district")] = "missing district; invalid or missing month"
    excluded_mask = reason != ""
    report.excluded_rows = df.loc[excluded_mask].drop(columns=["_month_norm"]).assign(
        exclusion_reason=reason[excluded_mask]
    )
    if excluded_mask.any():
        n_district = int(df["district"].isna().sum())
        n_month = int(bad_month.sum())
        add(
            "warning", "excluded_rows",
            f"{int(excluded_mask.sum())} row(s) cannot be analysed and were excluded (missing district: {n_district}; "
            f"invalid/missing month: {n_month}). They are listed in the excluded-rows table, not deleted from your file.",
            df.loc[excluded_mask, "_source_row"].tolist(),
        )
    if len(formats_used) > 1:
        add("info", "mixed_month_formats", f"Month values use {len(formats_used)} different formats; all were normalised to YYYY-MM.")

    df = df.loc[~excluded_mask].copy()
    if df.empty:
        return _fail(report, "no_usable_rows", "No rows have both a valid district and a valid month.")
    df["month"] = df["_month_norm"]
    df = df.drop(columns=["_month_norm"])

    # --- indicator detection ------------------------------------------------ #
    candidates = [c for c in df.columns if c not in ("month", "district") and c not in INTERNAL_COLUMNS]
    signed = set(signed_indicators)
    indicators, categorical = [], []
    for col in candidates:
        values = df[col]
        present = values.notna()
        n_present = int(present.sum())
        if n_present == 0:
            add("warning", "empty_column", f"Column '{col}' has no values and was ignored.")
            continue
        numeric = coerce_numeric(values)
        ok = numeric.notna()
        if ok.sum() / n_present >= min_numeric_fraction:
            bad = present & ~ok
            if bad.any():
                add(
                    "warning", "non_numeric_values",
                    f"Column '{col}': {int(bad.sum())} non-numeric value(s) (e.g. {list(values[bad].astype(str).unique()[:3])}) "
                    "were treated as missing. Fix them in the file if they should be numbers.",
                    df.loc[bad, "_source_row"].tolist(),
                )
            df[col] = numeric
            indicators.append(col)
        else:
            categorical.append(col)
            add(
                "info", "categorical_column",
                f"Column '{col}' is mostly non-numeric ({int(ok.sum())} of {n_present} values numeric) and is not used as an indicator.",
            )
    report.categorical_columns = categorical
    if not indicators:
        report.can_analyze = False
        return _fail(report, "no_indicators", "No usable numeric indicator column was found besides 'month' and 'district'.")

    # --- range checks (flag, never clip) ------------------------------------ #
    for col in list(indicators):
        values = df[col]
        if is_percent_indicator(col):
            out = (values < 0) | (values > 100)
            expected = "0-100"
        elif col not in signed:
            out = values < 0
            expected = ">= 0"
        else:
            continue
        if out.fillna(False).any():
            rows = df.loc[out.fillna(False), "_source_row"].tolist()
            action = "set to missing for analysis" if exclude_out_of_range else "kept as supplied (not clipped)"
            add(
                "warning", "out_of_range",
                f"Column '{col}': {len(rows)} value(s) outside the expected range {expected}; {action}. "
                "Check for data-entry errors.",
                rows,
            )
            if exclude_out_of_range:
                df.loc[out.fillna(False), col] = np.nan
    for col in list(indicators):
        if df[col].notna().sum() == 0:
            indicators.remove(col)
            add("warning", "no_valid_values", f"Indicator '{col}' has no valid values after cleaning and was dropped.")
    if not indicators:
        return _fail(report, "no_indicators", "No indicator has any valid value after cleaning.")
    report.indicators = indicators

    # --- duplicates --------------------------------------------------------- #
    dup_mask = df.duplicated(["district", "month"], keep=False)
    if dup_mask.any():
        report.duplicate_rows = df.loc[dup_mask].sort_values(["district", "month", "_source_row"])
        n_groups = int(df.loc[dup_mask].groupby(["district", "month"]).ngroups)
        if duplicate_policy == "error":
            report.issues.append(
                Issue(
                    "error", "duplicate_records",
                    f"{int(dup_mask.sum())} rows share a (district, month) key across {n_groups} group(s). "
                    "Trend analysis would be ambiguous. Fix the file, or choose an explicit policy in the sidebar "
                    "(keep_first, keep_last or mean).",
                    report.duplicate_rows["_source_row"].tolist(),
                )
            )
            report.can_analyze = False
            return None, report
        before = len(df)
        if duplicate_policy in ("keep_first", "keep_last"):
            df = df.drop_duplicates(["district", "month"], keep="first" if duplicate_policy == "keep_first" else "last")
        else:
            agg = {c: "mean" for c in indicators}
            agg.update({c: "first" for c in df.columns if c not in indicators and c not in ("district", "month")})
            agg["_source_row"] = "min"
            df = df.groupby(["district", "month"], as_index=False, sort=False).agg(agg)
        add(
            "info", "duplicates_resolved",
            f"Duplicate policy '{duplicate_policy}' applied: {before - len(df)} row(s) merged/removed across {n_groups} duplicated key(s).",
            report.duplicate_rows["_source_row"].tolist(),
        )

    # --- soft checks --------------------------------------------------------- #
    variants = df.groupby(df["district"].str.casefold())["district"].agg(lambda s: sorted(set(s)))
    for variant_list in variants:
        if len(variant_list) > 1:
            add("warning", "district_name_variants", f"District names differ only by case: {variant_list}. They are treated as different districts.")

    ordered = ["month", "district"] + indicators + categorical + list(INTERNAL_COLUMNS)
    clean = df[[c for c in ordered if c in df.columns]].sort_values(["district", "month"]).reset_index(drop=True)

    coverage = find_missing_periods(clean)
    gappy = coverage[coverage["missing_periods"].map(len) > 0] if not coverage.empty else coverage
    if len(gappy):
        names = ", ".join(f"{r.district} (missing {', '.join(r.missing_periods[:4])}{'...' if len(r.missing_periods) > 4 else ''})" for r in gappy.head(8).itertuples())
        add("warning", "missing_months", f"{len(gappy)} district(s) have gaps in their monthly series: {names}. Trends across a gap are labelled as such.")
    n_months = clean.groupby("district")["month"].nunique()
    if (n_months < 2).any():
        add("info", "single_observation", f"{int((n_months < 2).sum())} district(s) have only one reporting period; no trend can be computed for them.")
    if clean["district"].nunique() < 10 or clean["month"].nunique() < 3:
        add(
            "info", "small_dataset",
            f"Dataset has {clean['district'].nunique()} district(s) and {clean['month'].nunique()} reporting period(s). "
            "Recommended minimums are 3 months per district for trends and 10 districts for stable correlations.",
        )
    report.can_analyze = not report.errors
    return clean, report


def clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Drop internal bookkeeping columns for display."""
    return df.drop(columns=[c for c in INTERNAL_COLUMNS if c in df.columns])
