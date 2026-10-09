"""Month-over-month trend detection per (district, indicator)."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .utils import (
    EPS, SeverityConfig, find_missing_periods, format_number, format_period, format_value,
    month_to_ord, months_between, ord_to_month, pretty_label,
)

TREND_COLUMNS = [
    "district", "indicator", "prev_period", "current_period", "prev_value", "current_value",
    "change", "pct_change", "pct_status", "direction", "is_significant", "zero_baseline_review",
    "gap_periods", "missing_periods", "spans_gap", "severity", "explanation",
]


def percent_change(previous: float, current: float) -> float:
    """((current - previous) / |previous|) * 100, or NaN when previous == 0."""
    if previous == 0 or previous is None or np.isnan(previous) or np.isnan(current):
        return float("nan")
    return (current - previous) / abs(previous) * 100.0


def trend_explanation(row: pd.Series, threshold_pct: float) -> str:
    ind, dist = row["indicator"], row["district"]
    label = pretty_label(ind)
    p0, p1 = format_period(row["prev_period"]), format_period(row["current_period"])
    v0, v1 = format_value(ind, row["prev_value"]), format_value(ind, row["current_value"])
    gap = ""
    if int(row["gap_periods"]) > 0:
        gap = f" Note: this comparison spans {int(row['gap_periods'])} missing reporting period(s) ({row['missing_periods']})."
    if row["zero_baseline_review"]:
        return (
            f"{label} in {dist} changed from {v0} in {p0} to {v1} in {p1}. The percentage change is undefined "
            f"because the previous value is zero; flagged as a zero-baseline change requiring review.{gap}"
        )
    if np.isnan(row["pct_change"]):
        return f"{label} in {dist} was {v0} in {p0} and {v1} in {p1}; the percentage change is undefined (zero baseline).{gap}"
    pct = abs(row["pct_change"])
    if row["is_significant"]:
        verb = "increased" if row["change"] > 0 else "decreased"
        return (
            f"{label} in {dist} {verb} by {pct:.1f}% between {p0} and {p1} ({v0} to {v1}), "
            f"exceeding the configured {format_number(threshold_pct)}% significant-change threshold.{gap}"
        )
    direction = "rose" if row["change"] > 0 else "fell" if row["change"] < 0 else "was unchanged"
    if row["change"] == 0:
        return f"{label} in {dist} was unchanged between {p0} and {p1} ({v0}).{gap}"
    return (
        f"{label} in {dist} {direction} by {pct:.1f}% between {p0} and {p1} ({v0} to {v1}), "
        f"below the configured {format_number(threshold_pct)}% threshold.{gap}"
    )


def detect_trends(
    df: pd.DataFrame,
    indicators: Iterable[str],
    threshold_pct: float = 10.0,
    severity_cfg: Optional[SeverityConfig] = None,
) -> pd.DataFrame:
    """Compare consecutive *available* observations for every (district, indicator).

    Returns every comparison (significant or not) with ``is_significant`` set when
    ``abs(pct_change) >= threshold_pct``. A comparison that skips calendar months
    keeps ``gap_periods`` / ``missing_periods`` so it is never presented as
    month-over-month. Previous value 0 gives an undefined pct_change and, when the
    new value is non-zero, ``zero_baseline_review = True``.
    """
    if threshold_pct is None or not threshold_pct > 0:
        raise ValueError("Trend threshold must be greater than 0.")
    cfg = (severity_cfg or SeverityConfig()).validate()
    indicators = [i for i in indicators if i in df.columns]
    empty = pd.DataFrame(columns=TREND_COLUMNS)
    if df.empty or not indicators:
        return empty
    if df.duplicated(["district", "month"]).any():
        raise ValueError("Duplicate (district, month) records found; resolve duplicates before trend detection.")

    long = df.melt(id_vars=["district", "month"], value_vars=indicators, var_name="indicator", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    if long.empty:
        return empty
    long["ord"] = long["month"].map(month_to_ord).astype("int64")
    long = long.sort_values(["district", "indicator", "ord"], kind="mergesort").reset_index(drop=True)
    grp = long.groupby(["district", "indicator"], sort=False)
    long["prev_value"] = grp["value"].shift(1)
    long["prev_ord"] = grp["ord"].shift(1)
    t = long.dropna(subset=["prev_value"]).copy()
    if t.empty:
        return empty

    cur = t["value"].to_numpy(dtype=float)
    prev = t["prev_value"].to_numpy(dtype=float)
    change = cur - prev
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(prev != 0, change / np.abs(prev) * 100.0, np.nan)
    gap = (t["ord"].to_numpy() - t["prev_ord"].to_numpy() - 1).astype(int)
    sig = ~np.isnan(pct) & (np.abs(np.nan_to_num(pct)) >= threshold_pct - EPS)
    zero_base = (prev == 0) & (cur != 0)
    ratio = np.abs(np.nan_to_num(pct)) / threshold_pct
    severity = np.full(len(t), None, dtype=object)
    sev_sig = np.select(
        [ratio >= cfg.trend_high_mult - EPS, ratio >= cfg.trend_medium_mult - EPS], ["High", "Medium"], "Low"
    )
    severity[sig] = sev_sig[sig]
    severity[zero_base] = cfg.zero_baseline_severity

    out = pd.DataFrame(
        {
            "district": t["district"].to_numpy(),
            "indicator": t["indicator"].to_numpy(),
            "prev_period": [ord_to_month(o) for o in t["prev_ord"].astype(int)],
            "current_period": t["month"].to_numpy(),
            "prev_value": prev,
            "current_value": cur,
            "change": change,
            "pct_change": pct,
            "pct_status": np.where(prev != 0, "defined", "undefined_zero_baseline"),
            "direction": np.select([change > 0, change < 0], ["increase", "decrease"], "no change"),
            "is_significant": sig,
            "zero_baseline_review": zero_base,
            "gap_periods": gap,
            "missing_periods": [
                ";".join(months_between(ord_to_month(a), ord_to_month(b))) if g > 0 else ""
                for a, b, g in zip(t["prev_ord"].astype(int), t["ord"], gap)
            ],
            "spans_gap": gap > 0,
            "severity": severity,
        }
    )
    out["explanation"] = [trend_explanation(r, threshold_pct) for _, r in out.iterrows()]
    return out.sort_values(["district", "indicator", "current_period"]).reset_index(drop=True)[TREND_COLUMNS]


def flagged_trends(trends: pd.DataFrame) -> pd.DataFrame:
    """Significant changes plus zero-baseline changes that need review."""
    if trends.empty:
        return trends
    return trends[trends["is_significant"] | trends["zero_baseline_review"]].reset_index(drop=True)


def trend_warnings(df: pd.DataFrame, indicators: Iterable[str]) -> list:
    """Human-readable data-coverage warnings relevant to trend analysis."""
    warnings = []
    if df.empty:
        return warnings
    coverage = find_missing_periods(df)
    for r in coverage.itertuples():
        if r.missing_periods:
            warnings.append(f"{r.district}: missing reporting period(s) {', '.join(r.missing_periods)}; comparisons across the gap are marked.")
        if r.observed_periods < 2:
            warnings.append(f"{r.district}: only one reporting period, so no trend can be computed.")
    for ind in indicators:
        if ind in df.columns:
            n_missing = int(df[ind].isna().sum())
            if n_missing:
                warnings.append(f"{pretty_label(ind)}: {n_missing} missing value(s) were skipped; neighbouring available periods are compared (gap recorded).")
    return warnings
