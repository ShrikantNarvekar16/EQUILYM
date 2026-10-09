"""IQR and Z-score outlier detection.

Scope: the bounds / mean / standard deviation are computed on ``population_df``
(by default the whole eligible dataset, pooled across districts and months).
Observations are *reported* for ``report_df`` (defaults to the population).
An outlier is a statistical anomaly to review, not a proven error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .utils import (
    EPS, SeverityConfig, format_number, format_period, format_value, pretty_label, severity_from_ratio,
)

OUTLIER_COLUMNS = [
    "district", "indicator", "month", "value", "lower_bound", "upper_bound", "method", "z_score",
    "direction", "distance_beyond_bound", "population_mean", "population_n", "population_scope",
    "severity", "explanation",
]
STATS_COLUMNS = ["indicator", "n", "mean", "std", "q1", "q3", "iqr", "lower_bound", "upper_bound", "method", "status", "note"]
MIN_N = {"iqr": 4, "zscore": 3}


@dataclass
class OutlierResult:
    outliers: pd.DataFrame
    stats: pd.DataFrame
    notes: list = field(default_factory=list)


def _population_stats(values: pd.Series, method: str, iqr_multiplier: float, z_threshold: float) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(v))
    row = dict(n=n, mean=np.nan, std=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan, lower_bound=np.nan,
               upper_bound=np.nan, status="ok", note="")
    if n:
        row["mean"] = float(v.mean())
        row["std"] = float(v.std(ddof=1)) if n > 1 else np.nan
    if n < MIN_N[method]:
        row["status"] = "insufficient_data"
        row["note"] = f"Needs at least {MIN_N[method]} valid observations; found {n}. No bounds calculated."
        return row
    if method == "iqr":
        q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
        iqr = q3 - q1
        row.update(q1=q1, q3=q3, iqr=iqr, lower_bound=q1 - iqr_multiplier * iqr, upper_bound=q3 + iqr_multiplier * iqr)
        if iqr == 0:
            row["status"] = "zero_iqr"
            row["note"] = "IQR is 0 (at least half of the values are identical); any differing value falls outside the bounds."
    else:
        if not row["std"] > 0:
            row["status"] = "undefined_constant"
            row["note"] = "Standard deviation is 0 (all values identical), so Z-scores are undefined; no outliers flagged."
            return row
        row.update(lower_bound=row["mean"] - z_threshold * row["std"], upper_bound=row["mean"] + z_threshold * row["std"])
        max_z = (n - 1) / np.sqrt(n)
        if max_z < z_threshold:
            row["status"] = "threshold_unreachable"
            row["note"] = (
                f"With n={n}, no value can reach |z| = {z_threshold:g} (maximum possible is {max_z:.2f}); "
                "try a lower threshold or the IQR method."
            )
    return row


def outlier_explanation(r: pd.Series, iqr_multiplier: float, z_threshold: float) -> str:
    ind = r["indicator"]
    label, v = pretty_label(ind), format_value(ind, r["value"])
    base = f"{label} in {r['district']} was {v} in {format_period(r['month'])}"
    where = f"across {int(r['population_n'])} observations ({r['population_scope']})"
    tail = "This is a statistical anomaly that warrants review, not a confirmed error."
    if r["method"] == "iqr":
        side = "below the lower" if r["direction"] == "low" else "above the upper"
        return (
            f"{base}, which falls {side} interquartile-range bound "
            f"(bounds {format_value(ind, r['lower_bound'])} to {format_value(ind, r['upper_bound'])}, "
            f"multiplier {format_number(iqr_multiplier)}) calculated {where}. {tail}"
        )
    side = "below" if r["direction"] == "low" else "above"
    return (
        f"{base}, {abs(r['z_score']):.1f} standard deviations {side} the mean of {format_value(ind, r['population_mean'])} "
        f"calculated {where}, meeting the configured |z| >= {format_number(z_threshold)} threshold. {tail}"
    )


def detect_outliers(
    population_df: pd.DataFrame,
    indicators: Iterable[str],
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    severity_cfg: Optional[SeverityConfig] = None,
    report_df: Optional[pd.DataFrame] = None,
    population_scope: str = "all eligible observations",
) -> OutlierResult:
    method = method.lower().replace("-", "").replace("_", "")
    if method not in ("iqr", "zscore"):
        raise ValueError("method must be 'iqr' or 'zscore'.")
    if iqr_multiplier <= 0 or z_threshold <= 0:
        raise ValueError("IQR multiplier and Z-score threshold must be greater than 0.")
    cfg = (severity_cfg or SeverityConfig()).validate()
    report_df = population_df if report_df is None else report_df
    indicators = [i for i in indicators if i in population_df.columns]

    stats_rows, hits, notes = [], [], []
    for ind in indicators:
        st = _population_stats(population_df[ind], method, iqr_multiplier, z_threshold)
        stats_rows.append({"indicator": ind, "method": method, **st})
        if st["note"]:
            notes.append(f"{pretty_label(ind)}: {st['note']}")
        if st["status"] in ("insufficient_data", "undefined_constant") or report_df.empty or ind not in report_df.columns:
            continue
        sub = report_df[["district", "month", ind]].dropna(subset=[ind])
        if sub.empty:
            continue
        vals = sub[ind].astype(float)
        z = (vals - st["mean"]) / st["std"] if st["std"] > 0 else pd.Series(np.nan, index=sub.index)
        if method == "iqr":
            low, high = vals < st["lower_bound"], vals > st["upper_bound"]
        else:
            extreme = z.abs() >= z_threshold - EPS
            low, high = extreme & (z < 0), extreme & (z > 0)
        flagged = low | high
        for idx in sub.index[flagged]:
            value = float(vals[idx])
            is_low = bool(low[idx])
            dist = (st["lower_bound"] - value) if is_low else (value - st["upper_bound"])
            if method == "iqr":
                scale = st["iqr"] if st["iqr"] > 0 else (st["std"] if st["std"] > 0 else np.nan)
                severity = "Low" if np.isnan(scale) else severity_from_ratio(dist / scale, cfg.outlier_iqr_medium, cfg.outlier_iqr_high)
            else:
                severity = severity_from_ratio(abs(float(z[idx])) / z_threshold, cfg.outlier_z_medium_mult, cfg.outlier_z_high_mult)
            hits.append(
                {
                    "district": sub.at[idx, "district"], "indicator": ind, "month": sub.at[idx, "month"], "value": value,
                    "lower_bound": st["lower_bound"], "upper_bound": st["upper_bound"], "method": method,
                    "z_score": float(z[idx]) if not np.isnan(z[idx]) else np.nan,
                    "direction": "low" if is_low else "high", "distance_beyond_bound": float(dist),
                    "population_mean": st["mean"], "population_n": st["n"], "population_scope": population_scope,
                    "severity": severity,
                }
            )
    outliers = pd.DataFrame(hits, columns=[c for c in OUTLIER_COLUMNS if c != "explanation"])
    outliers["explanation"] = [outlier_explanation(r, iqr_multiplier, z_threshold) for _, r in outliers.iterrows()]
    outliers = outliers.sort_values(["indicator", "district", "month"]).reset_index(drop=True)[OUTLIER_COLUMNS]
    stats = pd.DataFrame(stats_rows, columns=STATS_COLUMNS)
    return OutlierResult(outliers=outliers, stats=stats, notes=notes)
