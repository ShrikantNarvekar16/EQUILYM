"""Optional absolute threshold rules (e.g. 'immunization must be >= 85').

A rule is a (min, max) limit per indicator; either side may be None. These are
user-configured rules, separate from the trend-change threshold, so a
``threshold_breach`` insight is never just a re-labelled trend.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .utils import SeverityConfig, format_number, format_period, format_value, is_percent_indicator, pretty_label

BREACH_COLUMNS = [
    "district", "indicator", "month", "value", "limit", "limit_type", "distance", "relative_pct",
    "severity", "explanation",
]


def normalize_rules(rules: Optional[Iterable]) -> dict:
    """Accept {indicator: (min, max)} or iterable of (indicator, min, max)."""
    out: dict = {}
    if not rules:
        return out
    items = rules.items() if isinstance(rules, dict) else ((r[0], (r[1], r[2])) for r in rules)
    for ind, (lo, hi) in items:
        lo = None if lo is None or (isinstance(lo, float) and np.isnan(lo)) else float(lo)
        hi = None if hi is None or (isinstance(hi, float) and np.isnan(hi)) else float(hi)
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"Rule for '{ind}' has minimum {lo} greater than maximum {hi}.")
        if lo is not None or hi is not None:
            out[ind] = (lo, hi)
    return out


def detect_threshold_breaches(df: pd.DataFrame, rules: Optional[Iterable], severity_cfg: Optional[SeverityConfig] = None) -> pd.DataFrame:
    cfg = (severity_cfg or SeverityConfig()).validate()
    rows = []
    for ind, (lo, hi) in normalize_rules(rules).items():
        if ind not in df.columns or df.empty:
            continue
        for _, r in df[["district", "month", ind]].dropna(subset=[ind]).iterrows():
            value = float(r[ind])
            for limit, kind in ((lo, "below minimum"), (hi, "above maximum")):
                if limit is None:
                    continue
                breached = value < limit if kind == "below minimum" else value > limit
                if not breached:
                    continue
                distance = abs(value - limit)
                rel = distance / abs(limit) * 100.0 if limit != 0 else np.nan
                if np.isnan(rel):
                    severity = "Low"  # relative size undefined for a zero limit
                elif rel >= cfg.breach_high_pct:
                    severity = "High"
                elif rel >= cfg.breach_medium_pct:
                    severity = "Medium"
                else:
                    severity = "Low"
                rows.append(
                    {
                        "district": r["district"], "indicator": ind, "month": r["month"], "value": value, "limit": limit,
                        "limit_type": kind, "distance": distance, "relative_pct": rel, "severity": severity,
                        "explanation": (
                            f"{pretty_label(ind)} in {r['district']} was {format_value(ind, value)} in {format_period(r['month'])}, "
                            f"{'below the configured minimum' if kind == 'below minimum' else 'above the configured maximum'} "
                            f"of {format_value(ind, limit)} by {format_number(distance)}{" percentage points" if is_percent_indicator(ind) else ""}."
                        ),
                    }
                )
    return pd.DataFrame(rows, columns=BREACH_COLUMNS).sort_values(["indicator", "district", "month"]).reset_index(drop=True) if rows else pd.DataFrame(columns=BREACH_COLUMNS)
