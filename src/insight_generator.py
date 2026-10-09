"""Turn computed trend / outlier / correlation / breach results into insight records.

All numbers come from the upstream result tables. Text is template-based.
Severity is delegated to the (configurable) rules recorded in SeverityConfig and the
basis for each severity is stored in ``supporting_data['severity_basis']``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from .correlation_analysis import CorrelationResult
from .trend_detection import flagged_trends
from .utils import (
    INSIGHT_TYPES, SEVERITY_LEVELS, SEVERITY_RANK, SeverityConfig, format_number, to_python,
)

NO_FINDINGS_MESSAGE = "No findings met the current thresholds."
CAUSATION_NOTE = "Correlation does not establish causation."


@dataclass
class Insight:
    insight_id: str
    type: str
    indicator: str
    entity: str
    period: str
    metric: Optional[float]
    value: Optional[float]
    prev_value: Optional[float]
    change: Optional[float]
    change_pct: Optional[float]
    severity: str
    explanation: str
    supporting_data: dict = field(default_factory=dict)
    metric_name: str = ""

    def to_dict(self) -> dict:
        return {
            "insight_id": self.insight_id, "type": self.type, "indicator": self.indicator, "entity": self.entity,
            "period": self.period, "metric": self.metric, "metric_name": self.metric_name, "value": self.value,
            "prev_value": self.prev_value, "change": self.change, "change_pct": self.change_pct,
            "severity": self.severity, "explanation": self.explanation, "supporting_data": self.supporting_data,
        }


def _clean(d: dict) -> dict:
    return {k: to_python(v) for k, v in d.items()}


def _trend_insights(trends: pd.DataFrame, threshold: float) -> list:
    out = []
    for r in flagged_trends(trends).itertuples(index=False):
        pct = None if np.isnan(r.pct_change) else float(r.pct_change)
        basis = (
            f"|change| = {abs(pct):.1f}% = {abs(pct) / threshold:.2f}x the {format_number(threshold)}% threshold"
            if pct is not None else "previous value is zero; fixed zero-baseline review severity"
        )
        out.append(
            Insight(
                "", "trend", r.indicator, r.district, r.current_period,
                pct if pct is not None else float(r.change), float(r.current_value), float(r.prev_value),
                float(r.change), pct, r.severity, r.explanation,
                _clean({
                    "previous_period": r.prev_period, "current_period": r.current_period,
                    "threshold_pct": threshold, "direction": r.direction, "spans_gap": bool(r.spans_gap),
                    "missing_periods": r.missing_periods, "zero_baseline_review": bool(r.zero_baseline_review),
                    "severity_basis": basis,
                }),
                "pct_change" if pct is not None else "absolute_change",
            )
        )
    return out


def _outlier_insights(outliers: pd.DataFrame) -> list:
    out = []
    for r in outliers.itertuples(index=False):
        if r.method == "iqr":
            basis = f"{r.distance_beyond_bound:.2f} beyond the {'lower' if r.direction == 'low' else 'upper'} IQR bound"
        else:
            basis = f"|z| = {abs(r.z_score):.2f}"
        out.append(
            Insight(
                "", "outlier", r.indicator, r.district, r.month, float(r.value), float(r.value), None, None, None,
                r.severity, r.explanation,
                _clean({
                    "method": r.method, "lower_bound": r.lower_bound, "upper_bound": r.upper_bound, "z_score": r.z_score,
                    "population_mean": r.population_mean, "population_n": r.population_n,
                    "population_scope": r.population_scope, "direction": r.direction,
                    "distance_beyond_bound": r.distance_beyond_bound, "severity_basis": basis,
                }),
                "observed_value",
            )
        )
    return out


def _correlation_insights(corr: Optional[CorrelationResult], period_range: str) -> list:
    out = []
    if corr is None or corr.pairs.empty:
        return out
    for r in corr.pairs.itertuples(index=False):
        out.append(
            Insight(
                "", "correlation", f"{r.indicator_a}:{r.indicator_b}", f"{corr.n_districts} districts (pooled)",
                period_range, float(r.pearson_r), float(r.pearson_r), None, None, None, r.severity, r.interpretation,
                _clean({
                    "coefficient": r.pearson_r, "abs_coefficient": r.abs_r, "direction": r.direction,
                    "threshold": r.threshold, "n_pairs": r.n_pairs, "p_value": r.p_value,
                    "n_districts": corr.n_districts, "n_periods": corr.n_periods,
                    "reliability_warning": r.reliability_warning, "causation_note": CAUSATION_NOTE,
                    "severity_basis": f"|r| = {r.abs_r:.2f}",
                }),
                "pearson_r",
            )
        )
    return out


def _breach_insights(breaches: Optional[pd.DataFrame]) -> list:
    out = []
    if breaches is None or breaches.empty:
        return out
    for r in breaches.itertuples(index=False):
        out.append(
            Insight(
                "", "threshold_breach", r.indicator, r.district, r.month, float(r.value), float(r.value), None, None, None,
                r.severity, r.explanation,
                _clean({
                    "limit": r.limit, "limit_type": r.limit_type, "distance": r.distance,
                    "relative_pct": r.relative_pct, "severity_basis": f"{r.relative_pct:.1f}% beyond the limit"
                    if not np.isnan(r.relative_pct) else "limit is zero; relative size undefined",
                }),
                "observed_value",
            )
        )
    return out


def generate_insights(
    trends: Optional[pd.DataFrame] = None,
    outliers: Optional[pd.DataFrame] = None,
    correlations: Optional[CorrelationResult] = None,
    breaches: Optional[pd.DataFrame] = None,
    trend_threshold: float = 10.0,
    period_range: str = "",
) -> list:
    """Build insight records; IDs are assigned deterministically (INS-0001...).

    Order: type (trend, outlier, threshold_breach, correlation), severity (High first),
    then entity / indicator / period - the same inputs always give the same IDs.
    """
    items: list = []
    if trends is not None and len(trends):
        items += _trend_insights(trends, trend_threshold)
    if outliers is not None and len(outliers):
        items += _outlier_insights(outliers)
    items += _breach_insights(breaches)
    items += _correlation_insights(correlations, period_range)
    type_order = {"trend": 0, "outlier": 1, "threshold_breach": 2, "correlation": 3}
    items.sort(key=lambda i: (type_order[i.type], -SEVERITY_RANK[i.severity], str(i.entity), str(i.indicator), str(i.period)))
    for n, item in enumerate(items, start=1):
        item.insight_id = f"INS-{n:04d}"
    return items


def summarize_insights(insights: list) -> dict:
    """Counts by severity (zero-filled) and by type (zero-filled)."""
    return {
        "severity": {s: sum(1 for i in insights if i.severity == s) for s in SEVERITY_LEVELS},
        "type": {t: sum(1 for i in insights if i.type == t) for t in INSIGHT_TYPES},
        "total": len(insights),
    }


def filter_insights(insights: list, types=None, severities=None, entities=None, indicators=None, periods=None) -> list:
    """Display-level filtering; IDs are never renumbered. Indicator pairs match if either indicator is selected."""
    def keep(i: Insight) -> bool:
        if types is not None and i.type not in types:
            return False
        if severities is not None and i.severity not in severities:
            return False
        if entities is not None and i.entity not in entities:
            return False
        if indicators is not None and not any(p in indicators for p in i.indicator.split(":")):
            return False
        if periods is not None and i.period not in periods:
            return False
        return True
    return [i for i in insights if keep(i)]
