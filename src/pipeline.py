"""End-to-end analysis: filtered scope -> trends / outliers / correlations -> insights."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

import pandas as pd

from .correlation_analysis import CorrelationResult, compute_correlations
from .insight_generator import Insight, generate_insights
from .outlier_detection import OutlierResult, detect_outliers
from .threshold_rules import detect_threshold_breaches, normalize_rules
from .trend_detection import detect_trends, trend_warnings
from .utils import SeverityConfig

STATS_SCOPES = ("global", "filtered")


@dataclass(frozen=True)
class AnalysisSettings:
    trend_threshold: float = 10.0
    outlier_method: str = "iqr"  # 'iqr' | 'zscore'
    iqr_multiplier: float = 1.5
    z_threshold: float = 3.0
    corr_threshold: float = 0.70
    stats_scope: str = "global"  # population for outlier bounds & correlations
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    threshold_rules: tuple = ()  # ((indicator, min|None, max|None), ...)

    def to_json(self) -> str:
        data = asdict(self)
        data["threshold_rules"] = [list(r) for r in self.threshold_rules]
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "AnalysisSettings":
        data = json.loads(text)
        data["severity"] = SeverityConfig.from_dict(data.get("severity"))
        data["threshold_rules"] = tuple(tuple(r) for r in data.get("threshold_rules", []))
        return cls(**data)


@dataclass
class AnalysisResults:
    settings: AnalysisSettings
    indicators: list
    scope_df: pd.DataFrame
    population_df: pd.DataFrame
    trends: pd.DataFrame
    outliers: OutlierResult
    correlations: CorrelationResult
    breaches: pd.DataFrame
    insights: list
    trend_warnings: list
    scope_notes: list
    population_label: str


def filter_scope(df: pd.DataFrame, districts: Optional[Iterable] = None, month_range: Optional[tuple] = None) -> pd.DataFrame:
    """Rows kept by the district and (inclusive) month-range filters."""
    out = df
    if districts is not None:
        out = out[out["district"].isin(list(districts))]
    if month_range is not None:
        lo, hi = month_range
        out = out[(out["month"] >= lo) & (out["month"] <= hi)]  # 'YYYY-MM' sorts chronologically
    return out


def run_analysis(
    df: pd.DataFrame,
    indicators: Iterable[str],
    settings: Optional[AnalysisSettings] = None,
    districts: Optional[Iterable] = None,
    month_range: Optional[tuple] = None,
) -> AnalysisResults:
    """Run every engine on a consistent scope.

    * Trends and threshold breaches use the filtered rows (district + month filters).
    * Outlier bounds and the correlation matrix use the population chosen by
      ``settings.stats_scope`` ('global' = all validated rows, 'filtered' = filtered rows).
      Outliers are *reported* only for filtered rows.
    """
    settings = settings or AnalysisSettings()
    settings.severity.validate()
    if settings.stats_scope not in STATS_SCOPES:
        raise ValueError(f"stats_scope must be one of {STATS_SCOPES}.")
    indicators = [i for i in indicators if i in df.columns]
    scope_df = filter_scope(df, districts, month_range)
    if settings.stats_scope == "global":
        population_df = df
        label = "all districts and months in the validated dataset"
    else:
        population_df = scope_df
        label = "only the rows selected by the current district/month filters"

    trends = detect_trends(scope_df, indicators, settings.trend_threshold, settings.severity)
    outliers = detect_outliers(
        population_df, indicators, settings.outlier_method, settings.iqr_multiplier, settings.z_threshold,
        settings.severity, report_df=scope_df, population_scope=label,
    )
    correlations = compute_correlations(population_df, indicators, settings.corr_threshold, settings.severity)
    breaches = detect_threshold_breaches(scope_df, normalize_rules(settings.threshold_rules), settings.severity)
    months = sorted(population_df["month"].unique()) if len(population_df) else []
    period_range = f"{months[0]}..{months[-1]}" if months else ""
    insights = generate_insights(trends, outliers.outliers, correlations, breaches, settings.trend_threshold, period_range)

    notes = [
        f"Trends and threshold rules use the {len(scope_df)} filtered row(s).",
        f"Outlier bounds and the correlation matrix are calculated on {label} ({len(population_df)} row(s)).",
    ]
    if settings.stats_scope == "global" and len(scope_df) != len(population_df):
        notes.append("Filters narrow which outliers are displayed but do not change the global bounds or correlations.")
    return AnalysisResults(
        settings, indicators, scope_df, population_df, trends, outliers, correlations, breaches, insights,
        trend_warnings(scope_df, indicators), notes, label,
    )
