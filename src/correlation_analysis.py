"""Pearson correlation matrix, flagged pairs and small-sample warnings."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .utils import EPS, SeverityConfig, format_number, pretty_label, severity_from_ratio

MIN_PAIR_OBS = 3  # below this a Pearson coefficient is not reported (n=2 is always +/-1)
PAIR_COLUMNS = [
    "indicator_a", "indicator_b", "pearson_r", "abs_r", "direction", "threshold", "meets_threshold",
    "n_pairs", "p_value", "strength", "severity", "reliability_warning", "interpretation",
]


@dataclass
class CorrelationResult:
    matrix: pd.DataFrame
    pair_counts: pd.DataFrame
    pairs: pd.DataFrame
    warnings: list = field(default_factory=list)
    n_rows: int = 0
    n_districts: int = 0
    n_periods: int = 0
    threshold: float = 0.70


def strength_label(abs_r: float) -> str:
    if abs_r >= 0.9:
        return "very strong"
    if abs_r >= 0.7:
        return "strong"
    if abs_r >= 0.5:
        return "moderate"
    if abs_r >= 0.3:
        return "weak"
    return "very weak"


def sample_warning(n_obs: int, n_districts: int, n_periods: int, min_obs: int = 30, min_districts: int = 10, min_periods: int = 3) -> Optional[str]:
    """Warning text when the sample is too small for a stable Pearson estimate."""
    reasons = []
    if n_obs < min_obs:
        reasons.append(f"only {n_obs} paired observation(s) (fewer than {min_obs})")
    if n_districts < min_districts:
        reasons.append(f"{n_districts} district(s) (the recommended minimum is {min_districts})")
    if n_periods < min_periods:
        reasons.append(f"{n_periods} reporting period(s) (the recommended minimum is {min_periods})")
    if not reasons:
        return None
    return (
        "Limited sample: " + "; ".join(reasons) + ". With so few points, one district can dominate r, so the coefficient "
        "is fragile and should not be read as reliable evidence of a general relationship. Repeated months for the same "
        "district are also not independent observations."
    )


def compute_correlations(
    df: pd.DataFrame,
    indicators: Iterable[str],
    threshold: float = 0.70,
    severity_cfg: Optional[SeverityConfig] = None,
    min_obs: int = 30,
    min_districts: int = 10,
    min_periods: int = 3,
) -> CorrelationResult:
    """Pearson correlations using pairwise-complete observations.

    Undefined coefficients (constant column or fewer than 3 paired values) stay NaN;
    they are never replaced by 0 and never flagged.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Correlation threshold must be between 0 and 1.")
    cfg = (severity_cfg or SeverityConfig()).validate()
    indicators = [i for i in dict.fromkeys(indicators) if i in df.columns]
    n_districts = int(df["district"].nunique()) if "district" in df and len(df) else 0
    n_periods = int(df["month"].nunique()) if "month" in df and len(df) else 0
    result = CorrelationResult(
        matrix=pd.DataFrame(index=indicators, columns=indicators, dtype=float),
        pair_counts=pd.DataFrame(0, index=indicators, columns=indicators, dtype=int),
        pairs=pd.DataFrame(columns=PAIR_COLUMNS), n_rows=int(len(df)), n_districts=n_districts,
        n_periods=n_periods, threshold=threshold,
    )
    if len(indicators) < 2:
        result.warnings.append("At least two numeric indicators are needed to compute correlations.")
        return result
    if df.empty:
        result.warnings.append("No rows in the analytical population; correlations are undefined.")
        return result

    data = df[indicators].apply(pd.to_numeric, errors="coerce")
    present = data.notna().astype(int)
    counts = present.T.dot(present)
    matrix = data.corr(method="pearson")
    off_diag = ~np.eye(len(indicators), dtype=bool)
    matrix = matrix.where(~(off_diag & (counts.to_numpy() < MIN_PAIR_OBS)))
    result.matrix, result.pair_counts = matrix, counts

    constant = [i for i in indicators if data[i].dropna().nunique() <= 1]
    if constant:
        result.warnings.append("Constant or empty indicator(s) have undefined correlations (shown as n/a, not 0): " + ", ".join(constant) + ".")
    thin = [(a, b) for a in indicators for b in indicators if a < b and counts.loc[a, b] < MIN_PAIR_OBS]
    if thin:
        result.warnings.append(f"{len(thin)} indicator pair(s) have fewer than {MIN_PAIR_OBS} paired observations; their coefficients are not reported.")
    general = sample_warning(int(len(df)), n_districts, n_periods, min_obs, min_districts, min_periods)
    if general:
        result.warnings.insert(0, general)

    rows = []
    for i, a in enumerate(indicators):
        for b in indicators[i + 1:]:
            r = matrix.loc[a, b]
            if not np.isfinite(r) or abs(r) < threshold - EPS:
                continue
            n = int(counts.loc[a, b])
            pair = data[[a, b]].dropna()
            try:
                p_value = float(stats.pearsonr(pair[a], pair[b]).pvalue)
            except Exception:
                p_value = np.nan
            direction = "positive" if r > 0 else "negative"
            strength = strength_label(abs(r))
            warn = sample_warning(n, n_districts, n_periods, min_obs, min_districts, min_periods)
            interp = (
                f"{pretty_label(a)} and {pretty_label(b)} show a {strength} {direction} Pearson correlation "
                f"(r = {r:.2f}, n = {n}) in the analysed dataset, meeting the configured |r| >= {format_number(threshold)} threshold."
            )
            interp += " The estimate is based on a limited sample. " if warn else " "
            interp += "Correlation does not establish causation."
            rows.append(
                {
                    "indicator_a": a, "indicator_b": b, "pearson_r": float(r), "abs_r": float(abs(r)),
                    "direction": direction, "threshold": threshold, "meets_threshold": True, "n_pairs": n,
                    "p_value": p_value, "strength": strength,
                    "severity": severity_from_ratio(abs(r), cfg.corr_medium, cfg.corr_high),
                    "reliability_warning": warn, "interpretation": interp,
                }
            )
    result.pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS).sort_values("abs_r", ascending=False).reset_index(drop=True)
    return result
