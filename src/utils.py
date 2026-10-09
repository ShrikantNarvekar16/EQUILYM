"""Shared constants, severity configuration and formatting helpers."""
from __future__ import annotations

import calendar
import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

import numpy as np
import pandas as pd

INSIGHT_TYPES = ("trend", "outlier", "correlation", "threshold_breach")
SEVERITY_LEVELS = ("Low", "Medium", "High")
SEVERITY_RANK = {"Low": 1, "Medium": 2, "High": 3}
EPS = 1e-9  # tolerance for ">=" comparisons on floating point values

# Tokens rendered in upper case when a column name is turned into a label.
ACRONYMS = frozenset(
    {"anc", "pnc", "hiv", "tb", "bcg", "opv", "dpt", "ipd", "opd", "icu", "sti", "ncd", "phc", "chc"}
)
# Column-name fragments that mark an indicator as a percentage (expected 0-100).
PERCENT_HINTS = ("pct", "percent", "%", "coverage", "proportion", "share", "immuniz", "immunis", "delivery")


@dataclass(frozen=True)
class SeverityConfig:
    """Boundaries that turn a computed quantity into Low / Medium / High.

    Trend:       |pct_change| / trend_threshold          -> medium at 1.5x, high at 2.0x
    IQR outlier: distance beyond bound / IQR             -> medium at 1.0,  high at 2.0
    Z outlier:   |z| / z_threshold                       -> medium at 1.25x, high at 1.5x
    Correlation: |r|                                     -> medium at 0.80, high at 0.90
    Breach:      distance beyond limit / |limit| * 100   -> medium at 10 %, high at 25 %
    Zero-baseline trends have no ratio, so they get a fixed review priority.
    """

    trend_medium_mult: float = 1.5
    trend_high_mult: float = 2.0
    outlier_iqr_medium: float = 1.0
    outlier_iqr_high: float = 2.0
    outlier_z_medium_mult: float = 1.25
    outlier_z_high_mult: float = 1.5
    corr_medium: float = 0.80
    corr_high: float = 0.90
    breach_medium_pct: float = 10.0
    breach_high_pct: float = 25.0
    zero_baseline_severity: str = "Medium"

    def validate(self) -> "SeverityConfig":
        pairs = [
            ("trend", self.trend_medium_mult, self.trend_high_mult),
            ("IQR outlier", self.outlier_iqr_medium, self.outlier_iqr_high),
            ("Z-score outlier", self.outlier_z_medium_mult, self.outlier_z_high_mult),
            ("correlation", self.corr_medium, self.corr_high),
            ("threshold breach", self.breach_medium_pct, self.breach_high_pct),
        ]
        for name, med, high in pairs:
            if not (0 < med <= high):
                raise ValueError(f"Severity boundaries for {name} must satisfy 0 < medium <= high (got {med}, {high}).")
        if self.zero_baseline_severity not in SEVERITY_LEVELS:
            raise ValueError("zero_baseline_severity must be Low, Medium or High.")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "SeverityConfig":
        if not data:
            return cls()
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def severity_from_ratio(ratio: float, medium: float, high: float) -> str:
    """Classify ``ratio`` against ``medium``/``high`` boundaries (inclusive lower edges)."""
    if ratio >= high - EPS:
        return "High"
    if ratio >= medium - EPS:
        return "Medium"
    return "Low"


# Months are normalised to 'YYYY-MM' strings throughout the project.
def month_to_ord(month: str) -> int:
    """'YYYY-MM' -> months since 1970-01 (consistent with pandas Period ordinals)."""
    return (int(month[:4]) - 1970) * 12 + int(month[5:7]) - 1


def ord_to_month(value: int) -> str:
    value = int(value)
    return f"{1970 + value // 12:04d}-{value % 12 + 1:02d}"


def months_between(prev_month: str, current_month: str) -> list[str]:
    """Calendar months strictly between two 'YYYY-MM' values."""
    a, b = month_to_ord(prev_month), month_to_ord(current_month)
    return [ord_to_month(o) for o in range(a + 1, b)]


def find_missing_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Per district, which calendar months between its first and last row are absent."""
    cols = ["district", "observed_periods", "first", "last", "missing_periods"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for district, grp in df.groupby("district", sort=True):
        ords = sorted({month_to_ord(m) for m in grp["month"]})
        have = set(ords)
        missing = [ord_to_month(o) for o in range(ords[0], ords[-1] + 1) if o not in have]
        rows.append(
            {
                "district": district,
                "observed_periods": len(ords),
                "first": ord_to_month(ords[0]),
                "last": ord_to_month(ords[-1]),
                "missing_periods": missing,
            }
        )
    return pd.DataFrame(rows, columns=cols)


def is_percent_indicator(name: str) -> bool:
    lowered = str(name).lower()
    return any(h in lowered for h in PERCENT_HINTS)


def pretty_label(name: str) -> str:
    """'anc_coverage' -> 'ANC coverage'; 'high_risk_cases' -> 'High risk cases'."""
    tokens = [t for t in str(name).replace("-", "_").split("_") if t]
    if not tokens:
        return str(name)
    out = []
    for i, tok in enumerate(tokens):
        if tok.lower() in ACRONYMS:
            out.append(tok.upper())
        elif i == 0:
            out.append(tok.capitalize())
        else:
            out.append(tok.lower())
    return " ".join(out)


def format_period(period: Any) -> str:
    """'2026-08' -> 'August 2026'. Non-month text is returned unchanged."""
    text = str(period)
    try:
        return f"{calendar.month_name[int(text[5:7])]} {int(text[:4])}"
    except (ValueError, IndexError):
        return text


def format_number(value: Any, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.{decimals}f}".rstrip("0").rstrip(".")


def format_value(indicator: str, value: Any, decimals: int = 2) -> str:
    text = format_number(value, decimals)
    if text != "n/a" and is_percent_indicator(indicator):
        return text + "%"
    return text


def to_python(value: Any) -> Any:
    """numpy scalars / NaN -> plain Python values (NaN -> None) for JSON & records."""
    if value is None:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if value is pd.NA or value is pd.NaT:
        return None
    return value
