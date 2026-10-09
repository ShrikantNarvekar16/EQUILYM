"""Plotly figure builders (presentation only; they never compute analytics)."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .utils import SEVERITY_LEVELS, month_to_ord, ord_to_month, pretty_label

SEVERITY_COLORS = {"Low": "#2e9e6b", "Medium": "#e0a526", "High": "#d64545"}
_LAYOUT = dict(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40), font=dict(size=13))


def severity_bar(counts: dict, title: str = "Insights by severity") -> go.Figure:
    values = [int(counts.get(s, 0)) for s in SEVERITY_LEVELS]
    fig = go.Figure(go.Bar(x=list(SEVERITY_LEVELS), y=values, text=values, textposition="outside",
                           marker_color=[SEVERITY_COLORS[s] for s in SEVERITY_LEVELS]))
    fig.update_layout(title=title, yaxis_title="Number of insights", xaxis_title="Severity", height=320, **_LAYOUT)
    fig.update_yaxes(range=[0, max(values + [1]) * 1.25], dtick=1 if max(values + [0]) <= 8 else None)
    return fig


def correlation_heatmap(matrix: pd.DataFrame, title: str = "Pearson correlation matrix") -> go.Figure:
    labels = [pretty_label(c) for c in matrix.columns]
    z = matrix.to_numpy(dtype=float)
    text = [["n/a" if np.isnan(v) else f"{v:.2f}" for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=labels, y=[pretty_label(i) for i in matrix.index], zmin=-1, zmax=1, zmid=0, colorscale="RdBu",
            text=text, texttemplate="%{text}", hoverongaps=False,
            hovertemplate="%{y} vs %{x}<br>r = %{text}<extra></extra>", colorbar=dict(title="r"),
        )
    )
    fig.update_yaxes(autorange="reversed")
    # NaN cells are transparent, so a grey plot background makes undefined values visibly different from 0.
    fig.update_layout(title=title, height=460, **{**_LAYOUT, "template": "plotly_white"}, plot_bgcolor="#cfcfcf")
    return fig


def district_line(df: pd.DataFrame, indicator: str, districts: Iterable[str]) -> go.Figure:
    """One line per district over a continuous month axis; missing months stay as gaps."""
    fig = go.Figure()
    sub = df[df["district"].isin(list(districts))]
    if sub.empty:
        fig.update_layout(title="No data for the current selection", height=380, **_LAYOUT)
        return fig
    ords = sub["month"].map(month_to_ord)
    months = [ord_to_month(o) for o in range(int(ords.min()), int(ords.max()) + 1)]
    for district, grp in sub.groupby("district", sort=True):
        series = grp.set_index("month")[indicator].reindex(months)
        fig.add_trace(go.Scatter(x=months, y=series.where(series.notna(), None).tolist(), mode="lines+markers",
                                 name=str(district), connectgaps=False,
                                 hovertemplate="%{x}<br>%{y}<extra>" + str(district) + "</extra>"))
    fig.update_layout(title=f"{pretty_label(indicator)} by month", xaxis_title="Month", yaxis_title=pretty_label(indicator),
                      xaxis=dict(type="category", categoryorder="array", categoryarray=months), height=420,
                      legend_title_text="District", **_LAYOUT)
    return fig


def outlier_scatter(df: pd.DataFrame, indicator: str, outliers: pd.DataFrame, stats_row: Optional[pd.Series]) -> go.Figure:
    fig = go.Figure()
    sub = df[["district", "month", indicator]].dropna(subset=[indicator])
    flagged = outliers[outliers["indicator"] == indicator] if len(outliers) else outliers
    keys = set(zip(flagged["district"], flagged["month"])) if len(flagged) else set()
    is_out = np.array([(d, m) in keys for d, m in zip(sub["district"], sub["month"])], dtype=bool)
    for name, mask, color in (("Within bounds", ~is_out, "#4c78a8"), ("Outlier (review)", is_out, "#d64545")):
        part = sub[mask]
        fig.add_trace(go.Scatter(x=part["month"], y=part[indicator], mode="markers", name=name, text=part["district"],
                                 marker=dict(color=color, size=10), hovertemplate="%{text}<br>%{x}: %{y}<extra></extra>"))
    if stats_row is not None:
        for key, label in (("lower_bound", "lower bound"), ("upper_bound", "upper bound")):
            if pd.notna(stats_row[key]):
                fig.add_hline(y=float(stats_row[key]), line_dash="dash", line_color="#888", annotation_text=label)
    fig.update_layout(title=f"{pretty_label(indicator)}: observations vs outlier bounds", xaxis_title="Month",
                      yaxis_title=pretty_label(indicator), height=380, **_LAYOUT)
    return fig


def trend_bar(flagged: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    data = flagged.dropna(subset=["pct_change"]).copy()
    if data.empty:
        fig.update_layout(title="No significant percentage changes", height=300, **_LAYOUT)
        return fig
    data["label"] = data["district"] + " - " + data["indicator"].map(pretty_label) + " (" + data["current_period"] + ")"
    data = data.sort_values("pct_change")
    fig.add_trace(go.Bar(x=data["pct_change"], y=data["label"], orientation="h",
                         marker_color=[SEVERITY_COLORS.get(s, "#888") for s in data["severity"]]))
    fig.update_layout(title="Significant changes vs previous period (colour = severity)", xaxis_title="% change",
                      height=max(300, 28 * len(data) + 120), **_LAYOUT)
    return fig
