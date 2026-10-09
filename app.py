"""EQUILYM - Automated Insight Generation Engine (Streamlit dashboard).

Run with:  streamlit run app.py
All analytics live in ``src/``; this file only wires widgets, caching and display.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

from src import charts
from src.data_loader import (
    SAMPLE_PATH, dataframe_info_text, dataset_overview, missing_value_counts, read_csv_content, schema_summary,
)
from src.export_utils import (
    INSIGHT_COLUMNS, correlation_matrix_to_csv_bytes, data_quality_report_json, frame_to_csv_bytes,
    insights_to_csv_bytes, insights_to_frame, insights_to_json,
)
from src.insight_generator import NO_FINDINGS_MESSAGE, filter_insights, summarize_insights
from src.pipeline import AnalysisSettings, run_analysis
from src.trend_detection import flagged_trends
from src.utils import INSIGHT_TYPES, SEVERITY_LEVELS, SeverityConfig, format_value, pretty_label
from src.validation import DUPLICATE_POLICIES, clean_view, validate_and_normalize

st.set_page_config(page_title="EQUILYM - Insight Engine", page_icon="📊", layout="wide")

DEFAULTS = dict(
    trend=10, method="IQR", iqr=1.5, z=3.0, corr=0.70, scope="Full validated dataset (global)",
    sev_trend_med=1.5, sev_trend_high=2.0, sev_iqr_med=1.0, sev_iqr_high=2.0, sev_z_med=1.25, sev_z_high=1.5,
    sev_corr_med=0.80, sev_corr_high=0.90, sev_breach_med=10.0, sev_breach_high=25.0,
)
POLICY_LABELS = {
    "error": "Stop and report duplicates (default)",
    "keep_first": "Keep first record per district-month",
    "keep_last": "Keep last record per district-month",
    "mean": "Average duplicate records",
}


# ----------------------------------------------------------------------------- helpers
def show_df(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_plot(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key)


def reset_filters() -> None:
    for k in list(st.session_state.keys()):
        if str(k).startswith(("flt_", "thr_", "sev_", "rule_")):
            del st.session_state[k]


def inject_css() -> None:
    st.markdown(
        """<style>
        .block-container {padding-top: 1.4rem;}
        div[data-testid="stMetric"] {background:#f6f8fb; border:1px solid #e3e8ef; border-radius:10px; padding:10px 14px;}
        .eq-title {font-size:2rem; font-weight:700; margin-bottom:0;}
        .eq-sub {color:#5b6675; margin-top:0;}
        </style>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_load(content: bytes):
    return read_csv_content(content)


@st.cache_data(show_spinner=False)
def cached_validate(content: bytes, policy: str, exclude_oor: bool, signed: tuple):
    load = read_csv_content(content)
    if load.raw is None:
        return None, None
    return validate_and_normalize(load.raw, load.header_duplicates, policy, exclude_oor, signed)


@st.cache_data(show_spinner="Analysing...")
def cached_analysis(clean: pd.DataFrame, indicators: tuple, settings_json: str, districts: tuple, month_range: tuple):
    return run_analysis(clean, list(indicators), AnalysisSettings.from_json(settings_json), list(districts), month_range)


# ----------------------------------------------------------------------------- sidebar
def sidebar_source():
    st.sidebar.header("1. Data")
    uploaded = st.sidebar.file_uploader("Upload district-level CSV", type=["csv"], key="uploader")
    if st.sidebar.button("Load sample dataset", key="btn_sample"):
        st.session_state["use_sample"] = True
    if uploaded is not None:
        st.session_state["use_sample"] = False
        return uploaded.getvalue(), uploaded.name
    if st.session_state.get("use_sample"):
        return SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
    return None, None


def sidebar_data_handling(content: bytes):
    with st.sidebar.expander("Data handling options"):
        policy = st.selectbox(
            "Duplicate (district, month) records", list(DUPLICATE_POLICIES), format_func=POLICY_LABELS.get, key="flt_policy",
            help="Duplicates make trends ambiguous, so analysis is blocked until you fix the file or choose a documented policy.",
        )
        exclude_oor = st.checkbox("Treat out-of-range values as missing", value=False, key="flt_exclude_oor",
                                  help="Off: values outside 0-100 (percent indicators) or below 0 are flagged but kept as supplied.")
        _, first_pass = cached_validate(content, "error", False, ())
        options = list(first_pass.indicators) if first_pass else []
        signed = st.multiselect("Indicators allowed to be negative", options, key="flt_signed")
    return policy, exclude_oor, tuple(signed)


def sidebar_filters(clean: pd.DataFrame, indicators: list, sig: str):
    st.sidebar.header("2. Filters")
    st.sidebar.button("Clear / reset filters and thresholds", on_click=reset_filters, key="btn_reset")
    all_districts = sorted(clean["district"].unique())
    months = sorted(clean["month"].unique())
    districts = st.sidebar.multiselect("Districts", all_districts, default=all_districts, key=f"flt_districts_{sig}")
    if len(months) > 1:
        month_range = st.sidebar.select_slider("Reporting period range", options=months, value=(months[0], months[-1]),
                                               key=f"flt_months_{sig}")
    else:
        month_range = (months[0], months[0])
        st.sidebar.caption(f"Only one reporting period available: {months[0]}")
    selected = st.sidebar.multiselect("Indicators", indicators, default=indicators, key=f"flt_inds_{sig}")
    return districts, tuple(month_range), selected


def sidebar_thresholds(clean: pd.DataFrame, selected: list):
    st.sidebar.header("3. Analysis thresholds")
    trend = st.sidebar.slider("Trend threshold (|% change|)", 1, 100, DEFAULTS["trend"], 1, key="thr_trend",
                              help="A month-over-month change is significant when its absolute % change is at least this value.")
    method = st.sidebar.radio("Outlier method", ["IQR", "Z-score"], key="thr_method", horizontal=True)
    iqr = st.sidebar.slider("IQR multiplier", 0.5, 5.0, DEFAULTS["iqr"], 0.1, key="thr_iqr", disabled=method != "IQR",
                            help="Bounds are Q1 - k*IQR and Q3 + k*IQR. Larger k flags fewer values.")
    z = st.sidebar.slider("Z-score threshold (|z|)", 1.0, 5.0, DEFAULTS["z"], 0.1, key="thr_z", disabled=method != "Z-score",
                          help="Flag values at least this many standard deviations from the mean. With n values |z| cannot exceed (n-1)/sqrt(n).")
    corr = st.sidebar.slider("Correlation threshold (|r|)", 0.10, 0.99, DEFAULTS["corr"], 0.01, key="thr_corr",
                             help="Indicator pairs with |Pearson r| at or above this value are flagged.")
    scope = st.sidebar.radio(
        "Population for outlier bounds & correlations",
        ["Full validated dataset (global)", "Current filters only"], key="thr_scope",
        help="Global: bounds/correlations use every district and month; filters only choose which outliers are shown. "
             "Current filters only: statistics are recomputed on the filtered rows.",
    )
    with st.sidebar.expander("Severity boundaries"):
        st.caption("Severity is an analytical review priority, not a clinical judgement.")
        tm = st.slider("Trend: Medium from (x threshold)", 1.0, 5.0, DEFAULTS["sev_trend_med"], 0.1, key="sev_trend_med")
        th = st.slider("Trend: High from (x threshold)", 1.0, 5.0, DEFAULTS["sev_trend_high"], 0.1, key="sev_trend_high")
        im = st.slider("IQR outlier: Medium from (IQRs beyond bound)", 0.1, 5.0, DEFAULTS["sev_iqr_med"], 0.1, key="sev_iqr_med")
        ih = st.slider("IQR outlier: High from (IQRs beyond bound)", 0.1, 5.0, DEFAULTS["sev_iqr_high"], 0.1, key="sev_iqr_high")
        zm = st.slider("Z outlier: Medium from (x threshold)", 1.0, 3.0, DEFAULTS["sev_z_med"], 0.05, key="sev_z_med")
        zh = st.slider("Z outlier: High from (x threshold)", 1.0, 3.0, DEFAULTS["sev_z_high"], 0.05, key="sev_z_high")
        cm = st.slider("Correlation: Medium from |r|", 0.1, 1.0, DEFAULTS["sev_corr_med"], 0.01, key="sev_corr_med")
        ch = st.slider("Correlation: High from |r|", 0.1, 1.0, DEFAULTS["sev_corr_high"], 0.01, key="sev_corr_high")
        bm = st.slider("Threshold breach: Medium from (% beyond limit)", 1.0, 100.0, DEFAULTS["sev_breach_med"], 1.0, key="sev_breach_med")
        bh = st.slider("Threshold breach: High from (% beyond limit)", 1.0, 100.0, DEFAULTS["sev_breach_high"], 1.0, key="sev_breach_high")
    pairs = [("trend", tm, th), ("IQR outlier", im, ih), ("Z outlier", zm, zh), ("correlation", cm, ch), ("breach", bm, bh)]
    fixed = []
    for name, med, high in pairs:
        if high < med:
            st.sidebar.warning(f"Severity '{name}': High boundary was below Medium; using Medium as High.")
            high = med
        fixed.append((med, high))
    sev = SeverityConfig(*[v for pair in fixed for v in pair])

    rules = []
    with st.sidebar.expander("Absolute threshold rules (optional)"):
        st.caption("Define limits to generate separate 'threshold_breach' insights. Off by default.")
        for ind in selected:
            median = float(clean[ind].median()) if clean[ind].notna().any() else 0.0
            lo_on = st.checkbox(f"{pretty_label(ind)}: minimum", key=f"rule_lo_on_{ind}")
            lo = st.number_input("Minimum", value=round(median, 2), key=f"rule_lo_{ind}", disabled=not lo_on) if lo_on else None
            hi_on = st.checkbox(f"{pretty_label(ind)}: maximum", key=f"rule_hi_on_{ind}")
            hi = st.number_input("Maximum", value=round(median, 2), key=f"rule_hi_{ind}", disabled=not hi_on) if hi_on else None
            if lo_on or hi_on:
                rules.append((ind, None if lo is None else float(lo), None if hi is None else float(hi)))
    return AnalysisSettings(
        trend_threshold=float(trend), outlier_method="iqr" if method == "IQR" else "zscore", iqr_multiplier=float(iqr),
        z_threshold=float(z), corr_threshold=float(corr), stats_scope="global" if scope.startswith("Full") else "filtered",
        severity=sev, threshold_rules=tuple(rules),
    )


# ----------------------------------------------------------------------------- sections
def render_data_quality(load, clean, report, raw_overview):
    st.subheader("Data quality and validation")
    for issue in report.issues:
        text = issue.message + (f"  \n_CSV lines: {', '.join(map(str, issue.rows[:20]))}{' ...' if len(issue.rows) > 20 else ''}_" if issue.rows else "")
        {"error": st.error, "warning": st.warning, "info": st.info}[issue.level](text)
    if report.can_analyze and not report.issues:
        st.success("No data-quality issues were found.")
    for note in load.notes:
        st.info(note)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Preview - `head()` of the file as loaded**")
        show_df(load.raw.head())
        st.markdown("**Missing values per column**")
        show_df(missing_value_counts(load.raw))
    with c2:
        st.markdown("**Schema - `info()` of the loaded file**")
        st.code(dataframe_info_text(load.raw))
        if clean is not None:
            st.markdown("**Schema after validation and normalisation**")
            show_df(schema_summary(clean_view(clean)))
    if not report.duplicate_rows.empty:
        st.markdown("**Duplicate (district, month) records**")
        show_df(clean_view(report.duplicate_rows))
    if not report.excluded_rows.empty:
        st.markdown("**Rows excluded from analysis (kept here for review)**")
        show_df(report.excluded_rows)


def render_insights(res):
    st.subheader("Automated insights")
    insights = res.insights
    if not insights:
        st.info(NO_FINDINGS_MESSAGE + " Try lowering the thresholds or widening the filters.")
        return []
    c = st.columns(5)
    types = c[0].multiselect("Type", INSIGHT_TYPES, default=list(INSIGHT_TYPES))
    sev = c[1].multiselect("Severity", SEVERITY_LEVELS, default=list(SEVERITY_LEVELS))
    entities = sorted({i.entity for i in insights})
    ent = c[2].multiselect("District / entity", entities, default=entities)
    inds = sorted({p for i in insights for p in i.indicator.split(":")})
    ind = c[3].multiselect("Indicator", inds, default=inds)
    periods = sorted({i.period for i in insights})
    per = c[4].multiselect("Period", periods, default=periods)
    shown = filter_insights(insights, types, sev, ent, ind, per)
    st.caption(f"Showing {len(shown)} of {len(insights)} insight(s) in the analysed scope. IDs are fixed for this report.")
    frame = insights_to_frame(shown)
    if frame.empty:
        st.warning("No insights match the selected insight filters.")
    else:
        show_df(frame)
        pick = st.selectbox("Inspect an insight's evidence", [i.insight_id for i in shown], key="inspect_id")
        chosen = next(i for i in shown if i.insight_id == pick)
        st.json(chosen.to_dict())
    show_plot(charts.severity_bar(summarize_insights(shown)["severity"], "Insights by severity (currently displayed)"), "sev_bar")
    return shown


def render_trends(res):
    st.subheader("Trend analysis")
    st.caption(f"Compares consecutive available periods per district and indicator. Threshold: |change| >= {res.settings.trend_threshold:g}%. "
               "Population: the filtered rows.")
    for w in res.trend_warnings:
        st.warning(w)
    if res.trends.empty:
        st.info("No consecutive comparisons are available (need at least two reporting periods per district in the filtered rows).")
        return
    show_all = st.checkbox("Show all consecutive comparisons (not only flagged)", value=False)
    table = res.trends if show_all else flagged_trends(res.trends)
    if table.empty:
        st.info(NO_FINDINGS_MESSAGE)
    else:
        show_df(table.drop(columns=["explanation"]).assign(explanation=table["explanation"]))
        show_plot(charts.trend_bar(flagged_trends(res.trends)), "trend_bar")


def render_outliers(res):
    st.subheader("Outlier analysis")
    method = "IQR" if res.settings.outlier_method == "iqr" else "Z-score"
    st.info(f"Method: **{method}**. Bounds are computed on **{res.population_label}** ({len(res.population_df)} rows). "
            "Outliers are listed for the filtered rows. An outlier is a statistical anomaly to review, not proof of an error.")
    for n in res.outliers.notes:
        st.warning(n)
    st.markdown("**Distribution statistics and bounds per indicator**")
    show_df(res.outliers.stats)
    if res.outliers.outliers.empty:
        st.info("No outliers met the current settings in the filtered rows.")
    else:
        show_df(res.outliers.outliers)
    if res.indicators and not res.scope_df.empty:
        ind = st.selectbox("Indicator to plot", res.indicators, key="outlier_ind")
        stats_row = res.outliers.stats[res.outliers.stats["indicator"] == ind].iloc[0]
        show_plot(charts.outlier_scatter(res.scope_df, ind, res.outliers.outliers, stats_row), "outlier_plot")


def render_correlations(res):
    st.subheader("Correlation analysis")
    corr = res.correlations
    st.caption(f"Pearson correlation on {res.population_label}: {corr.n_rows} rows, {corr.n_districts} district(s), "
               f"{corr.n_periods} period(s). Threshold |r| >= {corr.threshold:.2f}. Correlation is not causation.")
    for w in corr.warnings:
        st.warning(w)
    if corr.matrix.shape[0] >= 2:
        show_plot(charts.correlation_heatmap(corr.matrix), "heatmap")
        st.markdown("**Correlation matrix** (empty / n/a = undefined, never treated as 0)")
        st.dataframe(corr.matrix.round(4))
        st.markdown("**Paired observations used per cell**")
        st.dataframe(corr.pair_counts)
    if corr.pairs.empty:
        st.info("No indicator pair met the configured correlation threshold.")
    else:
        st.markdown("**Flagged pairs**")
        show_df(corr.pairs)


def render_explorer(res):
    st.subheader("District-level indicator exploration")
    if res.scope_df.empty or not res.indicators:
        st.info("No rows or indicators in the current selection.")
        return
    ind = st.selectbox("Indicator", res.indicators, format_func=pretty_label, key="explore_ind")
    options = sorted(res.scope_df["district"].unique())
    chosen = st.multiselect("Districts to plot", options, default=options, key="explore_districts")
    show_plot(charts.district_line(res.scope_df, ind, chosen), "line_chart")
    pivot = res.scope_df[res.scope_df["district"].isin(chosen)].pivot(index="district", columns="month", values=ind)
    st.dataframe(pivot)


def render_exports(res, shown, report, overview):
    st.subheader("Export and download")
    st.caption("Each file states its scope: 'all findings' = every insight in the current sidebar scope; "
               "'displayed' = after the insight-table filters.")
    settings = json.loads(res.settings.to_json())
    c1, c2 = st.columns(2)
    c1.download_button("Insights CSV - all findings in scope", insights_to_csv_bytes(res.insights), "insights_all.csv", "text/csv", key="dl_all")
    c1.download_button("Insights CSV - displayed (filtered) only", insights_to_csv_bytes(shown), "insights_filtered.csv", "text/csv", key="dl_filtered")
    c1.download_button("Insights JSON - all findings in scope", insights_to_json(res.insights, settings=settings), "insights_all.json", "application/json", key="dl_json")
    c2.download_button("Correlation matrix CSV", correlation_matrix_to_csv_bytes(res.correlations.matrix), "correlation_matrix.csv", "text/csv", key="dl_corr")
    c2.download_button("Trend comparisons CSV (all)", frame_to_csv_bytes(res.trends), "trends_all_comparisons.csv", "text/csv", key="dl_trends")
    c2.download_button("Outliers CSV", frame_to_csv_bytes(res.outliers.outliers), "outliers.csv", "text/csv", key="dl_outliers")
    st.download_button("Data-quality report JSON", data_quality_report_json(report, overview), "data_quality_report.json", "application/json", key="dl_dq")


# ----------------------------------------------------------------------------- main
def main() -> None:
    inject_css()
    st.markdown('<p class="eq-title">EQUILYM</p><p class="eq-sub">Automated insight generation for district healthcare performance data</p>', unsafe_allow_html=True)
    content, name = sidebar_source()
    if content is None:
        st.info("Upload a district-level CSV (columns: month, district and numeric indicators) or load the sample dataset from the sidebar.")
        st.code(SAMPLE_PATH.read_text(encoding="utf-8"), language="csv")
        return
    sig = hashlib.md5(content).hexdigest()[:8]
    load = cached_load(content)
    if load.raw is None:
        for err in load.errors:
            st.error(err)
        return
    policy, exclude_oor, signed = sidebar_data_handling(content)
    clean, report = cached_validate(content, policy, exclude_oor, signed)
    st.caption(f"Dataset: **{name}** ({len(load.raw)} data rows loaded)")

    if not report.can_analyze:
        st.error("The data cannot be analysed yet. Resolve the issues below.")
        render_data_quality(load, None, report, {})
        return

    districts, month_range, selected = sidebar_filters(clean, report.indicators, sig)
    settings = sidebar_thresholds(clean, selected)
    overview = dataset_overview(clean, report.indicators)
    if not districts or not selected:
        st.warning("The current filters select zero rows or no indicators. Select at least one district and one indicator, or reset the filters.")
        render_data_quality(load, clean, report, overview)
        return
    res = cached_analysis(clean, tuple(selected), settings.to_json(), tuple(districts), month_range)

    scope_df = res.scope_df
    cols = st.columns(5)
    cols[0].metric("Rows analysed (filtered)", len(scope_df))
    cols[1].metric("Districts", scope_df["district"].nunique() if len(scope_df) else 0)
    cols[2].metric("Reporting periods", scope_df["month"].nunique() if len(scope_df) else 0)
    cols[3].metric("Insights", len(res.insights))
    cols[4].metric("High severity", sum(1 for i in res.insights if i.severity == "High"))
    if len(scope_df) and scope_df["month"].nunique() < 2:
        st.caption("Only one reporting period is in scope, so no trends are available; counts above come from outlier/correlation/threshold results only.")

    tabs = st.tabs(["Overview", "Data quality", "Insights", "Trends", "Outliers", "Correlations", "District explorer", "Export"])
    with tabs[0]:
        st.subheader("Dashboard overview")
        scope_table = pd.DataFrame([
            {"Population": "Validated dataset", "Rows": len(clean), "Used for": "Source of all filters"},
            {"Population": "Filtered rows (districts + period range)", "Rows": len(scope_df),
             "Used for": "Trends, threshold rules, displayed outliers, district charts"},
            {"Population": "Statistics population", "Rows": len(res.population_df),
             "Used for": f"Outlier bounds and correlation matrix: {res.population_label}"},
        ])
        show_df(scope_table)
        for note in res.scope_notes:
            st.caption(note)
        st.markdown(f"**Detected numeric indicators:** {', '.join(report.indicators)}")
        if report.categorical_columns:
            st.markdown(f"**Non-indicator columns:** {', '.join(report.categorical_columns)}")
        st.markdown(f"**Whole file:** {overview['rows']} validated rows x {overview['columns']} columns, "
                    f"{overview['districts']} districts, {overview['periods']} reporting periods.")
        summary = summarize_insights(res.insights)
        show_plot(charts.severity_bar(summary["severity"], "Insights by severity (all findings in scope)"), "overview_sev")
        st.write({t: n for t, n in summary["type"].items()})
    with tabs[1]:
        render_data_quality(load, clean, report, overview)
    with tabs[2]:
        shown = render_insights(res)
    with tabs[3]:
        render_trends(res)
    with tabs[4]:
        render_outliers(res)
    with tabs[5]:
        render_correlations(res)
    with tabs[6]:
        render_explorer(res)
    with tabs[7]:
        render_exports(res, shown, report, overview)


try:
    main()
except Exception as exc:  # show a friendly message instead of a traceback
    st.error("Something went wrong while processing the data. Check the file and settings, or reset the filters.")
    st.caption(f"Details: {type(exc).__name__}: {exc}")
