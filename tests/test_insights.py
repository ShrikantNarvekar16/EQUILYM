import math

import pytest

from tests.helpers import frame, sample_clean
from src.insight_generator import NO_FINDINGS_MESSAGE, filter_insights, generate_insights, summarize_insights
from src.pipeline import AnalysisSettings, run_analysis
from src.utils import INSIGHT_TYPES, SEVERITY_LEVELS, SeverityConfig

REQUIRED = {"insight_id", "type", "indicator", "entity", "period", "metric", "value", "prev_value", "change",
            "change_pct", "severity", "explanation", "supporting_data"}


def _sample_results(**kw):
    clean, rep = sample_clean()
    return run_analysis(clean, rep.indicators, AnalysisSettings(**kw))


def test_ids_unique_sequential_and_stable():
    a, b = _sample_results().insights, _sample_results().insights
    ids = [i.insight_id for i in a]
    assert len(ids) == len(set(ids)) and ids[0] == "INS-0001" and ids == [f"INS-{n:04d}" for n in range(1, len(ids) + 1)]
    assert ids == [i.insight_id for i in b] and [i.explanation for i in a] == [i.explanation for i in b]


def test_required_fields_types_and_severity_values():
    for ins in _sample_results().insights:
        d = ins.to_dict()
        assert REQUIRED <= set(d)
        assert d["type"] in INSIGHT_TYPES and d["severity"] in SEVERITY_LEVELS and d["explanation"]


def test_trend_insight_values_match_computed_trend():
    res = _sample_results()
    ins = next(i for i in res.insights if i.type == "trend" and i.entity == "Ahmedabad" and i.indicator == "anc_coverage")
    t = res.trends[(res.trends.district == "Ahmedabad") & (res.trends.indicator == "anc_coverage")].iloc[0]
    assert (ins.value, ins.prev_value, ins.change) == (69.0, 85.0, -16.0)
    assert ins.change_pct == pytest.approx(t["pct_change"])
    assert ins.change_pct == pytest.approx(-18.8235, abs=1e-3)
    assert ins.period == "2026-08" and ins.supporting_data["previous_period"] == "2026-07"
    assert "decreased by 18.8%" in ins.explanation and "July 2026" in ins.explanation and "10%" in ins.explanation
    assert ins.severity == "Medium"  # 18.8 / 10 = 1.88x -> Medium under the default 1.5x / 2.0x rule


def test_severity_follows_configured_boundaries():
    res = _sample_results(severity=SeverityConfig(trend_medium_mult=1.2, trend_high_mult=1.8))
    ins = next(i for i in res.insights if i.type == "trend" and i.entity == "Ahmedabad" and i.indicator == "anc_coverage")
    assert ins.severity == "High"
    res = _sample_results(trend_threshold=40.0)
    mehsana = next(i for i in res.insights if i.type == "trend" and i.entity == "Mehsana" and i.indicator == "anc_coverage")
    assert mehsana.severity == "Low"  # 50% / 40% = 1.25x threshold, below the 1.5x Medium boundary


def test_outlier_and_trend_types_remain_distinct():
    res = _sample_results()
    types = {i.type for i in res.insights}
    assert {"trend", "outlier", "correlation"} <= types
    outlier = next(i for i in res.insights if i.type == "outlier" and i.entity == "Mehsana" and i.indicator == "anc_coverage")
    assert outlier.prev_value is None and outlier.change is None and outlier.change_pct is None
    assert outlier.value == 42.0 and "interquartile-range" in outlier.explanation and "warrants review" in outlier.explanation
    row = res.outliers.outliers.query("district == 'Mehsana' and indicator == 'anc_coverage'").iloc[0]
    assert outlier.supporting_data["lower_bound"] == pytest.approx(row.lower_bound)
    assert outlier.supporting_data["upper_bound"] == pytest.approx(row.upper_bound)


def test_correlation_insight_fields_and_wording():
    res = _sample_results()
    ins = next(i for i in res.insights if i.type == "correlation" and i.indicator == "anc_coverage:high_risk_cases")
    r = res.correlations.matrix.loc["anc_coverage", "high_risk_cases"]
    assert ins.value == pytest.approx(r) and ins.supporting_data["n_pairs"] == 12
    assert ins.period == "2026-07..2026-08" and ins.prev_value is None and ins.change is None
    assert "negative" in ins.explanation and "limited sample" in ins.explanation and "does not establish causation" in ins.explanation
    assert ins.supporting_data["reliability_warning"]


def test_no_findings_does_not_fabricate():
    df = frame([("A", "2026-01", 10), ("A", "2026-02", 10.1), ("B", "2026-01", 20), ("B", "2026-02", 20.1)])
    res = run_analysis(df, ["ind"], AnalysisSettings(outlier_method="zscore"))
    assert res.insights == [] and summarize_insights(res.insights)["total"] == 0
    assert "No findings" in NO_FINDINGS_MESSAGE
    assert generate_insights() == []


def test_threshold_breach_is_a_separate_type_with_its_own_rule():
    clean, rep = sample_clean()
    off = run_analysis(clean, rep.indicators, AnalysisSettings())
    assert not any(i.type == "threshold_breach" for i in off.insights)  # a crossed trend threshold is not a breach
    on = run_analysis(clean, rep.indicators, AnalysisSettings(threshold_rules=(("immunization", 90.0, None),)))
    breaches = [i for i in on.insights if i.type == "threshold_breach"]
    assert {i.entity for i in breaches} == {"Bhavnagar"} and len(breaches) == 2
    assert all(i.value < 90 and "below the configured minimum" in i.explanation for i in breaches)


def test_summary_counts_zero_fill_and_filtering_keeps_ids():
    res = _sample_results()
    s = summarize_insights(res.insights)
    assert set(s["severity"]) == set(SEVERITY_LEVELS) and sum(s["severity"].values()) == len(res.insights)
    only_trends = filter_insights(res.insights, types=["trend"])
    assert all(i.type == "trend" for i in only_trends) and only_trends[0].insight_id == "INS-0001"
    assert filter_insights(res.insights, severities=[]) == []
    pair = filter_insights(res.insights, indicators=["high_risk_cases"])
    assert any(i.type == "correlation" for i in pair)


def test_zero_baseline_insight_has_null_pct_and_fixed_severity():
    df = frame([("A", "2026-01", 0), ("A", "2026-02", 7)])
    res = run_analysis(df, ["ind"])
    ins = next(i for i in res.insights if i.type == "trend")
    assert ins.change_pct is None and ins.change == 7 and ins.severity == "Medium"
    assert "undefined" in ins.explanation and ins.metric_name == "absolute_change"
