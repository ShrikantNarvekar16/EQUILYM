import math

import pandas as pd
import pytest

from tests.helpers import frame, sample_clean
from src.trend_detection import detect_trends, flagged_trends, percent_change, trend_warnings


def test_percent_change_formula_and_zero_baseline():
    assert percent_change(85, 69) == pytest.approx(-18.8235, abs=1e-3)
    assert percent_change(-10, -5) == pytest.approx(50.0)  # |previous| denominator
    assert math.isnan(percent_change(0, 5))


def test_ahmedabad_anc_from_sample_through_general_algorithm():
    clean, rep = sample_clean()
    t = detect_trends(clean, rep.indicators, 10.0)
    row = t[(t.district == "Ahmedabad") & (t.indicator == "anc_coverage")].iloc[0]
    assert row["change"] == -16 and row["pct_change"] == pytest.approx(-18.8235, abs=1e-3)
    assert row.direction == "decrease" and row.is_significant
    assert row.prev_period == "2026-07" and row.current_period == "2026-08"
    assert "18.8%" in row.explanation and "decreased" in row.explanation


def test_positive_and_negative_changes_and_threshold_boundary():
    df = frame([("A", "2026-01", 100), ("A", "2026-02", 110), ("B", "2026-01", 100), ("B", "2026-02", 89)])
    t = detect_trends(df, ["ind"], 10.0).set_index("district")
    assert t.loc["A", "direction"] == "increase" and t.loc["A", "is_significant"]  # exactly 10% counts
    assert t.loc["B", "direction"] == "decrease" and t.loc["B", "is_significant"]
    t2 = detect_trends(df, ["ind"], 10.5).set_index("district")
    assert not t2.loc["A", "is_significant"] and t2.loc["B", "is_significant"]
    assert pd.isna(t2.loc["A", "severity"])


def test_threshold_must_be_positive():
    with pytest.raises(ValueError):
        detect_trends(frame([("A", "2026-01", 1), ("A", "2026-02", 2)]), ["ind"], 0)


def test_zero_previous_value_is_undefined_and_flagged_for_review():
    df = frame([("A", "2026-01", 0), ("A", "2026-02", 7), ("B", "2026-01", 0), ("B", "2026-02", 0)])
    t = detect_trends(df, ["ind"]).set_index("district")
    assert math.isnan(t.loc["A", "pct_change"]) and t.loc["A", "pct_status"] == "undefined_zero_baseline"
    assert t.loc["A", "zero_baseline_review"] and not t.loc["A", "is_significant"]
    assert not t.loc["B", "zero_baseline_review"] and t.loc["B", "direction"] == "no change"
    assert len(flagged_trends(t.reset_index())) == 1


def test_single_observation_per_district_gives_no_trend():
    t = detect_trends(frame([("A", "2026-01", 1), ("B", "2026-01", 2)]), ["ind"])
    assert t.empty
    assert any("only one reporting period" in w for w in trend_warnings(frame([("A", "2026-01", 1)]), ["ind"]))


def test_missing_periods_recorded_not_hidden():
    df = frame([("A", "2026-01", 100), ("A", "2026-04", 50)])
    t = detect_trends(df, ["ind"]).iloc[0]
    assert t.gap_periods == 2 and t.spans_gap and t.missing_periods == "2026-02;2026-03"
    assert "missing reporting period" in t.explanation
    assert any("2026-02" in w for w in trend_warnings(df, ["ind"]))


def test_chronological_sorting_regardless_of_input_order():
    df = frame([("A", "2026-03", 130), ("A", "2026-01", 100), ("A", "2026-02", 120)])
    t = detect_trends(df, ["ind"])
    assert t.prev_period.tolist() == ["2026-01", "2026-02"] and t.current_period.tolist() == ["2026-02", "2026-03"]
    assert t.change.tolist() == [20, 10]


def test_missing_values_skipped_with_gap_recorded():
    df = frame([("A", "2026-01", 100), ("A", "2026-02", float("nan")), ("A", "2026-03", 150)])
    t = detect_trends(df, ["ind"])
    assert len(t) == 1 and t.iloc[0].gap_periods == 1


def test_trend_severity_uses_threshold_multiples():
    df = frame([("A", "2026-01", 100), ("A", "2026-02", 112), ("B", "2026-01", 100), ("B", "2026-02", 116),
                ("C", "2026-01", 100), ("C", "2026-02", 125)])
    t = detect_trends(df, ["ind"], 10.0).set_index("district")
    assert (t.loc["A", "severity"], t.loc["B", "severity"], t.loc["C", "severity"]) == ("Low", "Medium", "High")


def test_duplicate_records_rejected():
    with pytest.raises(ValueError):
        detect_trends(frame([("A", "2026-01", 1), ("A", "2026-01", 2)]), ["ind"])
