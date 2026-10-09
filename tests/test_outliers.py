import math

import numpy as np
import pytest

from tests.helpers import frame
from src.outlier_detection import detect_outliers


def _series(values):
    return frame([(f"D{i}", "2026-01", v) for i, v in enumerate(values)])


BASE = [10, 11, 12, 13, 14, 15, 16, 17, 18, 100]


def test_known_iqr_outlier_detected_and_ordinary_values_not():
    res = detect_outliers(_series(BASE), ["ind"], "iqr")
    assert res.outliers["value"].tolist() == [100.0]
    row = res.outliers.iloc[0]
    assert row.district == "D9" and row.direction == "high" and row.method == "iqr"
    q1, q3 = np.percentile(BASE, [25, 75])
    assert row.upper_bound == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert row.lower_bound == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert "warrants review" in row.explanation and "D9" in row.explanation


def test_low_outlier_direction():
    res = detect_outliers(_series([-80] + BASE[:-1]), ["ind"], "iqr")
    assert res.outliers.iloc[0].direction == "low"


def test_changing_iqr_multiplier_changes_results():
    values = BASE[:-1] + [26]
    assert len(detect_outliers(_series(values), ["ind"], "iqr", iqr_multiplier=0.5).outliers) > len(
        detect_outliers(_series(values), ["ind"], "iqr", iqr_multiplier=3.0).outliers
    )
    assert detect_outliers(_series(values), ["ind"], "iqr", iqr_multiplier=3.0).outliers.empty


def test_zscore_method_flags_expected_value():
    values = [50] * 15 + [51] * 15 + [200]
    res = detect_outliers(_series(values), ["ind"], "zscore", z_threshold=3.0)
    assert res.outliers["value"].tolist() == [200.0]
    v = np.array(values, float)
    assert res.outliers.iloc[0].z_score == pytest.approx((200 - v.mean()) / v.std(ddof=1))
    assert detect_outliers(_series(values), ["ind"], "zscore", z_threshold=6.0).outliers.empty


def test_zscore_unreachable_threshold_is_explained():
    res = detect_outliers(_series([1, 2, 3, 4, 100]), ["ind"], "zscore", z_threshold=3.0)
    assert res.outliers.empty and any("no value can reach" in n for n in res.notes)


def test_constant_column_is_handled_without_division_errors():
    res = detect_outliers(_series([5.0] * 8), ["ind"], "zscore")
    assert res.outliers.empty and res.stats.iloc[0].status == "undefined_constant"
    assert any("undefined" in n for n in res.notes)
    res = detect_outliers(_series([5.0] * 8), ["ind"], "iqr")
    assert res.outliers.empty and res.stats.iloc[0].status == "zero_iqr"


def test_missing_values_ignored_and_too_few_observations_explained():
    values = BASE + [float("nan")] * 3
    res = detect_outliers(_series(values), ["ind"], "iqr")
    assert res.stats.iloc[0].n == 10 and len(res.outliers) == 1
    res = detect_outliers(_series([1, 2]), ["ind"], "iqr")
    assert res.outliers.empty and res.stats.iloc[0].status == "insufficient_data"


def test_population_vs_reporting_scope():
    df = _series(BASE)
    res = detect_outliers(df, ["ind"], "iqr", report_df=df.iloc[:5], population_scope="all rows")
    assert res.outliers.empty  # D9 is outside the reported rows, but bounds still use all 10
    assert res.stats.iloc[0].n == 10


def test_severity_scales_with_distance_beyond_bound():
    near = detect_outliers(_series(BASE[:-1] + [24]), ["ind"], "iqr")
    far = detect_outliers(_series(BASE), ["ind"], "iqr")
    if not near.outliers.empty:
        assert near.outliers.iloc[0].severity in ("Low", "Medium")
    assert far.outliers.iloc[0].severity == "High"


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        detect_outliers(_series(BASE), ["ind"], "median")
    with pytest.raises(ValueError):
        detect_outliers(_series(BASE), ["ind"], "iqr", iqr_multiplier=0)
