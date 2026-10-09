import numpy as np
import pandas as pd
import pytest

from src.correlation_analysis import compute_correlations, sample_warning


def _df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "district": [f"D{i % 20}" for i in range(n)],
            "month": ["2026-01" if i < n // 2 else "2026-02" for i in range(n)],
            "x": x, "pos": 2 * x + 1, "neg": -3 * x, "noise": rng.normal(size=n), "const": 5.0,
        }
    )


def test_matrix_matches_pandas_and_is_symmetric():
    df = _df()
    res = compute_correlations(df, ["x", "pos", "neg", "noise"], 0.7)
    expected = df[["x", "pos", "neg", "noise"]].corr()
    pd.testing.assert_frame_equal(res.matrix, expected)
    assert np.allclose(res.matrix.values, res.matrix.values.T, equal_nan=True)


def test_positive_and_negative_pairs_flagged_without_duplicates_or_self_pairs():
    res = compute_correlations(_df(), ["x", "pos", "neg", "noise"], 0.7)
    pairs = {(r.indicator_a, r.indicator_b): r for r in res.pairs.itertuples()}
    assert pairs[("x", "pos")].direction == "positive" and pairs[("x", "pos")].pearson_r == pytest.approx(1.0)
    assert pairs[("x", "neg")].direction == "negative" and pairs[("pos", "neg")].direction == "negative"
    assert all(a != b for a, b in pairs) and len({frozenset(k) for k in pairs}) == len(pairs) == 3
    assert all(r.abs_r >= 0.7 and r.meets_threshold for r in pairs.values())


def test_threshold_changes_which_pairs_are_flagged():
    df = _df()
    df["partial"] = df["x"] + np.random.default_rng(1).normal(scale=1.0, size=len(df))
    r_val = abs(df["x"].corr(df["partial"]))
    low = compute_correlations(df, ["x", "partial"], max(r_val - 0.05, 0))
    high = compute_correlations(df, ["x", "partial"], min(r_val + 0.05, 1))
    assert len(low.pairs) == 1 and high.pairs.empty


def test_constant_column_is_nan_not_zero_and_never_flagged():
    res = compute_correlations(_df(), ["x", "const"], 0.0)
    assert np.isnan(res.matrix.loc["x", "const"]) and res.pairs.empty
    assert any("Constant" in w for w in res.warnings)


def test_insufficient_data_and_single_indicator():
    tiny = pd.DataFrame({"district": ["A", "B"], "month": ["2026-01"] * 2, "a": [1.0, 2.0], "b": [2.0, 4.0]})
    res = compute_correlations(tiny, ["a", "b"], 0.7)
    assert np.isnan(res.matrix.loc["a", "b"]) and res.pairs.empty
    assert compute_correlations(tiny, ["a"], 0.7).pairs.empty
    assert compute_correlations(tiny.iloc[0:0], ["a", "b"], 0.7).pairs.empty


def test_missing_values_use_pairwise_complete_observations():
    df = _df(20)
    df.loc[:4, "pos"] = np.nan
    res = compute_correlations(df, ["x", "pos"], 0.7)
    assert res.pair_counts.loc["x", "pos"] == 15


def test_small_sample_warning_and_no_causal_claim():
    df = _df(12)
    df["district"] = [f"D{i % 6}" for i in range(12)]
    res = compute_correlations(df, ["x", "pos"], 0.7)
    assert res.warnings and "Limited sample" in res.warnings[0] and "6 district" in res.warnings[0]
    row = res.pairs.iloc[0]
    assert row.reliability_warning and "does not establish causation" in row.interpretation
    assert "causes" not in row.interpretation.lower().replace("does not establish causation", "")
    assert sample_warning(500, 50, 12) is None


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        compute_correlations(_df(), ["x", "pos"], 1.5)
