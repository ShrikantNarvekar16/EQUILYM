import pytest

from tests.helpers import csv_clean, sample_clean
from src.pipeline import AnalysisSettings, filter_scope, run_analysis


def test_sample_end_to_end_finds_ahmedabad_anc_drop():
    clean, rep = sample_clean()
    res = run_analysis(clean, rep.indicators, AnalysisSettings(trend_threshold=10.0))
    hit = [i for i in res.insights if i.type == "trend" and i.entity == "Ahmedabad" and i.indicator == "anc_coverage"]
    assert len(hit) == 1 and round(hit[0].change_pct, 1) == -18.8


def test_district_and_month_filters_affect_trends_but_global_stats_stay_wide():
    clean, rep = sample_clean()
    res = run_analysis(clean, rep.indicators, AnalysisSettings(), districts=["Surat"])
    assert set(res.trends["district"]) == {"Surat"} and len(res.scope_df) == 2 and len(res.population_df) == 12
    assert res.outliers.outliers.empty and res.correlations.n_rows == 12
    local = run_analysis(clean, rep.indicators, AnalysisSettings(stats_scope="filtered"), districts=["Surat"])
    assert local.correlations.n_rows == 2 and local.correlations.pairs.empty


def test_month_range_filter_and_empty_scope():
    clean, rep = sample_clean()
    one_month = run_analysis(clean, rep.indicators, month_range=("2026-07", "2026-07"))
    assert one_month.trends.empty
    empty = run_analysis(clean, rep.indicators, districts=[])
    assert empty.insights == [] or all(i.type == "correlation" for i in empty.insights)
    assert len(filter_scope(clean, None, None)) == 12


def test_threshold_changes_update_results():
    clean, rep = sample_clean()
    strict = run_analysis(clean, rep.indicators, AnalysisSettings(trend_threshold=50.0))
    loose = run_analysis(clean, rep.indicators, AnalysisSettings(trend_threshold=5.0))
    assert len(loose.trends[loose.trends.is_significant]) > len(strict.trends[strict.trends.is_significant])


def test_new_dataset_with_extra_indicator_and_new_district_needs_no_code_changes():
    text = "month,district,region,beds,anc_coverage\n" + "".join(
        f"2026-0{m},Zedpur,East,{b},{a}\n" for m, b, a in [(1, 50, 80), (2, 40, 60), (3, 41, 61)]
    )
    clean, rep = csv_clean(text)
    assert rep.indicators == ["beds", "anc_coverage"]
    res = run_analysis(clean, rep.indicators)
    assert any(i.entity == "Zedpur" and i.indicator == "beds" and i.type == "trend" for i in res.insights)


def test_settings_json_round_trip_and_bad_scope():
    s = AnalysisSettings(trend_threshold=15, threshold_rules=(("a", 1.0, None),))
    assert AnalysisSettings.from_json(s.to_json()) == s
    clean, rep = sample_clean()
    with pytest.raises(ValueError):
        run_analysis(clean, rep.indicators, AnalysisSettings(stats_scope="nope"))
