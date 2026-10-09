import json

import pandas as pd

from tests.helpers import sample_clean
from src.export_utils import (
    INSIGHT_COLUMNS, correlation_matrix_to_csv_bytes, frame_to_csv_bytes, insights_to_csv_bytes, insights_to_frame,
    insights_to_json, sanitize_cell, write_all_outputs,
)
from src.insight_generator import filter_insights
from src.pipeline import AnalysisSettings, run_analysis
import io


def _res():
    clean, rep = sample_clean()
    return run_analysis(clean, rep.indicators, AnalysisSettings())


def _read(data):
    return pd.read_csv(io.StringIO(data.decode("utf-8-sig")), dtype={"period": str})


def test_insights_csv_columns_order_and_values():
    res = _res()
    df = _read(insights_to_csv_bytes(res.insights))
    assert list(df.columns) == INSIGHT_COLUMNS == "insight_id,type,indicator,entity,period,value,prev_value,change,change_pct,severity,explanation".split(",")
    assert len(df) == len(res.insights) and df["insight_id"].is_unique
    row = df[(df.entity == "Ahmedabad") & (df.indicator == "anc_coverage")].iloc[0]
    assert (row.value, row.prev_value, row.change) == (69, 85, -16) and abs(row.change_pct - (-18.8235)) < 1e-3


def test_missing_fields_serialise_as_empty_cells():
    text = insights_to_csv_bytes(_res().insights).decode("utf-8-sig")
    outlier_line = next(l for l in text.splitlines() if ",outlier," in l)
    assert ",,," in outlier_line  # prev_value, change, change_pct are empty, not 'nan' / 'None'
    assert "nan" not in text.lower().replace("explanation", "") or "nan" not in outlier_line.split('"')[0]
    assert "None" not in outlier_line.split('"')[0]


def test_filtered_export_matches_scope():
    res = _res()
    subset = filter_insights(res.insights, types=["outlier"])
    df = _read(insights_to_csv_bytes(subset))
    assert set(df["type"]) == {"outlier"} and len(df) == len(subset) < len(res.insights)
    assert len(_read(insights_to_csv_bytes([]))) == 0
    assert list(_read(insights_to_csv_bytes([])).columns) == INSIGHT_COLUMNS


def test_correlation_matrix_csv_has_labels_and_round_trips():
    res = _res()
    df = pd.read_csv(io.StringIO(correlation_matrix_to_csv_bytes(res.correlations.matrix).decode("utf-8-sig")), index_col=0)
    assert list(df.index) == list(df.columns) == res.correlations.matrix.index.tolist()
    assert abs(df.loc["anc_coverage", "high_risk_cases"] - res.correlations.matrix.loc["anc_coverage", "high_risk_cases"]) < 1e-4


def test_json_is_valid_scoped_and_has_nulls_not_nan():
    res = _res()
    text = insights_to_json(res.insights, scope_label="all findings", settings={"trend_threshold": 10})
    payload = json.loads(text)
    assert payload["count"] == len(res.insights) and payload["scope"] == "all findings"
    assert "NaN" not in text
    outlier = next(i for i in payload["insights"] if i["type"] == "outlier")
    assert outlier["prev_value"] is None and "lower_bound" in outlier["supporting_data"]


def test_formula_injection_is_neutralised():
    assert sanitize_cell("=SUM(A1)") == "'=SUM(A1)" and sanitize_cell("safe") == "safe" and sanitize_cell(5) == 5
    frame = pd.DataFrame({"district": ["=cmd|' /C calc'!A0"], "v": [1.0]})
    assert frame_to_csv_bytes(frame).decode("utf-8-sig").splitlines()[1].startswith("'=")


def test_write_all_outputs(tmp_path):
    res = _res()
    files = write_all_outputs(res, tmp_path)
    names = {f.name for f in files}
    assert {"insights_all.csv", "insights_all.json", "correlation_matrix.csv"} <= names
    assert all(f.stat().st_size > 0 for f in files)
