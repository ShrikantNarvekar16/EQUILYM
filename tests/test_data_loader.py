import pytest

from src.data_loader import (
    SAMPLE_PATH, dataframe_info_text, dataset_overview, load_sample, missing_value_counts, read_csv_content,
)


def test_sample_file_matches_spec_exactly():
    lines = SAMPLE_PATH.read_text().strip().splitlines()
    assert lines[0] == "month,district,anc_coverage,institutional_delivery,immunization,high_risk_cases"
    assert len(lines) == 13
    assert lines[1] == "2026-07,Ahmedabad,85,91,93,10"
    assert lines[10] == "2026-08,Mehsana,42,88,91,28"


def test_load_sample_shape_and_info():
    res = load_sample()
    assert res.ok and res.raw.shape == (12, 6)
    assert "RangeIndex: 12 entries" in dataframe_info_text(res.raw)


def test_missing_value_counts_per_column():
    res = read_csv_content("month,district,a\n2026-01,X,1\n2026-02,,\n")
    counts = missing_value_counts(res.raw).set_index("column")
    assert counts.loc["district", "missing"] == 1 and counts.loc["a", "missing"] == 1
    assert counts.loc["month", "missing"] == 0


def test_empty_and_header_only_and_binary_and_malformed():
    assert not read_csv_content("").ok
    assert not read_csv_content("   \n").ok
    assert not read_csv_content(b"a,b\n\x00\x01").ok
    header_only = read_csv_content("month,district,a\n")
    assert header_only.raw is not None and len(header_only.raw) == 0
    bad = read_csv_content("month,district,a\n2026-01,X,1,EXTRA\n")
    assert not bad.ok and "Malformed" in bad.errors[0]


def test_duplicate_headers_are_detected_after_normalisation():
    res = read_csv_content("Month,month,district,a\n2026-01,2026-01,X,1\n")
    assert res.header_duplicates == ["month"]


def test_latin1_fallback_and_normalised_names():
    res = read_csv_content("Month, District ,Anc Coverage\n2026-01,Caf\xe9,5\n".encode("latin-1"))
    assert list(res.raw.columns) == ["month", "district", "anc_coverage"]
    assert any("Windows-1252" in n for n in res.notes)


def test_file_like_input_and_overview():
    import io

    res = read_csv_content(io.BytesIO(b"month,district,a\n2026-01,X,1\n"))
    assert res.ok
    ov = dataset_overview(res.raw, ["a"])
    assert ov["rows"] == 1 and ov["indicators"] == ["a"]
