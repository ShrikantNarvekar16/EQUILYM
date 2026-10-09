import numpy as np

from tests.helpers import csv_clean, sample_clean
from src.validation import parse_month_value


def test_sample_validates_cleanly():
    clean, rep = sample_clean()
    assert rep.can_analyze and not rep.errors and not rep.warnings
    assert rep.indicators == ["anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]
    assert clean["month"].tolist()[:2] == ["2026-07", "2026-08"]
    assert str(clean["anc_coverage"].dtype) == "float64"


def test_missing_required_column():
    clean, rep = csv_clean("district,a\nX,1\n")
    assert clean is None and not rep.can_analyze
    assert "month" in rep.errors[0].message


def test_header_only_and_duplicate_columns_are_errors():
    clean, rep = csv_clean("month,district,a\n")
    assert clean is None and rep.errors[0].code == "no_rows"
    clean, rep = csv_clean("month,month,district,a\n2026-01,2026-01,X,1\n")
    assert clean is None and rep.errors[0].code == "duplicate_columns"


def test_month_formats_normalised_and_invalid_excluded_not_hidden():
    text = "month,district,a\n2026-01,X,1\nFeb 2026,X,2\n03/2026,X,3\nnot-a-month,X,4\n2026-04,,5\n"
    clean, rep = csv_clean(text)
    assert clean["month"].tolist() == ["2026-01", "2026-02", "2026-03"]
    assert len(rep.excluded_rows) == 2
    assert set(rep.excluded_rows["exclusion_reason"]) == {"invalid or missing month", "missing district"}
    assert any(i.code == "excluded_rows" for i in rep.warnings)
    assert parse_month_value("2026-13")[0] is None


def test_strings_percent_and_thousands_become_numbers():
    clean, rep = csv_clean('month,district,a\n2026-01,X,"1,200"\n2026-02,X,85%\n2026-03,X, 7 \n')
    assert clean["a"].tolist() == [1200.0, 85.0, 7.0]


def test_non_numeric_values_reported_and_treated_missing():
    text = "month,district,a\n" + "".join(f"2026-0{m},X,{v}\n" for m, v in [(1, 5), (2, 6), (3, 7), (4, 8), (5, "oops")])
    clean, rep = csv_clean(text)
    issue = next(i for i in rep.warnings if i.code == "non_numeric_values")
    assert issue.rows == [6]
    assert np.isnan(clean["a"].iloc[-1])


def test_categorical_columns_are_not_indicators():
    clean, rep = csv_clean("month,district,region,a\n2026-01,X,North,1\n2026-02,X,North,2\n")
    assert rep.indicators == ["a"] and rep.categorical_columns == ["region"]


def test_additional_numeric_indicator_detected_dynamically():
    clean, rep = csv_clean("month,district,a,b,c\n2026-01,Zed,1,2,3\n2026-02,Zed,2,3,4\n")
    assert rep.indicators == ["a", "b", "c"]


def test_duplicates_block_until_policy_chosen():
    text = "month,district,a\n2026-01,X,10\n2026-01,X,20\n2026-02,X,30\n"
    clean, rep = csv_clean(text)
    assert clean is None and not rep.can_analyze and rep.errors[0].code == "duplicate_records"
    assert len(rep.duplicate_rows) == 2
    clean, rep = csv_clean(text, duplicate_policy="keep_last")
    assert clean["a"].tolist() == [20.0, 30.0] and any(i.code == "duplicates_resolved" for i in rep.infos)
    clean, rep = csv_clean(text, duplicate_policy="mean")
    assert clean["a"].tolist() == [15.0, 30.0]
    clean, rep = csv_clean(text, duplicate_policy="keep_first")
    assert clean["a"].tolist() == [10.0, 30.0]


def test_out_of_range_flagged_not_clipped():
    clean, rep = csv_clean("month,district,anc_coverage,cases\n2026-01,X,120,-3\n2026-02,X,50,4\n")
    codes = [i.code for i in rep.warnings]
    assert codes.count("out_of_range") == 2
    assert clean["anc_coverage"].iloc[0] == 120.0 and clean["cases"].iloc[0] == -3.0
    clean, rep = csv_clean("month,district,anc_coverage\n2026-01,X,120\n2026-02,X,50\n", exclude_out_of_range=True)
    assert np.isnan(clean["anc_coverage"].iloc[0])


def test_signed_indicator_allows_negatives():
    clean, rep = csv_clean("month,district,delta\n2026-01,X,-3\n2026-02,X,4\n", signed_indicators=["delta"])
    assert not any(i.code == "out_of_range" for i in rep.issues)


def test_missing_months_and_single_observation_reported():
    clean, rep = csv_clean("month,district,a\n2026-01,X,1\n2026-04,X,2\n2026-01,Y,3\n")
    assert any(i.code == "missing_months" and "2026-02" in i.message for i in rep.warnings)
    assert any(i.code == "single_observation" for i in rep.infos)


def test_no_numeric_indicator_is_an_error():
    clean, rep = csv_clean("month,district,note\n2026-01,X,hello\n")
    assert clean is None and rep.errors[0].code == "no_indicators"


def test_district_case_variants_warned():
    clean, rep = csv_clean("month,district,a\n2026-01,Surat,1\n2026-02,surat,2\n")
    assert any(i.code == "district_name_variants" for i in rep.warnings)
